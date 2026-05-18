import base64
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import lmdb
import torch
from PIL import Image

import openai

from serve import global_vars
from serve.utils_general import get_from_cache, save_to_cache
from serve.utils_llm import ensure_openai_credentials, get_llm_api_base

logging.basicConfig(level=logging.INFO)

_vlm_cache: Optional[lmdb.Environment] = None
_vlm_cache_path: Optional[str] = None


def _get_vlm_cache() -> lmdb.Environment:
    global _vlm_cache, _vlm_cache_path
    cache_path = str(global_vars.get_vlm_cache_path())
    if _vlm_cache is not None and _vlm_cache_path == cache_path:
        return _vlm_cache

    if _vlm_cache is not None:
        _vlm_cache.close()
    _vlm_cache_path = cache_path
    _vlm_cache = lmdb.open(cache_path, map_size=int(1e11))
    return _vlm_cache


@dataclass
class _BlipRuntime:
    model: torch.nn.Module
    vis_processors: Dict[str, callable]
    device: torch.device


_blip_captioner: Optional[_BlipRuntime] = None
_blip_feature: Optional[_BlipRuntime] = None
_blip_captioner_lock = threading.Lock()
_blip_feature_lock = threading.Lock()


def _load_blip_captioner() -> _BlipRuntime:
    global _blip_captioner
    if _blip_captioner is not None:
        return _blip_captioner
    with _blip_captioner_lock:
        if _blip_captioner is not None:
            return _blip_captioner
        try:
            from lavis.models import load_model_and_preprocess
        except ImportError as exc:
            raise RuntimeError(
                "BLIP captioning requires salesforce-lavis. "
                "Install it or use a non-BLIP captioner model."
            ) from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
        )
        _blip_captioner = _BlipRuntime(model=model, vis_processors=vis_processors, device=device)
    return _blip_captioner


def _load_blip_feature_model() -> _BlipRuntime:
    global _blip_feature
    if _blip_feature is not None:
        return _blip_feature
    with _blip_feature_lock:
        if _blip_feature is not None:
            return _blip_feature
        try:
            from lavis.models import load_model_and_preprocess
        except ImportError as exc:
            raise RuntimeError(
                "BLIP feature proposer requires salesforce-lavis. "
                "Install it or disable the BLIP feature proposer."
            ) from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
        )
        _blip_feature = _BlipRuntime(model=model, vis_processors=vis_processors, device=device)
    return _blip_feature


def _prepare_image_tensor(image_path: str, runtime: _BlipRuntime) -> torch.Tensor:
    raw_image = Image.open(image_path).convert("RGB")
    tensor = runtime.vis_processors["eval"](raw_image).unsqueeze(0).to(runtime.device)
    return tensor


def get_embed_caption_blip(
    sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
) -> List[str]:
    runtime = _load_blip_feature_model()
    key = json.dumps([sampled_dataset1, sampled_dataset2, 1])
    cache_env = _get_vlm_cache()
    cached_value = get_from_cache(key, cache_env)
    if cached_value is not None:
        logging.debug("VLM Cache Hit")
        return json.loads(cached_value)

    embeds1 = []
    embeds2 = []
    with torch.inference_mode():
        for item1, item2 in zip(sampled_dataset1, sampled_dataset2):
            image1 = _prepare_image_tensor(item1["path"], runtime)
            image2 = _prepare_image_tensor(item2["path"], runtime)
            embeds1.append(runtime.model.embed_image({"image": image1}))
            embeds2.append(runtime.model.embed_image({"image": image2}))

    mean_embeds1 = torch.mean(torch.stack(embeds1), dim=0)
    mean_embeds2 = torch.mean(torch.stack(embeds2), dim=0)
    dif_embed = mean_embeds1 - mean_embeds2

    # Use a blank canvas to decode the difference embedding into text.
    blank_image = Image.new("RGB", (256, 256), color=(0, 0, 0))
    ex_image = runtime.vis_processors["eval"](blank_image).unsqueeze(0).to(runtime.device)

    with torch.inference_mode():
        outputs = [
            runtime.model.generate({"image": ex_image}, image_embeds=dif_embed)[0]
            for _ in range(10)
        ]

    cache_env = _get_vlm_cache()
    save_to_cache(key, json.dumps(outputs), cache_env)
    return outputs


def _get_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


@dataclass
class _LLaVAArgs:
    model_path: str = os.getenv("VISDIFF_LLAVA_MODEL", "liuhaotian/llava-v1.5-13b")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temperature: float = 0.2
    max_new_tokens: int = 512
    image_aspect_ratio: str = "pad"


_llava_state: Optional[Tuple] = None
_llava_lock = threading.Lock()


def _load_llava_runtime() -> Tuple:
    global _llava_state
    if _llava_state is not None:
        return _llava_state
    with _llava_lock:
        if _llava_state is not None:
            return _llava_state
        args = _LLaVAArgs()
        llava_path = global_vars.LLAVA_CODE_PATH
        if os.path.isdir(llava_path) and llava_path not in sys.path:
            sys.path.append(llava_path)
        try:
            from llava.constants import (
                DEFAULT_IM_END_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IMAGE_TOKEN,
                IMAGE_TOKEN_INDEX,
            )
            from llava.conversation import SeparatorStyle, conv_templates
            from llava.mm_utils import (
                KeywordsStoppingCriteria,
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
            )
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
        except ImportError as exc:
            raise RuntimeError(
                "LLaVA support requires the llava package. Install it or disable the LLaVA option."
            ) from exc

        disable_torch_init()
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, None, model_name, False, False, device=args.device
        )
        _llava_state = (
            args,
            tokenizer,
            model,
            image_processor,
            context_len,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
            SeparatorStyle,
            conv_templates,
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
    return _llava_state


def _run_llava(image_path: str, prompt: str) -> str:
    (
        args,
        tokenizer,
        model,
        image_processor,
        _context_len,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
        SeparatorStyle,
        conv_templates,
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    ) = _load_llava_runtime()

    model_name = get_model_name_from_path(args.model_path)
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ("user", "assistant")
    else:
        roles = conv.roles

    raw_image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([raw_image], image_processor, args)
    if isinstance(image_tensor, list):
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = prompt
    if model.config.mm_use_im_start_end:
        inp = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + inp
        )
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + inp

    conv.append_message(roles[0], inp)
    conv.append_message(roles[1], None)
    conv_prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(
            conv_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(model.device)
    )
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
    return outputs


def _run_blip(image_path: str, prompt: str) -> str:
    runtime = _load_blip_captioner()
    image_tensor = _prepare_image_tensor(image_path, runtime)
    with torch.inference_mode():
        result = runtime.model.generate({"image": image_tensor, "prompt": prompt})[0]
    return result


def _run_openai_vision(image_path: str, prompt: str, model: str) -> str:
    ensure_openai_credentials()
    openai.api_base = get_llm_api_base(model)
    base64_image = _get_image_base64(image_path)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }
    completion = openai.ChatCompletion.create(**payload)
    return completion["choices"][0]["message"]["content"]


def _is_openai_vision_model(model: str) -> bool:
    prefixes = (
        "gpt-4o",
        "gpt-4.1",
        "gpt-4.5",
        "gpt-5",
    )
    return model == "gpt-4-vision-preview" or model.startswith(prefixes)


def get_vlm_output(image: str, prompt: str, model: str) -> str:
    key = json.dumps([model, image, prompt])
    cache_env = _get_vlm_cache()
    cached_value = get_from_cache(key, cache_env)
    if cached_value is not None:
        logging.debug("VLM Cache Hit")
        return cached_value

    if model == "blip":
        output = _run_blip(image, prompt)
    elif model == "llava":
        output = _run_llava(image, prompt)
    elif _is_openai_vision_model(model):
        output = _run_openai_vision(image, prompt, model)
    else:
        raise NotImplementedError(f"VLM model {model} not implemented.")

    cache_env = _get_vlm_cache()
    save_to_cache(key, output, cache_env)
    return output


def captioning(image: str, model: str) -> str:
    caption = get_vlm_output(image, "Describe this image in detail.", model)
    return caption


def vqa(image: str, question: str, model: str) -> str:
    answer = get_vlm_output(image, question, model)
    return answer


def test_get_vlm_output():
    image = "data/teaser.png"
    model = "blip"

    caption = captioning(image, model)
    print(f"{caption=}")
    question = "Is there a table in the image?"
    answer = vqa(image, question, model)
    print(f"{answer=}")


def test_get_vlm_output_parallel():
    threads = []

    for _ in range(3):
        thread = threading.Thread(target=test_get_vlm_output)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    test_get_vlm_output()
    # test_get_vlm_output_parallel()
