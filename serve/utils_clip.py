import json
import logging
import os
import threading
from typing import Dict, List, Tuple

import lmdb
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from serve import global_vars
from serve.utils_general import get_from_cache, save_to_cache

_clip_cache = None
_clip_cache_path = None


def _get_clip_cache():
    global _clip_cache, _clip_cache_path
    cache_path = str(global_vars.get_clip_cache_path())
    if _clip_cache is not None and _clip_cache_path == cache_path:
        return _clip_cache

    if _clip_cache is not None:
        _clip_cache.close()
    _clip_cache_path = cache_path
    _clip_cache = lmdb.open(cache_path, map_size=int(1e11))
    return _clip_cache

_CLIP_MODELS: Dict[Tuple[str, str], Tuple[torch.nn.Module, callable, callable, str]] = {}
_CLIP_LOCK = threading.Lock()


def _load_clip_model(model_name: str, pretrained: str) -> Tuple[torch.nn.Module, callable, callable, str]:
    key = (model_name, pretrained)
    if key in _CLIP_MODELS:
        return _CLIP_MODELS[key]

    with _CLIP_LOCK:
        if key in _CLIP_MODELS:
            return _CLIP_MODELS[key]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        _CLIP_MODELS[key] = (model, preprocess, tokenizer, device)
    return _CLIP_MODELS[key]


def get_embeddings(
    inputs: List[str],
    model: str,
    modality: str,
    pretrained: str,
) -> np.ndarray:
    input_to_embeddings = {}
    for inp in inputs:
        cache_env = _get_clip_cache()
        key = json.dumps([inp, model, pretrained, modality])
        cached_value = get_from_cache(key, cache_env)
        if cached_value is not None:
            logging.debug("CLIP Cache Hit")
            input_to_embeddings[inp] = json.loads(cached_value)

    uncached_inputs = [inp for inp in inputs if inp not in input_to_embeddings]

    if uncached_inputs:
        clip_model, preprocess, tokenizer, device = _load_clip_model(model, pretrained)
        batch_embeddings = []
        batch_size = 32
        for i in range(0, len(uncached_inputs), batch_size):
            batch = uncached_inputs[i : i + batch_size]
            if modality == "image":
                images = torch.stack(
                    [preprocess(Image.open(img).convert("RGB")) for img in batch]
                ).to(device)
                with torch.inference_mode():
                    feats = clip_model.encode_image(images)
            elif modality == "text":
                text_inputs = tokenizer(batch).to(device)
                with torch.inference_mode():
                    feats = clip_model.encode_text(text_inputs)
            else:
                raise ValueError(f"Unknown modality {modality}")
            feats = F.normalize(feats, dim=-1).cpu().numpy()
            batch_embeddings.extend(feats)

        for inp, embedding in zip(uncached_inputs, batch_embeddings):
            input_to_embeddings[inp] = embedding.tolist()
            key = json.dumps([inp, model, pretrained, modality])
            cache_env = _get_clip_cache()
            save_to_cache(key, json.dumps(embedding.tolist()), cache_env)

    input_embeddings = [input_to_embeddings[inp] for inp in inputs]
    return np.array(input_embeddings)


if __name__ == "__main__":
    embeddings = get_embeddings(
        ["data/teaser.png"],
        "ViT-bigG-14",
        "image",
        "laion2b_s39b_b160k",
    )
    print(embeddings)

    embeddings = get_embeddings(
        ["haha", "hello world"], "ViT-bigG-14", "text", "laion2b_s39b_b160k"
    )
    print(embeddings)
