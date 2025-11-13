import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import lmdb
import openai

from serve import global_vars
from serve.utils_general import get_from_cache, save_to_cache

logging.basicConfig(level=logging.INFO)

_llm_cache = None
_llm_cache_path = None


def _get_llm_cache():
    global _llm_cache, _llm_cache_path
    cache_path = str(global_vars.get_llm_cache_path())
    if _llm_cache is not None and _llm_cache_path == cache_path:
        return _llm_cache

    if _llm_cache is not None:
        _llm_cache.close()
    _llm_cache_path = cache_path
    _llm_cache = lmdb.open(cache_path, map_size=int(1e11))
    return _llm_cache


@dataclass
class LLMSettings:
    api_key: Optional[str] = None
    host: str = global_vars.DEFAULT_LLM_HOST
    path: str = global_vars.DEFAULT_LLM_PATH


_settings = LLMSettings()
_DEFAULT_MODEL_BASE: Dict[str, str] = {
    "gpt-3.5-turbo": "https://api.openai.com/v1",
    "gpt-4": "https://api.openai.com/v1",
    "gpt-4-vision-preview": "https://api.openai.com/v1",
    "vicuna": os.getenv("VICUNA_API_BASE", "http://localhost:8000/v1"),
}


def configure_llm(api_key: Optional[str] = None, host: Optional[str] = None, path: Optional[str] = None) -> None:
    """Override the default LLM connection parameters."""

    if api_key is not None:
        _settings.api_key = api_key
    if host is not None:
        _settings.host = host
    if path is not None:
        _settings.path = path


def _resolve_api_base(model: str) -> str:
    # If host/path were explicitly configured, respect them.
    if (_settings.host, _settings.path) != (
        global_vars.DEFAULT_LLM_HOST,
        global_vars.DEFAULT_LLM_PATH,
    ):
        base = _settings.host.rstrip("/")
        path = _settings.path if _settings.path.startswith("/") else f"/{_settings.path}"
        return f"{base}{path}"

    default = _DEFAULT_MODEL_BASE.get(model)
    if default:
        return default
    # Fall back to OpenAI-compatible endpoint.
    return f"{global_vars.DEFAULT_LLM_HOST.rstrip('/')}{global_vars.DEFAULT_LLM_PATH}"


def get_llm_api_base(model: str) -> str:
    return _resolve_api_base(model)


def ensure_openai_credentials() -> None:
    api_key = _settings.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No LLM API key available. Set the OPENAI_API_KEY environment variable "
            "or supply one via the VisDiff API."
        )
    openai.api_key = api_key


def get_llm_output(prompt: str, model: str) -> str:
    api_base = get_llm_api_base(model)
    openai.api_base = api_base

    # Always format messages as chat messages for OpenAI-compatible endpoints
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    key = json.dumps([model, messages])

    llm_cache = _get_llm_cache()
    cached_value = get_from_cache(key, llm_cache)
    if cached_value is not None:
        logging.debug("LLM Cache Hit")
        return cached_value

    for _ in range(3):
        try:
            ensure_openai_credentials()
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
            )
            response = completion["choices"][0]["message"]["content"]
            save_to_cache(key, response, llm_cache)
            return response

        except Exception as e:
            logging.error(f"LLM Error: {e}")
            continue
    return "LLM Error: Cannot get response."


def prompt_differences(captions1: List[str], captions2: List[str]) -> str:
    caption1_concat = "\n".join(
        [f"Image {i + 1}: {caption}" for i, caption in enumerate(captions1)]
    )
    caption2_concat = "\n".join(
        [f"Image {i + 1}: {caption}" for i, caption in enumerate(captions2)]
    )
    prompt = f"""Here are two groups of images:

Group 1:
```
{caption1_concat}
```

Group 2:
```
{caption2_concat}
```

What are the differences between the two groups of images?
Think carefully and summarize each difference in JSON format, such as:
```
{{"difference": several words, "rationale": group 1... while group 2...}}
```
Output JSON only. Do not include any other information.
"""
    return prompt


def get_differences(captions1: List[str], captions2: List[str], model: str) -> str:
    prompt = prompt_differences(captions1, captions2)
    differences = get_llm_output(prompt, model)
    try:
        differences = json.loads(differences)
    except Exception as e:
        logging.error(f"Difference Error: {e}")
    return differences


def test_get_llm_output():
    prompt = "hello"
    model = "gpt-4"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "gpt-3.5-turbo"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "vicuna"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")


def test_get_llm_output_parallel():
    threads = []

    for _ in range(3):
        thread = threading.Thread(target=test_get_llm_output)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def test_get_differences():
    captions1 = [
        "A cat is sitting on a table",
        "A dog is sitting on a table",
        "A pig is sitting on a table",
    ]
    captions2 = [
        "A cat is sitting on the floor",
        "A dog is sitting on the floor",
        "A pig is sitting on the floor",
    ]
    differences = get_differences(captions1, captions2, "gpt-4")
    print(f"{differences=}")


if __name__ == "__main__":
    test_get_llm_output()
    test_get_llm_output_parallel()
    test_get_differences()
