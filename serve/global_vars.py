"""Global configuration paths and defaults for the VisDiff runtime."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Union


def _prepare_cache_dir(path: Union[str, Path]) -> Path:
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


_ENV_CACHE_DIR = os.getenv("VISDIFF_CACHE_DIR")
_DEFAULT_CACHE_DIR = (
    Path(_ENV_CACHE_DIR)
    if _ENV_CACHE_DIR
    else Path(tempfile.gettempdir()) / "visdiff-cache"
)
_DEFAULT_CACHE_DIR = _prepare_cache_dir(_DEFAULT_CACHE_DIR)
_CACHE_DIR = _DEFAULT_CACHE_DIR
DEFAULT_CACHE_DIR = _CACHE_DIR


def set_cache_dir(cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Update the cache directory used across the runtime."""

    global _CACHE_DIR
    target = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
    _CACHE_DIR = _prepare_cache_dir(target)
    return _CACHE_DIR


def get_cache_dir() -> Path:
    return _CACHE_DIR


def _subdir(name: str) -> Path:
    path = get_cache_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_llm_cache_path() -> Path:
    return _subdir("cache_llm")


def get_vlm_cache_path() -> Path:
    return _subdir("cache_vlm")


def get_clip_cache_path() -> Path:
    return _subdir("cache_clip")


# Default LLM host/path used for OpenAI-compatible APIs. These can be
# overridden via the public API.
DEFAULT_LLM_HOST = os.getenv("VISDIFF_LLM_HOST", "https://api.openai.com")
DEFAULT_LLM_PATH = os.getenv("VISDIFF_LLM_PATH", "/v1")

# Optional path to a local LLaVA checkout. Only required if users enable LLaVA
# based captioning or ranking.
LLAVA_CODE_PATH = os.getenv("VISDIFF_LLAVA_PATH", "./LLaVA")
