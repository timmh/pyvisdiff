"""Public API for programmatic access to VisDiff."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from serve.utils_llm import configure_llm
from serve import global_vars

from .pipeline import build_dataset_records, load_config, run_pipeline


def run_visdiff(
    dataset_a_images: Sequence[str],
    dataset_b_images: Sequence[str],
    dataset_a_description: str,
    dataset_b_description: str,
    config_overrides: Optional[Dict] = None,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_dir: Optional[Union[str, Path]] = None,
    llm_api_key: Optional[str] = None,
    llm_host: Optional[str] = None,
    llm_path: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    """Run VisDiff on two in-memory datasets.

    Args:
        dataset_a_images: Iterable of file paths for dataset A.
        dataset_b_images: Iterable of file paths for dataset B.
        dataset_a_description: Natural language description for dataset A.
        dataset_b_description: Natural language description for dataset B.
        config_overrides: Optional dictionary merged into the base config.
        wandb_entity: Entity to use for Weights & Biases logging.
        wandb_project: Project to use for Weights & Biases logging.
        wandb_dir: Directory passed to `wandb.init(dir=...)`.
        llm_api_key: API key for the LLM provider.
        llm_host: Host for the LLM provider (e.g. https://api.openai.com).
        llm_path: API path for the LLM provider (e.g. /v1).
        cache_dir: Directory for VisDiff caches. Accepts str or `pathlib.Path`.

    Returns:
        Dictionary with ranked hypotheses and evaluation metrics.
    """

    if cache_dir is not None:
        global_vars.set_cache_dir(cache_dir)

    configure_llm(api_key=llm_api_key, host=llm_host, path=llm_path)

    args = load_config(
        overrides=config_overrides,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        disable_wandb_if_missing=True,
    )

    args.setdefault("data", {})
    args["data"]["group1"] = dataset_a_description
    args["data"]["group2"] = dataset_b_description
    args["data"].setdefault("name", "Custom Dataset")
    if wandb_dir is not None:
        args["wandb_dir"] = str(Path(wandb_dir))

    dataset1 = build_dataset_records(dataset_a_images, dataset_a_description, dataset_a_description)
    dataset2 = build_dataset_records(dataset_b_images, dataset_b_description, dataset_b_description)

    ranked_hypotheses, metrics = run_pipeline(
        args,
        dataset1,
        dataset2,
        [dataset_a_description, dataset_b_description],
    )
    return {"ranked_hypotheses": ranked_hypotheses, "metrics": metrics}
