"""Core helpers for running the VisDiff pipeline."""

from __future__ import annotations

import logging
from importlib import resources
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from omegaconf import OmegaConf

import wandb
from components.evaluator import GPTEvaluator, NullEvaluator
from components.proposer import (
    LLMProposer,
    LLMProposerDiffusion,
    ManualProposer,
    VLMFeatureProposer,
    VLMProposer,
)
from components.ranker import CLIPRanker, LLMRanker, NullRanker, VLMRanker


def _load_base_config() -> OmegaConf:
    with resources.files("configs").joinpath("base.yaml").open("r") as handle:
        return OmegaConf.load(handle)


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict] = None,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    disable_wandb_if_missing: bool = False,
) -> Dict:
    base_cfg = _load_base_config()
    merged_cfg = base_cfg
    if config_path:
        cfg = OmegaConf.load(config_path)
        merged_cfg = OmegaConf.merge(base_cfg, cfg)
    if overrides:
        merged_cfg = OmegaConf.merge(merged_cfg, OmegaConf.create(overrides))

    args = OmegaConf.to_container(merged_cfg, resolve=True)
    if wandb_entity is not None:
        args["entity"] = wandb_entity
    if wandb_project is not None:
        args["project"] = wandb_project
    if disable_wandb_if_missing and (not wandb_entity or not wandb_project):
        args["wandb"] = False
    args["config"] = config_path
    return args


def load_data_from_csv(args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    data_args = args["data"]
    df = pd.read_csv(f"{data_args['root']}/{data_args['name']}.csv")

    if data_args.get("subset"):
        old_len = len(df)
        df = df[df["subset"] == data_args["subset"]]
        logging.info(
            "Taking %s subset (dataset size reduced from %s to %s)",
            data_args["subset"],
            old_len,
            len(df),
        )

    dataset1 = df[df["group_name"] == data_args["group1"]].to_dict("records")
    dataset2 = df[df["group_name"] == data_args["group2"]].to_dict("records")
    group_names = [data_args["group1"], data_args["group2"]]

    purity = data_args.get("purity", 1)
    if purity < 1:
        logging.warning("Purity is set to %s. Swapping groups.", purity)
        assert len(dataset1) == len(dataset2), "Groups must be of equal size"
        n_swap = int((1 - purity) * len(dataset1))
        dataset1 = dataset1[n_swap:] + dataset2[:n_swap]
        dataset2 = dataset2[n_swap:] + dataset1[:n_swap]
    return dataset1, dataset2, group_names


def _maybe_init_wandb(args: Dict) -> None:
    if not args.get("wandb"):
        return
    wandb.init(
        project=args.get("project"),
        entity=args.get("entity"),
        dir=args.get("wandb_dir"),
        name=args["data"].get("name"),
        group=f"{args['data']['group1']} - {args['data']['group2']} ({args['data'].get('purity', 1.0)})",
        config=args,
    )


def propose(args: Dict, dataset1: List[Dict], dataset2: List[Dict]) -> List[str]:
    proposer_args = args["proposer"]
    proposer_args["seed"] = args["seed"]
    proposer_args["captioner"] = args["captioner"]

    proposer = eval(proposer_args["method"])(proposer_args)
    hypotheses, logs, images = proposer.propose(dataset1, dataset2)
    if args.get("wandb"):
        wandb.log({"logs": wandb.Table(dataframe=pd.DataFrame(logs))})
        for i in range(len(images)):
            wandb.log(
                {
                    f"group 1 images ({dataset1[0]['group_name']})": images[i][
                        "images_group_1"
                    ],
                    f"group 2 images ({dataset2[0]['group_name']})": images[i][
                        "images_group_2"
                    ],
                }
            )
    return hypotheses


def rank(
    args: Dict,
    hypotheses: List[str],
    dataset1: List[Dict],
    dataset2: List[Dict],
    group_names: List[str],
) -> List[Dict]:
    ranker_args = args["ranker"]
    ranker_args["seed"] = args["seed"]

    ranker = eval(ranker_args["method"])(ranker_args)

    scored_hypotheses = ranker.rerank_hypotheses(hypotheses, dataset1, dataset2)
    if args.get("wandb"):
        table_hypotheses = wandb.Table(dataframe=pd.DataFrame(scored_hypotheses))
        wandb.log({"scored hypotheses": table_hypotheses})
        for i in range(min(5, len(scored_hypotheses))):
            wandb.summary[f"top_{i + 1}_difference"] = scored_hypotheses[i][
                "hypothesis"
            ].replace('"', "")
            wandb.summary[f"top_{i + 1}_score"] = scored_hypotheses[i]["auroc"]

    scored_groundtruth = ranker.rerank_hypotheses(
        group_names,
        dataset1,
        dataset2,
    )
    if args.get("wandb"):
        table_groundtruth = wandb.Table(dataframe=pd.DataFrame(scored_groundtruth))
        wandb.log({"scored groundtruth": table_groundtruth})

    return scored_hypotheses


def evaluate(args: Dict, ranked_hypotheses: List[str], group_names: List[str]) -> Dict:
    evaluator_args = args["evaluator"]

    evaluator = eval(evaluator_args["method"])(evaluator_args)

    metrics, evaluated_hypotheses = evaluator.evaluate(
        ranked_hypotheses,
        group_names[0],
        group_names[1],
    )

    evaluation_result = {"summary": metrics, "details": evaluated_hypotheses}

    if args.get("wandb") and evaluator_args["method"] != "NullEvaluator":
        table_evaluated_hypotheses = wandb.Table(
            dataframe=pd.DataFrame(evaluated_hypotheses)
        )
        wandb.log({"evaluated hypotheses": table_evaluated_hypotheses})
        wandb.log(metrics)
    return evaluation_result


def run_pipeline(
    args: Dict,
    dataset1: List[Dict],
    dataset2: List[Dict],
    group_names: List[str],
) -> Tuple[List[str], Dict]:
    _maybe_init_wandb(args)
    hypotheses = propose(args, dataset1, dataset2)
    ranked_hypotheses = rank(args, hypotheses, dataset1, dataset2, group_names)
    evaluation = evaluate(
        args,
        [entry["hypothesis"] for entry in ranked_hypotheses],
        group_names,
    )
    return ranked_hypotheses, evaluation


def build_dataset_records(
    image_paths: Sequence[str],
    group_name: str,
    group_description: Optional[str] = None,
) -> List[Dict]:
    records = []
    for path in image_paths:
        records.append(
            {
                "path": path,
                "group_name": group_name,
                "group_description": group_description or group_name,
            }
        )
    return records
