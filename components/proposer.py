import hashlib
import json
import os
import random
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image

import components.prompts as prompts
import wandb
from serve.utils_general import save_data_diff_image
from serve.utils_llm import get_llm_output
from serve.utils_vlm import get_embed_caption_blip, get_vlm_output


def _resolve_prompt(prompt_value: str) -> str:
    """Return custom prompt text or look up predefined templates by name."""

    if isinstance(prompt_value, str) and hasattr(prompts, prompt_value):
        return getattr(prompts, prompt_value)
    return prompt_value


class Proposer:
    def __init__(self, args: Dict):
        self.args = args

    def propose(
        self, dataset1: List[Dict], dataset2: List[Dict]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        all_hypotheses = []
        all_logs = []
        all_images = []
        random.seed(self.args["seed"])
        if len(dataset1) == 0:
            raise ValueError("Dataset1 is empty")
        if len(dataset2) == 0:
            raise ValueError("Dataset2 is empty")
        for i in range(self.args["num_rounds"]):
            sampled_dataset1 = self.sample(dataset1, min(len(dataset1), self.args["num_samples"]))
            sampled_dataset2 = self.sample(dataset2, min(len(dataset2), self.args["num_samples"]))
            hypotheses, logs = self.get_hypotheses(sampled_dataset1, sampled_dataset2)
            images = self.visualize(sampled_dataset1, sampled_dataset2)
            all_hypotheses += hypotheses
            all_logs.append(logs)
            all_images.append(images)
        return all_hypotheses, all_logs, all_images

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        raise NotImplementedError

    def sample(self, dataset: List[Dict], n: int) -> List[Dict]:
        return random.sample(dataset, n)

    def visualize(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Dict:
        images1 = [
            wandb.Image(
                Image.open(item["path"]).convert("RGB").resize((224, 224)),
                caption=item.get("caption", ""),
            )
            for item in sampled_dataset1
        ]
        images2 = [
            wandb.Image(
                Image.open(item["path"]).convert("RGB").resize((224, 224)),
                caption=item.get("caption", ""),
            )
            for item in sampled_dataset2
        ]
        images = {"images_group_1": images1, "images_group_2": images2}
        return images

    def captioning(self, dataset: List[Dict]):
        for item in dataset:
            item["caption"] = get_vlm_output(
                item["path"],
                self.args["captioner"]["prompt"],
                self.args["captioner"]["model"],
            )


class LLMProposer(Proposer):
    def __init__(self, args: Dict):
        super().__init__(args)
        self.prompt = _resolve_prompt(args["prompt"])

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        self.captioning(sampled_dataset1)
        self.captioning(sampled_dataset2)
        captions1 = [
            f"Group A: {item['caption']}".replace("\n", " ").strip()
            for item in sampled_dataset1
        ]
        captions2 = [
            f"Group B: {item['caption']}".replace("\n", " ").strip()
            for item in sampled_dataset2
        ]
        caption_concat = "\n".join(captions1 + captions2)
        prompt = self.prompt.format(text=caption_concat)
        output = get_llm_output(prompt, self.args["model"])
        hypotheses = [line.replace("* ", "") for line in output.splitlines()]
        logs = {"prompt": prompt, "output": output}
        return hypotheses, logs


class LLMProposerDiffusion(LLMProposer):
    def propose(
        self, dataset1: List[Dict], dataset2: List[Dict]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        assert "prompt" in dataset1[0].keys(), "'prompt' column not in dataset"
        all_hypotheses = []
        all_logs = []
        all_images = []
        random.seed(self.args["seed"])
        for i in range(self.args["num_rounds"]):
            sampled_dataset1 = self.sample(dataset1, self.args["num_samples"])
            sampled_prompts = [
                item["prompt"] for item in sampled_dataset1
            ]  # BIG CHANGE HERE
            sampled_dataset2 = [
                item for item in dataset2 if item["prompt"] in sampled_prompts
            ]  # BIG CHANGE HERE
            sampled_dataset1 = sorted(
                sampled_dataset1, key=lambda k: k["prompt"]
            )  # BIG CHANGE HERE
            sampled_dataset2 = sorted(
                sampled_dataset2, key=lambda k: k["prompt"]
            )  # BIG CHANGE HERE
            hypotheses, logs = self.get_hypotheses(sampled_dataset1, sampled_dataset2)
            images = self.visualize(sampled_dataset1, sampled_dataset2)
            all_hypotheses += hypotheses
            all_logs.append(logs)
            all_images.append(images)
        return all_hypotheses, all_logs, all_images

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        # make sure 'prompt' is in dataset
        assert "prompt" in sampled_dataset1[0].keys(), "'prompt' column not in dataset"
        self.captioning(sampled_dataset1)
        self.captioning(sampled_dataset2)
        captions = []
        for item1, item2 in zip(sampled_dataset1, sampled_dataset2):
            assert item1["prompt"] == item2["prompt"], "Prompt mismatch"
            prompt_a = f"Group A: {item1['caption']}".replace("\n", " ").strip()
            prompt_b = f"Group B: {item2['caption']}".replace("\n", " ").strip()
            captions += [f"\nPrompt: {item1['prompt']}\n{prompt_a}\n{prompt_b}"]
        caption_concat = "\n".join(captions)
        prompt = self.prompt.format(text=caption_concat)
        output = get_llm_output(prompt, self.args["model"])
        hypotheses = [line.replace("* ", "") for line in output.splitlines()]
        logs = {"prompt": prompt, "output": output}
        return hypotheses, logs


class VLMProposer(Proposer):
    """
    Concatenate images and ask VLM to find differences
    """

    def __init__(self, args: Dict):
        super().__init__(args)
        self.prompt = _resolve_prompt(args["prompt"])

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        assert len(sampled_dataset1) == len(
            sampled_dataset2
        ), "Groups must be of equal size"
        assert len(sampled_dataset1) <= 20, "Groups must be smaller than 20"
        filenames = [item["path"] for item in sampled_dataset1 + sampled_dataset2]
        save_name = hashlib.sha256(json.dumps(filenames).encode()).hexdigest()

        image_path = f"cache/images/{save_name}.png"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        save_data_diff_image(sampled_dataset1, sampled_dataset2, image_path)
        output = get_vlm_output(image_path, self.prompt, self.args["model"])
        output = output.replace("</s>", " ").strip()  # remove </s> token for llava
        hypotheses = [line.replace("* ", "") for line in output.splitlines()]
        logs = {"image": image_path, "prompt": self.prompt, "output": output}
        return hypotheses, logs


class VLMFeatureProposer(Proposer):
    def __init__(self, args: Dict):
        super().__init__(args)

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        diff_caption = get_embed_caption_blip(sampled_dataset1, sampled_dataset2)
        logs = {"output": diff_caption}
        return diff_caption, logs


class ManualProposer(Proposer):
    """Proposer that returns a fixed list of hypotheses.

    Useful for testing/demo scenarios where no LLM should be invoked.
    """

    def __init__(self, args: Dict):
        super().__init__(args)
        hypotheses = args.get("hypotheses")
        if not hypotheses:
            raise ValueError("ManualProposer requires a non-empty 'hypotheses' list")
        self._hypotheses = hypotheses

    def propose(
        self, dataset1: List[Dict], dataset2: List[Dict]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        logs = {"source": "manual"}
        return list(self._hypotheses), [logs], []


def test_proposers():
    dataset = pd.read_csv("data/diffusion_plates.csv")
    dataset = dataset.to_dict("records")
    dataset1 = [item for item in dataset if item["set"] == "a_plate"]
    dataset2 = [item for item in dataset if item["set"] == "a_dinner_plate"]

    args = {
        "num_rounds": 2,
        "num_samples": 10,
        "num_hypotheses": 10,
        "seed": 0,
        "prompt": "CLIP_FRIENDLY",
        "model": "gpt-4",
        "captioner": {
            "prompt": "Describe this image",
            "model": "llava",
        },
    }

    proposer = LLMProposer(args)
    hypotheses, _, _ = proposer.propose(dataset1, dataset2)
    print(hypotheses)

    args = {
        "num_rounds": 2,
        "num_samples": 10,
        "num_hypotheses": 10,
        "seed": 0,
        "prompt": "VLM_PROMPT",
        "model": "llava",
    }

    proposer = VLMProposer(args)
    hypotheses, _, _ = proposer.propose(dataset1, dataset2)
    print(hypotheses)

    args = {
        "num_rounds": 1,
        "num_samples": 10,
        "seed": 0,
    }

    proposer = VLMFeatureProposer(args)
    hypotheses, _, _ = proposer.propose(dataset1, dataset2)
    print(hypotheses)


if __name__ == "__main__":
    test_proposers()
