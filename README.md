# pyvisdiff: An easy-to-use Pythion API for VisDiff

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Pytorch](https://img.shields.io/badge/Pytorch-2.1-red.svg)](https://pytorch.org/get-started/previous-versions/#v21)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

This repo is based on the [original VisDiff repository](https://github.com/Understanding-Visual-Datasets/VisDiff) for the paper: [Describing Differences in Image Sets with Natural Language](https://arxiv.org/abs/2312.02974) (**CVPR 2024 Oral**). It adds an easy-to-use Python API to enable programmatic integration of VisDiff into other codebases. All the credit belongs to the original authors, this is merely a slight adaptation of their work.

## 🚀 Installing
Run `pip install git+https://github.com/timmh/pyvisdiff.git` to install.

## 🧪 Python API

Using this repository, you can run VisDiff directly from Python without creating dataset CSV files.
The `visdiff.run_visdiff` helper loads all required models in-process and accepts lists of image
paths plus dataset descriptions:

```python
from pyvisdiff import run_visdiff

result = run_visdiff(
    dataset_a_images=["/path/to/a1.jpg", "/path/to/a2.jpg"],
    dataset_b_images=["/path/to/b1.jpg", "/path/to/b2.jpg"],
    dataset_a_description="People practicing yoga in a mountainous setting",
    dataset_b_description="People meditating in a mountainous setting",
    config_overrides={
        "proposer": {"method": "LLMProposer"},
        "ranker": {"method": "CLIPRanker"},
    },
    wandb_entity="my-team",
    wandb_project="visdiff-demo",
    llm_api_key="sk-your-key",
)
print(result["ranked_hypotheses"][0])
```

- `config_overrides` mirrors the keys in `configs/base.yaml` so you can switch
  proposer/ranker/evaluator implementations or adjust hyperparameters.
- Pass `wandb_entity` and `wandb_project` to enable logging; otherwise logging
  is disabled automatically.
- Use `llm_host`, `llm_path`, and `llm_api_key` to point at any OpenAI-compatible
  endpoint.
- Supply `cache_dir` (string or `Path`) if you want VisDiff to store cached
  embeddings/results outside of the default temporary directory.

For offline/testing scenarios set the proposer to `ManualProposer`, the ranker to
`NullRanker`, and the evaluator to `NullEvaluator`. This configuration avoids
external service calls and is what our unit tests exercise using the sample data
in `data/examples`.

## 🎯 Acknowledgements & Citation

This repo is based on the [original VisDiff repository](https://github.com/Understanding-Visual-Datasets/VisDiff) and is merely a slight adaptation of the work of the original authors. If useful, please consider starring the original repositoy and please citing the original paper as follows:
```
@inproceedings{VisDiff,
  title={Describing Differences in Image Sets with Natural Language},
  author={Dunlap, Lisa and Zhang, Yuhui and Wang, Xiaohan and Zhong, Ruiqi and Darrell, Trevor and Steinhardt, Jacob and Gonzalez, Joseph E. and Yeung-Levy, Serena},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
