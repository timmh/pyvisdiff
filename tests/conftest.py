import pytest


@pytest.fixture
def manual_overrides():
    return {
        "wandb": False,
        "proposer": {
            "method": "ManualProposer",
            "hypotheses": [
                "Group A features more yoga poses than Group B",
                "Group B focuses on meditation scenes",
            ],
        },
        "ranker": {"method": "NullRanker", "max_num_samples": 4, "classify_threshold": 0.5},
        "evaluator": {"method": "NullEvaluator"},
    }
