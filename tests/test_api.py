from pathlib import Path

from pyvisdiff import run_visdiff


def test_run_visdiff_manual_proposer(manual_overrides):
    repo_root = Path(__file__).resolve().parents[1]
    set_a = sorted((repo_root / "data" / "examples" / "set_a").glob("*.jpg"))[:4]
    set_b = sorted((repo_root / "data" / "examples" / "set_b").glob("*.jpg"))[:4]

    result = run_visdiff(
        [str(path) for path in set_a],
        [str(path) for path in set_b],
        "People practicing yoga in a mountainous setting",
        "People meditating in a mountainous setting",
        config_overrides=manual_overrides,
    )

    assert "ranked_hypotheses" in result
    top_hyp = result["ranked_hypotheses"][0]
    assert top_hyp["hypothesis"].startswith("Group A features more yoga")
    assert "evaluation" in result
    eval_block = result["evaluation"]
    assert "summary" in eval_block
    assert "details" in eval_block
