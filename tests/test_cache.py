from pathlib import Path

from serve import global_vars
from pyvisdiff import run_visdiff


def test_run_visdiff_custom_cache_dir(tmp_path, manual_overrides):
    repo_root = Path(__file__).resolve().parents[1]
    set_a = sorted((repo_root / "data" / "examples" / "set_a").glob("*.jpg"))[:2]
    set_b = sorted((repo_root / "data" / "examples" / "set_b").glob("*.jpg"))[:2]

    custom_cache = tmp_path / "visdiff-cache"
    result = run_visdiff(
        [str(path) for path in set_a],
        [str(path) for path in set_b],
        "People practicing yoga in a mountainous setting",
        "People meditating in a mountainous setting",
        config_overrides=manual_overrides,
        cache_dir=custom_cache,
    )

    assert result["ranked_hypotheses"]
    assert "evaluation" in result
    assert global_vars.get_cache_dir() == custom_cache.resolve()
    assert custom_cache.exists()

    # Reset cache dir to default to avoid leaking state between tests.
    global_vars.set_cache_dir(None)
