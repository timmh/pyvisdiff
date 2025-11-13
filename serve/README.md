# Serve Module

The helper modules in this directory are now loaded directly by the VisDiff
runtime—no standalone Flask servers are required. The previous server scripts
are retained for reference only.

- `utils_llm.py`, `utils_vlm.py`, and `utils_clip.py` lazily load the
  respective models in-process and cache their outputs on disk.
- Configuration such as the LLM host/path can be supplied via the
  `visdiff.run_visdiff` API or environment variables (see `serve/global_vars.py`).
- Optional dependencies (e.g. [LLaVA](https://github.com/haotian-liu/LLaVA))
  are only needed if you explicitly select those models in your config.

For most workflows you can ignore this directory entirely and either use the
CLI (`python main.py --config ...`) or the Python API (`visdiff.run_visdiff`).
