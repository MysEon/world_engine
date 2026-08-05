from __future__ import annotations

from pathlib import Path

import yaml


def load_preset(path: str | Path) -> dict:
    """Load a world preset YAML file into a dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"预设文件不存在: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"预设文件格式错误: {p}")
    return data
