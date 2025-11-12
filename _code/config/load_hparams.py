# _code/config/load_hparams.py
from pathlib import Path
import yaml

def load_hparams(path: str | Path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"HParams YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
