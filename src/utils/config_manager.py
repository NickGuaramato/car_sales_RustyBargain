#config_manager
import yaml
from pathlib import Path

from typing import Dict, Any

import os

def load_config(file_name: str) -> Dict[str, Any]:
    config_path = Path(__file__).parent.parent.parent / "config" / file_name
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    return config

def load_paths() -> Dict[str, Path]:
    """Devuelve diccionario de PATH objects"""
    config = load_config("paths.yaml")
    return {
        "dirs": {k : Path(v) for k, v in config["dirs"].items()},
        "files": {k: Path(v) for k, v in config["files"].items()}
    }

def load_params() -> Dict[str, Any]:
    "Devuelve parametros"
    return load_config("params.yaml")

def get_config() -> Dict[str, Any]:
    return {"paths": load_paths(),
            "params": load_params()
            }