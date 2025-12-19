# src/config_loader.py
import yaml
from pathlib import Path

def load_config(path):
    """
    Loads YAML config at `path`. If the config contains `include_env`,
    it loads that file and stores it under key '_env'. Relative paths resolved.
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())

    if "include_env" in cfg:
        env_path = Path(cfg["include_env"])
        # resolve relative to the config file location
        if not env_path.is_absolute():
            env_path = path.parent.joinpath(env_path)
        env = yaml.safe_load(env_path.read_text())
        cfg["_env"] = env
    else:
        # if no include, allow top-level env keys
        cfg["_env"] = cfg.get("_env", {})
    return cfg
