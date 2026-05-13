from omegaconf import OmegaConf, DictConfig


def load_config(path: str) -> DictConfig:
    """Load a YAML config file via OmegaConf."""
    return OmegaConf.load(path)
