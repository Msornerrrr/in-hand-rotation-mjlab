"""RSL-RL config helpers shared across scripts and reset-time policy loading."""

from __future__ import annotations

from dataclasses import asdict


def prepare_agent_cfg(agent_cfg) -> dict:
    """Convert dataclass config to runner dict and drop unused model fields."""
    cfg_dict = asdict(agent_cfg)
    for model_key in ("actor", "critic"):
        model_cfg = cfg_dict.get(model_key)
        if isinstance(model_cfg, dict):
            class_name = model_cfg.get("class_name", "MLPModel")
            if class_name != "CNNModel":
                model_cfg.pop("cnn_cfg", None)
            if class_name != "RNNModel":
                model_cfg.pop("rnn_type", None)
                model_cfg.pop("rnn_hidden_dim", None)
                model_cfg.pop("rnn_num_layers", None)
    return cfg_dict
