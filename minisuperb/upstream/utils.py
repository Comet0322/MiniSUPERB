import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass, is_dataclass

import torch

from minisuperb.util.download import _urls_to_filepaths

logger = logging.getLogger(__name__)


def load_fairseq_ckpt(source: str, **override):
    from fairseq.checkpoint_utils import load_checkpoint_to_cpu
    from omegaconf import OmegaConf

    source = str(source)
    if source.startswith("http"):
        fairseq_path = _urls_to_filepaths(source)
    else:
        fairseq_path = source

    state = load_checkpoint_to_cpu(fairseq_path, arg_overrides=override)
    cfg = OmegaConf.to_container(state["cfg"])

    assert type(cfg) == dict
    return state, cfg


def merge_with_parent(dc: dataclass, cfg: dict):

    assert is_dataclass(dc)
    assert type(cfg) == dict
    cfg = deepcopy(cfg)

    def fix_cfg(cfg):
        target_keys = set(dc.__dataclass_fields__.keys())
        for k in list(cfg.keys()):
            if k not in target_keys:
                del cfg[k]

    fix_cfg(cfg)
    assert len(cfg) > 0
    return dc(**cfg)

