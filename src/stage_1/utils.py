import json
import os
import random
from typing import Sequence

import numpy as np
import rich.syntax
import rich.tree
import torch
from omegaconf import DictConfig, OmegaConf


class JsonlReader:
    '''
    Class to help load JSONL files
    '''

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        with open(self.fname, encoding="utf-8", errors="ignore") as rf:
            for jsonl in rf:
                jsonl = jsonl.strip()
                if not jsonl:
                    continue
                yield json.loads(jsonl)


def target_directory(proj_root_dir, model_path_datetime, model_type='roberta'):
    target_log_dir = os.path.join(proj_root_dir, 'logs', 'experiments', 'stage_3_doc_re', model_path_datetime)
    best_model_path = os.path.join(target_log_dir, 'ce_checkpoint', f'{model_type}_best.bin')
    return target_log_dir, best_model_path


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)


def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
                "trainer",
                "model",
                "name",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)
