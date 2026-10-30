"""Experiment configs.

Each trial YAML says which config it `extends` and lists only its `changes`, so a
trial file reads as "what changed". `extends: default` = configs/default.yaml,
`extends: T3` = experiments/T3/config.yaml. The fully resolved pipeline is saved
next to the results so every run stays reproducible on its own.
"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base, changes):
    out = copy.deepcopy(base)
    for k, v in (changes or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def config_path(name):
    return ROOT / "configs" / "default.yaml" if name == "default" else ROOT / "experiments" / name / "config.yaml"


def load_trial(path):
    """Trial YAML -> dict with the resolved `pipeline`."""
    path = Path(path)
    trial = yaml.safe_load(path.read_text())
    if "pipeline" in trial:  # configs/default.yaml is already a full pipeline
        return trial
    parent = load_trial(config_path(trial["extends"]))
    trial["pipeline"] = deep_merge(parent["pipeline"], trial.get("changes"))
    return trial


def set_by_path(cfg, dotted, value):
    """set_by_path(p, 'skin.cr', [130, 175]) -> returns a modified copy (for sweeps)."""
    out = copy.deepcopy(cfg)
    node = out
    *parents, last = dotted.split(".")
    for k in parents:
        node = node[k]
    node[last] = value
    return out
