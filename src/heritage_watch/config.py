"""Validated, portable site configuration; raster metadata supplies the CRS."""
from dataclasses import dataclass, field, fields
import os
from pathlib import Path
import re

import yaml


@dataclass
class SiteConfig:
    imagery_dir: Path
    annotation_csv: Path
    scene_date_regex: str
    classes: list[str]
    drop_categories: list[str] = field(default_factory=list)
    columns: dict[str, str] = field(default_factory=lambda: {
        "fid": "fid", "category": "category", "layer": "layer",
        "year": "year", "x": "X ", "y": "Y"})
    pairing: str = "month"
    tolerance_days: int = 75
    chip_size: int = 64
    blocks: int = 6
    folds: int = 5
    replicates: int = 8
    min_effect: float = 0.02

    def __post_init__(self):
        for name in ("imagery_dir", "annotation_csv"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value):
                raise ValueError(f"{name} must be a nonempty path")
            setattr(self, name, Path(value))
        if self.pairing not in ("month", "year"):
            raise ValueError("pairing must be 'month' or 'year'")
        try:
            pattern = re.compile(self.scene_date_regex)
        except (re.error, TypeError) as exc:
            raise ValueError(f"invalid scene_date_regex: {exc}") from exc
        if pattern.groups != 3:
            raise ValueError("scene_date_regex must have three capture groups: year, month, day")
        for name in ("classes", "drop_categories"):
            value = getattr(self, name)
            if not isinstance(value, list) or any(not isinstance(c, str) or not c.strip() for c in value):
                raise ValueError(f"{name} must be a list of nonempty strings")
            if len(set(value)) != len(value):
                raise ValueError(f"{name} contains duplicates")
        if len(self.classes) < 2:
            raise ValueError("classes must contain at least two distinct labels")
        if set(self.classes) & set(self.drop_categories):
            raise ValueError("classes and drop_categories must be disjoint")
        required = {"fid", "category", "layer", "year", "x", "y"}
        if (not isinstance(self.columns, dict) or set(self.columns) != required
                or any(not isinstance(v, str) or not v for v in self.columns.values())
                or len(set(self.columns.values())) != len(required)):
            raise ValueError(f"columns must map exactly {sorted(required)} to distinct CSV headers")
        for name, minimum in (("chip_size", 1), ("blocks", 1), ("folds", 2),
                              ("replicates", 2), ("tolerance_days", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.min_effect) not in (float, int) or not 0 < self.min_effect <= 1:
            raise ValueError("min_effect must be a number in (0, 1]")

    def evaluation_kwargs(self):
        return dict(classes=self.classes, blocks=self.blocks, folds=self.folds,
                    replicates=self.replicates)


def _expand(value: str) -> str:
    def replace(match):
        name, default = match.group(1), match.group(2)
        result = os.environ.get(name, default)
        if result is None:
            raise ValueError(f"set environment variable {name} used in the site config")
        return result
    return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}", replace, value)


def load_config(path) -> SiteConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("site config must be a YAML mapping")
    unknown = set(raw) - {f.name for f in fields(SiteConfig)}
    if unknown:
        raise ValueError(f"unknown site config fields: {sorted(unknown)}")
    for name in ("imagery_dir", "annotation_csv"):
        if name in raw and isinstance(raw[name], str):
            p = Path(_expand(raw[name])).expanduser()
            raw[name] = p if p.is_absolute() else (path.parent / p).resolve()
    try:
        return SiteConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"invalid site config: {exc}") from exc
