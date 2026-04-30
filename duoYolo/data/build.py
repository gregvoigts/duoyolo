"""Dataset builders for multitask models."""

from typing import Any
from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.utils import colorstr

from duoYolo.data.dataset import ClassificationDataset, DuoYOLODataset

def build_duoYolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
):
    """Build DuoYOLO multitask dataset from configuration."""
    return DuoYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )

def build_classify_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
):
    """Build classification dataset from configuration."""
    return ClassificationDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        augment=mode == "train",  # augmentation
        hyp=cfg,  
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )