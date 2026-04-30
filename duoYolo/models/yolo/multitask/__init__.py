"""Multitask YOLO trainer, validator, and predictor exports."""

from duoYolo.models.yolo.multitask.predict import MultitaskPredictor
from duoYolo.models.yolo.multitask.train import MultitaskTrainer
from duoYolo.models.yolo.multitask.val import MultitaskValidator

__all__ = "MultitaskPredictor", "MultitaskTrainer", "MultitaskValidator"