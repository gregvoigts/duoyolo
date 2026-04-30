"""Classification trainer and validator exports for DuoYOLO."""

from duoYolo.models.yolo.classify.train import ClassificationTrainer
from duoYolo.models.yolo.classify.val import ClassificationValidator

__all__ = "ClassificationTrainer", "ClassificationValidator"