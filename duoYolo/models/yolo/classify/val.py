"""Classification validator with DuoYOLO dataset and metrics integration."""

from typing import Any

import torch

from duoYolo.data.build import build_classify_dataset
from duoYolo.engine.validator import DuoYoloValidatorMixin
from duoYolo.utils.metrics import UpdatedClassifyMetrics
from ultralytics.models.yolo.classify.val import ClassificationValidator as BaseClassificationValidator
from ultralytics.utils import LOGGER

class ClassificationValidator(DuoYoloValidatorMixin, BaseClassificationValidator):
    """
    Overrides the ultralytics Classification Validator to use the Custom Dataset which allows for class labels in txt files
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize ClassificationValidator with dataloader, save directory, and other parameters.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (str | Path, optional): Directory to save results.
            args (dict, optional): Arguments containing model and validation configuration.
            _callbacks (list, optional): List of callback functions to be called during validation.

        Examples:
            >>> from ultralytics.models.yolo.classify import ClassificationValidator
            >>> args = dict(model="yolo11n-cls.pt", data="imagenet10")
            >>> validator = ClassificationValidator(args=args)
            >>> validator()
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.metrics = UpdatedClassifyMetrics()

    def get_desc(self) -> str:
        """Return a formatted string summarizing classification metrics."""
        return ("%22s" + "%11s" * 3) % ("classes", "top1_acc", "top5_acc", "bacc(recall)")

    def build_dataset(self, img_path: str, mode: str = "val", batch=None):
        return build_classify_dataset(self.args, img_path, self.data, stride=self.stride)

    def get_stats(self) -> dict[str, Any]:
        """
        Calculate and return metrics statistics.

        Returns:
            (dict[str, Any]): Dictionary containing metrics results.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (torch.Tensor): Model predictions for the current batch.
            batch (dict[str, Any]): Batch data containing ground truth.
        """        
        n5 = min(len(self.names), 5)
        preds = preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu()
        targets = batch["cls"].type(torch.int32).cpu()

        self.pred.append(preds)
        self.targets.append(targets)            
        # update metrics after all tasks are processed
        self.metrics.update_stats({
            "pred": [preds],
            "target": [targets]
        })

    def print_results(self) -> None:
        """Print evaluation metrics for the classification model."""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", *self.metrics.mean_results()))

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize confusion matrix, class names, and tracking containers for predictions and targets."""
        super().init_metrics(model)
        self.metrics.names = model.names
        