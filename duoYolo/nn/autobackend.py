"""Backend and class name utilities for multitask model inference."""

from pathlib import Path
from ultralytics.utils import YAML, ROOT
from ultralytics.utils.checks import check_yaml
from ultralytics.nn.autobackend import AutoBackend as _AutoBackend, check_class_names, default_class_names
import torch

def check_duo_class_names(names: list | dict) -> list[dict[int, str]] |  dict[int, str]:
    """
    Check class names and convert to dict format if needed.
    Supports names in multitask format where names is a list of dicts

    Args:
        names (list | dict): Class names as list or dict format.

    Returns:
        (dict): Class names in dict format with integer keys and string values.

    Raises:
        KeyError: If class indices are invalid for the dataset size.
    """
    if isinstance(names, list) and all(isinstance(i, dict) for i in names):  # names is a list of dicts
        return [check_class_names(n) for n in names]
    
    return check_class_names(names)


def default_duo_class_names(data: list[str] | str | Path | None = None) -> list[dict[int, str]] |  dict[int, str]:
    """
    Apply default class names to an input YAML file or return numerical class names.

    Args:
        data (str | Path, optional): Path to YAML file containing class names.

    Returns:
        (dict): Dictionary mapping class indices to class names.
    """
    if not isinstance(data, list):
        return default_class_names(data)
    
    return [default_class_names(d) for d in data]


class AutoBackend(_AutoBackend):
    """Duo Yolo AutoBackend supporting name dicts in multitask format"""

    @torch.no_grad()
    def __init__(
        self,
        model: str | torch.nn.Module = "yolo11n.pt",
        device: torch.device = torch.device("cpu"),
        dnn: bool = False,
        data: str | Path | None = None,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the AutoBackend for inference.

        Args:
            model (str | torch.nn.Module): Path to the model weights file or a module instance.
            device (torch.device): Device to run the model on.
            dnn (bool): Use OpenCV DNN module for ONNX inference.
            data (str | Path, optional): Path to the additional data.yaml file containing class names.
            fp16 (bool): Enable half-precision inference. Supported only on specific backends.
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization.
            verbose (bool): Enable verbose logging.
        """
        super().__init__(model, device, dnn, data, fp16, fuse, verbose)

        # reassign names
        if self.nn_module or self.pt:
            names = self.model.module.names if hasattr(self.model, "module") else self.model.names
        
        # Check names
        if "names" not in locals():  # names missing
            names = default_duo_class_names(data)
        names = check_duo_class_names(names)
        self.names = names

        self.end2end = getattr(self.model, "end2end", False)