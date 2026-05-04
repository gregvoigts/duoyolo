"""
Multitask model architecture and configuration parsing.

Defines MultitaskModel that supports multiple task heads (detect, segment, classify, pose, obb)
and provides utilities for model initialization, task detection, and weight loading.
"""

import contextlib
from pathlib import Path
import re
from typing import List
from ultralytics.nn.modules.head import WorldDetect, YOLOEDetect, v10Detect
from ultralytics.nn.tasks import BaseModel, guess_model_scale, torch_safe_load, yaml_model_load, parse_model
from ultralytics.nn.modules import Detect, Classify, Segment, Pose, OBB, YOLOESegment
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.torch_utils import initialize_weights, scale_img
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.checks import check_yaml as ultralytics_check_yaml
from copy import deepcopy
import torch

from duoYolo.utils.checks import check_yaml
from duoYolo.utils.loss import MultitaskLoss

class MultitaskModel(BaseModel):
    """Multitask model supporting multiple detection heads."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

    def __init__(self, cfg="duoyolov8n-od-cls.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the YOLO multiclass model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        self.yaml["channels"] = ch  # save channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_multitask_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = [{i: f"{i}" for i in range(classes)} for classes in self.yaml["nc"]]  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.heads = self._find_heads(self.model)
        self.end2end = {f"task_{i}": getattr(m, "end2end", False) for i, m in enumerate(self.heads)}
        self.lambda_list = None

        # Build strides
        s = 256  # 2x min stride
        
        self.model.eval()  # Avoid changing batch statistics until training begins
        for name, idx in self.heads:
            m = self.model[idx]
            m.inplace = self.inplace
            m.training = True  # Setting it to True to properly return strides
        model_out = self.forward(torch.zeros(1, ch, s, s))
        self.stride = []
        for i, (name, idx) in enumerate(self.heads):
            m = self.model[idx]
            if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment
                if self.end2end[f"task_{i}"]:
                    out = model_out[i]["one2many"]
                else:
                    out = model_out[i][0] if isinstance(m, (Segment, YOLOESegment, Pose, OBB)) else model_out[i]
                m.stride = torch.tensor([s / x.shape[-2] for x in out])  
                self.stride.append(m.stride)
                m.bias_init()  # only run once
            elif isinstance(m, (Classify)):
                m.stride = torch.Tensor([1])
                self.stride.append(m.stride)
            else:
                m.stride = torch.Tensor([32])  # default stride for i.e. RTDETR
                self.stride.append(m.stride)
                m.bias_init()  # only run once
        self.model.train()  # Set model back to training(default) mode
        # Pad strides to same shape before stacking
        max_len = max(s.shape[0] for s in self.stride)
        self.stride = torch.stack([
            torch.nn.functional.pad(s, (0, max_len - s.shape[0]))
            for s in self.stride
        ])

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")


    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Overrrides the predict function to return multiple relevant outputs for the different Tasks

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            List[torch.Tensor]: List of output tensors from the model for each suptask.
        """
        y, dt, embeddings = [], [], []  # outputs
        outputs = []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            if isinstance(m, (Detect, Classify, Segment, Pose, OBB)):
                outputs.append(x)
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize) # type: ignore
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return outputs
    
    def _predict_augment(self, x):
        """
        Perform augmentations on input image x and return augmented inference and train outputs.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (torch.Tensor): Augmented inference output.
        """
        LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
        return self._predict_once(x)
        # img_size = x.shape[-2:]  # height, width #TODO: implement multi-scale inference for multitask
        # s = [1, 0.83, 0.67]  # scales
        # f = [None, 3, None]  # flips (2-ud, 3-lr)
        # y = []  # outputs
        # for si, fi in zip(s, f):
        #     xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
        #     yi = super().predict(xi)[0]  # forward
        #     yi = [self._descale_pred(yij, fi, si, img_size) for yij in yi]
        #     y.append(yi)
        # y = self._clip_augmented(y)  # clip augmented tails
        # return torch.cat(y, -1), None  # augmented inference, train
    
    def get_heads(self) -> List[str]:
        """
        Get the list of heads present in the multitask model.

        Returns:
            List[Tuple[str, int]]: List of head names and their index.
        """
        return self.heads

    def get_loss_names(self) -> List[str]:
        """
        Get the list of loss names for each head in the multitask model.

        Returns:
            List[str]: List of loss names.
        """
        if hasattr(self, 'criterion'):
            return self.criterion.loss_names
        loss_names = []
        for idx, (head,i) in enumerate(self.get_heads()):
            if head == "detect":
                loss_names.extend([f"task_{idx}_box", f"task_{idx}_cls", f"task_{idx}_dfl"])
            elif head == "classify":
                loss_names.append(f"task_{idx}_cls")
            elif head == "pose":
                loss_names.extend([f"task_{idx}_box",f"task_{idx}_pose",f"task_{idx}_kobj", f"task_{idx}_cls", f"task_{idx}_dfl"])
            elif head == "segment":
                loss_names.extend([f"task_{idx}_box",f"task_{idx}_seg", f"task_{idx}_cls", f"task_{idx}_dfl"])
            elif head == "obb":
                loss_names.extend([f"task_{idx}_box", f"task_{idx}_cls", f"task_{idx}_dfl"])
        return loss_names
    
    @staticmethod
    def _find_heads(model) -> List[str]:
        """
        Find and return the list of heads present in the multitask model.

        Args:
            model (torch.nn.Module): The multitask model.
        """
        heads = []
        for i, m in enumerate(model):
            if isinstance(m, Classify):
                heads.append(("classify",i))
            elif isinstance(m, Segment) or isinstance(m, YOLOESegment):
                heads.append(("segment",i))
            elif isinstance(m, Pose):
                heads.append(("pose",i))
            elif isinstance(m, OBB):
                heads.append(("obb",i))
            elif isinstance(m, Detect):
                heads.append(("detect",i))
        return heads


    # @staticmethod
    # def _descale_pred(p, flips, scale, img_size, dim=1):
    #     """
    #     De-scale predictions following augmented inference (inverse operation).

    #     Args:
    #         p (torch.Tensor): Predictions tensor.
    #         flips (int): Flip type (0=none, 2=ud, 3=lr).
    #         scale (float): Scale factor.
    #         img_size (tuple): Original image size (height, width).
    #         dim (int): Dimension to split at.

    #     Returns:
    #         (torch.Tensor): De-scaled predictions.
    #     """
    #     p[:, :4] /= scale  # de-scale
    #     x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
    #     if flips == 2:
    #         y = img_size[0] - y  # de-flip ud
    #     elif flips == 3:
    #         x = img_size[1] - x  # de-flip lr
    #     return torch.cat((x, y, wh, cls), dim)

    # def _clip_augmented(self, y):
    #     """
    #     Clip YOLO augmented inference tails.

    #     Args:
    #         y (list[torch.Tensor]): List of detection tensors.

    #     Returns:
    #         (list[torch.Tensor]): Clipped detection tensors.
    #     """
    #     nl = self.model[-1].nl  # number of detection layers (P3-P5)
    #     g = sum(4**x for x in range(nl))  # grid points
    #     e = 1  # exclude layer count
    #     for j in range(len(y[0])):
    #         i = (y[0][j].shape[-1] // g) * sum(4**x for x in range(e))  # indices
    #         y[0][j] = y[0][j][..., :-i]  # large
    #     for j in range(len(y[-1])):
    #         i = (y[-1][j].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
    #         y[-1][j] = y[-1][j][..., i:]  # small
    #     return y
    
    def init_criterion(self):
        return MultitaskLoss(self, lambda_list=self.lambda_list)

class MultitaskClassCount:
    """Helper class to handle multiple class counts for multitask datasets. 
    Supports comparison with integers and lists of integers."""
    
    def __init__(self, nc):
        self.nc = nc
    
    def __eq__(self, value):
        if self.nc == value:
            return True
        if isinstance(self.nc, list) and value in self.nc:
            return True
        return False        

def parse_multitask_model(d, ch, verbose=True):
    """Parse a multitask model dictionary. Replaces the 'nc' strings in the head args with the correct number of classes for each task.

    Args:
        d (dict): Model configuration dictionary.
        ch (int): Number of input channels.
        verbose (bool): Whether to display model information.

    Returns:
        MultitaskModel: The parsed multitask model.
    """
    nc = d['nc']
    d['nc'] = MultitaskClassCount(nc)  # wrap nc in helper class for flexible comparison in head parsing
    for (f, n, m, args) in d["head"]:  # from, number, module, args
        for arg in args:
            if isinstance(arg, str) and arg.startswith("nc"):  # number of classes
                task_idx = int(arg[2:]) if len(arg) > 2 else 0
                args[args.index(arg)] = nc[task_idx]  # nc per task
                    
    return parse_model(d, ch, verbose)



def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb', 'multitask').
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        head_m = [x[-2].lower() for x in cfg["head"]]
        tasks = set()
        if any(m in {"classify", "classifier", "cls", "fc"} for m in head_m):
            tasks.add("classify")
        if any("detect" in m for m in head_m):
            tasks.add("detect")
        if any("segment" in m for m in head_m):
            tasks.add("segment")
        if any(m == "pose" for m in head_m):
            tasks.add("pose")
        if any(m == "obb" for m in head_m):
            tasks.add("obb")
        if len(tasks) == 1:
            return tasks.pop()
        elif len(tasks) > 1:
            return "multitask"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        tasks = set()
        for m in model.modules():
            if isinstance(m, (Segment, YOLOESegment)):
                tasks.add("segment")
            elif isinstance(m, Classify):
                tasks.add("classify")
            elif isinstance(m, Pose):
                tasks.add("pose")
            elif isinstance(m, OBB):
                tasks.add("obb")
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                tasks.add("detect")
        if len(tasks) == 1:
            return tasks.pop()
        elif len(tasks) > 1:
            return "multitask"
        
    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        tasks = set()
        if "-seg" in model.stem or "segment" in model.parts:
            tasks.add("segment")
        elif "-cls" in model.stem or "classify" in model.parts:
            tasks.add("classify")
        elif "-pose" in model.stem or "pose" in model.parts:
            tasks.add("pose")
        elif "-obb" in model.stem or "obb" in model.parts:
            tasks.add("obb")
        elif "detect" in model.parts:
            tasks.add("detect")
        if len(tasks) == 1:
            return tasks.pop()
        elif len(tasks) > 1:
            return "multitask"

    # Unable to determine task from model
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect

def load_partial_weights(weights: str, model: torch.nn.Module):
    """
    Load partial weights from a checkpoint into the model.

    Args:
        weights (str): Path to the weights file.
        model (torch.nn.Module): The model to load weights into.
    """
    checkpoint, _ = torch_safe_load(weights)
    state_dict = checkpoint["model"].state_dict() 
    model_state_dict = model.state_dict()

    # Filter out unnecessary keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

    # Load the filtered state dict into the model
    model.load_state_dict(filtered_state_dict, strict=False)
    LOGGER.info(f"Loaded {len(filtered_state_dict)}/{len(model_state_dict)} layers from {weights}")

def yaml_model_load(path):
    """
    Load a DuoYolo model from a YAML file. If non found call for ultralytics yaml_model_load

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    try:
        yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    except (FileNotFoundError):
        yaml_file = ultralytics_check_yaml(unified_path, hard=False) or ultralytics_check_yaml(path)  # fallback to ultralytics yaml_model_load which also checks for pretrained weights
    d = YAML.load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(yaml_file)
    return d