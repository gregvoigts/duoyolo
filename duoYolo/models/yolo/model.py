"""DuoYOLO model interface supporting single and multitask detection."""

from pathlib import Path
from typing import Any, List
from ultralytics.engine.results import Results
from ultralytics.models.yolo.model import YOLO as BaseYolo
from ultralytics.utils import LOGGER, SETTINGS, checks
from PIL import Image
import numpy as np
import torch

from duoYolo.models.yolo.multitask import MultitaskPredictor, MultitaskTrainer, MultitaskValidator
from duoYolo.models.yolo.classify import ClassificationTrainer, ClassificationValidator
from duoYolo.models.yolo import DetectionTrainer, DetectionValidator, DetectionPredictor
from duoYolo.models.yolo import SegmentationTrainer, SegmentationValidator, SegmentationPredictor
from duoYolo.models.yolo import ClassificationPredictor
from duoYolo.models.yolo import OBBTrainer, OBBValidator, OBBPredictor
from duoYolo.models.yolo import PoseTrainer, PoseValidator, PosePredictor
from duoYolo.nn.tasks import MultitaskModel, guess_model_task, load_partial_weights, yaml_model_load
from duoYolo.nn import DetectionModel, SegmentationModel, ClassificationModel, OBBModel, PoseModel
import duoYolo.cfg # noqa: F401 to register 'multitask' task

class DuoYOLO(BaseYolo):
    """
    DuoYOLO model class extending the YOLO base class to also support multi task models to be loaded.

    This class provides a unified interface for YOLO models and DuoYolo models, automatically switching based on the model filename. It supports various computer vision tasks including object
    detection, segmentation, classification, pose estimation, oriented bounding box detection or a combination of these.

    Attributes:
        model: The loaded DuoYolo model instance.
        task: The task type (detect, segment, classify, pose, obb).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize a DuoYolo model with automatic type detection.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.

    Examples:
        Initialize from a YAML configuration
        >>> model = DuoYOLO("duoyolov8n-od-cls.yaml")
    """

    def __init__(self, model: str | Path = "config/models/v8/duoyolov8n-od-cls.yaml", task: str | None = None, verbose: bool = False):
        """
        Initialize a DuoYOLO model.

        Args:
            model (str | Path): Path to the model file or configuration.
            task (str | None): The task type for the model. When none is infered from the model
            verbose (bool): Whether to display model information.
        """
        super().__init__(model=model, task=task, verbose=verbose)
        if self.task == "multitask":
            self.tasks = [name for name, _ in self.model.get_heads()]

    def _check_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Normalize kwargs for multitask (dict data) vs single-task (str data). Remove task-specific args."""
        if self.task == "multitask":
            kwargs["tasks"] = self.tasks
            
            if isinstance(kwargs.get("data", None), str):
                kwargs["data"] = {"task_0": kwargs["data"]}

        if self.task != "multitask":
            if "lambda_list" in kwargs:
                kwargs.pop("lambda_list")  # remove lambda_list for non-multitask models
            if "tasks" in kwargs:
                kwargs.pop("tasks")  # remove tasks for non-multitask models
            
            if isinstance(kwargs.get("data", None), dict):
                if len(kwargs["data"].items()) != 1:
                    raise ValueError("Data argument must be a single dataset for non-multitask models.")
                kwargs["data"] = list(kwargs["data"].values())[0]

    def train(
        self,
        trainer=None,
        **kwargs: Any,
    ):
        """
        Train the model using the specified dataset and training configuration.

        This method facilitates model training with a range of customizable settings. It supports training with a
        custom trainer or the default training approach. The method handles scenarios such as resuming training
        from a checkpoint, integrating with Ultralytics HUB, and updating model and configuration after training.

        When using Ultralytics HUB, if the session has a loaded model, the method prioritizes HUB training
        arguments and warns if local arguments are provided. It checks for pip updates and combines default
        configurations, method-specific defaults, and user-provided arguments to configure the training process.

        This Override adds the 'tasks' list from the model to the training kwargs. This is needed during the data loading.

        Args:
            trainer (BaseTrainer, optional): Custom trainer instance for model training. If None, uses default.
            **kwargs (Any): Arbitrary keyword arguments for training configuration. Common options include:
                data (str): Path to dataset configuration file.
                epochs (int): Number of training epochs.
                batch (int): Batch size for training.
                imgsz (int): Input image size.
                device (str): Device to run training on (e.g., 'cuda', 'cpu').
                workers (int): Number of worker threads for data loading.
                optimizer (str): Optimizer to use for training.
                lr0 (float): Initial learning rate.
                patience (int): Epochs to wait for no observable improvement for early stopping of training.

        Returns:
            (dict | None): Training metrics if available and training is successful; otherwise, None.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.train(data="coco8.yaml", epochs=3)
        """
        self._check_kwargs(kwargs)
        return super().train(trainer=trainer, **kwargs)
    
    def predict(
        self,
        source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> list[Results]:
        """
        Perform predictions on the given image source using the DuoYOLO model.
        This method facilitates the prediction process, allowing various configurations through keyword arguments.
        It supports predictions with custom predictors or the default predictor method. The method handles different
        types of image sources and can operate in a streaming mode.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | list | tuple): The source
                of the image(s) to make predictions on. Accepts various types including file paths, URLs, PIL
                images, numpy arrays, and torch tensors.
            stream (bool): If True, treats the input source as a continuous stream for predictions.
            predictor (BasePredictor, optional): An instance of a custom predictor class for making predictions.
                If None, the method uses a default predictor.
            **kwargs (Any): Additional keyword arguments for configuring the prediction process.

        Returns:
            (dict[str, list[ultralytics.engine.results.Results]]): A dictionary mapping task names to lists of prediction results, each encapsulated in a
                Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.predict(source="path/to/image.jpg", conf=0.25)
            >>> for r in results:
            ...     print(r.boxes.data)  # print detection bounding boxes

        Notes:
            - If 'source' is not provided, it defaults to the ASSETS constant with a warning.
            - The method sets up a new predictor if not already present and updates its arguments with each call.
            - For SAM-type models, 'prompts' can be passed as a keyword argument.
        """
        self._check_kwargs(kwargs)
        if self.task == "multitask":            
            if isinstance(kwargs.get("project", None), str):
                kwargs["project"] = [kwargs["project"]]
            
            if isinstance(kwargs.get("save_dir", None), str):
                kwargs["save_dir"] = [kwargs["save_dir"]]

        if self.task != "multitask":            
            if isinstance(kwargs.get("project", None), list):
                if len(kwargs["project"]) != 1:
                    raise ValueError("Project argument must be a single project for non-multitask models.")
                kwargs["project"] = kwargs["project"][0]

            if isinstance(kwargs.get("save_dir", None), list):
                if len(kwargs["save_dir"]) != 1:
                    raise ValueError("Save_dir argument must be a single save_dir for non-multitask models.")
                kwargs["save_dir"] = kwargs["save_dir"][0]  
        return super().predict(source, stream, predictor,**kwargs)

    def val(
        self,
        validator=None,
        **kwargs: Any,
    ):
        """
        Validate the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for customization through various settings. It
        supports validation with a custom validator or the default validation approach. The method combines default
        configurations, method-specific defaults, and user-provided arguments to configure the validation process.

        Args:
            validator (ultralytics.engine.validator.BaseValidator, optional): An instance of a custom validator class
                for validating the model.
            **kwargs (Any): Arbitrary keyword arguments for customizing the validation process.

        Returns:
            (ultralytics.utils.metrics.DetMetrics): Validation metrics obtained from the validation process.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # Print mAP50-95
        """        
        # Handle data argument, check for correct type
        self._check_kwargs(kwargs)
        return super().val(validator=validator, **kwargs)
    
    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "multitask":{
                "model": MultitaskModel,
                "trainer": MultitaskTrainer,
                "validator": MultitaskValidator,
                "predictor": MultitaskPredictor,
            },
            # update complete task map to enforce import of duo yolo config
            "classify": {
                "model": ClassificationModel,
                "trainer": ClassificationTrainer,
                "validator": ClassificationValidator,
                "predictor": ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": DetectionTrainer,
                "validator": DetectionValidator,
                "predictor": DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": SegmentationTrainer,
                "validator": SegmentationValidator,
                "predictor": SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": PoseTrainer,
                "validator": PoseValidator,
                "predictor": PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": OBBTrainer,
                "validator": OBBValidator,
                "predictor": OBBPredictor,
            },
        }
    
    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        Initialize a new model and infer the task type from model definitions.

        Creates a new model instance based on the provided configuration file. Loads the model configuration, infers
        the task type if not specified, and initializes the model using the appropriate class from the task map.

        Args:
            cfg (str): Path to the model configuration file in YAML format.
            task (str, optional): The specific task for the model. If None, it will be inferred from the config.
            model (torch.nn.Module, optional): A custom model instance. If provided, it will be used instead of
                creating a new one.
            verbose (bool): If True, displays model information during loading.

        Raises:
            ValueError: If the configuration file is invalid or the task cannot be inferred.
            ImportError: If the required dependencies for the specified task are not installed.

        Examples:
            >>> model = Model()
            >>> model._new("yolo11n.yaml", task="detect", verbose=True)
        """
        # use modified version of model task inference to include 'multitask'
        cfg_dict = yaml_model_load(cfg)
        task = task or guess_model_task(cfg_dict)
        super()._new(cfg_dict["yaml_file"], task=task, model=model, verbose=verbose)
    
    def _load(self, weights: str, task=None) -> None:
        """
        Load a model from a checkpoint file or initialize it from a weights file.

        This method handles loading models from either .pt checkpoint files or other weight file formats. It sets
        up the model, task, and related attributes based on the loaded weights.

        Args:
            weights (str): Path to the model weights file to be loaded.
            task (str, optional): The task associated with the model. If None, it will be inferred from the model.

        Raises:
            FileNotFoundError: If the specified weights file does not exist or is inaccessible.
            ValueError: If the weights file format is unsupported or invalid.

        Examples:
            >>> model = Model()
            >>> model._load("yolo11n.pt")
            >>> model._load("path/to/weights.pth", task="detect")
        """

        if str(weights).rpartition(".")[-1] != "pt":
            task = task or guess_model_task(checks.check_file(weights))

        super()._load(weights, task=task)

    def load_partial_weights(self, weights: str) -> None:
        """
        Load weights partially from a given weights file into the current model.

        This method allows for loading weights from a specified file into the existing model instance without
        overwriting the entire model. It is useful for scenarios where only certain layers or components of the
        model need to be updated with pre-trained weights.

        Args:
            weights (str): Path to the weights file from which to load the weights.

        Examples:
            >>> model = Model()
            >>> model.load_partial_weights("partial_weights.pt")
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])  # download and return local file
        weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolo11n -> yolo11n.pt

        if str(weights).rpartition(".")[-1] == "pt":
            load_partial_weights(weights, self.model)

        else:
            LOGGER.warning(f"Partial weight loading from {weights} not supported for non-.pt files.")