"""Multitask trainer for training models with multiple detection heads."""

from __future__ import annotations

import math
import random
from copy import copy
import time
from types import SimpleNamespace
from typing import Any
import warnings
from torch import distributed as dist

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from duoYolo.utils import DEFAULT_CFG
from duoYolo.data.utils import  check_duo_datasets
from duoYolo.models.yolo.multitask.val import MultitaskValidator
from duoYolo.nn.tasks import MultitaskModel
from duoYolo.data.build import build_duoYolo_dataset

from ultralytics.cfg import get_cfg
from ultralytics.models import yolo
from ultralytics.data import build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, emojis, colorstr
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import autocast, torch_distributed_zero_first, unset_deterministic, unwrap_model
from ultralytics.utils.tqdm import TQDM

class MultitaskTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a multitask model.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).
        tasks (list): List of tasks for the multitask model.

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """
        Initialize a MultitaskTrainer object for training DuoYolo model training.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.
        """
        self.tasks = overrides.get("tasks", []) if overrides else []
        super().__init__(cfg, overrides, _callbacks)

        # Create task directories
        if RANK in (-1, 0):
            for i, _ in enumerate(self.tasks):  
                task_dir = self.save_dir / f"task_{i}"
                task_dir.mkdir(parents=True, exist_ok=True)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return build_duoYolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
    
    def get_dataset(self):
        """
        Get train and validation datasets from data dictionary.

        Overrides the base method to use check_duo_datasets for multitask datasets.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
                data = check_duo_datasets(self.args.data, self.tasks)
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{self.args.data}' error ❌ {e}")) from e
        return data
    
    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """
        Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, torch.Tensor):
                        v[kk] = vv.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self): # TODO: Check when finished building dataset configuration for multitask
        """
        Set model attributes based on dataset information.

        This method configures the model with dataset-specific attributes including the number of classes,
        class names, and training hyperparameters. Called after dataset initialization to ensure the model
        is properly configured for the specific task(s) being trained.

        Sets:
            model.nc (int): Number of classes from dataset.
            model.names (list): Class names from dataset.
            model.args: Training arguments/hyperparameters.

        Note:
            This method is a placeholder for future enhancement. Currently commented out:
            - Dynamic scaling of box, cls losses based on detection layers
            - Scaling based on image size and number of layers
            - Class weight initialization from dataset labels
        """
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """
        Return a DuoYOLO model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (MultiTaskModel): YOLO detection model.
        """
        model = MultitaskModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """
        Return a MultitaskValidator for YOLO model validation.

        Initializes and returns a MultitaskValidator configured with the training setup.
        Extracts loss names from the model and passes them to the validator for metric tracking.

        Returns:
            MultitaskValidator: Validator instance configured with:
                - test_loader: Validation dataloader
                - save_dir: Directory for validation outputs
                - args: Copy of training arguments
                - callbacks: Training callbacks for event handling

        Attributes Set:
            self.loss_names (list): Loss component names extracted from model for logging

        Raises:
            AttributeError: If the model doesn't have get_loss_names method.
        """
        
        self.loss_names = unwrap_model(self.model).get_loss_names()
        return MultitaskValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """
        Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (list[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """
        Return a formatted string of training progress header.

        Generates a formatted header string for progress logging during training. The header includes
        epoch, GPU memory usage, task-specific loss components, batch size, and image size.

        Returns:
            str: Formatted header string containing:
                - Epoch number
                - GPU memory usage
                - Loss names (dynamic based on model.loss_names)
                - Number of instances
                - Image size

        Example:
            >>> trainer = MultitaskTrainer(...)
            >>> header = trainer.progress_string()
            >>> print(header)
            "       Epoch      GPU_mem  box_loss  cls_loss  dfl_loss Instances      Size"
        """
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """
        Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        for i, task in enumerate(self.tasks):
            task_key = f"task_{i}"
            filtered_batch = {k: v[task_key] if isinstance(v, dict) else v for k, v in batch.items()}
            # Create a temporary object to hold necessary attributes for plotting
            constructed_self = SimpleNamespace(
                save_dir=self.save_dir / task_key,
                on_plot=self.on_plot,
            )

            # use task-specific plotting functions with constructed object
            if task == "classify":
                for k in {"bboxes", "conf", "masks", "keypoints"}:
                    if k in filtered_batch:
                        del filtered_batch[k]  # remove non classify keys for plotting, so plotting function recognizes it as classify batch

            yolo.detect.DetectionTrainer.plot_training_samples(constructed_self, filtered_batch, ni)

    def plot_training_labels(self):
        """
        Create a labeled training plot of the YOLO model for each task.

        Generates and saves visualization plots for training labels for each task in the multitask model.
        Concatenates bounding boxes and class labels from the training dataset and creates plots showing
        the distribution of labels across the dataset. One plot is created per task.

        This method:
        - Iterates through each task in the multitask model
        - Extracts task-specific bounding boxes and class labels from dataset
        - Calls plot_labels to generate and save visualization
        - Saves plots in task-specific directories

        Data Requirements:
            self.train_loader.dataset.labels must have structure:
            {
                "bboxes": {"task_0": array, "task_1": array, ...},
                "cls": {"task_0": array, "task_1": array, ...},
                ...
            }

        Raises:
            KeyError: If dataset labels don't contain expected task keys.
            FileNotFoundError: If save directory cannot be created.
            ValueError: If bboxes or cls arrays are empty for a task.

        Note:
            Generates PNG plots saved to self.save_dir/task_i/labels.png
        """
        for i, task in enumerate(self.tasks):            
            boxes = np.concatenate([lb["bboxes"][f"task_{i}"] for lb in self.train_loader.dataset.labels], 0)
            cls = np.concatenate([lb["cls"][f"task_{i}"] for lb in self.train_loader.dataset.labels], 0)
            plot_labels(boxes, cls.squeeze(), names=self.data["names"][i], save_dir=self.save_dir / f"task_{i}", on_plot=self.on_plot)

    def auto_batch(self):
        """
        Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(max(len(label["cls"][f"task_{i}"]) for i in range(len(self.tasks))) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj)
    
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Construct an optimizer for the given model. Adapted to support multitask models.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", [10])  # number of classes
            nc = max(nc)  # get max number of classes by task
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer
    
    def setup_model(self):
        """
        Set up model for training with multitask-specific configuration.

        Calls the base setup_model method and then attaches the lambda weight list to the model.
        The lambda list controls the relative weighting of losses for different tasks during training.

        Returns:
            dict: Checkpoint dictionary from base setup_model containing:
                - Model weights and architecture
                - Training state information

        Sets:
            model.lambda_list (list): Task loss weights from args.lambda_list, used during loss computation
                to balance multiple task objectives. Format: [det_weight, seg_weight, cls_weight, ...]

        Note:
            lambda_list values should typically sum to expected total loss per batch.
            Missing or incomplete lambda_list will default to equal weighting.

        Example:
            >>> trainer = MultitaskTrainer(overrides={"lambda_list": [1.0, 0.5, 0.3]})
            >>> ckpt = trainer.setup_model()
            >>> print(trainer.model.lambda_list)
            [1.0, 0.5, 0.3]
        """
        ckpt = super().setup_model()
        self.model.lambda_list = self.args.lambda_list
        return ckpt
    
    def _do_train(self):
        """
        Execute the main training loop for multitask model.

        Orchestrates the complete training process including:
        - DDP setup for distributed training (if world_size > 1)
        - Training pipeline initialization
        - Epoch-based training loop with batch iteration
        - Validation and checkpoint saving
        - Learning rate scheduling and early stopping

        Training Flow:
        1. Setup DDP if distributed training
        2. Initialize training with _setup_train()
        3. For each epoch:
           - Set epoch in sampler (DDP)
           - Iterate through batches
           - Perform warmup if early epochs
           - Forward pass with autocast (mixed precision)
           - Backward pass and gradient accumulation
           - Optimize weights
           - Log training metrics
           - Plot training samples (if configured)
        4. After epoch:
           - Validate model (if configured)
           - Save metrics and model checkpoint
           - Check early stopping criteria
           - Update learning rate schedule

        Attributes Modified:
            self.epoch (int): Current training epoch number
            self.loss (torch.Tensor): Current batch loss value
            self.loss_items (torch.Tensor): Breakdown of loss components
            self.tloss (torch.Tensor): Accumulated loss for the epoch
            self.metrics (dict): Validation metrics
            self.fitness (float): Model fitness score
            self.stop (bool): Flag indicating training should stop
            self.lr (dict): Current learning rates by param group

        Raises:
            RuntimeError: If DDP synchronization fails or training is interrupted.
            OutOfMemoryError: If CUDA memory is exhausted (caught and handled by autocast).

        Note:
            - Supports time-based training (args.time in hours)
            - Supports epoch-based training
            - Handles mosaic augmentation closing after n epochs
            - Synchronizes stop signal across DDP ranks
            - Implements gradient accumulation for large batches
            - Supports mixed precision training with autocast

        Implementation Details:
            - Warmup: Learning rate linearly increases for first nw iterations
            - Gradient accumulation: Accumulates gradients before optimization step
            - DDP: Broadcasts stop signal to ensure all ranks stop together
            - Memory management: Clears cache if memory utilization > 50%
            - Checkpoint saving: Saves best and last models only in rank 0
        """
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    if self.args.compile:
                        # Decouple inference and loss calculations for improved compile performance
                        preds = self.model(batch["img"])
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model(batch)
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"]["task_0"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)  # prevent VRAM spike
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")