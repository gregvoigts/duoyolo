"""Multitask validator for evaluating models with multiple detection heads."""

import json
from multiprocessing.pool import ThreadPool
from types import SimpleNamespace
from typing import Any
import numpy as np
import pandas as pd
from pyparsing import Path
import torch
import torch.nn.functional as F

from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, NUM_THREADS, RANK, TQDM, callbacks, colorstr, ops, nms
from ultralytics.utils.checks import check_imgsz, check_requirements
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode, unwrap_model
from ultralytics.utils.metrics import OKS_SIGMA, box_iou, kpt_iou, mask_iou
from ultralytics.data import build_dataloader
from ultralytics.models import yolo
from duoYolo.engine.validator import DuoYoloValidatorMixin
from duoYolo.utils.ops import AdvancedProfile
from duoYolo.data.utils import check_duo_datasets
from duoYolo.utils.metrics import MultitaskConfusionMatrix, MultitaskMetrics
from duoYolo.data.build import build_duoYolo_dataset
from duoYolo.nn.autobackend import AutoBackend

class MultitaskValidator(DuoYoloValidatorMixin, BaseValidator):


    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize multitask validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict[str, Any], optional): Arguments for the validator.
            _callbacks (list[Any], optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.sigma = []
        self.kpt_shape = []
        self.process = None
        self.args.task = "multitask"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = MultitaskMetrics(self.args.tasks)

        # Create task directories
        if RANK in (-1, 0):
            for i, _ in enumerate(self.args.tasks):  
                task_dir = self.save_dir / f"task_{i}"
                task_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess batch of images for YOLO validation.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.

        Returns:
            (dict[str, Any]): Preprocessed batch.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
            if k in {"bboxes", "cls", "segments", "obb", "batch_idx"}:
                batch[k] = {task: val.to(self.device, non_blocking=self.device.type == "cuda") for task, val in v.items()}
            if k in {"cls"}:
                batch[k] = {task: val.to(self.device, non_blocking=self.device.type == "cuda").long() for task, val in v.items()}
            if k in {"keypoints", "masks"}:
                batch[k] = {task: val.to(self.device, non_blocking=self.device.type == "cuda").float() for task, val in v.items()}

        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for DuoYOLO validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        self.class_map = [list(range(1, len(n) + 1)) for n in model.names]
        self.names = model.names
        self.nc = [len(n) for n in model.names]
        self.end2end = getattr(model, "end2end", False)
        self.seen = {f"task_{i}": 0 for i in range(len(self.args.tasks))}
        self.jdict = []
        self.metrics.names = model.names
        self.confusion_matrix = MultitaskConfusionMatrix(tasks=self.args.tasks, names=model.names, save_matches=self.args.plots and self.args.visualize)

        for i, task in enumerate(self.args.tasks):
            if task != "pose":
                self.kpt_shape.append(None)
                self.sigma.append(None)
                continue
            self.kpt_shape.append(self.data["tasks"][i]["kpt_shape"])
            is_pose = self.kpt_shape[-1] == [17,3]
            nkpt = self.kpt_shape[-1][0]
            self.sigma.append(OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt)

        if "segment" in self.args.tasks: # only needed for segmentation
            if self.args.save_json:
                check_requirements("faster-coco-eval>=1.6.7")
            # More accurate vs faster
            self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask

    def get_desc(self) -> list[str]:
        """Return a formatted string summarizing class metrics of YOLO model."""
        formatted = []
        formatted.append(("%22s" + "%11s" * 2) % ("Class", "Images", "Instances"))
        for idx, task in enumerate(self.args.tasks):
            if task == "classify":
                formatted.append(("%13s" + "%13s") % (f"task_{idx}_acc(top1", "top5)"))
            elif task == "pose":
                formatted.append(("%13s" * 8) % (
                                f"task_{idx}_Box(P",
                                "R",
                                "mAP50",
                                "mAP50-95)",
                                f"task_{idx}_Pose(P",
                                "R",
                                "mAP50",
                                "mAP50-95)",
                            ))
            elif task == "segment":
                formatted.append(("%13s" * 8) % (
                                f"task_{idx}_Box(P",
                                "R",
                                "mAP50",
                                "mAP50-95)",
                                f"task_{idx}_Mask(P",
                                "R",
                                "mAP50",
                                "mAP50-95)",
                            ))
            else:  # detect or obb
                formatted.append(("%13s" * 4) % (f"task_{idx}_Box(P", "R", "mAP50", "mAP50-95)"))
        return str.join("", formatted)
            
    def postprocess(self, preds: list[torch.Tensor]) -> dict[str, list[dict[str, torch.Tensor]]]:
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains
                'bboxes', 'conf', 'cls', and 'extra' tensors.
        """
        assert len(preds) == len(self.args.tasks), "Number of predictions must match number of tasks."
        outputs= {}
        for i,(pred, task) in enumerate(zip(preds, self.args.tasks)):
            if task == "classify":
                pred = pred[0] if isinstance(pred, (list,tuple)) else pred
                n5 = min(self.nc[i], 5)
                sorted = pred.argsort(1, descending=True)[:, :n5]
                outputs[f"task_{i}"] = [{"cls":o, "conf":p} for o,p in zip(sorted.type(torch.int32).cpu().unbind(dim=0), pred.type(torch.float32).cpu().unbind(dim=0))]          
                continue

            if task == "segment":
                proto = pred[1][-1] if len(pred[1]) == 3 else pred[1]

            outputs_i = nms.non_max_suppression(
                pred,
                self.args.conf,
                self.args.iou,
                nc=0 if task == "detect" else self.nc[i],
                multi_label=True,
                agnostic=self.args.single_cls or self.args.agnostic_nms,
                max_det=self.args.max_det,
                end2end=self.end2end[f"task_{i}"],
                rotated=task == "obb",
            )
            outputs_i = [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs_i]
            if task == "pose":
                for out in outputs_i:
                    out["keypoints"] = out.pop("extra").view(-1, *self.kpt_shape[i])
            if task == "segment":
                outputs_i = self._segment_postprocess(outputs_i, proto)
            
            outputs[f"task_{i}"] = outputs_i
        return outputs

    def _segment_postprocess(self, preds: list[dict[str, torch.Tensor]], proto) -> list[dict[str, torch.Tensor]]:
        """
        Post-process YOLO predictions specifically for segmentation and return output detections with proto.

        Args:
            preds (list[dict[str, torch.Tensor]]): predictions postprocessed by the default handler

        Returns:
            list[dict[str, torch.Tensor]]: Processed detection predictions with masks.
        """
        imgsz = [4 * x for x in proto.shape[2:]]  # get image size from proto
        for i, pred in enumerate(preds):
            coefficient = pred.pop("extra")
            pred["masks"] = (
                self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
                if coefficient.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
        return preds

    def _prepare_batch(self, si: int, task_key: str, task_idx: int, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare a batch of images and annotations for validation.

        Args:
            si (int): Batch index.
            task (str): Task Key e.g 'task_0'
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"][task_key] == si
        cls = batch["cls"][task_key][idx].squeeze(-1)
        bbox = batch["bboxes"][task_key][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        pbatch =  {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }
        if self.args.tasks[task_idx] == "pose":
            kpts = batch["keypoints"][task_key][idx]
            h, w = pbatch["imgsz"]
            kpts = kpts.clone()
            kpts[..., 0] *= w
            kpts[..., 1] *= h
            pbatch["keypoints"] = kpts

        if self.args.tasks[task_idx] == "segment":
            nl = pbatch["cls"].shape[0]
            if self.args.overlap_mask:
                masks = batch["masks"][task_key][si]
                index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
                masks = (masks == index).float()
            else:
                masks = batch["masks"][task_key][si]
                masks = masks.unsqueeze(0)
            if nl:
                mask_size = tuple(s if self.process is ops.process_mask_native else s // 4 for s in pbatch["imgsz"])
                if masks.shape[1:] != mask_size:
                    masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0]
                    masks = masks.gt_(0.5)
            pbatch["masks"] = masks
        return pbatch

    def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Prepare predictions for evaluation against ground truth.

        Args:
            pred (dict[str, torch.Tensor]): Post-processed predictions from the model.

        Returns:
            (dict[str, torch.Tensor]): Prepared predictions in native space.
        """
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def update_metrics(self, preds: dict[str,list[dict[str, torch.Tensor]]], batch: dict[str, Any]) -> None:
        """
        Update metrics with new predictions and ground truth for all tasks.

        Processes predictions vs ground truth for each task individually, computing TP/FP/confidence
        scores and confusion matrices. Handles task-specific metric computation including detection,
        segmentation, pose, and classification metrics.

        Args:
            preds (dict[str, list[dict[str, torch.Tensor]]]): Dictionary mapping task names to prediction lists:
                {
                    "task_0": [
                        {"bboxes": ..., "conf": ..., "cls": ..., "keypoints": ..., "masks": ...},
                        ...  # one per image
                    ],
                    "task_1": [...],
                    ...
                }
            batch (dict[str, Any]): Ground truth batch with structure:
                {
                    "img": (B, 3, H, W),
                    "bboxes": {"task_i": tensor},
                    "cls": {"task_i": tensor},
                    "batch_idx": {"task_i": tensor},
                    "ori_shape": [(h, w), ...],
                    "ratio_pad": [(r, (pl, pt)), ...],
                    "keypoints": {"task_i": tensor} [optional],
                    "masks": {"task_i": tensor} [optional],
                    ...
                }

        Updates:
            self.seen (dict): Increments image count per task
            self.confusion_matrix: Processes batch for confusion matrix
            self.metrics: Updates internal statistics
            self.jdict (list): Appends JSON predictions if saving
            Saved files: TXT predictions if args.save_txt

        Processing Per Task:
            1. For each image in batch
            2. Extract task-specific predictions and ground truth
            3. Prepare batch (scale coordinates, interpolate masks, etc.)
            4. Prepare predictions (apply single-class mode if needed)
            5. Compute TP/FP for detection/pose tasks or accuracy for classification
            6. Save predictions to JSON/TXT if configured
            7. Update confusion matrix and metrics

        Note:
            - Classification stats keyed as: {task_name}_target, {task_name}_pred
            - Detection stats: tp, fp, conf, pred_cls, ground_truth_cls
            - Pose stats augment detection stats with keypoint metrics
            - Segmentation stats augment detection stats with mask metrics
        """
        # enumerate to also get task index (task order must match self.args.tasks)
        stats = {}
        for task_idx, (k, task_preds) in enumerate(preds.items()):                    
            for si, pred in enumerate(task_preds):
                self.seen[f"task_{task_idx}"] += 1
                pbatch = self._prepare_batch(si, k, task_idx, batch)
                predn = self._prepare_pred(pred)

                cls = pbatch["cls"].cpu().numpy()
                no_pred = predn["cls"].shape[0] == 0
                # call task-specific processing so pose/segment can augment the base tp matrix
                if self.args.tasks[task_idx] == "classify":
                    stats = {
                        **stats,
                        f"{k}_target": stats.get(f"{k}_target", []) + [pbatch["cls"].type(torch.int32).cpu()],
                        f"{k}_pred": stats.get(f"{k}_pred", []) + [predn["cls"].type(torch.int32).unsqueeze(0).cpu()],
                    }
                else:
                    temp_stats = {
                            **{f"{k}_{key}": value for key, value in self._process_batch(predn, pbatch, task_idx).items()},
                            f"{k}_target_cls": cls,
                            f"{k}_target_img": np.unique(cls),
                            f"{k}_conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                            f"{k}_pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                        }
                    for key, value in temp_stats.items():
                        if key in stats:
                            stats[key] = np.concatenate((stats[key], value), axis=0)
                        else:
                            stats[key] = value
                # Evaluate
                if self.args.plots:
                    self.confusion_matrix.process_batch(k, predn, pbatch, conf=self.args.conf)
                    if self.args.visualize:
                        self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

                if no_pred:
                    continue

                # Save
                if self.args.save_json or self.args.save_txt:
                    predn_scaled = self.scale_preds(predn, pbatch, self.args.tasks[task_idx])
                if self.args.save_json:
                    self.pred_to_json(predn_scaled, pbatch, task_idx)
                if self.args.save_txt:
                    self.save_one_txt(
                        predn_scaled,
                        self.args.save_conf,
                        pbatch["ori_shape"],
                        task_idx,
                        self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",                        
                    )
        # update metrics after all tasks are processed
        self.metrics.update_stats(
            stats)

    def finalize_metrics(self) -> None:
        """
        Finalize metrics computation and prepare for results reporting.

        Performs final metric calculations after all batches have been processed:
        - Computes per-class AP, mAP for detection/segmentation/pose tasks
        - Computes top-1/top-5 accuracy for classification tasks
        - Plots confusion matrices if configured
        - Attaches speed metrics and confusion matrix to metrics object

        Updates:
            self.metrics.speed: Inference/NMS timing statistics
            self.metrics.confusion_matrix: Final confusion matrix
            self.metrics.save_dir: Results directory

        Side Effects:
            - Saves confusion matrix plots to self.save_dir/confusion_matrix.png
            - Saves normalized CM if args.plots=True
        """
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def get_stats(self) -> dict[str, Any]:
        """
        Calculate and return metrics statistics.

        Returns:
            (dict[str, Any]): Dictionary containing metrics results.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def print_results(self) -> None:
        """
        Print validation results for all tasks and per-class metrics.

        Logs summary statistics including:
        - Mean metrics across all tasks (mAP, accuracy, etc.)
        - Per-class metrics for each task if available
        - Total images seen and object instances found

        Output Format:
            Header: "Class" "Images" "Instances" [task-specific metric names...]
            Summary: "all" [global totals] [mean metrics for all tasks...]
            Per-class: class_name [class-specific metrics...]

        Raises:
            Warning: Logged if no labels found in the dataset

        Note:
            - Output varies by task type (detect, segment, classify, pose)
            - Verbose mode required to print per-class metrics
            - Metrics include P (precision), R (recall), mAP50, mAP75, mAP50-95, etc.
        """
        pf = "%22s" + "%11i" * 2 + "%13.3g" * len(self.metrics.keys)  # print format
        classes = 0
        for nt_class in self.metrics.nt_per_class.values():
            classes += nt_class.sum()
        seen = max(self.seen.values())
        mean_stats = []
        for res in self.metrics.mean_results().values():
            mean_stats.extend(res)
        LOGGER.info(pf % ("all", seen, classes, *mean_stats))
        if classes == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and max(self.nc) > 1 and len(self.metrics.stats):
            for j, (task_key, class_indices) in enumerate(self.metrics.ap_class_index.items()):
                for i, c in enumerate(class_indices):
                    LOGGER.info(
                        pf
                        % (
                            self.names[j][c],
                            self.metrics.nt_per_image[task_key][c],
                            self.metrics.nt_per_class[task_key][c],
                            *self.metrics.class_result(task_key,i),
                        )
                    )

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any], task_idx: int | None = None) -> dict[str, np.ndarray]:
        """
        Unified _process_batch that dispatches to appropriate per-task computation.

        If task_idx is provided, it will compute detection TP and optionally pose/mask TP
        depending on the task type at that index in self.args.tasks.
        """
        # default detection-like behavior
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            base_tp = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
            tp = {"tp": base_tp}
        else:
            iou = box_iou(batch["bboxes"], preds["bboxes"])
            tp = {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

        # if no task index given, return detection only
        if task_idx is None:
            return tp

        task = self.args.tasks[task_idx]
        # pose-specific augmentation (same logic as ultralytics PoseValidator)
        if task == "pose":
            gt_cls = batch["cls"]
            if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
                tp_p = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
            else:
                area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
                iou_k = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
                tp_p = self.match_predictions(preds["cls"], gt_cls, iou_k).cpu().numpy()
            tp.update({"tp_p": tp_p})

        # segment-specific augmentation (same logic as ultralytics SegmentationValidator)
        if task == "segment":
            gt_cls = batch["cls"]
            if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
                tp_m = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
            else:
                iou_m = mask_iou(batch["masks"].flatten(1), preds["masks"].flatten(1))
                tp_m = self.match_predictions(preds["cls"], gt_cls, iou_m).cpu().numpy()
            tp.update({"tp_m": tp_m})

        return tp

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        return build_duoYolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): Dataloader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset, batch_size, self.args.workers, shuffle=False, rank=-1, drop_last=self.args.compile
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int, task_key:str, task_idx:int) -> None:
        """
        Plot validation image samples.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            ni (int): Batch index.
            task_key (str): Task Key e.g 'task_0'
            task_idx (int): Task index.
        """
        # Create a temporary object to hold necessary attributes for plotting
        constructed_self = SimpleNamespace(
            names=self.names[task_idx],
            save_dir=self.save_dir / task_key,
            on_plot=self.on_plot,
            args=self.args
        )

        batch_copy = batch.copy() # create a copy to avoid modifying the original batch

        task = self.args.tasks[task_idx]
        # use task-specific plotting functions with constructed object
        if task == "classify":
            for k in {"bboxes", "conf", "masks", "keypoints"}:
                    if k in batch_copy:
                        del batch_copy[k]  # remove non classify keys for plotting, so plotting function recognizes it as classify batch
        yolo.detect.DetectionValidator.plot_val_samples(constructed_self, batch_copy, ni)

    def plot_predictions(
        self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int, task_key: str, task_idx:int, max_det: int | None = None
    ) -> None:
        """
        Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
            task_key (str): Task Key e.g 'task_0'
            task_idx (int): Task index.
            max_det (Optional[int]): Maximum number of detections to plot.
        """
        # Create a temporary object to hold necessary attributes for plotting
        constructed_self = SimpleNamespace(
            names=self.names[task_idx],
            save_dir=self.save_dir / task_key,
            on_plot=self.on_plot,
            args=self.args,
        )

        task = self.args.tasks[task_idx]
        # use task-specific plotting functions with constructed object
        if task == "classify":
            cls_preds = torch.stack([v["cls"] for v in preds])  # convert list of dicts to tensor
            v = yolo.classify.ClassificationValidator.__new__(yolo.classify.ClassificationValidator)
            v.__dict__.update(constructed_self.__dict__)
            v.plot_predictions(batch, cls_preds, ni)
            return
        if task == "segment":
            v = yolo.segment.SegmentationValidator.__new__(yolo.segment.SegmentationValidator)
            v.__dict__.update(constructed_self.__dict__)
            v.plot_predictions(batch, preds, ni)
            return
        if task == "obb":
            v = yolo.obb.OBBValidator.__new__(yolo.obb.OBBValidator)
            v.__dict__.update(constructed_self.__dict__)
            v.plot_predictions(batch, preds, ni)
            return
        v = yolo.detect.DetectionValidator.__new__(yolo.detect.DetectionValidator)
        v.__dict__.update(constructed_self.__dict__)
        v.plot_predictions(batch, preds, ni)

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], task_idx:int, file: Path) -> None:
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', and 'cls'.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            task_idx (int): Task index.
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results
        task = self.args.tasks[task_idx]
        save_bbox = task in {"detect", "obb", "pose", "segment"}
        save_mask = task == "segment"
        save_probs = task == "classify"

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names[task_idx],
            probs=predn["conf"] if save_probs else None,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1) if save_bbox else None,
            masks=torch.as_tensor(predn["masks"], dtype=torch.uint8) if save_mask else None,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any], task_idx:int) -> None:
        """
        Serialize YOLO predictions to COCO json format.

        Args:
            predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys
                with bounding box coordinates, confidence scores, and class predictions.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
            task_idx (int): Task index.

        Examples:
             >>> result = {
             ...     "image_id": 42,
             ...     "file_name": "42.jpg",
             ...     "category_id": 18,
             ...     "bbox": [258.15, 41.29, 348.26, 243.78],
             ...     "score": 0.236,
             ... }
        """
        task = self.args.tasks[task_idx]
        save_bbox = task in {"detect", "obb", "pose", "segment"}

        rles = None
        if task == "segment":
            from faster_coco_eval.core.mask import encode  # noqa

            def single_encode(x):
                """Encode predicted masks as RLE and append results to jdict."""
                rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                return rle

            pred_masks = np.transpose(predn["masks"], (2, 0, 1))
            with ThreadPool(NUM_THREADS) as pool:
                rles = pool.map(single_encode, pred_masks)

        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem

        box = None
        conf = None
        if save_bbox:
            box = ops.xyxy2xywh(predn["bboxes"])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            box = box.tolist()
            conf = predn["conf"].tolist()

        for idx, c in enumerate(predn["cls"].tolist()):
            j ={
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[task_idx][int(c)]
                }
            if save_bbox and box is not None:
                j["bbox"] = [round(x, 3) for x in box[idx]]
                j["score"] = round(conf[idx], 5)
            if rles is not None:
                j["segmentation"] = rles[idx]
            self.jdict[f'task_{task_idx}'].append(
                j
            )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any], task: str) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        scale_bbox = task in {"detect", "obb", "pose", "segment"}
        scale_mask = task == "segment"

        ret = {**predn}
        if scale_bbox and "bboxes" in predn:
            ret["bboxes"] = ops.scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            )
        if scale_mask and "masks" in predn:
            ret["masks"] = ops.scale_image(
            torch.as_tensor(predn["masks"], dtype=torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy(),
            pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"],
        )
        return ret
        

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate YOLO output in JSON format and return performance statistics.

        Args:
            stats (dict[str, Any]): Current statistics dictionary.

        Returns:
            (dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
        """
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return self.coco_evaluate(stats, pred_json, anno_json)

    def coco_evaluate(
        self,
        stats: dict[str, Any],
        pred_json: str,
        anno_json: str,
        iou_types: str | list[str] = "bbox",
        suffix: str | list[str] = "Box",
    ) -> dict[str, Any]:
        """
        Evaluate COCO/LVIS metrics using faster-coco-eval library.

        Performs evaluation using the faster-coco-eval library to compute mAP metrics
        for object detection. Updates the provided stats dictionary with computed metrics
        including mAP50, mAP50-95, and LVIS-specific metrics if applicable.

        Args:
            stats (dict[str, Any]): Dictionary to store computed metrics and statistics.
            pred_json (str | Path]): Path to JSON file containing predictions in COCO format.
            anno_json (str | Path]): Path to JSON file containing ground truth annotations in COCO format.
            iou_types (str | list[str]]): IoU type(s) for evaluation. Can be single string or list of strings.
                Common values include "bbox", "segm", "keypoints". Defaults to "bbox".
            suffix (str | list[str]]): Suffix to append to metric names in stats dictionary. Should correspond
                to iou_types if multiple types provided. Defaults to "Box".

        Returns:
            (dict[str, Any]): Updated stats dictionary containing the computed COCO/LVIS evaluation metrics.
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster # type: ignore

                anno = COCO(anno_json)
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
                    )
                    val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    # update mAP50-95 and mAP50
                    stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]

                    if self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]  # always use box mAP50-95 for fitness
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats
    
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod  # validate non-compiled original model to avoid issues
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit = model.stride, model.pt, model.jit
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            self.data = check_duo_datasets(self.args.data, self.args.tasks)

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device)
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            AdvancedProfile(device=self.device),
            AdvancedProfile(device=self.device),
            AdvancedProfile(device=self.device),
            AdvancedProfile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = { f'task_{i}': [] for i,k in enumerate(self.args.tasks ) }  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                # print plots for each task, when args.plots is a list check if true for each task
                for task_idx, task_key in enumerate(preds.keys()):
                    if isinstance(self.args.plots, list) and self.args.plots[task_idx] is False:
                        continue
                    filtered_batch = {k: v[task_key] if isinstance(v, dict) else v for k, v in batch.items()}
                    self.plot_val_samples(filtered_batch, batch_i, task_key, task_idx)
                    self.plot_predictions(filtered_batch, preds[task_key], batch_i, task_key, task_idx)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.store_timings(dt)
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats