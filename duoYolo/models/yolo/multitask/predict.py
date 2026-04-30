"""Multitask predictor routing raw outputs to task-specific predictors."""

from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any
import cv2

import torch
from duoYolo.engine.multitask_result import MultitaskResults
from duoYolo.utils import DEFAULT_CFG
from ultralytics.engine.predictor import BasePredictor
from ultralytics.models import yolo
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.models.yolo.obb.predict import OBBPredictor
from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import MACOS, WINDOWS


class MultitaskPredictor(BasePredictor):
    """
    A Class Extending the BasePredictor to support multitask models. 

    This Predictor get the used heads on the model and stores instances of the relevant predictors for each head.
    Then Routes the raw prediction outputs to the relevant predictor for postprocessing and combines the results.

    Attributes:
        model: The multitask model to perform predictions with.
        predictors: A dictionary mapping task names to their corresponding predictor instances.

    Methods:
        postprocess: Postprocess the raw outputs from the multitask model using the appropriate predictors.
    """
    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict[str, list[callable]] | None = None,
    ):
        super().__init__(cfg, overrides, _callbacks)
        

    def postprocess(self, preds, img, orig_img):
        """
        Postprocess the raw outputs from the multitask model using the appropriate predictors.

        Args:
            preds (List[torch.Tensor]): The raw output tensors from the model for each subtask.
            img (torch.Tensor): The preprocessed input image tensor.
            orig_img (np.ndarray): The original input image before preprocessing.
        
        Returns:
            (list(dict[str, Result])): A list of postprocessed predictions from each head. Containing results for each img in the batch.
        """
        constructed_self = SimpleNamespace(
            args=self.args,
            model=self.model,
            _feats=getattr(self, "_feats", None),
            batch=self.batch     
        )
        results = ()
        for idx, (pred, task) in enumerate(zip(preds, self.args.tasks)):
            if task == "detect":
                v = DetectionPredictor.__new__(DetectionPredictor)
                v.__dict__.update(constructed_self.__dict__)                
                res = v.postprocess(pred, img, orig_img)
                results += (res,)
            elif task == "classify":
                v = ClassificationPredictor.__new__(ClassificationPredictor)
                v.__dict__.update(constructed_self.__dict__)
                res = v.postprocess( pred, img, orig_img) 
                results += (res,)
            elif task == "obb":
                v = OBBPredictor.__new__(OBBPredictor)
                v.__dict__.update(constructed_self.__dict__)
                res = v.postprocess(pred, img, orig_img)
                results += (res,)
            elif task == "pose":
                v = PosePredictor.__new__(PosePredictor)
                v.__dict__.update(constructed_self.__dict__)
                res = v.postprocess(pred, img, orig_img)
                results += (res,)
            elif task == "segment":
                protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
                v = SegmentationPredictor.__new__(SegmentationPredictor)
                v.__dict__.update(constructed_self.__dict__)
                res = v.postprocess(pred, img, orig_img, protos=protos)
                results += (res,)

            results_objs = [MultitaskResults(result) for result in zip(*results)]
            return results_objs

    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
        """
        Write inference results to a file or directory.

        Args:
            i (int): Index of the current image in the batch.
            p (Path): Path to the current image.
            im (torch.Tensor): Preprocessed image tensor.
            s (list[str]): List of result strings.

        Returns:
            (str): String with result information.
        """
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 0 if frame undetermined

        results = self.results[i]

        for task, result in results.results_dict.items():
            self.txt_path = self.save_dir / "labels" / task / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
            string += "{:g}x{:g} ".format(*im.shape[2:])
            result.save_dir = self.save_dir.__str__()  # used in other locations
            string += f"{result.verbose()}"

            # Add predictions to image
            if self.args.save or self.args.show:
                self.plotted_img = result.plot(
                    line_width=self.args.line_width,
                    boxes=self.args.show_boxes,
                    conf=self.args.show_conf,
                    labels=self.args.show_labels,
                    im_gpu=None if self.args.retina_masks else im[i],
                )

            # Save results
            if self.args.save_txt:
                result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
            if self.args.save_crop:
                result.save_crop(save_dir=self.save_dir / "crops" / task, file_name=self.txt_path.stem)
            if self.args.show:
                self.show(str(p))
            if self.args.save:
                self.save_predicted_images(self.save_dir / p.name, task, frame)

        string += f"{results.speed['inference']:.1f}ms"
        return string
    
    def save_predicted_images(self, save_path: Path, task_id: str, frame: int = 0):
        """
        Save video predictions as mp4 or images as jpg at specified path.

        Args:
            save_path (Path): Path to save the results.
            frame (int): Frame number for video mode.
        """
        im = self.plotted_img

        # Save videos and streams
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = self.save_dir / task_id / f"{save_path.stem}_frames"  # save frames to a separate directory
            if save_path not in self.vid_writer:  # new video
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # integer required, floats produce error in MP4 codec
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)
                )

            # Save video
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}/{save_path.stem}_{frame}.jpg", im)

        # Save images
        else:
            cv2.imwrite(str(save_path.with_suffix(".jpg")), im)  # save to JPG for best support