from pyparsing import Any
import numpy as np
import torch
from duoYolo.utils.instance import MultitaskInstances
from ultralytics.data.augment import Compose, Format, Mosaic, RandomPerspective, MixUp, CutMix, Albumentations, RandomHSV, RandomFlip, CopyPaste, LetterBox
from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import xyxyxyxy2xywhr
from ultralytics.utils import LOGGER, IterableSimpleNamespace

def duo_v8_transforms(dataset, imgsz: int, hyp: IterableSimpleNamespace, stretch: bool = False):
    """
    Apply a series of image transformations for training.

    This function creates a composition of image augmentation techniques to prepare images for YOLO training.
    It includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    This function is adapted for transformations before applying multitask training.

    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (IterableSimpleNamespace): A dictionary of hyperparameters controlling various aspects of the transformations.
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.

    Returns:
        (MultitaskCompose): A composition of image transformations to be applied to the dataset.

    """
    mosaic = MultitaskMosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    mosaic.pre_transform = MultitaskConcatenate()
    affine = MultitaskRandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )

    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        mosaic2 = MultitaskMosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
        mosaic2.pre_transform = MultitaskConcatenate()
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([mosaic2, affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if any(t.get("task", "detect") == "pose" for t in dataset.data.get("tasks",[])):
        kpt_shape = dataset.data.get("kpt_shape", None) # TODO: Fix for multitask
        if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
            hyp.fliplr = hyp.flipud = 0.0  # both fliplr and flipud require flip_idx
            LOGGER.warning("No 'flip_idx' array defined in data.yaml, disabling 'fliplr' and 'flipud' augmentations.")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            MultitaskConcatenate(),
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms

class ClassificationPreparer:
    """Prepares the label dict provided to the transformation pipeline for the classification transformations"""

    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Apply transformations and convert cls to tensor."""
        transformed_images = self.transformations(labels["img"])
        cls_tensor = torch.from_numpy(labels["cls"])
        return {**labels, "img": transformed_images, "cls": cls_tensor}

class MultitaskMosaic(Mosaic):
    """
    A class for performing mosaic augmentation on multitask datasets.
    """

    def _cat_labels(self, mosaic_labels: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Concatenate and process labels for mosaic augmentation.

        This method combines labels from multiple images used in mosaic augmentation, clips instances to the
        mosaic border, and removes zero-area boxes.

        Args:
            mosaic_labels (list[dict[str, Any]]): A list of label dictionaries for each image in the mosaic.

        Returns:
            (dict[str, Any]): A dictionary containing concatenated and processed labels for the mosaic image, including:
                - im_file (str): File path of the first image in the mosaic.
                - ori_shape (tuple[int, int]): Original shape of the first image.
                - resized_shape (tuple[int, int]): Shape of the mosaic image (imgsz * 2, imgsz * 2).
                - cls (np.ndarray): Concatenated class labels.
                - instances (Instances): Concatenated instance annotations.
                - mosaic_border (tuple[int, int]): Mosaic border size.
                - texts (list[str], optional): Text labels if present in the original labels.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640)
            >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(4)]
            >>> result = mosaic._cat_labels(mosaic_labels)
            >>> print(result.keys())
            dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances', 'mosaic_border'])
        """
        if not mosaic_labels:
            return {}
        cls = []
        instances = []
        split_indices = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            split_indices.append(labels["split_indices"])
            instances.append(labels["instances"])
        # Final labels
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "split_indices": np.concatenate(split_indices, 0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["split_indices"] = final_labels["split_indices"][good]
        final_labels["cls"] = final_labels["cls"][good]

        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels
    
class MultitaskRandomPerspective(RandomPerspective):
    """Random affine transformations for multitask annotations (bboxes, segments, keypoints)."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """
        Apply random perspective and affine transformations to an image and its associated labels.

        This method performs a series of transformations including rotation, translation, scaling, shearing,
        and perspective distortion on the input image and adjusts the corresponding bounding boxes, segments,
        and keypoints accordingly.

        Args:
            labels (dict[str, Any]): A dictionary containing image data and annotations.
                Must include:
                    'img' (np.ndarray): The input image.
                    'cls' (np.ndarray): Class labels.
                    'instances' (Instances): Object instances with bounding boxes, segments, and keypoints.
                May include:
                    'mosaic_border' (tuple[int, int]): Border size for mosaic augmentation.

        Returns:
            (dict[str, Any]): Transformed labels dictionary containing:
                - 'img' (np.ndarray): The transformed image.
                - 'cls' (np.ndarray): Updated class labels.
                - 'instances' (Instances): Updated object instances.
                - 'resized_shape' (tuple[int, int]): New image shape after transformation.

        Examples:
            >>> transform = RandomPerspective()
            >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> labels = {
            ...     "img": image,
            ...     "cls": np.array([0, 1, 2]),
            ...     "instances": Instances(bboxes=np.array([[10, 10, 50, 50], [100, 100, 150, 150]])),
            ... }
            >>> result = transform(labels)
            >>> assert result["img"].shape[:2] == result["resized_shape"]
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # do not need ratio pad

        img = labels["img"]
        cls = labels["cls"]
        split_indices = labels["split_indices"]
        instances = labels.pop("instances")
        # Make sure the coord formats are right
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # Scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        
        # get indices of first dimension where values are not NaN (valid segments)
        valid_segments = ~np.isnan(segments).all(axis=(1, 2))
        # Update bboxes if there are segments.
        if any(valid_segments):            
            bboxes[valid_segments], segments[valid_segments] = self.apply_segments(segments[valid_segments], M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        # Clip
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["split_indices"] = split_indices[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

class MultitaskConcatenate:
    """
    A class to concatenate multiple labels dictionaries for multitask learning.

    Methods:
        __call__: Concatenate multiple labels dictionaries into a single dictionary.
    """

    def __call__(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Concatenate multiple labels dictionaries into a single dictionary.

        This method combines the class labels and instances from multiple tasks into a single structure.

        Args:
            labels_list (list[dict[str, Any]]): A list of labels dictionaries to be concatenated.

        Returns:
            (dict[str, Any]): A single concatenated labels dictionary.
        """
        # Combine labels for all tasks
        combined_data = data.copy()
        combined_data["split_indices"] = np.concatenate([np.full(len(a), i) for i, a in enumerate(data["cls"].values())])
        combined_data["cls"] = np.concatenate(list(data["cls"].values()), axis=0, dtype=np.float16)
        combined_data["instances"] = MultitaskInstances.concatenate(list(data["instances"].values()))
        combined_data["task_count"] = len(data["cls"])
        return combined_data
    
class MultitaskSplit:
    """
    A class to split concatenated labels dictionary back into multiple labels dictionaries for multitask learning.

    Methods:
        __call__: Split a concatenated labels dictionary into multiple dictionaries.
    """

    def __init__(self):
        self.concatenator = MultitaskConcatenate()

    def __call__(self, combined_data: dict[str, Any]) -> dict[str, Any]:
        """
        Split a concatenated labels dictionary into multiple dictionaries.

        This method separates the combined class labels and instances back into their respective tasks.

        Args:
            combined_data (dict[str, Any]): A concatenated labels dictionary.
        """

        # Separate labels back into their respective tasks
        split_indicies = combined_data.pop("split_indices")
        task_count = combined_data.pop("task_count")
        data = combined_data.copy()
        data["instances"] = self.split_instances(combined_data["instances"], split_indicies, task_count)
        data["cls"] = {f"task_{i}": combined_data["cls"][split_indicies == i] for i in range(task_count)}
        return data
    
    def split_instances(self, instances: Instances, indices: np.ndarray, task_count: int) -> dict[str, Instances]:
        """Split combined Instances back into a list of Instances for each task."""
        split_instances = {}
        for i in range(task_count):
            split_instances[f"task_{i}"] = instances[indices == i]
        return split_instances    

class MultitaskFormat(Format):
    """
    This override extends the original Format class to handle multitask scenarios by formatting labels for multiple tasks.

    A class for formatting image annotations for object detection, instance segmentation, and pose estimation tasks.

    This class standardizes image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader.    

    Attributes:
        bbox_format (str): Format for bounding boxes. Options are 'xywh' or 'xyxy'.
        normalize (bool): Whether to normalize bounding boxes.
        return_mask (dict[str, bool]): Whether to return instance masks for segmentation per task.
        return_keypoint (dict[str, bool]): Whether to return keypoints for pose estimation per task.
        return_obb (dict[str, bool]): Whether to return oriented bounding boxes per task.
        mask_ratio (dict[str, int] | int): Downsample ratio for masks per task or overall.
        mask_overlap (dict[str, bool] | bool): Whether to overlap masks per task or overall.
        batch_idx (bool): Whether to keep batch indexes.
        bgr (float): The probability to return BGR images.

    Methods:
        __call__: Format labels dictionary with image, classes, bounding boxes, and optionally masks and keypoints.
        _format_img: Convert image from Numpy array to PyTorch tensor.
        _format_segments: Convert polygon points to bitmap masks.

    Examples:
        >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=[True, False])
        >>> formatted_labels = formatter(labels)
        >>> img = formatted_labels["img"]
        >>> bboxes = formatted_labels["bboxes"]
        >>> masks = formatted_labels["masks"]
    """

    def __init__(
        self,
        bbox_format: str = "xywh",
        normalize: bool = True,
        return_mask: dict[str, bool] = [False],
        return_keypoint: dict[str, bool] = [False],
        return_obb: dict[str, bool] = [False],
        mask_ratio: dict[str, int] | int = 4,
        mask_overlap: dict[str, bool] | bool = True,
        batch_idx: bool = True,
        bgr: float = 0.0,
    ):
        """
        Initialize the Format class with given parameters for image and instance annotation formatting.

        This class standardizes image and instance annotations for object detection, instance segmentation, and pose
        estimation tasks, preparing them for use in PyTorch DataLoader's `collate_fn`.

        Args:
            bbox_format (str): Format for bounding boxes. Options are 'xywh', 'xyxy', etc.
            normalize (bool): Whether to normalize bounding boxes to [0,1].
            return_mask (dict[str, bool]): If True, returns instance masks for segmentation tasks per task.
            return_keypoint (dict[str, bool]): If True, returns keypoints for pose estimation tasks per task.
            return_obb (dict[str, bool]): If True, returns oriented bounding boxes per task.
            mask_ratio (dict[str, int] | int): Downsample ratio for masks per task or overall.
            mask_overlap (dict[str, bool] | bool): If True, allows mask overlap per task or overall.
            batch_idx (bool): If True, keeps batch indexes.
            bgr (float): Probability of returning BGR images instead of RGB.

        Attributes:
            bbox_format (str): Format for bounding boxes.
            normalize (bool): Whether bounding boxes are normalized.
            return_mask (dict[str, bool]): Whether to return instance masks per task.
            return_keypoint (dict[str, bool]): Whether to return keypoints per task.
            return_obb (dict[str, bool]): Whether to return oriented bounding boxes per task.
            mask_ratio (dict[str, int] | int): Downsample ratio for masks per task or overall.
            mask_overlap (dict[str, bool] | bool): Whether masks can overlap per task or overall.
            batch_idx (bool): Whether to keep batch indexes.
            bgr (float): The probability to return BGR images.
        """
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # set False when training detection only
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio if isinstance(mask_ratio, dict) else {f"task_{i}": mask_ratio for i in range(len(return_mask))}
        self.mask_overlap = mask_overlap if isinstance(mask_overlap, dict) else {f"task_{i}": mask_overlap for i in range(len(return_mask))}
        self.batch_idx = batch_idx  # keep the batch indexes
        self.bgr = bgr

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """
        Format image annotations for object detection, instance segmentation, and pose estimation tasks.

        This method standardizes the image and instance annotations to be used by the `collate_fn` in PyTorch
        DataLoader. It processes the input labels dictionary, converting annotations to the specified format and
        applying normalization if required.

        Args:
            labels (dict[str, Any]): A dictionary containing image and annotation data with the following keys:
                - 'img': The input image as a numpy array.
                - 'cls': Dictionary of class labels for instances per task.
                - 'instances': Dictionary of instance objects containing bounding boxes, segments, and keypoints per task.

        Returns:
            (dict[str, Any]): A dictionary with formatted data, including:
                - 'img': Formatted image tensor.
                - 'cls': Dictionary of class label tensors per task.
                - 'bboxes': Dictionary of bounding box tensors in the specified format per task.
                - 'masks': Dictionary of instance mask tensors per task (if return_mask is True).
                - 'keypoints': Dictionary of keypoint tensors per task (if return_keypoint is True).
                - 'batch_idx': Batch index tensor (if batch_idx is True).

        Examples:
            >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=[True])
            >>> labels = {"img": np.random.rand(640, 640, 3), "cls": np.array([0, 1]), "instances": Instances(...)}
            >>> formatted_labels = formatter(labels)
            >>> print(formatted_labels.keys())
        """
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        for instance in instances.values():
            instance.convert_bbox(self.bbox_format)
            instance.denormalize(w, h)
        nl = {key: len(task) for key, task in instances.items()}

        labels["img"] = self._format_img(img)

        for key in cls.keys(): # Iterate over each task
            if self.return_mask[key]:
                if nl[key]:
                    masks, instances[key], cls[key] = self._format_segments(instances[key], cls[key], w, h, key)
                    masks = torch.from_numpy(masks)
                else:
                    masks = torch.zeros(
                        1 if self.mask_overlap[key] else nl[key], img.shape[0] // self.mask_ratio[key], img.shape[1] // self.mask_ratio[key]
                    )
                labels["masks"] = {key: masks, **labels.get("masks", {})}
            elif any(self.return_mask.values()):
                # If at least one task needs a mask we need to add the task keys for all tasks to keep dict splits consistent
                labels["masks"] = {key: torch.empty(0), **labels.get("masks", {})}
                

            labels["cls"] = {key: torch.from_numpy(cls[key]) if nl[key] else torch.zeros(nl[key], 1), **labels.get("cls", {})}
            labels["bboxes"] = {key: torch.from_numpy(instances[key].bboxes) if nl[key] else torch.zeros((nl[key], 4)), **labels.get("bboxes", {})}

            if self.return_keypoint[key]:
                keypoint = (
                    torch.empty(0, 3) if instances[key].keypoints is None else torch.from_numpy(instances[key].keypoints)
                )
                if self.normalize:
                    keypoint[..., 0] /= w
                    keypoint[..., 1] /= h
                labels["keypoints"] = {key: keypoint, **labels.get("keypoints", {})}

            if self.return_obb[key]:
                labels["bboxes"] = {
                    key: xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5)),
                    **labels.get("bboxes", {}),
                }

            # NOTE: need to normalize obb in xywhr format for width-height consistency
            if self.normalize:
                labels["bboxes"][key][:, [0, 2]] /= w
                labels["bboxes"][key][:, [1, 3]] /= h

            # Then we can use collate_fn
            if self.batch_idx:
                labels["batch_idx"] = {key: torch.zeros(nl[key]), **labels.get("batch_idx", {})}
        return labels
    
    def _format_segments(
        self, instances: Instances, cls: np.ndarray, w: int, h: int, task_key: str
    ) -> tuple[np.ndarray, Instances, np.ndarray]:
        """
        Convert polygon segments to bitmap masks.

        Args:
            instances (Instances): Object containing segment information.
            cls (np.ndarray): Class labels for each instance.
            w (int): Width of the image.
            h (int): Height of the image.

        Returns:
            masks (np.ndarray): Bitmap masks with shape (N, H, W) or (1, H, W) if mask_overlap is True.
            instances (Instances): Updated instances object with sorted segments if mask_overlap is True.
            cls (np.ndarray): Updated class labels, sorted if mask_overlap is True.

        Notes:
            - If self.mask_overlap is True, masks are overlapped and sorted by area.
            - If self.mask_overlap is False, each mask is represented separately.
            - Masks are downsampled according to self.mask_ratio.
        """
        segments = instances.segments
        if self.mask_overlap[task_key]:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio[task_key])
            masks = masks[None]  # (640, 640) -> (1, 640, 640)
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio[task_key])

        return masks, instances, cls