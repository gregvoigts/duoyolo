"""Multitask dataset classes for YOLO format data loading."""

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
import uuid
from ultralytics.utils.patches import imread
from PIL import Image
import cv2
import numpy as np
import torch
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import (
    load_dataset_cache_file,
    get_hash,
    save_dataset_cache_file,
    HELP_URL,
        
)
from ultralytics.utils import NUM_THREADS, LOGGER, TQDM, LOCAL_RANK
from ultralytics.data.augment import (
    LetterBox,
    Compose,
    classify_augmentations,
    classify_transforms
)
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.instance import Instances

from duoYolo.data.utils import img2label_paths, verify_image_label
from duoYolo.data.augment import ClassificationPreparer, MultitaskConcatenate, MultitaskFormat, duo_v8_transforms, MultitaskSplit

class DuoYOLODataset(YOLODataset):
    """
    Dataset class for loading labels for multiple tasks in YOLO format.

    This class supports loading data for object detection, segmentation, pose estimation, and oriented bounding box
    (OBB), classification tasks using the YOLO format.

    Attributes:
        use_segments (bool): Invalidated by the override. Is calculated per task in the methods using.
        use_keypoints (bool): Invalidated by the override. Is calculated per task in the methods using.
        use_obb (bool): Invalidated by the override. Is calculated per task in the methods using.
        data (List[dict]): List of Dataset configuration dictionaries for each task.

    Methods:
        cache_labels: Cache dataset labels, check images and read shapes.

    Examples:
        >>> dataset = DuoYOLODataset(img_path="path/to/images", data=[{"names": {0: "person"}}])
        >>> dataset.get_labels()
    """

    def cache_labels(self, data: dict, path: Path = Path("./labels.cache"), task: str = "detect") -> dict:
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.
            data (dict): Dataset configuration dictionary for a single task.
            task (str): Type of task ('detect', 'segment', 'pose', 'obb', 'classify').

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        use_segments = task == "segment"
        use_keypoints = task == "pose"
        use_obb = task == "obb"
        use_classify = task == "classify"
        nkpt, ndim = data.get("kpt_shape", (0, 0))
        if use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(use_keypoints),
                    repeat(use_classify),
                    repeat(len(data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> list[list[dict]]:
        """
        Return dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (dict): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_location = Path(self.img_path).relative_to(self.data.get("path")).with_suffix(".cache")
        labels_list = []
        for task in self.data.get("tasks", []):
            task_base = task.get("path", f"/tmp/{uuid.uuid4()}")
            self.label_files = img2label_paths(self.im_files, str(task_base))
            cache_path = task_base / cache_location
            try:
                cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
                assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
                assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
            except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
                cache, exists = self.cache_labels(task, cache_path, task.get("task", "detect")), False  # run cache ops

            # Display cache
            nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
            if exists and LOCAL_RANK in {-1, 0}:
                d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
                TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings

            # Read cache
            [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
            labels = cache["labels"]
            if not labels:
                raise RuntimeError(
                    f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
                )
            # Check image files
            for img, lb in zip(self.im_files, labels):
                if img != lb["im_file"]:
                    raise RuntimeError(f"Image file paths do not match: {img} != {lb['im_file']}")

            # Check if the dataset is all boxes or all segments
            lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
            len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
            if len_segments and len_boxes != len_segments:
                LOGGER.warning(
                    f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                    f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                    "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
                )
                for lb in labels:
                    lb["segments"] = []
            if len_cls == 0:
                LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
            labels_list.append(labels)

        # check labels_list
        assert all(len(labels_list[0]) == len(l) for l in labels_list[1:]), "All tasks must have the same number of images."
        for i in range(len(labels_list[0])):
            im_file = labels_list[0][i]["im_file"]
            for j, task_labels in enumerate(labels_list[1:]):
                if im_file != task_labels[i]["im_file"]:
                    LOGGER.warning(
                        f"Image file paths do not match for {i}th image between task 0 and {j+1}: {im_file} != {task_labels[i]['im_file']}. ")

        # restructure labels_list to have one entry per image containing all tasks
        restructured_labels_list = []
        for i in range(len(labels_list[0])):
            combined_label = {**labels_list[0][i], "cls": {}, "bboxes": {}, "segments": {}, "keypoints": {}}
            for j, task_labels in enumerate(labels_list):
                task_key = f"task_{j}"
                combined_label["cls"] = {**combined_label["cls"], **{task_key: task_labels[i]["cls"]}}
                combined_label["bboxes"] = {**combined_label["bboxes"], **{task_key: task_labels[i]["bboxes"]}}
                combined_label["segments"] = {**combined_label["segments"], **{task_key: task_labels[i]["segments"]}}
                combined_label["keypoints"] = {**combined_label["keypoints"], **{task_key: task_labels[i]["keypoints"]}}
                
            restructured_labels_list.append(combined_label)

        return restructured_labels_list
    
    def update_labels_info(self, label: dict) -> dict:
        """
        Update label format for different tasks.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        label["instances"] = {}
        for i, task in enumerate(self.data.get("tasks", [])):
            use_obb = task == "obb"
            segment_resamples = 100 if use_obb else 1000
            if len(segments[f"task_{i}"]) > 0:
                # make sure segments interpolate correctly if original length is greater than segment_resamples
                max_len = max(len(s) for s in segments[f"task_{i}"])
                segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
                # list[np.array(segment_resamples, 2)] * num_samples
                task_segments = np.stack(resample_segments(segments[f"task_{i}"], n=segment_resamples), axis=0)
            else:
                task_segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
            label["instances"][f"task_{i}"] = Instances(bboxes[f"task_{i}"], task_segments, keypoints[f"task_{i}"], bbox_format=bbox_format, normalized=normalized)
        return label

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """
        Build and append transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        tasks = [t["task"] for t in self.data.get("tasks", [])]        

        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            if "classify" in tasks:
                LOGGER.warning("Mosaic/Augmentations are disabled when multitask contains a classification task.")
                hyp.mosaic = 0.0  # disable mosaic for classification task
            transforms = Compose(duo_v8_transforms(self, self.imgsz, hyp))
        else:
            transforms = Compose([MultitaskConcatenate(), LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

        transforms.append(MultitaskSplit())
        transforms.append(
            MultitaskFormat(
                bbox_format="xywh",
                normalize=True,
                return_mask={f"task_{i}":"segment" == task for i, task in enumerate(tasks)},
                return_keypoint={f"task_{i}":"pose" == task for i, task in enumerate(tasks)},
                return_obb={f"task_{i}":"obb" == task for i, task in enumerate(tasks)},
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Collate data samples into batches.

        Args:
            batch (list[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}: # value is tuple[dict[str, Tensor]]
                # concat all Tensors for same task key across the batch
                value = {task_key: torch.cat([v[task_key] for v in value], 0) for task_key in value[0].keys()}
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            for task_key in new_batch["batch_idx"][i].keys():
                new_batch["batch_idx"][i][task_key] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = {task_key: torch.cat([v[task_key] for v in new_batch["batch_idx"]], 0) for task_key in new_batch["batch_idx"][0].keys()}
        return new_batch


class ClassificationDataset(DuoYOLODataset):
    """
    Dataset class for loading labels for classification tasks in YOLO format.
    This class supports loading data for classification tasks using the YOLO format.
    For every image the class expects a label txt file with a single class index.
    """

    def __init__(self, *args, data, task, augment = True, cache: bool | str = False, **kwargs):

        cache_ram = cache is True or str(cache).lower() == "ram"  # cache images into RAM
        if cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache=False`."
            )
            cache = False

        super().__init__(*args, data=data, task=task, cache=cache, **kwargs)

    def get_labels(self) -> list[list[dict]]:
        """
        Return dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (dict): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_location = Path(self.img_path).relative_to(self.data.get("path")).with_suffix(".cache")
        
        base = self.data.get("path")
        self.label_files = img2label_paths(self.im_files, str(base))
        cache_path = base / cache_location
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(self.data, cache_path, "classify"), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        # Check image files
        for img, lb in zip(self.im_files, labels):
            if img != lb["im_file"]:
                raise RuntimeError(f"Image file paths do not match: {img} != {lb['im_file']}")

        return labels
    
    def update_labels_info(self, label: dict) -> dict:
        """
        pop label information except cls.
        """   
        label.pop("bboxes", None)
        label.pop("segments", None)
        label.pop("keypoints", None)
        label.pop("bbox_format", None)
        label.pop("normalized", None)

        if label["cls"].shape[0] == 0:
            label["cls"] = np.array([[-100]])  # set background to -100 for ignore index in loss calculation

        return label  

    def build_transforms(self, hyp = None):
        """
        Build and append transforms to the list.
        """
        scale = (1.0 - hyp.scale, 1.0)  # (0.08, 1.0)
        transforms = (
            classify_augmentations(
                size=hyp.imgsz,
                scale=scale,
                hflip=hyp.fliplr,
                vflip=hyp.flipud,
                erasing=hyp.erasing,
                auto_augment=hyp.auto_augment,
                hsv_h=hyp.hsv_h,
                hsv_s=hyp.hsv_s,
                hsv_v=hyp.hsv_v,
            )
            if self.augment
            else classify_transforms(size=hyp.imgsz)
            )
        self.torch_transforms = transforms
        return ClassificationPreparer(transforms)

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Collate data samples into batches.

        Args:
            batch (list[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img"}:
                value_final = torch.stack(value, 0)
            if k in {"cls"}: 
                # concat all Tensors for same task key across the batch
                value1 = torch.cat(value, 0)
                value_final = torch.squeeze(value1).type(torch.long)
            new_batch[k] = value_final
        return new_batch

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """
        Load an image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool): Whether to use rectangular resizing.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (tuple[int, int]): Resized image dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, im.shape[:2], im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None


            return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), im.shape[:2], im.shape[:2]

        return Image.fromarray(cv2.cvtColor(self.ims[i], cv2.COLOR_BGR2RGB)), self.im_hw0[i], self.im_hw[i]


