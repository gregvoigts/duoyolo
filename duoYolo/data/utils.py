from copy import deepcopy
import os
from typing import Any, List
from pyparsing import Path
from duoYolo.utils.checks import check_file
from ultralytics.data.utils import check_det_dataset, exif_size, IMG_FORMATS, FORMATS_HELP_MSG, segments2boxes
import numpy as np
from PIL import Image, ImageOps


def img2label_paths(img_paths: list[str], base_path: str) -> list[str]:
    """Convert image paths to label paths appending the image filename to the lable base path and extension with '.txt'."""
    sa, sl = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return [base_path + sl + (x.rsplit(sa, 1)[1].rsplit(".", 1)[0] + ".txt") for x in img_paths]

def check_duo_datasets(dataset: dict[str,str],tasks: List[str], autodownload: bool = True) -> dict[str, Any]:
    """
    Download, verify, and/or unzip datasets if not found locally.

    This function checks the availability of a list of dataset, and if a dataset is not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset. Finally, it merges multiple datasets into a single data dictionary.

    Args:
        dataset (dict[str, str]): Paths to the datasets or dataset descriptors (like a YAML file).
        tasks (List[str]): List of tasks corresponding to each dataset.
        autodownload (bool, optional): Whether to automatically download the dataset if not found.

    Returns:
        (dict[str, Any]): Parsed dataset information and paths.
    """
    assert len(dataset) == len(tasks), "Number of datasets must match number of tasks."

    for task in dataset.keys():
        d = dataset[task]
        try:
            dataset[task] = check_file(d)
        except FileNotFoundError:
            pass # dataset not found in duoYolo directory, check_det_dataset will handle downloading and search in ultralytics directory

    data_dicts = {task: check_det_dataset(d, autodownload) if d else None for task, d in sorted(dataset.items())}

    first_complete_dataset = next((d for d in data_dicts.values() if d is not None), None)
    if first_complete_dataset is None:
        raise RuntimeError("No valid dataset found. Please check the provided dataset paths.")
    
    data = deepcopy(first_complete_dataset)

    data["nc"] = [d["nc"] if d and "nc" in d else None for _,d in data_dicts.items()]  # List of number of classes
    data["names"] = [d["names"] if d and "names" in d else None for _,d in data_dicts.items()]  # List of names dicts

    # Merge multiple datasets
    data["tasks"] = [{**(d if d else {"names": []}), "task": task} for task, d in zip(tasks, data_dicts.values())]

    return data  # dictionary

def check_single_dataset(dataset: str, autodownload: bool = True) -> dict[str, Any]:
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Most of the logic is handled by check_det_dataset, this function is a simple wrapper to support files located under duoYolo root.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found.

    Returns:
        (dict[str, Any]): Parsed dataset information and paths.
    """
    try:
        dataset = check_file(dataset)
    except FileNotFoundError:
        pass # dataset not found in duoYolo directory, check_det_dataset will handle downloading and search in ultralytics directory
    
    return check_det_dataset(dataset, autodownload)


def verify_image_label(args: tuple) -> list:
    """Verify one image-label pair. Works also for classification label files."""
    im_file, lb_file, prefix, keypoint, classify, num_cls, nkpt, ndim, single_cls = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                elif classify:
                    assert nl == 1, "classification label file must have only one row"
                    assert lb.shape[1] == 1, f"classification labels require 1 column, {lb.shape[1]} columns detected"
                    lb = np.hstack((lb, np.array([0.5,0.5,1,1]).reshape(1,4)))  # add bb over complete image for compatibility
                    points = lb[:, 1:]
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                # Coordinate points check with 1% tolerance
                assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
                assert lb.min() >= -0.01, f"negative class labels or coordinate {lb[lb < -0.01]}"

                # All labels
                max_cls = 0 if single_cls else lb[:, 0].max()  # max label count
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}{im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)          
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]