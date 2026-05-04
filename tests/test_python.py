# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import csv
import tarfile
import urllib
import zipfile
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from duoYolo.utils.metrics import MultitaskConfusionMatrix, MultitaskMetrics
from tests import CFG, MODEL, MODELS, SOURCE, SOURCES_LIST, TASK_MODEL_DATA
from duoYolo import DuoYOLO
from duoYolo.cfg import TASK2DATA, TASKS
from duoYolo.utils import DEFAULT_CFG, DEFAULT_CFG_PATH
from ultralytics import RTDETR
from ultralytics.data.build import load_inference_source
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import (
    ARM64,
    ASSETS,
    ASSETS_URL,    
    IS_JETSON,
    IS_RASPBERRYPI,
    LINUX,
    LOGGER,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    YAML,
    checks,
    is_github_action_running,
)
from ultralytics.utils.downloads import download, safe_download
from ultralytics.utils.torch_utils import TORCH_1_11, TORCH_1_13

from duoYolo.engine.multitask_result import MultitaskResults


def test_model_forward():
    """Test the forward pass of the YOLO model."""
    model = DuoYOLO(CFG)
    model(source=None, imgsz=32, augment=True)  # also test no source and augment


def test_model_methods():
    """Test various methods and properties of the YOLO model to ensure correct functionality."""
    model = DuoYOLO(MODEL)

    # Model methods
    model.info(verbose=True, detailed=True)
    model = model.reset_weights()
    model = model.load(MODEL)
    model.to("cpu")
    model.fuse()
    model.clear_callback("on_train_start")
    model.reset_callbacks()

    # Model properties
    _ = model.names
    _ = model.device
    _ = model.transforms
    _ = model.task_map

def test_predict_txt(tmp_path):
    """Test YOLO predictions with file, directory, and pattern sources listed in a text file."""
    file = tmp_path / "sources_multi_row.txt"
    with open(file, "w") as f:
        for src in SOURCES_LIST:
            f.write(f"{src}\n")
    results = DuoYOLO(MODEL)(source=file, imgsz=32)
    assert len(results) == 7, f"Expected 7 results from source list, got {len(results)}"


@pytest.mark.skipif(True, reason="disabled for testing")
def test_predict_csv_multi_row(tmp_path):
    """Test YOLO predictions with sources listed in multiple rows of a CSV file."""
    file = tmp_path / "sources_multi_row.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source"])
        writer.writerows([[src] for src in SOURCES_LIST])
    results = DuoYOLO(MODEL)(source=file, imgsz=32)
    assert len(results) == 7, f"Expected 7 results from multi-row CSV, got {len(results)}"


@pytest.mark.skipif(True, reason="disabled for testing")
def test_predict_csv_single_row(tmp_path):
    """Test YOLO predictions with sources listed in a single row of a CSV file."""
    file = tmp_path / "sources_single_row.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SOURCES_LIST)
    results = DuoYOLO(MODEL)(source=file, imgsz=32)
    assert len(results) == 7, f"Expected 7 results from single-row CSV, got {len(results)}"


@pytest.mark.parametrize("model_name", MODELS)
def test_predict_img(model_name):
    """Test YOLO model predictions on various image input types and sources, including online images."""
    channels = 1 if model_name == "yolo11n-grayscale.pt" else 3
    model = DuoYOLO(WEIGHTS_DIR / model_name)
    im = cv2.imread(str(SOURCE), flags=cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)  # uint8 NumPy array
    assert len(model(source=Image.open(SOURCE), save=True, verbose=True, imgsz=32)) == 1  # PIL
    assert len(model(source=im, save=True, save_txt=True, imgsz=32)) == 1  # ndarray
    assert len(model(torch.rand((2, channels, 32, 32)), imgsz=32)) == 2  # batch-size 2 Tensor, FP32 0.0-1.0 RGB order
    assert len(model(source=[im, im], save=True, save_txt=True, imgsz=32)) == 2  # batch
    assert len(list(model(source=[im, im], save=True, stream=True, imgsz=32))) == 2  # stream
    assert len(model(torch.zeros(320, 640, channels).numpy().astype(np.uint8), imgsz=32)) == 1  # tensor to numpy
    batch = [
        str(SOURCE),  # filename
        Path(SOURCE),  # Path
        "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/im/zidane.jpg?token=123" if ONLINE else SOURCE,  # URI
        im,  # OpenCV
        Image.open(SOURCE),  # PIL
        np.zeros((320, 640, channels), dtype=np.uint8),  # numpy
    ]
    assert len(model(batch, imgsz=32, classes=0)) == len(batch)  # multiple sources in a batch


@pytest.mark.parametrize("model", MODELS)
def test_predict_visualize(model):
    """Test model prediction methods with 'visualize=True' to generate prediction visualizations."""
    DuoYOLO(WEIGHTS_DIR / model)(SOURCE, imgsz=32, visualize=True)


def test_predict_gray_and_4ch(tmp_path):
    """Test YOLO prediction on SOURCE converted to grayscale and 4-channel images with various filenames."""
    im = Image.open(SOURCE)

    source_grayscale = tmp_path / "grayscale.jpg"
    source_rgba = tmp_path / "4ch.png"
    source_non_utf = tmp_path / "non_UTF_测试文件_tést_image.jpg"
    source_spaces = tmp_path / "image with spaces.jpg"

    im.convert("L").save(source_grayscale)  # grayscale
    im.convert("RGBA").save(source_rgba)  # 4-ch PNG with alpha
    im.save(source_non_utf)  # non-UTF characters in filename
    im.save(source_spaces)  # spaces in filename

    # Inference
    model = DuoYOLO(MODEL)
    for f in source_rgba, source_grayscale, source_non_utf, source_spaces:
        for source in Image.open(f), cv2.imread(str(f)), f:
            results = model(source, save=True, verbose=True, imgsz=32)
            assert len(results) == 1, f"Expected 1 result for {f.name}, got {len(results)}"
        f.unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_predict_all_image_formats():
    """Predict on all 12 image formats (AVIF, BMP, DNG, HEIC, JP2, JPEG, JPG, MPO, PNG, TIF, TIFF, WebP)."""
    # Download dataset if needed
    data = check_det_dataset("coco12-formats.yaml")
    dataset_path = Path(data["path"])

    # Collect all images from train and val
    expected = {"avif", "bmp", "dng", "heic", "jp2", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp"}
    images = [im for im in (dataset_path / "images" / "train").glob("*.*") if im.suffix.lower().lstrip(".") in expected]
    images += [im for im in (dataset_path / "images" / "val").glob("*.*") if im.suffix.lower().lstrip(".") in expected]
    assert len(images) == 12, f"Expected 12 images, found {len(images)}"

    # Verify all format extensions are represented
    extensions = {img.suffix.lower().lstrip(".") for img in images}
    assert extensions == expected, f"Missing formats: {expected - extensions}"

    # Run inference on all images
    model = DuoYOLO(MODEL)
    results = model(images, imgsz=32)
    assert len(results) == 12, f"Expected 12 results, got {len(results)}"


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.skipif(is_github_action_running(), reason="No auth https://github.com/JuanBindez/pytubefix/issues/166")
def test_youtube():
    """Test YOLO model on a YouTube video stream, handling potential network-related errors."""
    model = DuoYOLO(MODEL)
    try:
        model.predict("https://youtu.be/G17sBkb38XQ", imgsz=96, save=True)
    # Handle internet connection errors and 'urllib.error.HTTPError: HTTP Error 429: Too Many Requests'
    except (urllib.error.HTTPError, ConnectionError) as e:
        LOGGER.error(f"YouTube Test Error: {e}")


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.parametrize("model", MODELS)
def test_track_stream(model, tmp_path):
    """Test streaming tracking on a short 10 frame video using ByteTrack tracker and different GMC methods.

    Note imgsz=160 required for tracking for higher confidence and better matches.
    """
    if model == "yolo11n-cls.pt" or model =="duoyolo11n-od-seg.pt":  # classification model not supported for tracking
        return
    video_url = f"{ASSETS_URL}/decelera_portrait_min.mov"
    model = DuoYOLO(model)
    model.track(video_url, imgsz=160, tracker="bytetrack.yaml")
    model.track(video_url, imgsz=160, tracker="botsort.yaml", save_frames=True)  # test frame saving also

    # Test Global Motion Compensation (GMC) methods and ReID
    for gmc, reidm in zip(["orb", "sift", "ecc"], ["auto", "auto", "yolo11n-cls.pt"]):
        default_args = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
        custom_yaml = tmp_path / f"botsort-{gmc}.yaml"
        YAML.save(custom_yaml, {**default_args, "gmc_method": gmc, "with_reid": True, "model": reidm})
        model.track(video_url, imgsz=160, tracker=custom_yaml)


@pytest.mark.parametrize(
    "task,weight,data,data_key",
    TASK_MODEL_DATA,
    ids=[f"{task}-{weight.name}-{data_key}" for task, weight, _, data_key in TASK_MODEL_DATA],
)
def test_val(task: str, weight: str, data: str, data_key: str) -> None:
    """Test the validation mode of the YOLO model."""
    model = DuoYOLO(weight)
    for plots in {True, False}:  # Test both cases i.e. plots=True and plots=False
        metrics = model.val(data=data, imgsz=32, plots=plots)
        if task == "multitask":
            assert isinstance(metrics, MultitaskMetrics), f"Expected MultitaskMetrics for multitask task, got {type(metrics)}"
            assert len(metrics.metrics) == len(data), f"Expected metrics for {len(data)} tasks, got {len(metrics.metrics)}"
            assert isinstance(metrics.confusion_matrix, MultitaskConfusionMatrix), f"Expected confusion_matrix to be a MultitaskConfusionMatrix for multitask, got {type(metrics.confusion_matrix)}"
            assert len(metrics.confusion_matrix.matrices) == len(data), f"Expected confusion matrices for {len(data)} tasks, got {len(metrics.confusion_matrix.matrices)}"
        metrics.to_df()
        metrics.to_csv()
        metrics.to_json()
        # Tests for confusion matrix export
        metrics.confusion_matrix.to_df()
        metrics.confusion_matrix.to_csv()
        metrics.confusion_matrix.to_json()


# @pytest.mark.skipif(not ONLINE, reason="environment is offline")
# @pytest.mark.skipif(IS_JETSON or IS_RASPBERRYPI, reason="Edge devices not intended for training")
# def test_train_scratch():
#     """Test training the YOLO model from scratch on 12 different image types in the COCO12-Formats dataset."""
#     model = DuoYOLO(CFG)
#     model.train(data="coco12-formats.yaml", epochs=2, imgsz=32, cache="disk", batch=-1, close_mosaic=1, name="model")
#     model(SOURCE)


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_train_ndjson():
    """Test training the YOLO model using NDJSON format dataset."""
    model = DuoYOLO(WEIGHTS_DIR / "yolo11n.pt")
    model.train(data=f"{ASSETS_URL}/coco8-ndjson.ndjson", epochs=1, imgsz=32)


@pytest.mark.parametrize("scls", [False, True])
def test_train_pretrained(scls):
    """Test training of the YOLO model starting from a pre-trained checkpoint."""
    model = DuoYOLO(WEIGHTS_DIR / "yolo11n-seg.pt")
    model.train(
        data="coco8-seg.yaml", epochs=1, imgsz=32, cache="ram", copy_paste=0.5, mixup=0.5, name=0, single_cls=scls
    )
    model(SOURCE)


def test_all_model_yamls():
    """Test YOLO model creation for all available YAML configurations in the `cfg/models` directory."""
    for m in (ROOT / "cfg" / "models").rglob("*.yaml"):
        if "rtdetr" in m.name:
            if TORCH_1_11:
                _ = RTDETR(m.name)(SOURCE, imgsz=640)  # must be 640
        else:
            DuoYOLO(m.name)


@pytest.mark.skipif(WINDOWS, reason="Windows slow CI export bug https://github.com/ultralytics/ultralytics/pull/16003")
def test_workflow():
    """Test the complete workflow including training, validation, prediction, and exporting."""
    model = DuoYOLO(MODEL)
    model.train(data="coco8.yaml", epochs=1, imgsz=32, optimizer="SGD")
    model.val(imgsz=32)
    model.predict(SOURCE, imgsz=32)
    model.export(format="torchscript")  # WARNING: Windows slow CI export bug


def test_predict_callback_and_setup():
    """Test callback functionality during YOLO prediction setup and execution."""

    def on_predict_batch_end(predictor):
        """Callback function that handles operations at the end of a prediction batch."""
        path, im0s, _ = predictor.batch
        im0s = im0s if isinstance(im0s, list) else [im0s]
        bs = [predictor.dataset.bs for _ in range(len(path))]
        predictor.results = zip(predictor.results, im0s, bs)  # results is list[batch_size]

    model = DuoYOLO(MODEL)
    model.add_callback("on_predict_batch_end", on_predict_batch_end)

    dataset = load_inference_source(source=SOURCE)
    bs = dataset.bs  # access predictor properties
    results = model.predict(dataset, stream=True, imgsz=160)  # source already setup
    for r, im0, bs in results:
        print("test_callback", im0.shape)
        print("test_callback", bs)
        boxes = r.boxes  # Boxes object for bbox outputs
        print(boxes)


@pytest.mark.parametrize("model", MODELS)
def test_results(model: str, tmp_path):
    """Test YOLO model results processing and output in various formats."""
    im = "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/im/boats.jpg" if model == "yolo11n-obb.pt" else SOURCE
    results = DuoYOLO(WEIGHTS_DIR / model)([im, im], imgsz=160)

    def check_results(r):
        assert len(r), f"'{model}' results should not be empty!"
        r = r.cpu().numpy()
        print(r, len(r), r.path)  # print numpy attributes
        r = r.to(device="cpu", dtype=torch.float32)
        r.save_txt(txt_file=tmp_path / "runs/tests/label.txt", save_conf=True)
        r.save_crop(save_dir=tmp_path / "runs/tests/crops/")
        r.to_df(decimals=3)  # Align to_ methods: https://docs.ultralytics.com/modes/predict/#working-with-results
        r.to_csv()
        r.to_json(normalize=True)
        r.plot(pil=True, save=True, filename=tmp_path / "results_plot_save.jpg")
        r.plot(conf=True, boxes=True)
        print(r, len(r), r.path)  # print after methods
    
    for r in results:
        if type(r) is MultitaskResults:
            for task, r_t in r.results_dict.items():
                check_results(r_t)
        else:        
            check_results(r)


def test_labels_and_crops():
    """Test output from prediction args for saving YOLO detection labels and crops."""
    imgs = [SOURCE, ASSETS / "zidane.jpg"]
    results = DuoYOLO(WEIGHTS_DIR / "yolo11n.pt")(imgs, imgsz=320, save_txt=True, save_crop=True)
    save_path = Path(results[0].save_dir)
    for r in results:
        im_name = Path(r.path).stem
        cls_idxs = r.boxes.cls.int().tolist()
        # Check that detections are made (at least 2 detections per image expected)
        assert len(cls_idxs) >= 2, f"Expected at least 2 detections, got {len(cls_idxs)}"
        # Check label path
        labels = save_path / f"labels/{im_name}.txt"
        assert labels.exists(), f"Label file {labels} does not exist"
        # Check detections match label count
        label_count = len([line for line in labels.read_text().splitlines() if line])
        assert len(r.boxes.data) == label_count, f"Box count {len(r.boxes.data)} != label count {label_count}"
        # Check crops path and files
        crop_dirs = list((save_path / "crops").iterdir())
        crop_files = [f for p in crop_dirs for f in p.glob("*")]
        # Crop directories match detections
        crop_dir_names = {d.name for d in crop_dirs}
        assert all(r.names.get(c) in crop_dir_names for c in cls_idxs), (
            f"Crop dirs {crop_dir_names} don't match classes {cls_idxs}"
        )
        # Same number of crops as detections
        crop_count = len([f for f in crop_files if im_name in f.name])
        assert crop_count == len(r.boxes.data), f"Crop count {crop_count} != detection count {len(r.boxes.data)}"


def test_cfg_init():
    """Test configuration initialization utilities from the 'ultralytics.cfg' module."""
    from ultralytics.cfg import check_dict_alignment, copy_default_cfg, smart_value

    with contextlib.suppress(SyntaxError):
        check_dict_alignment({"a": 1}, {"b": 2})
    copy_default_cfg()
    (Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")).unlink(missing_ok=False)

    # Test smart_value() with comprehensive cases
    # Test None conversion
    assert smart_value("none") is None
    assert smart_value("None") is None
    assert smart_value("NONE") is None

    # Test boolean conversion
    assert smart_value("true") is True
    assert smart_value("True") is True
    assert smart_value("TRUE") is True
    assert smart_value("false") is False
    assert smart_value("False") is False
    assert smart_value("FALSE") is False

    # Test numeric conversion (ast.literal_eval)
    assert smart_value("42") == 42
    assert smart_value("-42") == -42
    assert smart_value("3.14") == 3.14
    assert smart_value("-3.14") == -3.14
    assert smart_value("1e-3") == 0.001

    # Test list/tuple conversion (ast.literal_eval)
    assert smart_value("[1, 2, 3]") == [1, 2, 3]
    assert smart_value("(1, 2, 3)") == (1, 2, 3)
    assert smart_value("[640, 640]") == [640, 640]

    # Test dict conversion (ast.literal_eval)
    assert smart_value("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}

    # Test string fallback (when ast.literal_eval fails)
    assert smart_value("some_string") == "some_string"
    assert smart_value("path/to/file") == "path/to/file"
    assert smart_value("hello world") == "hello world"

    # Test that code injection is prevented (ast.literal_eval safety)
    # These should return strings, not execute code
    # assert smart_value("__import__('os').system('ls')") == "__import__('os').system('ls')"
    # assert smart_value("eval('1+1')") == "eval('1+1')"
    # assert smart_value("exec('x=1')") == "exec('x=1')"

@pytest.fixture
def image():
    """Load and return an image from a predefined source (OpenCV BGR)."""
    return cv2.imread(str(SOURCE))

@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_model_tune():
    """Tune YOLO model for performance improvement."""
    DuoYOLO("yolo11n.pt").tune(
        data=["coco8.yaml", "coco8-grayscale.yaml"], plots=False, imgsz=32, epochs=1, iterations=2, device="cpu"
    )
    DuoYOLO("yolo11n-pose.pt").tune(data="coco8-pose.yaml", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")
    DuoYOLO("yolo11n-cls.pt").tune(data="imagenet10", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE or not checks.IS_PYTHON_MINIMUM_3_10, reason="environment is offline")
def test_model_tune_ray():
    """Tune YOLO model for performance improvement."""
    DuoYOLO("yolo11n-cls.pt").tune(
        data="imagenet10",
        use_ray=True,
        plots=False,
        imgsz=32,
        epochs=1,
        iterations=2,
        search_alg="random",
        device="cpu",
    )


def test_model_embeddings():
    """Test YOLO model embeddings extraction functionality."""
    model_detect = DuoYOLO(MODEL)
    model_segment = DuoYOLO(WEIGHTS_DIR / "yolo11n-seg.pt")

    for batch in [SOURCE], [SOURCE, SOURCE]:  # test batch size 1 and 2
        assert len(model_detect.embed(source=batch, imgsz=32)) == len(batch)
        assert len(model_segment.embed(source=batch, imgsz=32)) == len(batch)


@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="YOLOWorld with CLIP is not supported in Python 3.12")
@pytest.mark.skipif(
    checks.IS_PYTHON_3_8 and LINUX and ARM64,
    reason="YOLOWorld with CLIP is not supported in Python 3.8 and aarch64 Linux",
)
def test_yolo_world():
    """Test YOLO world models with CLIP support."""
    model = DuoYOLO(WEIGHTS_DIR / "yolov8s-world.pt")  # no YOLO11n-world model yet
    model.set_classes(["tree", "window"])
    model(SOURCE, conf=0.01)

    model = DuoYOLO(WEIGHTS_DIR / "yolov8s-worldv2.pt")  # no YOLO11n-world model yet
    # Training from a pretrained model. Eval is included at the final stage of training.
    # Use dota8.yaml which has fewer categories to reduce the inference time of CLIP model
    model.train(
        data="dota8.yaml",
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
    )

    # test WorWorldTrainerFromScratch
    from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

    model = DuoYOLO("yolov8s-worldv2.yaml")  # no YOLO11n-world model yet
    model.train(
        data={"train": {"yolo_data": ["dota8.yaml"]}, "val": {"yolo_data": ["dota8.yaml"]}},
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
        trainer=WorldTrainerFromScratch,
    )


# @pytest.mark.skipif(not TORCH_1_13, reason="YOLOE with CLIP requires torch>=1.13")
# @pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="YOLOE with CLIP is not supported in Python 3.12")
# @pytest.mark.skipif(
#     checks.IS_PYTHON_3_8 and LINUX and ARM64,
#     reason="YOLOE with CLIP is not supported in Python 3.8 and aarch64 Linux",
# )
# def test_yoloe(tmp_path):
#     """Test YOLOE models with MobileCLIP support."""
#     # Predict
#     # text-prompts
#     model = DuoYOLO(WEIGHTS_DIR / "yoloe-11s-seg.pt")
#     model.set_classes(["person", "bus"])
#     model(SOURCE, conf=0.01)

#     from ultralytics import YOLOE
#     from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

#     # visual-prompts
#     visuals = dict(
#         bboxes=np.array([[221.52, 405.8, 344.98, 857.54], [120, 425, 160, 445]]),
#         cls=np.array([0, 1]),
#     )
#     model.predict(
#         SOURCE,
#         visual_prompts=visuals,
#         predictor=YOLOEVPSegPredictor,
#     )

#     # Val
#     model = YOLOE(WEIGHTS_DIR / "yoloe-11s-seg.pt")
#     # text prompts
#     model.val(data="coco128-seg.yaml", imgsz=32)
#     # visual prompts
#     model.val(data="coco128-seg.yaml", load_vp=True, imgsz=32)

#     # Train, fine-tune
#     from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer, YOLOESegTrainerFromScratch

#     model = YOLOE("yoloe-11s-seg.pt")
#     model.train(
#         data="coco128-seg.yaml",
#         epochs=1,
#         close_mosaic=1,
#         trainer=YOLOEPESegTrainer,
#         imgsz=32,
#     )
#     # Train, from scratch
#     data_dict = dict(train=dict(yolo_data=["coco128-seg.yaml"]), val=dict(yolo_data=["coco128-seg.yaml"]))
#     data_yaml = tmp_path / "yoloe-data.yaml"
#     YAML.save(data=data_dict, file=data_yaml)
#     for data in [data_dict, data_yaml]:
#         model = YOLOE("yoloe-11s-seg.yaml")
#         model.train(
#             data=data,
#             epochs=1,
#             close_mosaic=1,
#             trainer=YOLOESegTrainerFromScratch,
#             imgsz=32,
#         )

#     # prompt-free
#     # predict
#     model = YOLOE(WEIGHTS_DIR / "yoloe-11s-seg-pf.pt")
#     model.predict(SOURCE)
#     # val
#     model = YOLOE("yoloe-11s-seg.pt")  # or select yoloe-m/l-seg.pt for different sizes
#     model.val(data="coco128-seg.yaml", imgsz=32)


def test_yolov10():
    """Test YOLOv10 model training, validation, and prediction functionality."""
    model = DuoYOLO("yolov10n.yaml")
    # train/val/predict
    model.train(data="coco8.yaml", epochs=1, imgsz=32, close_mosaic=1, cache="disk")
    model.val(data="coco8.yaml", imgsz=32)
    model.predict(imgsz=32, save_txt=True, save_crop=True, augment=True)
    model(SOURCE)


def test_multichannel():
    """Test YOLO model multi-channel training, validation, and prediction functionality."""
    model = DuoYOLO("yolo11n.pt")
    model.train(data="coco8-multispectral.yaml", epochs=1, imgsz=32, close_mosaic=1, cache="disk")
    model.val(data="coco8-multispectral.yaml")
    im = np.zeros((32, 32, 10), dtype=np.uint8)
    model.predict(source=im, imgsz=32, save_txt=True, save_crop=True, augment=True)
    model.export(format="onnx")


# @pytest.mark.parametrize(
#     "task,model,data",
#     TASK_MODEL_DATA,
#     ids=[f"{task}-{model.name}" for task, model, _ in TASK_MODEL_DATA],
# )
# def test_grayscale(task: str, model: str, data: str, tmp_path) -> None:
#     """Test YOLO model grayscale training, validation, and prediction functionality."""
#     if task == "classify":  # not support grayscale classification yet
#         return
#     grayscale_data = tmp_path / f"{Path(data).stem}-grayscale.yaml"
#     data = check_det_dataset(data)
#     data["channels"] = 1  # add additional channels key for grayscale
#     YAML.save(data=data, file=grayscale_data)
#     # remove npy files in train/val splits if exists, might be created by previous tests
#     for split in {"train", "val"}:
#         for npy_file in (Path(data["path"]) / data[split]).glob("*.npy"):
#             npy_file.unlink()

#     model = DuoYOLO(model)
#     model.train(data=grayscale_data, epochs=1, imgsz=32, close_mosaic=1, cache="ram")
#     model.val(data=grayscale_data)
#     im = np.zeros((32, 32, 1), dtype=np.uint8)
#     model.predict(source=im, imgsz=32, save_txt=True, save_crop=True, augment=True)
#     export_model = model.export(format="onnx")

#     model = DuoYOLO(export_model, task=task)
#     model.predict(source=im, imgsz=32)
