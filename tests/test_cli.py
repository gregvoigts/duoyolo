# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import subprocess
from pathlib import Path

import pytest

from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE, MODELS, TASK_MODEL_DATA
from ultralytics.utils import ASSETS, WEIGHTS_DIR
from ultralytics.utils.torch_utils import TORCH_1_11


def run(cmd: str) -> None:
    """Execute a shell command using subprocess."""
    subprocess.run(cmd.split(), check=True)


def test_special_modes() -> None:
    """Test various special command-line modes for YOLO functionality."""
    run("duoyolo help")
    run("duoyolo checks")
    run("duoyolo version")
    run("duoyolo settings reset")
    run("duoyolo cfg")


@pytest.mark.parametrize(
    "task,model,data",
    TASK_MODEL_DATA,
    ids=[f"{task}-{model.name}" for task, model, _ in TASK_MODEL_DATA],
)
def test_train(task: str, model: str, data: str | dict[str, str]) -> None:
    """Test YOLO training for different tasks, models, and datasets."""
    if isinstance(data, dict):
        run(f"duoyolo train {task} model={model} {' '.join(f'data{i}={v}' for i, v in enumerate(data.values())) } imgsz=32 epochs=1 cache=disk")
    else:
        run(f"duoyolo train {task} model={model} data={data} imgsz=32 epochs=1 cache=disk")


@pytest.mark.parametrize(
    "task,model,data",
    TASK_MODEL_DATA,
    ids=[f"{task}-{model.name}" for task, model, _ in TASK_MODEL_DATA],
)
def test_val(task: str, model: str, data: str | dict[str, str]) -> None:
    """Test YOLO validation process for specified task, model, and data using a shell command."""
    for end2end in {False, True}:
        if isinstance(data, dict):
            run(
                f"duoyolo val {task} model={model} {' '.join(f'data{i}={v}' for i, v in enumerate(data.values())) } imgsz=32 save_txt save_json visualize end2end={end2end} max_det=100 agnostic_nms"
            )
        else:
            run(
                f"duoyolo val {task} model={model} data={data} imgsz=32 save_txt save_json visualize end2end={end2end} max_det=100 agnostic_nms"
            )


@pytest.mark.parametrize(
    "task,model,data",
    TASK_MODEL_DATA,
    ids=[f"{task}-{model.name}" for task, model, _ in TASK_MODEL_DATA],
)
def test_predict(task: str, model: str, data: str) -> None:
    """Test YOLO prediction on provided sample assets for specified task and model."""
    for end2end in {False, True}:
        run(
            f"duoyolo {task} predict model={model} source={ASSETS} imgsz=32 save save_crop save_txt visualize end2end={end2end} max_det=100"
        )


@pytest.mark.parametrize("model", MODELS)
def test_export(model: str) -> None:
    """Test exporting a YOLO model to TorchScript format."""
    for end2end in {False, True}:
        run(f"duoyolo export model={model} format=torchscript imgsz=32 end2end={end2end} max_det=100")


# @pytest.mark.skipif(not TORCH_1_11, reason="RTDETR requires torch>=1.11")
# def test_rtdetr(task: str = "detect", model: Path = WEIGHTS_DIR / "rtdetr-l.pt", data: str = "coco8.yaml") -> None:
#     """Test the RTDETR functionality within Ultralytics for detection tasks using specified model and data."""
#     # Add comma, spaces, fraction=0.25 args to test single-image training
#     run(f"duoyolo predict {task} model={model} source={ASSETS / 'bus.jpg'} imgsz=160 save save_crop save_txt")
#     run(f"duoyolo train {task} model={model} data={data} --imgsz= 160 epochs =1, cache = disk fraction=0.25")


# Slow Tests -----------------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize(
    "task,model,data",
    TASK_MODEL_DATA,
    ids=[f"{task}-{model.name}" for task, model, _ in TASK_MODEL_DATA],
)
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
@pytest.mark.skipif(CUDA_DEVICE_COUNT < 2, reason="DDP is not available")
def test_train_gpu(task: str, model: str, data: str) -> None:
    """Test YOLO training on GPU(s) for various tasks and models."""
    run(f"duoyolo train {task} model={model} data={data} imgsz=32 epochs=1 device=0")  # single GPU
    run(f"duoyolo train {task} model={model} data={data} imgsz=32 epochs=1 device=0,1")  # multi GPU


@pytest.mark.parametrize(
    "solution",
    ["count", "blur", "workout", "heatmap", "isegment", "visioneye", "speed", "queue", "analytics", "trackzone"],
)
def test_solutions(solution: str) -> None:
    """Test yolo solutions command-line modes."""
    run(f"duoyolo solutions {solution} verbose=False")
