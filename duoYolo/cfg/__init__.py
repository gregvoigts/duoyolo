"""
Configuration and CLI entrypoint for DuoYolo.

Extends Ultralytics configuration to support multitask models. Handles command-line argument
parsing, model initialization, and task execution (train, val, predict, export).
"""

from pathlib import Path
import sys

from duoYolo import __version__
from duoYolo.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_PATH
import ultralytics.cfg as ul_cfg
from ultralytics.cfg import SOLUTION_MAP, TASK2DATA, TASK2METRIC, TASK2MODEL, TASKS, MODES, check_dict_alignment, copy_default_cfg, handle_yolo_hub, handle_yolo_settings, handle_yolo_solutions, merge_equals_args, parse_key_value_pair
from ultralytics.utils import ASSETS, IS_VSCODE, LOGGER, SETTINGS, YAML, checks, colorstr, vscode_msg

# Extend the TASK2DATA mapping to include 'multitask' task
TASK2DATA["multitask"] = {"Task_1": "coco8.yaml", "Task_2": "coco8-seg.yaml"}

# Extend the TASK2MODEL mapping to include 'multitask' task
TASK2MODEL["multitask"] = "duoyolo11n-od-seg.yaml"

# Extend the TASK2METRIC mapping to include 'multitask' task
TASK2METRIC["multitask"] = "metrics/multitask"

# Add 'multitask' to the list of supported tasks
TASKS = TASKS.union({"multitask"})

ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []

ul_cfg.CLI_HELP_MSG = f"""
    Arguments received: {str(["yolo"] + ARGV[1:])}. DuoYolo 'yolo' commands use the following syntax:

        duoYolo TASK MODE ARGS

        Where   TASK (optional) is one of {list(TASKS)}
                MODE (required) is one of {list(MODES)}
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                    See all ARGS with 'duoYolo cfg'

    1. Train a multitask model with 3 tasks for 1 epoch
        duoYolo train model=config/models/11/duoyolo11n-od-seg-cls.yaml epochs=1 data1=... data2=... data3=...
    
    2. Train a detection model for 10 epochs with an initial learning_rate of 0.01
        duoYolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01

    3. Predict a YouTube video using a pretrained segmentation model at image size 320:
        duoYolo predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

    4. Val a pretrained detection model at batch-size 1 and image size 640:
        duoYolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

    5. Export a YOLO11n classification model to ONNX format at image size 224 by 128 (no TASK required)
        duoYolo export model=yolo11n-cls.pt format=onnx imgsz=224,128

    6. Ultralytics solutions usage
        duoYolo solutions count or in {list(SOLUTION_MAP.keys())[1:-1]} source="path/to/video.mp4"

    7. Run special commands:
        duoYolo help
        duoYolo checks
        duoYolo version
        duoYolo settings
        duoYolo copy-cfg
        duoYolo cfg
        duoYolo solutions help

    Docs: https://docs.ultralytics.com
    Solutions: https://docs.ultralytics.com/solutions/
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

def entrypoint(debug: str = "") -> None:
    """
    DuoYolo entrypoint function for parsing and executing command-line arguments.

    This function serves as the main entry point for the DuoYolo CLI, parsing command-line arguments and
    executing the corresponding tasks such as training, validation, prediction, exporting models, and more.

    Args:
        debug (str): Space-separated string of command-line arguments for debugging purposes.

    Examples:
        Train a detection model for 10 epochs with an initial learning_rate of 0.01:
        >>> entrypoint("train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01")

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        >>> entrypoint("predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320")

        Validate a pretrained detection model at batch-size 1 and image size 640:
        >>> entrypoint("val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640")

    Notes:
        - If no arguments are passed, the function will display the usage help message.
        - For a list of all available commands and their arguments, see the provided help messages and the
          Ultralytics documentation at https://docs.ultralytics.com.
    """
    args = (debug.split(" ") if debug else ARGV)[1:]
    if not args:  # no arguments passed
        LOGGER.info(ul_cfg.CLI_HELP_MSG)
        return

    special = {
        "checks": checks.collect_system_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_yolo_settings(args[1:]),
        "cfg": lambda: YAML.print(DEFAULT_CFG_PATH),
        "hub": lambda: handle_yolo_hub(args[1:]),
        "login": lambda: handle_yolo_hub(args),
        "logout": lambda: handle_yolo_hub(args),
        "copy-cfg": copy_default_cfg,
        "solutions": lambda: handle_yolo_solutions(args[1:]),
        "help": lambda: LOGGER.info(ul_cfg.CLI_HELP_MSG),  # help below hub for -h flag precedence
    }
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    # Define common misuses of special commands, i.e. -h, -help, --help
    special.update({k[0]: v for k, v in special.items()})  # singular
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})  # singular
    special = {**special, **{f"-{k}": v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    overrides = {}  # basic overrides, i.e. imgsz=320
    for a in merge_equals_args(args):  # merge spaces around '=' sign
        if a.startswith("--"):
            LOGGER.warning(f"argument '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"argument '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:  # custom.yaml passed
                    LOGGER.info(f"Overriding {DEFAULT_CFG_PATH} with {v}")
                    overrides = {k: val for k, val in YAML.load(checks.check_yaml(v)).items() if k != "cfg"}
                elif k.startswith("data") and len(k) > 4 and v is not None:  # support data1=val1, data2=val2 for multi-data configs
                    idx = k.replace("data", "")
                    if not idx.isdigit():
                        raise ValueError(f"Invalid data argument '{a}'. Expected format 'data=path' or 'dataN=path' where N is an integer.")
                    if overrides.get("data") is not None and not isinstance(overrides["data"], dict):
                        raise ValueError(f"Conflicting 'data' argument '{a}' passed. Previous 'data' value: {overrides['data']}. Expected one argument like 'data=path' or multiple 'dataN=path' where N is an integer.")
                    overrides['data'] = {**overrides.get('data', {}), idx: v}  # support multiple data args like data1=val1 data2=val2
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)

        elif a in TASKS:
            overrides["task"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True  # auto-True for default bool args, i.e. 'yolo show' sets show=True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{ul_cfg.CLI_HELP_MSG}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})

    # Check keys
    check_dict_alignment(full_args_dict, overrides)

    # Mode
    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_CFG.mode or "predict"
        LOGGER.warning(f"'mode' argument is missing. Valid modes are {list(MODES)}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {list(MODES)}.\n{ul_cfg.CLI_HELP_MSG}")

    # Task
    task = overrides.pop("task", None)
    if task:
        if task not in TASKS:
            if task == "track":
                LOGGER.warning(
                    f"invalid 'task=track', setting 'task=detect' and 'mode=track'. Valid tasks are {list(TASKS)}.\n{ul_cfg.CLI_HELP_MSG}."
                )
                task, mode = "detect", "track"
            else:
                raise ValueError(f"Invalid 'task={task}'. Valid tasks are {list(TASKS)}.\n{ul_cfg.CLI_HELP_MSG}")
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task]

    # Model
    model = overrides.pop("model", DEFAULT_CFG.model)
    if model is None:
        model = "yolo11n.pt"
        LOGGER.warning(f"'model' argument is missing. Using default 'model={model}'.")
    overrides["model"] = model
    stem = Path(model).stem.lower()
    if "rtdetr" in stem:  # guess architecture
        from ultralytics import RTDETR

        model = RTDETR(model)  # no task argument
    elif "fastsam" in stem:
        from ultralytics import FastSAM

        model = FastSAM(model)
    elif "sam_" in stem or "sam2_" in stem or "sam2.1_" in stem:
        from ultralytics import SAM

        model = SAM(model)
    else: 
        if "yoloe" in stem or "world" in stem:
            from duoYolo import YOLO
            model = YOLO(model, task=task)
            cls_list = overrides.pop("classes", DEFAULT_CFG.classes)
            if cls_list is not None and isinstance(cls_list, str):
                model.set_classes(cls_list.split(","))  # convert "person, bus" -> ['person', ' bus'].
        else:
            from duoYolo import DuoYOLO
            model = DuoYOLO(model, task=task)
    # Task Update
    if task != model.task:
        if task:
            LOGGER.warning(
                f"conflicting 'task={task}' passed with 'task={model.task}' model. "
                f"Ignoring 'task={task}' and updating to 'task={model.task}' to match model."
            )
        task = model.task

    # Mode
    if mode in {"predict", "track"} and "source" not in overrides:
        overrides["source"] = (
            "https://ultralytics.com/images/boats.jpg" if task == "obb" else DEFAULT_CFG.source or ASSETS
        )
        LOGGER.warning(f"'source' argument is missing. Using default 'source={overrides['source']}'.")
    elif mode in {"train", "val"}:
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"'data' argument is missing. Using default 'data={overrides['data']}'.")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_CFG.format or "torchscript"
            LOGGER.warning(f"'format' argument is missing. Using default 'format={overrides['format']}'.")

    # Run command in python
    getattr(model, mode)(**overrides)  # default args from model

    # Show help
    LOGGER.info(f"💡 Learn more at https://docs.ultralytics.com/modes/{mode}")

    # Recommend VS Code extension
    if IS_VSCODE and SETTINGS.get("vscode_msg", True):
        LOGGER.info(vscode_msg())