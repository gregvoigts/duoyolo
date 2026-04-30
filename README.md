# duoYolo

DuoYolo is a Python package developed to support multitask training with YOLO models. It extends the Ultralytics YOLO framework and is build to be compatible with the original codebase, allowing users to leverage existing features while adding multitask capabilities. The package is designed to be flexible and modular, enabling users to easily integrate it into their workflows for various computer vision tasks.

I created duoYolo as part of my master's thesis on multitask learning on railway data.

Repository: https://github.com/gregvoigts/duoyolo
Ultralytics: https://github.com/ultralytics/ultralytics

## What duoYolo does

The package mirrors key parts of the Ultralytics structure and adapts them for multitask learning:

- Multitask dataset handling: multiple data configuration files can be passed, images are loaded from the first config, and labels are mapped by task index.
- Multitask augmentation: labels are flattened to pass through Ultralytics-style transforms and then reassembled by task.
- Multitask model support: YAML model configs can define multiple heads; outputs are returned per head.
- Multitask training pipeline: predictor, trainer, and validator variants expose outputs and labels for each task.
- Multitask metrics interface: combines multiple Ultralytics metrics behind one interface and forwards unpacked task-specific data.
- Interchangeable usage mode: when a non-multitask model is used, DuoYolo falls back to the standard Ultralytics components.
- CLI adaptation: the command-line workflow mirrors to Ultralytics cli and extended for multitask data arguments.
- Weighted multitask loss: task losses are weighted and summed.
- Classification: adapted the original ultralytics classification data structure to match with the other tasks.

## Weighted multitask loss

The training objective is:

$$
\mathcal{L}_{\text{total}} = \sum_{i=1}^{N} \lambda_i \mathcal{L}_i
$$

where:

- $N$ is the number of tasks
- $\mathcal{L}_i$ is the loss for task $i$
- $\lambda_i$ is the weight for task $i$

## Installation

Install in editable mode from this repository root:

```bash
pip install -e .
```

## Quick usage

### Python

```python
from duoYolo import DuoYOLO

model = DuoYOLO("path/to/model.yaml")
results = model.train(data={"1": "task1.yaml", "2": "task2.yaml"}, epochs=1)
```

### CLI

```bash
duoyolo train model=path/to/model.yaml data1=task1.yaml data2=task2.yaml epochs=1
```

For available options:

```bash
duoyolo help
```

## Implementation scope and limitations

DuoYolo is intended to provide Ultralytics-like functionality by reusing the original implementation where possible, but not all paths are fully validated yet.

Validated in thesis experiments:

- Classification adaptation.
- Multitask combinations were tested for detection, segmentation, and classification.

Not fully validated yet:

- Full feature parity across every Ultralytics mode and argument combination.
- Extensive CLI coverage beyond a standard train command and a multitask train command.

## License

This repository is licensed under AGPL-3.0. See LICENSE for details.
