from typing import Any
import numpy as np
from itertools import repeat
from pyparsing import Path
import torch
from ultralytics.utils import LOGGER, DataExportMixin, SimpleClass, TryExcept, plt_settings
from ultralytics.utils.metrics import DetMetrics, SegmentMetrics, ClassifyMetrics, PoseMetrics, OBBMetrics, ConfusionMatrix

class MultitaskConfusionMatrix(DataExportMixin):
    """
    Wrapper for multiple Confusion Matrices, one per task.
    
    Manages per-task confusion matrices for multitask validation. Supports classification,
    detection, segmentation, pose, and OBB tasks.
    """

    def __init__(self, names: list[dict[int, str]] = [], tasks: list[str] = [], save_matches: bool = False):
        """
        Initialize MultitaskConfusionMatrix with task-specific matrices.

        Args:
            names: Class names dict for each task
            tasks: Task type list (detect, classify, segment, pose, obb)
            save_matches: Whether to save matches for visualization
        """
        assert len(names) == len(tasks), "Length of names and tasks must be the same."
        self.matrices = {}
        for i, (task, name) in enumerate(zip(tasks, names)):
            matrix_name = f"task_{i}"
            self.matrices[matrix_name] = ConfusionMatrix(names=name, task=task, save_matches=save_matches)

    def _append_matches(self, mtype: str, batch: list[dict[str, Any]], idx: int) -> None:
        """
        Append the matches to TP, FP, FN or GT list for the last batch.

        This method updates the matches dictionary by appending specific batch data
        to the appropriate match type (True Positive, False Positive, or False Negative).

        Args:
            mtype (str): Match type identifier ('TP', 'FP', 'FN' or 'GT').
            batch (list[dict[str, Any]]): Batch data containing detection results with keys
                like 'bboxes', 'cls', 'conf', 'keypoints', 'masks'.
            idx (int): Index of the specific detection to append from the batch.
        """
        for i, batchi in enumerate(batch):
            self.matrices[f"task_{i}"]._append_matches(mtype, batchi, idx)

    def process_cls_preds(self, task_key: str, preds: list[torch.Tensor], targets: list[torch.Tensor]) -> None:
        """
        Update confusion matrix for classification task.

        Args:
            preds (list[N, min(nc,5)]): Predicted class labels.
            targets (list[N, 1]]): Ground truth class labels.
        """
        self.matrices[task_key].process_cls_preds(preds, targets)

    def process_batch(
        self,
        task_key:str,
        detections: dict[str, torch.Tensor],
        batch: dict[str, Any],
        conf: list[float] | float = 0.25,
        iou_thres: list[float] | float = 0.45,
    ) -> None:
        """
        Update confusion matrix for object detection task.

        Args:
            task_key (str): Task Key e.g 'task_0'
            detections (dict[str, torch.Tensor]): Dictionary containing detected bounding boxes and their associated information.
                                       Should contain 'cls', 'conf', and 'bboxes' keys, where 'bboxes' can be
                                       Array[N, 4] for regular boxes or Array[N, 5] for OBB with angle.
            batch (dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' (Array[M, 4]| Array[M, 5]) and
                'cls' (Array[M]) keys, where M is the number of ground truth objects.
            conf (float, optional): Confidence threshold for detections.
            iou_thres (float, optional): IoU threshold for matching detections to ground truth.
        """
        matrix = self.matrices[task_key]
        if matrix.task == "classify":
            matrix.process_cls_preds([detections['cls'].unsqueeze(0)], [batch['cls']])
        else:
            matrix.process_batch(detections, batch, conf=conf, iou_thres=iou_thres)

    def matrix(self):
        """Return the confusion matrix per task."""
        return [m.matrix for m in self.matrices.values()]

    def tp_fp(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Return true positives and false positives.

        Returns:
        list[
            tp (np.ndarray): True positives.
            fp (np.ndarray): False positives.]
        """
        return [m.tp_fp() for m in self.matrices.values()]
    
    def plot_matches(self, img: torch.Tensor, im_file: str, save_dir: Path) -> None:
        """
        Plot grid of GT, TP, FP, FN for each image.

        Args:
            img (torch.Tensor): Image to plot onto.
            im_file (str): Image filename to save visualizations.
            save_dir (Path): Location to save the visualizations to.
        """
        for i, m in enumerate(self.matrices.values()):
            m.plot_matches(img, im_file, save_dir / f"task_{i}")

    def plot(self, normalize: bool = True, save_dir: str = "", on_plot=None):
        """
        Plot the confusion matrix using matplotlib and save it to a file.

        Args:
            normalize (bool, optional): Whether to normalize the confusion matrix.
            save_dir (str, optional): Directory where the plot will be saved.
            on_plot (callable, optional): An optional callback to pass plots path and data when they are rendered.
        """
        for name, m in self.matrices.items():
            m.plot(normalize=normalize, save_dir=Path(save_dir) / name, on_plot=on_plot)

    def print(self):
        """Print the confusion matrix to the console."""
        for name, m in self.matrices.items():
            LOGGER.info(f"Confusion Matrix for {name}:")
            m.print()

    def summary(self, normalize: bool = False, decimals: int = 5) -> dict[str, list[dict[str, float]]]:
        """
        Generate a summarized representation of the confusion matrix as a list of dictionaries, with optional
        normalization. This is useful for exporting the matrix to various formats such as CSV, XML, HTML, JSON, or SQL.

        Args:
            normalize (bool): Whether to normalize the confusion matrix values.
            decimals (int): Number of decimal places to round the output values to.

        Returns:
            (dict[str,list[dict[str, float]]]): A list of dictionaries, each representing one predicted class with corresponding values for all actual classes for each class
        """
        summary = {}
        for name, m in self.matrices.items():
            summary[name] = m.summary(normalize=normalize, decimals=decimals)
        return summary
        

class UpdatedClassifyMetrics(ClassifyMetrics):
    """
    Classification metrics for multitask models.
    
    Extends ClassifyMetrics to track per-class top-1, top-5 accuracy and balanced accuracy.
    """
    def __init__(self, names : dict[int, str] = None):
        super().__init__()
        self.stats = dict(target=[], pred=[])
        self.names = names
        self.nt_per_class = None
        self.top1_per_class = None
        self.top5_per_class = None
        self.recall_per_class = None
        self.bacc = 0

    def update_stats(self, stat: dict[str, Any]) -> None:
        """
        Update statistics by appending new values to existing stat collections.

        Args:
            stat (dict[str, any]): Dictionary containing new statistical values to append.
                         Keys should match existing keys in self.stats.
        """
        for k in self.stats.keys():
            self.stats[k].extend(stat[k])
    
    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """
        Process predicted results and update metrics.
        """

        self.nt_per_class = np.bincount(np.concatenate(self.stats['target']).astype(int), minlength=len(self.names))

        self.top1_per_class = np.zeros(len(self.names))
        self.top5_per_class = np.zeros(len(self.names))
        self.recall_per_class = np.zeros(len(self.names))

        pred, targets = torch.cat(self.stats['pred']), torch.cat(self.stats['target'])
        if targets.shape[0] == 0:
            LOGGER.warning("No statistics to process for classification. Returning empty results.")
            return {'top1': np.array(0), 'top5': np.array(0)}
        
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
        self.top1, self.top5 = acc.mean(0).tolist()
        
        # Calculate per-class accuracy
        for i in range(len(self.names)):
            # Get indices where target equals class i
            class_mask = targets == i
            if class_mask.sum() > 0:  # Only calculate if there are samples for this class
                self.top1_per_class[i] = acc[class_mask, 0].mean().item()
                self.top5_per_class[i] = acc[class_mask, 1].mean().item()

                tp = correct[class_mask, 0].sum().item()  # True Positives for class i
                actual_pos = class_mask.sum().item()  # Actual positives for class i
                self.recall_per_class[i] = tp / (actual_pos)

        self.bacc = self.recall_per_class.mean().item() # Balanced accuracy is the mean of per-class recall

        return {'top1': np.array(self.top1), 'top5': np.array(self.top5), 'bacc': np.array(self.bacc)}
    
    def clear_stats(self) -> None:
        """
        Clear statistics by resetting all stats to None.
        """
        self.stats = dict(target=[], pred=[])

    def mean_results(self) -> list[float]:
        """Returns results as a list: [top1, top5, bacc]."""
        return [self.top1, self.top5, self.bacc]

    def class_result(self, i: int) -> tuple[float, float, float]:
        """Return result of evaluating the performance of a specific class."""
        return (self.top1_per_class[i], self.top5_per_class[i], self.recall_per_class[i])
    
    @property
    def maps(self) -> np.ndarray:
        """Classification does not have mAP, so return an empty array."""
        return np.array([])

    @property
    def ap_class_index(self) -> np.ndarray:
        """Classification metrics have an entry for every class in the correct range. So just return the full range."""
        return np.array(range(len(self.names)))
    
    @property
    def nt_per_image(self) -> np.ndarray:
        """Return number of targets per image."""
        return self.nt_per_class  # For classification, nt_per_image is equivalent to nt_per_class

    @property
    def results_dict(self) -> dict[str, float]:
        """Return a dictionary with model's performance metrics and fitness score."""
        return {**super().results_dict, "metrics/bacc": self.bacc}

    @property
    def keys(self) -> list[str]:
        """Return a list of keys for the results_dict property."""
        return super().keys + ["metrics/bacc"]
    
    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, float]]:
        """
        Generate a summarized representation of per-class classification metrics as a list of dictionaries. (Top-1 and Top-5 accuracy).

        Args:
            normalize (bool): For Classify metrics, everything is normalized  by default [0-1].
            decimals (int): Number of decimal places to round the metrics values to.

        Returns:
            (list[dict[str, float]]): A list with one dictionary containing Top-1 and Top-5 classification accuracy.
        """
        return [
            {
                "Class": name,
                "Images": self.nt_per_image[i],
                "Instances": self.nt_per_class[i],
                "top1": round(self.class_result(i)[0], decimals),
                "top5": round(self.class_result(i)[1], decimals),
                "recall": round(self.class_result(i)[2], decimals)
            }
            for i,name in self.names.items()
        ]

class MultitaskMetrics(SimpleClass):
    """
    Aggregates metrics across multiple tasks in multitask models.
    
    Creates and manages separate metric instances for each task type (detect, segment,
    classify, pose, obb) and provides unified interface for accessing results.
    """

    def __init__(self, tasks: list[str]):
        """
        Initialize a MultitaskMetrics object. Creates a metrics object for each task in the multitask setup.
        
        Args:
            tasks (list[str]): A list of task types as strings. Supported tasks include 'detect', 'segment', 'classify', 'pose', and 'obb'.
        """
        super().__init__()
        self.task = "multitask"
        self.tasks = tasks
        self.metrics: dict[str, DetMetrics | SegmentMetrics | UpdatedClassifyMetrics | PoseMetrics | OBBMetrics] = {}
        for i, task in enumerate(tasks):
            name = f"task_{i}"
            if task == 'detect':
                self.metrics[name] = DetMetrics()
            elif task == 'segment':
                self.metrics[name] = SegmentMetrics()
            elif task == 'classify':
                self.metrics[name] = UpdatedClassifyMetrics()
            elif task == 'pose':
                self.metrics[name] = PoseMetrics()
            elif task == 'obb':
                self.metrics[name] = OBBMetrics()
            else:
                raise ValueError(f"Unsupported task type: {task}")
            
    def update_stats(self, stat: dict[str, Any]) -> None:
        """
        Update statistics by appending new values to existing stat collections.

        Args:
            stat (dict[str, Any]): Dictionary containing new statistical values to append.
                         Keys should match existing keys in self.stats.
        """
        for name, metric in self.metrics.items():
            metric.update_stats({k[len(f"{name}_"):]: v for k, v in stat.items() if k.startswith(f"{name}_")})

    @property
    def names(self) -> list[dict[int, str]]:
        """
        Get the names of all classes for each task.

        Returns:
            list[dict[int, str]]: A list of dictionaries, each containing class names for a task.
        """
        names_list = []
        for name, metric in self.metrics.items():
            names_list.append(metric.names)
        return names_list

    @names.setter
    def names(self, names: list[dict[int, str]]) -> None:
        """
        Set the names of all classes for each task.

        Args:
            names (list[dict[int, str]]): A list of dictionaries, each containing class names for a task.
        """
        for (name, metric), task_names in zip(self.metrics.items(), names):
            metric.names = task_names 

    @property
    def stats(self) -> dict[str, Any]:
        """
        Aggregate statistics from all task-specific metrics.

        Returns:
            dict[str, Any]: A dictionary containing aggregated statistics from all tasks.
        """
        aggregated_stats = {}
        for name, metric in self.metrics.items():
            aggregated_stats.update({f"{name}_{k}": v for k, v in metric.stats.items()})
        return aggregated_stats
    
    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """
        Process predicted results for all tasks and update metrics.

        Args:
            save_dir (Path): Directory to save plots. Defaults to Path(".").
            plot (bool): Whether to plot precision-recall curves. Defaults to False.
            on_plot (callable | dict[str, callable], optional): Function or dictionary of functions to call after plots are generated. Defaults to None.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing concatenated statistics arrays.
        """
        stats = {}
        for name, metric in self.metrics.items():
            if isinstance(on_plot, dict):
                if name not in on_plot:
                    LOGGER.warning(f"No on_plot function provided for task '{name}'. Skipping on_plot for this task.")
                metric_on_plot = on_plot[name] if name in on_plot else None
            else:
                metric_on_plot = on_plot

            metric_stats = metric.process(save_dir=save_dir, plot=plot, on_plot=metric_on_plot)
            stats.update({f"{name}_{k}": v for k, v in metric_stats.items()})
        return stats
    
    def clear_stats(self) -> None:
        """
        Clear statistics for all task-specific metrics.
        """
        for metric in self.metrics.values():
            metric.clear_stats()

    @property
    def keys(self) -> list[str]:
        """
        Get a list of all statistic keys from all task-specific metrics.

        Returns:
            list[str]: A list of all statistic keys.
        """
        all_keys = []
        for name, metric in self.metrics.items():
            all_keys.extend([f"{name}_{k}" for k in metric.keys])
        return all_keys
    
    def mean_results(self) -> dict[str, list[float]]:
        """
        Calculate mean results for all task-specific metrics.

        Returns:
            dict[str, list[float]]: A dictionary containing mean results from all tasks.
        """
        mean_results = {}
        for name, metric in self.metrics.items():
            mean_results[name] = metric.mean_results()
        return mean_results
    
    def class_result(self, task: str, i:int) -> list[float]:
        """Return the result of evaluating the performance of a specific task on a specific class."""
        if task not in self.metrics:
            raise ValueError(f"Task '{task}' not found in metrics.")
        result = []
        for task_key, metric in self.metrics.items():
            if task_key == task:
                result.extend(metric.class_result(i))
            else:
                result.extend([float('nan')] * len(metric.keys))
        return result
    
    @property
    def maps(self) -> dict[str, np.ndarray]:
        """Return mean Average Precision (mAP) scores per class."""
        maps = {}
        for name, metric in self.metrics.items():
            maps[name] = metric.maps
        return maps

    @property
    def fitness(self) -> dict[str, float]:
        """Return the fitness of box object."""
        fitness = {}
        for name, metric in self.metrics.items():
            fitness[name] = metric.fitness
        return fitness

    @property
    def ap_class_index(self) -> dict[str, list]:
        """Return the average precision index per class."""
        ap_class_index = {}
        for name, metric in self.metrics.items():
            ap_class_index[name] = metric.ap_class_index
        return ap_class_index

    @property
    def results_dict(self) -> dict[str, float]:
        """Return dictionary of computed performance metrics and statistics."""
        keys = self.keys + [f"{name}_fitness" for name in self.metrics.keys()]
        results = []
        for r in self.mean_results().values():
            results.extend(r)
        for f in self.fitness.values():
            results.append(f)
        values = ((float(x) if hasattr(x, "item") else x) for x in results)
        return dict(zip(keys, values))

    @property
    def curves(self) -> list[str]:
        """Return a list of curves for accessing specific metrics curves."""
        curves = []
        for name, metric in self.metrics.items():
            curves.extend([f"{name}_{curve}" for curve in metric.curves])
        return curves

    @property
    def curves_results(self) -> list[list]:
        """Return a list of computed performance metrics and statistics."""
        results = []
        for name, metric in self.metrics.items():
            results.extend(metric.curves_results)
        return results

    @property
    def nt_per_class(self) -> np.ndarray:
        """Return number of targets per class."""
        nt_per_class = {}
        for name, metric in self.metrics.items():
            nt_per_class[name] = metric.nt_per_class
        return nt_per_class
    
    @property
    def nt_per_image(self) -> np.ndarray:
        """Return number of targets per image."""
        nt_per_image = {}
        for name, metric in self.metrics.items():
            nt_per_image[name] = metric.nt_per_image
        return nt_per_image
    
    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """
        Generate a summarized representation of per-class metrics for each task as a list of dictionaries. Includes shared
        scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for each class.

        Args:
           normalize (bool): For Detect metrics, everything is normalized  by default [0-1].
           decimals (int): Number of decimal places to round the metrics values to.

        Returns:
           (list[dict[str, Any]]): A list of dictionaries, each representing one class with corresponding metric values.

        """
        summary = {}
        for name, metric in self.metrics.items():
            summary[name] = metric.summary(normalize=normalize, decimals=decimals)
        return summary
    
    def to_df(self, normalize=False, decimals=5):
        """
        Create a polars DataFrame from the prediction results summary or validation metrics for every metric.

        Args:
            normalize (bool, optional): Normalize numerical values for easier comparison.
            decimals (int, optional): Decimal places to round floats.

        Returns:
            (Dict[str, DataFrame]): DataFrame containing the summary data.
        """
        dataframes = {}
        for name, metric in self.metrics.items():
            dataframes[name] = metric.to_df(normalize=normalize, decimals=decimals)
        
        return dataframes

    def to_csv(self, normalize=False, decimals=5):
        """
        Export results or metrics to CSV string format for every metric.

        Args:
           normalize (bool, optional): Normalize numeric values.
           decimals (int, optional): Decimal precision.

        Returns:
           (Dict[str, str]): Dictionary with CSV content as string for each metric.
        """
        csv_data = {}
        for name, metric in self.metrics.items():
            csv_data[name] = metric.to_csv(normalize=normalize, decimals=decimals)
        return csv_data

    def to_json(self, normalize=False, decimals=5):
        """
        Export results to JSON format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.

        Returns:
            (Dict[str, str]): Dictionary with JSON-formatted string of the results for each metric.
        """
        json_data = {}
        for name, metric in self.metrics.items():
            json_data[name] = metric.to_json(normalize=normalize, decimals=decimals)
        return json_data