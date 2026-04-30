from typing import Any, TYPE_CHECKING
import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
from ultralytics.utils.tal import TaskAlignedAssigner, RotatedTaskAlignedAssigner
from ultralytics.utils.loss import E2EDetectLoss, v8ClassificationLoss, v8DetectionLoss, v8PoseLoss, v8SegmentationLoss, BboxLoss, v8OBBLoss, RotatedBboxLoss, KeypointLoss
from ultralytics.utils.metrics import OKS_SIGMA

if TYPE_CHECKING:
    # Only import for type checking to avoid circular runtime import with
    # `duoYolo.nn.tasks` which imports this module.
    from duoYolo.nn.tasks import MultitaskModel  # pragma: no cover


class MultitaskLoss:
    """
    Criterion class for computing multitask loss across multiple heads.

    This class computes the combined loss for multitask models with multiple detection heads
    (detection, segmentation, classification, pose, OBB). Each head's loss is computed independently
    and then combined using a weighted sum with task-specific lambda weights that balance contributions.

    This class purposely avoids importing `MultitaskModel` at runtime to prevent a circular import:
    `nn.tasks -> utils.loss -> nn.tasks`.

    Attributes:
        criteria (list): List of loss criterion objects, one per detection head.
        loss_names (list[str]): Names of loss components for logging (e.g., ["task_0_box", "task_0_cls", ...])
        lambda_list (list[float]): Weights to apply to each task's total loss (default 1.0 for each).
    """

    def __init__(self, model: "MultitaskModel", lambda_list: list[float] | None = None):
        """
        Initialize MultitaskLoss with one criterion per head in the multitask model.

        Inspects the model architecture to determine which task heads are present and creates
        appropriate loss criteria. Automatically appends loss names for logging and initializes
        lambda weights for task balancing.

        Args:
            model: The multitask model instance (typed only for linting/type-checkers).
                Must have method `get_heads()` returning (head_name, head_index) tuples.
            lambda_list (list[float], optional): Task loss weights for balancing. If None, defaults to
                [1.0] for each task. Length can be less than number of tasks (fills with 1.0) or
                greater (truncates).

        Supported Heads:
            - "detect": Standard object detection (creates v8MultiDetectionLoss or MultitaskE2EDetectLoss)
            - "classify": Classification (creates v8MultiClassificationLoss)
            - "pose": Keypoint detection (creates v8MultiPoseLoss)
            - "segment": Instance segmentation (creates v8MultiSegmentationLoss)
            - "obb": Oriented bounding box (creates v8MultiOBBLoss)

        Attributes Set:
            self.criteria (list): Loss criterion objects for each head
            self.loss_names (list[str]): Names of loss components (e.g., "task_0_box", "task_0_cls")
            self.lambda_list (list[float]): Final lambda weights after validation

        Note:
            - Loss names format: "task_{idx}_{component}" (e.g., task_0_box, task_0_cls, task_0_dfl)
            - Lambda list is validated: warnings logged if length doesn't match task count
            - Missing lambda values filled with 1.0; extra values truncated
        """
        self.criteria = []
        self.loss_names = []
        for idx,(head, i) in enumerate(model.get_heads()):
            if head == "detect":
                end2end = getattr(model, "end2end", False)
                if isinstance(end2end, dict):
                    end2end = end2end.get(f"task_{idx}", False)
                self.criteria.append(MultitaskE2EDetectLoss(model, i) if end2end else v8MultiDetectionLoss(model, i))
                self.loss_names.extend([f"task_{idx}_box", f"task_{idx}_cls", f"task_{idx}_dfl"])
            elif head == "classify":
                self.criteria.append(v8MultiClassificationLoss())
                self.loss_names.append(f"task_{idx}_cls")
            elif head == "pose":
                self.criteria.append(v8MultiPoseLoss(model, i))
                self.loss_names.extend([f"task_{idx}_box",f"task_{idx}_pose",f"task_{idx}_kobj", f"task_{idx}_cls", f"task_{idx}_dfl"])
            elif head == "segment":
                self.criteria.append(v8MultiSegmentationLoss(model, i))
                self.loss_names.extend([f"task_{idx}_box",f"task_{idx}_seg", f"task_{idx}_cls", f"task_{idx}_dfl"])
            elif head == "obb":
                self.criteria.append(v8MultiOBBLoss(model, i))
                self.loss_names.extend([f"task_{idx}_box", f"task_{idx}_cls", f"task_{idx}_dfl"])
        
        self.lambda_list = lambda_list if lambda_list is not None else [1.0 for _ in self.criteria]
        if len(self.criteria) > len(self.lambda_list):
            LOGGER.warning(f"Length of lambda_list ({len(self.lambda_list)}) is smaller than number of tasks ({len(self.criteria)}). Filling up with ones.")
            self.lambda_list.extend([1.0 for _ in range(len(self.criteria) - len(self.lambda_list))])
        if len(self.criteria) < len(self.lambda_list):
            LOGGER.warning(f"Length of lambda_list ({len(self.lambda_list)}) is larger than number of tasks ({len(self.criteria)}). Truncating.")
            self.lambda_list = self.lambda_list[:len(self.criteria)]
        
    def __call__(self, preds: Any, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the total multitask loss as the weighted sum of individual head losses.

        Iterates through each task head, extracts task-specific predictions and ground truth,
        computes the loss for that task, applies lambda weighting, and accumulates the total.
        Only the total loss is scaled by lambda; per-component losses are preserved for logging.

        Args:
            preds (Any): List of predictions from each head. Each element format depends on head type:
                - Detection/OBB: (B, N, 6+) tensor [x, y, w, h, conf, cls, ...]
                - Segmentation: (B, N, 6+) tensor with proto
                - Pose: (B, N, 6+kpt_size) tensor
                - Classification: (B, num_classes) tensor
            batch (dict[str, torch.Tensor | dict[str, torch.Tensor]]): Ground truth batch with format:
                For task-specific fields (bboxes, cls, etc.): {task_key: tensor}
                For shared fields (img, ori_shape, etc.): single tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of:
                - total_loss: (num_tasks,) concatenated scaled losses from all heads
                - loss_items: (num_tasks,) concatenated unscaled loss components for logging

        Example:
            >>> model = MultitaskModel(...)
            >>> criterion = MultitaskLoss(model, lambda_list=[1.0, 0.5, 0.3])
            >>> preds = model(images)
            >>> total_loss, loss_items = criterion(preds, batch)
            >>> loss_items  # [task_0_loss, task_1_loss, task_2_loss] - unscaled
            >>> total_loss  # scaled version: [task_0*1.0, task_1*0.5, task_2*0.3]

        Note:
            - Lambda scaling only applied to total loss, not loss_items (to preserve logging)
            - Loss items concatenated across tasks for logging purposes
            - Task indexing: extracting batch[f"task_{idx}"] from dict fields
        """
        total_loss = None
        for idx, (criterion, pred, lambda_val) in enumerate(zip(self.criteria, preds, self.lambda_list)):
            task_batch = {k: v[f"task_{idx}"] if isinstance(v,dict) else v for k, v in batch.items()}
            loss = criterion(pred, task_batch)
            loss = (loss[0] * lambda_val, loss[1]) # scale only total loss, not loss items so we can still display and log the original loss
            if total_loss:
                total_loss = (torch.cat((total_loss[0], loss[0])), torch.cat((total_loss[1], loss[1])))
            else:
                total_loss = loss
        return total_loss

class v8MultiClassificationLoss(v8ClassificationLoss):
    """
    Extension of v8ClassificationLoss to support multitask models.

    Adapts the standard YOLOv8 classification loss to work within the multitask framework.
    Ensures output format consistency with other multitask loss criteria by adding an extra
    dimension to loss tensors.

    Inherits all functionality from v8ClassificationLoss while only overriding __call__ to:
    1. Reshape class labels to expected format
    2. Compute classification loss
    3. Ensure output tensors have consistent shape [(1,), (1,)] for compatibility
    """
    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute classification loss with output formatting for multitask integration.

        Args:
            preds (Any): Predicted class logits of shape (B, num_classes)
            batch (dict[str, torch.Tensor]): Batch with "cls" key containing ground truth labels

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Shape (1,) tensors:
                - loss: (1,) scalar loss value
                - loss_items: (1,) scalar loss components for logging

        Note:
            - Input cls squeezed and converted to long type for loss computation
            - Output wrapped in unsqueeze(0) to match criterion output format
        """
        batch["cls"] = torch.squeeze(batch["cls"]).type(torch.long)
        loss = super().__call__(preds, batch)
        loss = (loss[0].unsqueeze(0),  # make sure loss tensor has 1 dimension
                loss[1].unsqueeze(0))
        return loss

class v8MultiDetectionLoss(v8DetectionLoss):
    """
    Extension of v8DetectionLoss to support multitask detection models.

    Adapts YOLOv8's detection loss computation to work with multitask model architectures.
    Supports task-aligned assignment strategy for improved training stability and convergence.

    Key Components:
        - Task-aligned assigner: Assigns predictions to ground truth based on classification
          and localization scores
        - DFL (Distribution Focal Loss) for shape regression when reg_max > 1
        - BCE loss for classification confidence

    Inherits from v8DetectionLoss and adds multitask-specific initialization.
    """

    def __init__(self, model, head_index, tal_topk: int = 10):  # model must be de-paralleled
        """
        Initialize v8DetectionLoss with model parameters and task-aligned assignment settings.

        Args:
            model: Multitask model instance (must be de-paralleled)
            head_index (int): Index of the detection head in model.model[head_index]
            tal_topk (int, optional): Top-k value for task-aligned assignment. Determines how many
                predictions are matched to each ground truth object. Default: 10

        Attributes Initialized:
            self.bce: Binary cross-entropy loss for classification (reduction='none')
            self.hyp: Hyperparameters from model.args
            self.stride: Model stride values from detection module
            self.nc: Number of classes
            self.no: Number of outputs per prediction (nc + reg_max * 4)
            self.reg_max: Regression max value for DFL
            self.device: Device same as model parameters
            self.use_dfl: Whether to use DFL (True if reg_max > 1)
            self.assigner: TaskAlignedAssigner for matching
            self.bbox_loss: BboxLoss criterion
            self.proj: Projection tensor for DFL [0, 1, 2, ..., reg_max-1]
        """
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[head_index]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
    
class MultitaskE2EDetectLoss(E2EDetectLoss):
    """
    Extension of E2EDetectLoss to support multitask models.

    Implements end-to-end detection loss combining one-to-many and one-to-one matching strategies.
    The one-to-many strategy provides stable supervision early in training, while one-to-one
    provides precise matching for final refinement.

    Attributes:
        one2many: Detection loss with top-k=10 (multiple predictions per target)
        one2one: Detection loss with top-k=1 (one prediction per target)
    """

    def __init__(self, model, head_index):
        """
        Initialize E2EDetectLoss with one-to-many and one-to-one detection losses.

        Creates two detection loss instances with different task-aligned assignment settings
        to enable hybrid matching strategy during training.

        Args:
            model: Multitask model instance (must be de-paralleled)
            head_index (int): Index of the detection head in model.model[head_index]

        Attributes:
            self.one2many: v8MultiDetectionLoss with tal_topk=10
            self.one2one: v8MultiDetectionLoss with tal_topk=1
        """
        self.one2many = v8MultiDetectionLoss(model, head_index, tal_topk=10)
        self.one2one = v8MultiDetectionLoss(model, head_index, tal_topk=1)

class v8MultiOBBLoss(v8OBBLoss, v8MultiDetectionLoss):
    """
    Extension of v8OBBLoss to support multitask oriented bounding box detection.

    Specialized loss for oriented bounding box (OBB) tasks that includes rotation angles
    in addition to standard bounding box coordinates. Uses rotated task-aligned assignment
    to match rotated predictions with rotated ground truth.

    Key Differences from v8MultiDetectionLoss:
        - Uses RotatedTaskAlignedAssigner instead of standard assigner
        - Uses RotatedBboxLoss for rotation-aware bounding box regression
    """
    def __init__(self, model, head_index):  # model must be de-paralleled
        """
        Initialize v8OBBLoss with rotated task-aligned assignment.

        Calls parent v8MultiDetectionLoss.__init__ and then replaces the standard assigner
        with a rotation-aware assigner and rotated bounding box loss.

        Args:
            model: Multitask model instance (must be de-paralleled)
            head_index (int): Index of the OBB detection head in model.model[head_index]

        Attributes Overridden:
            self.assigner: RotatedTaskAlignedAssigner for rotation-aware matching
            self.bbox_loss: RotatedBboxLoss for rotation-aware bbox regression
        """
        v8MultiDetectionLoss.__init__(self, model, head_index)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

class v8MultiSegmentationLoss(v8SegmentationLoss, v8MultiDetectionLoss):
    """
    Extension of v8SegmentationLoss to support multitask instance segmentation.

    Combines bounding box detection loss with mask segmentation loss to predict both object
    locations and instance-level masks. Inherits detection loss computation and adds mask
    prediction loss during training.

    Key Attributes:
        - Inherits all detection components from v8MultiDetectionLoss
        - self.overlap: Whether masks can overlap (controls mask encoding)
    """
    def __init__(self, model, head_index):
        """
        Initialize v8SegmentationLoss with model parameters.

        Initializes detection loss components via parent and configures mask prediction
        settings from model arguments.

        Args:
            model: Multitask model instance (must be de-paralleled)
            head_index (int): Index of the segmentation head in model.model[head_index]

        Attributes:
            self.overlap: From model.args.overlap_mask - whether masks can overlap
        """
        v8MultiDetectionLoss.__init__(self, model, head_index)
        self.overlap = model.args.overlap_mask

class v8MultiPoseLoss(v8PoseLoss, v8MultiDetectionLoss):
    """
    Extension of v8PoseLoss to support multitask keypoint/pose detection.

    Combines bounding box detection loss with keypoint localization loss to predict both
    object locations and keypoint positions. Supports arbitrary keypoint configurations
    (e.g., COCO 17-point pose, custom skeletons).

    Key Components:
        - Inherits detection loss from v8MultiDetectionLoss
        - Keypoint loss: Computed with task-specific OKS (Object Keypoint Similarity) sigma values
        - Keypoint objectness loss: BCE loss for keypoint visibility prediction

    Attributes:
        self.kpt_shape: Keypoint configuration (e.g., [17, 3] for COCO)
        self.keypoint_loss: Keypoint regression loss with OKS sigma weighting
        self.bce_pose: Binary cross-entropy for keypoint visibility
    """
    def __init__(self, model, head_index):
        """
        Initialize v8PoseLoss with model parameters and keypoint-specific configuration.

        Initializes detection loss components and creates keypoint loss criterion with
        appropriate sigma values based on keypoint configuration (COCO pose vs custom).

        Args:
            model: Multitask model instance (must be de-paralleled)
            head_index (int): Index of the pose detection head in model.model[head_index]

        Attributes:
            self.kpt_shape: Keypoint shape from model (e.g., [17, 3])
            self.bce_pose: Binary cross-entropy loss for keypoint visibility
            self.keypoint_loss: KeypointLoss with OKS sigma values
                - Uses OKS_SIGMA constant if kpt_shape == [17, 3] (COCO)
                - Otherwise uses uniform sigma (1 / num_keypoints)

        Note:
            COCO pose format has 17 keypoints with 3 values each (x, y, confidence)
        """
        v8MultiDetectionLoss.__init__(self, model, head_index)
        self.kpt_shape = model.model[head_index].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)