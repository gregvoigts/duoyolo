"""YOLO task-specific trainer, validator, and predictor wrappers for DuoYOLO."""

from duoYolo.engine.validator import DuoYoloValidatorMixin
from duoYolo.utils import DEFAULT_CFG_DICT
from ultralytics.models.yolo.classify import ClassificationPredictor
from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.segment import SegmentationPredictor, SegmentationTrainer, SegmentationValidator
from ultralytics.models.yolo.obb import OBBPredictor, OBBTrainer, OBBValidator
from ultralytics.models.yolo.pose import PosePredictor, PoseTrainer, PoseValidator

# Define childclasses in this Namespace to ensure DuoYolo config initialization
class DetectionTrainer(DetectionTrainer):
    pass
class DetectionValidator(DuoYoloValidatorMixin, DetectionValidator):
    pass
class DetectionPredictor(DetectionPredictor):
    pass
class SegmentationTrainer(SegmentationTrainer):
    pass
class SegmentationValidator(DuoYoloValidatorMixin, SegmentationValidator):
    pass
class SegmentationPredictor(SegmentationPredictor):
    pass
class ClassificationPredictor(ClassificationPredictor):
    pass
class OBBTrainer(OBBTrainer):
    pass
class OBBValidator(DuoYoloValidatorMixin, OBBValidator):
    pass
class OBBPredictor(OBBPredictor):   
    pass
class PoseTrainer(PoseTrainer):
    pass
class PoseValidator(DuoYoloValidatorMixin, PoseValidator):
    pass
class PosePredictor(PosePredictor):
    pass