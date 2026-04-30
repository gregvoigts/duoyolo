"""Neural network model wrappers that ensure DuoYOLO config registration."""

from duoYolo.utils import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, ClassificationModel, OBBModel, PoseModel

# Define childclasses in this Namespace to ensure DuoYolo config initialization
class DetectionModel(DetectionModel):
    pass
class SegmentationModel(SegmentationModel):
    pass
class ClassificationModel(ClassificationModel):
    pass
class OBBModel(OBBModel):
    pass
class PoseModel(PoseModel):
    pass