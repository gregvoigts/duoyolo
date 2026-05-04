"""Classification trainer using DuoYOLO datasets."""

from duoYolo.data.build import build_classify_dataset

from ultralytics.models.yolo.classify.train import ClassificationTrainer as BaseClassificationTrainer
from ultralytics.utils import emojis
from ultralytics.data.utils import check_cls_dataset
from ultralytics.utils.torch_utils import unwrap_model

from duoYolo.data.utils import check_single_dataset

class ClassificationTrainer(BaseClassificationTrainer):
    """
    Overrides the ultralytics Classification Trainer to use the DouYolo Dataset.
    """

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
            gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
            return build_classify_dataset(self.args, img_path, self.data, mode=mode, rect=mode == "val", stride=gs)
        else:
            return super().build_dataset(img_path, mode=mode, batch=batch)      


    def get_dataset(self):
        """
        Get train and validation datasets from data dictionary.

        Overrides the base method to use check_duo_datasets for multitask datasets.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """      

        try:
            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                data = check_single_dataset(self.args.data)
            else:
                data = check_cls_dataset(self.args.data, split=self.args.split)
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{self.args.data}' error ❌ {e}")) from e
        return data