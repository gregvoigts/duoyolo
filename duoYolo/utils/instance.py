import numpy as np

from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments

class MultitaskInstances(Instances):
    
    @classmethod
    def concatenate(cls, instances_list: list[Instances], axis=0) -> Instances:
        """
        Concatenate a list of Instances objects into a single Instances object. 
        Handles concatenation of bounding boxes, segments, and keypoints if present also when some instances 
        dont have some of these properties.

        Args:
            instances_list (list[Instances]): A list of Instances objects to concatenate.
            split_indices (np.ndarray): An array of indices indicating to witch index holds data from which list element
            axis (int, optional): The axis along which the arrays will be concatenated.

        Returns:
            (Instances): A new Instances object containing the concatenated bounding boxes, segments, and keypoints
                if present.

        Notes:
            The `Instances` objects in the list should have the same properties, such as the format of the bounding
            boxes, whether keypoints are present, and if the coordinates are normalized.
        """

        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) or isinstance(instance, MultitaskInstances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized

        # handle bboxes        
        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        samples_per_instance = [len(ins) for ins in instances_list]

        seg_len = [b.segments.shape[1] for b in instances_list]
        max_len = max(seg_len)
            
        cat_segments = np.concatenate(
            [
                # resample segments if there's different length
                (resample_segments(list(b.segments), max_len) if len(frozenset(seg_len)) > 1 else b.segments)
                if len(b.segments)
                else np.full((samples_per_instance[i], max_len, 2), np.nan, dtype=np.float32)  # fillup empty segments with empty arrays
                for i, b in enumerate(instances_list)
            ],
            axis=axis,
        )

        cat_keypoints = np.concatenate([
                                            b.keypoints 
                                            if len(b.keypoints)
                                            else np.full((samples_per_instance[i], max_len, 2), np.nan, dtype=np.float32) 
                                            for i, b in enumerate(instances_list)], 
                                    axis=axis) if use_keypoint else None
        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)
    