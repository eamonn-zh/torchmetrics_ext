from ._spatial_vqa import SpatialVQAMetric


class VSIBenchMetric(SpatialVQAMetric):
    r"""
    Compute the accuracy for the VSI-Bench spatial VQA benchmark.

    References:
        - VSI-Bench: https://vision-x-nyu.github.io/thinking-in-space.github.io/

    Example:
        >>> from torchmetrics_ext.metrics.vqa import VSIBenchMetric
        >>> metric = VSIBenchMetric(split="test")
        >>> # preds is a dictionary mapping each unique question identifier "id" to a predicted answer
        >>> preds = {
        ...     0: "3",
        ...     1: "A",
        ...     ...
        ... }
        >>> result = metric(preds)
    """

    mcq_question_types = (
        "object_rel_distance",
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "route_planning",
        "obj_appearance_order",
    )

    numeric_question_types = (
        "object_counting",
        "object_abs_distance",
        "object_size_estimation",
        "room_size_estimation",
    )

    merged_question_types = {
        "object_rel_direction": (
            "object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"
        ),
    }

    def __init__(self, split="test", dataset_path="nyu-visionx/VSI-Bench", dir_name=None):
        super().__init__(split=split, dataset_path=dataset_path, dataset_name=dir_name)
        self.dir_name = dir_name
