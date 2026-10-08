from ._spatial_vqa import SpatialVQAMetric


class ReVSIMetric(SpatialVQAMetric):
    r"""
    Compute the accuracy for the ReVSI spatial VQA benchmark.

    References:
        - ReVSI: https://3dlg-hcvc.github.io/revsi/

    Example:
        >>> from torchmetrics_ext.metrics.vqa import ReVSIMetric
        >>> metric = ReVSIMetric(subset="all_frame")
        >>> # preds is a dictionary mapping each unique question identifier "id" to a predicted answer
        >>> preds = {
        ...     0: "3",
        ...     1: "A",
        ...     ...
        ... }
        >>> result = metric(preds)
    """

    mcq_question_types = (
        "object_rel_direction_forward_easy",
        "object_rel_direction_backward_easy",
        "object_rel_direction_forward_hard",
        "object_rel_direction_backward_hard",
        "object_rel_distance_closest",
        "object_rel_distance_farthest",
        "route_planning",
    )

    numeric_question_types = (
        "object_counting_single",
        "object_counting_multiple",
        "object_abs_distance",
        "object_size_estimation",
        "room_size_estimation_single",
        "room_size_estimation_multiple",
    )

    merged_question_types = {
        "object_rel_direction": (
            "object_rel_direction_forward_easy",
            "object_rel_direction_backward_easy",
            "object_rel_direction_forward_hard",
            "object_rel_direction_backward_hard",
        ),
        "object_counting": ("object_counting_single", "object_counting_multiple"),
        "object_rel_distance": ("object_rel_distance_closest", "object_rel_distance_farthest"),
        "room_size_estimation": ("room_size_estimation_single", "room_size_estimation_multiple"),
    }

    def __init__(self, split="test", dataset_path="3dlg-hcvc/ReVSI", subset="all_frame"):
        super().__init__(split=split, dataset_path=dataset_path, dataset_name=subset)
        self.subset = subset
