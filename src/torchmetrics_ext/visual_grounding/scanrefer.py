import torch
from torchmetrics import Metric
from typing import Dict, Sequence


class ScanReferMetric(Metric):
    r"""
    Compute the Acc@kIoU for the ScanRefer 3D visual grounding task.
    Please refer to https://daveredrum.github.io/ScanRefer/ for more details.

    Example:
        >>> import torch
        >>> from src.evaluation.scanrefer_metric import ScanReferMetric
        >>> metric = ScanReferMetric()
        >>> pred_aabbs = torch.rand(size=(3, 2, 3), dtype=torch.float32)
        >>> gt_aabbs = torch.rand(size=(3, 2, 3), dtype=torch.float32)
        >>> eval_types = ("unique", "multiple", "multiple")
        >>> metric(pred_aabbs, gt_aabbs, eval_types)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_types_mapping = {"unique": 0, "multiple": 1}
        self.add_state("unique_tp_thresh_25", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("unique_tp_thresh_50", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("multiple_tp_thresh_25", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("multiple_tp_thresh_50", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("all_tp_thresh_25", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("all_tp_thresh_50", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("unique_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("multiple_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("all_total", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def _get_batch_aabb_pair_ious_optimized(batch_boxes_1_bound: torch.Tensor, batch_boxes_2_bound: torch.Tensor) -> torch.Tensor:
        """
        :param batch_boxes_1_bound: a batch of axis-aligned bounding boxes (B, 2, 3)
        :param batch_boxes_2_bound: a batch of axis-aligned bounding boxes (B, 2, 3)
        :return: IoU values for each pair of axis-aligned bounding boxes (B, )
        """
        # directly unpack the min and max without splitting
        box_1_x_min, box_1_y_min, box_1_z_min = batch_boxes_1_bound[:, 0].unbind(dim=1)
        box_1_x_max, box_1_y_max, box_1_z_max = batch_boxes_1_bound[:, 1].unbind(dim=1)

        box_2_x_min, box_2_y_min, box_2_z_min = batch_boxes_2_bound[:, 0].unbind(dim=1)
        box_2_x_max, box_2_y_max, box_2_z_max = batch_boxes_2_bound[:, 1].unbind(dim=1)

        # calculate intersections directly
        x_a = torch.maximum(box_1_x_min, box_2_x_min)
        y_a = torch.maximum(box_1_y_min, box_2_y_min)
        z_a = torch.maximum(box_1_z_min, box_2_z_min)
        x_b = torch.minimum(box_1_x_max, box_2_x_max)
        y_b = torch.minimum(box_1_y_max, box_2_y_max)
        z_b = torch.minimum(box_1_z_max, box_2_z_max)

        # simplify volume calculations
        intersection_volume = torch.clamp((x_b - x_a), min=0) * torch.clamp((y_b - y_a), min=0) * torch.clamp(
            (z_b - z_a), min=0
        )
        box_1_volume = (box_1_x_max - box_1_x_min) * (box_1_y_max - box_1_y_min) * (box_1_z_max - box_1_z_min)
        box_2_volume = (box_2_x_max - box_2_x_min) * (box_2_y_max - box_2_y_min) * (box_2_z_max - box_2_z_min)

        # IoU calculation with epsilon to prevent division by zero
        ious = intersection_volume / (box_1_volume + box_2_volume - intersection_volume + torch.finfo(torch.float32).eps)
        return ious.flatten()

    def _convert_eval_types_to_idx(self, eval_types: Sequence[str], device) -> torch.Tensor:
        eval_types_tensor = torch.empty(size=(len(eval_types), ), dtype=torch.bool, device=device)
        for i, eval_type in enumerate(eval_types):
            eval_types_tensor[i] = self.eval_types_mapping[eval_type]
        return eval_types_tensor

    def update(self, preds: torch.Tensor, targets: torch.Tensor, eval_types: Sequence[str]) -> None:
        """
        :param preds: predicted axis-aligned bounding boxes (B, 2, 3)
        :param targets: ground truth axis-aligned bounding boxes (B, 2, 3)
        :param eval_types: a sequence of "unique" or "multiple" labels (B, )
        """

        # check input sizes
        if preds.shape != targets.shape or preds.shape[0] != len(eval_types):
            raise ValueError("preds, targets and eval_types must have the same length")

        # convert evaluation types to numerical values for convenience
        eval_types_tensor = self._convert_eval_types_to_idx(eval_types, preds.device)

        # calculate axis-aligned bounding boxes between predictions and GTs
        ious = self._get_batch_aabb_pair_ious_optimized(preds, targets)

        # count true positives above the IoU thresholds
        tp_thresh_25_mask = ious >= 0.25
        tp_thresh_50_mask = ious >= 0.50

        eval_type_unique_mask = eval_types_tensor == self.eval_types_mapping["unique"]
        eval_type_multiple_mask = eval_types_tensor == self.eval_types_mapping["multiple"]

        # update metrics
        self.all_total += targets.shape[0]
        self.unique_total += torch.count_nonzero(eval_type_unique_mask)
        self.multiple_total += torch.count_nonzero(eval_type_multiple_mask)

        self.all_tp_thresh_25 += torch.count_nonzero(tp_thresh_25_mask)
        self.all_tp_thresh_50 += torch.count_nonzero(tp_thresh_50_mask)

        self.unique_tp_thresh_25 += torch.count_nonzero(tp_thresh_25_mask & eval_type_unique_mask)
        self.unique_tp_thresh_50 += torch.count_nonzero(tp_thresh_50_mask & eval_type_unique_mask)

        self.multiple_tp_thresh_25 += torch.count_nonzero(tp_thresh_25_mask & eval_type_multiple_mask)
        self.multiple_tp_thresh_50 += torch.count_nonzero(tp_thresh_50_mask & eval_type_multiple_mask)

    def compute(self) -> Dict[str, torch.Tensor]:
        return {
            "unique_0.25": self.unique_tp_thresh_25 / self.unique_total,
            "unique_0.5": self.unique_tp_thresh_50 / self.unique_total,
            "multiple_0.25": self.multiple_tp_thresh_25 / self.multiple_total,
            "multiple_0.5": self.multiple_tp_thresh_50 / self.multiple_total,
            "all_0.25": self.all_tp_thresh_25 / self.all_total,
            "all_0.5": self.all_tp_thresh_50 / self.all_total
        }
