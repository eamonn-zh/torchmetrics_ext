import torch
from torchmetrics import Metric
from typing import Dict, Sequence
from torchmetrics_ext.tool import get_batch_aabb_pair_ious


class ScanReferMetric(Metric):
    r"""
    Compute the Acc@kIoU for the ScanRefer 3D visual grounding task.
    Please refer to https://daveredrum.github.io/ScanRefer/ for more details.

    Example:
        >>> import torch
        >>> from torchmetrics_ext.visual_grounding import ScanReferMetric
        >>> metric = ScanReferMetric()
        >>> # min max bounds of 3D axis-aligned bounding boxes (B, 2, 3)
        >>> pred_aabbs = torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [2., 2., 2.]]], dtype=torch.float32)
        >>> gt_aabbs = torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [1.5, 1.5, 1.5]]], dtype=torch.float32)
        >>> gt_eval_types = ("unique", "multiple")
        >>> metric(pred_aabbs, gt_aabbs, gt_eval_types)
        {'unique_0.25': tensor(1.),
         'unique_0.5': tensor(1.),
         'multiple_0.25': tensor(1.),
         'multiple_0.5': tensor(0.),
         'all_0.25': tensor(1.),
         'all_0.5': tensor(0.5000)}
    """
    def __init__(self, iou_thresholds: Sequence[float] = (0.25, 0.5), eval_types: Sequence[str] = ("unique", "multiple")):
        super().__init__()
        self.iou_thresholds = iou_thresholds

        self.eval_types_mapping = {}
        for i, eval_type in enumerate(eval_types):
            self.eval_types_mapping[eval_type] = i
            self.add_state(name=f"{eval_type}_total", default=torch.tensor(0), dist_reduce_fx="sum")

            for iou_threshold in self.iou_thresholds:
                self.add_state(
                    name=f"{eval_type}_tp_thresh_{iou_threshold}", default=torch.tensor(0), dist_reduce_fx="sum"
                )

        for iou_threshold in self.iou_thresholds:
            self.add_state(name=f"all_tp_thresh_{iou_threshold}", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state(name="all_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _convert_eval_types_to_idx(self, eval_types: Sequence[str], device) -> torch.Tensor:
        eval_types_tensor = torch.empty(size=(len(eval_types), ), dtype=torch.uint8, device=device)
        for i, eval_type in enumerate(eval_types):
            eval_types_tensor[i] = self.eval_types_mapping[eval_type]
        return eval_types_tensor

    def update(self, preds: torch.Tensor, targets: torch.Tensor, eval_types: Sequence[str]) -> None:
        """
        :param preds: predicted axis-aligned bounding box min max bounds (B, 2, 3)
        :param targets: ground truth axis-aligned bounding box min max bounds (B, 2, 3)
        :param eval_types: a sequence of evaluation type labels (B, )
        """

        # check input sizes
        if preds.shape != targets.shape or preds.shape[0] != len(eval_types):
            raise ValueError("preds, targets and eval_types must have the same length")

        # convert evaluation types to numerical values for convenience
        eval_types_tensor = self._convert_eval_types_to_idx(eval_types, preds.device)

        # calculate axis-aligned bounding boxes between predictions and GTs
        ious = get_batch_aabb_pair_ious(preds, targets)

        # count true positives above the IoU thresholds
        tp_thresh_masks = {}
        for iou_threshold in self.iou_thresholds:
            tp_thresh_masks[f"tp_thresh_{iou_threshold}_mask"] = ious >= iou_threshold

        eval_type_masks = {}
        for eval_type in self.eval_types_mapping.keys():
            eval_type_masks[eval_type] = eval_types_tensor == self.eval_types_mapping[eval_type]

        # update metrics
        self.all_total += targets.shape[0]

        for eval_type in self.eval_types_mapping.keys():
            self.__dict__[f"{eval_type}_total"] += torch.count_nonzero(eval_type_masks[eval_type])

        for iou_threshold in self.iou_thresholds:
            for eval_type in self.eval_types_mapping.keys():
                tp_thresh_mask = tp_thresh_masks[f"tp_thresh_{iou_threshold}_mask"]
                self.__dict__[f"{eval_type}_tp_thresh_{iou_threshold}"] += torch.count_nonzero(tp_thresh_mask & eval_type_masks[eval_type])
            self.__dict__[f"all_tp_thresh_{iou_threshold}"] += torch.count_nonzero(tp_thresh_mask & eval_type_masks[eval_type])


    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute Acc@kIoU based on inputs passed in to ``update`` previously."""

        output_dict = {}
        for iou_threshold in self.iou_thresholds:
            for eval_type in self.eval_types_mapping.keys():
                output_dict[f"{eval_type}_{iou_threshold}"] = self.__dict__[f"{eval_type}_tp_thresh_{iou_threshold}"] / self.__dict__[f"{eval_type}_total"]
            output_dict[f"all_{iou_threshold}"] = self.__dict__[f"all_tp_thresh_{iou_threshold}"] / self.all_total
        return output_dict
