import torch
import numpy as np
from typing import Dict, Sequence
from torchmetrics import Metric
from scipy.optimize import linear_sum_assignment
from torchmetrics_ext.util import get_aabb_per_pair_ious


class MultiTargetGroundingMetric(Metric):
    r"""
    Base class for 3D visual grounding benchmarks where each description refers to zero, one or multiple target objects.
    Computes F1-scores at multiple IoU thresholds (F1@kIoU), matching predicted and ground truth axis-aligned bounding
    boxes with the Hungarian algorithm based on IoUs.

    Subclasses must define ``eval_types`` and implement ``_load_gt_data`` to populate ``self.gt_data``, a dictionary
    mapping each description identifier to ``{"gt_aabbs": np.ndarray of shape (M, 2, 3) or None, "eval_type": str}``.
    Evaluation types starting with ``"zt"`` (zero target) are reported without IoU thresholds.
    """

    iou_thresholds: Sequence[float] = (0.25, 0.5)
    eval_types: Sequence[str] = ()

    def __init__(self, split: str, strict: bool = False):
        super().__init__()
        self.strict = strict

        # initialize metrics
        for eval_type in (*self.eval_types, "all"):
            self.add_state(name=f"{eval_type}_total", default=torch.tensor(0), dist_reduce_fx="sum")
            for iou_threshold in self.iou_thresholds:
                self.add_state(
                    name=f"{eval_type}_f1_thresh_{iou_threshold}", default=torch.tensor(0.0), dist_reduce_fx="sum"
                )

        # initialize dataset
        self.gt_data = {}
        self._load_gt_data(split=split)

    def _load_gt_data(self, split: str) -> None:
        raise NotImplementedError

    def get_all_data_ids(self):
        return list(self.gt_data.keys())

    def _eval_zt(self, pred: torch.Tensor) -> np.ndarray:
        f1_score = 1.0 if len(pred) == 0 else 0.0
        return np.full(len(self.iou_thresholds), f1_score)

    def _eval_st_or_mt(self, pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        if len(pred) == 0:
            return np.zeros(len(self.iou_thresholds))

        # initialize the cost matrix
        square_matrix_len = max(len(target), len(pred))
        iou_matrix = np.zeros(shape=(square_matrix_len, square_matrix_len), dtype=np.float64)

        # calculate ious for all combinations
        ious = get_aabb_per_pair_ious(target.to(device=pred.device, dtype=pred.dtype), pred)
        iou_matrix[: ious.shape[0], : ious.shape[1]] = ious.cpu().numpy()

        # apply matching algorithm
        row_idx, col_idx = linear_sum_assignment(iou_matrix, maximize=True)
        matched_ious = iou_matrix[row_idx, col_idx]

        iou_thresholds = np.array(self.iou_thresholds, dtype=np.float32)[:, None]
        tp = (matched_ious >= iou_thresholds).sum(axis=1)

        # calculate f1-scores for each iou threshold
        return 2 * tp / (len(pred) + len(target))

    def update(self, preds: Dict[str, torch.Tensor]) -> None:
        """
        Processes a batch of predicted results, evaluates them against ground truth, and updates
        internal F1-score statistics for all IoU thresholds.

        Args:
            preds (dict):
            A dictionary mapping each unique description identifier to its predicted axis-aligned bounding boxes.
            Each value is a tensor of shape (N, 2, 3), representing the min and max 3D coordinates for N predicted
            boxes. Use an empty tensor to predict that no object matches the description.
        """
        if self.strict and preds.keys() != self.gt_data.keys():
            raise ValueError("Mismatched IDs between predictions and dataset")

        for key, pred_aabbs in preds.items():
            if key not in self.gt_data:
                raise KeyError(f"id {key} is not in the ground truth dataset")
            eval_type = self.gt_data[key]["eval_type"]

            if eval_type.startswith("zt"):
                f1_scores = self._eval_zt(pred_aabbs)
            else:
                f1_scores = self._eval_st_or_mt(pred_aabbs, torch.from_numpy(self.gt_data[key]["gt_aabbs"]))

            for prefix in (eval_type, "all"):
                self.__dict__[f"{prefix}_total"] += 1
                for iou_threshold, f1_score in zip(self.iou_thresholds, f1_scores):
                    name = f"{prefix}_f1_thresh_{iou_threshold}"
                    self.__dict__[name] += f1_score

    def compute(self) -> Dict[str, torch.Tensor]:
        output_dict = {}
        for eval_type in (*self.eval_types, "all"):
            total = self.__dict__[f"{eval_type}_total"]
            if eval_type.startswith("zt"):
                # the zt case doesn't depend on IoU thresholds
                output_dict[eval_type] = self.__dict__[f"{eval_type}_f1_thresh_{self.iou_thresholds[0]}"] / total
                continue
            for iou_threshold in self.iou_thresholds:
                output_dict[f"{eval_type}_{iou_threshold}"] = self.__dict__[f"{eval_type}_f1_thresh_{iou_threshold}"] / total
        return output_dict
