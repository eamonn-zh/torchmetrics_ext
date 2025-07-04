import torch
import numpy as np
from torchmetrics import Metric
from typing import Dict, Sequence
from torchmetrics.utilities import dim_zero_cat
from torchmetrics_ext.tools import get_aabb_per_pair_ious


class MeanAveragePrecisionMetric(Metric):
    r"""
    Compute the mean Average Precision for 3D detection task.
    Applicable datasets & tasks:
        - ScanNet
        - S3DIS
        - MultiScan
        - ARKitScenes
        - 3RScan
        - ScanNet++
        ...

    (Under development)
    """

    def __init__(self, semantic_classes: Sequence[int], iou_thresholds: Sequence[float]=(0.25, 0.5)):
        super().__init__()
        self.iou_thresholds = iou_thresholds
        self.semantic_classes = semantic_classes

        for sem_class in semantic_classes:
            self.add_state(name=f"total_{sem_class}", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(name=f"score_{sem_class}", default=[], dist_reduce_fx="cat")
            for iou_threshold in iou_thresholds:
                self.add_state(name=f"tp_{sem_class}_{iou_threshold}", default=[], dist_reduce_fx="cat")

    def update(
        self, pred_boxes: torch.Tensor, pred_classes: torch.Tensor, pred_scores: torch.Tensor, pred_batch_idx: torch.Tensor,
        target_boxes: torch.Tensor, target_classes: torch.Tensor, target_batch_idx: torch.Tensor
    ) -> None:

        pred_boxes = pred_boxes.to(torch.float64)  # for numerical stability
        target_boxes = target_boxes.to(torch.float64)  # for numerical stability

        unique_batch_idx = target_batch_idx.unique()

        for batch_idx in unique_batch_idx:
            current_target_mask = target_batch_idx == batch_idx
            current_pred_mask = pred_batch_idx == batch_idx

            current_target_boxes = target_boxes[current_target_mask]
            current_target_classes = target_classes[current_target_mask]

            current_pred_boxes = pred_boxes[current_pred_mask]
            current_pred_classes = pred_classes[current_pred_mask]

            current_pred_scores = pred_scores[current_pred_mask]

            current_pred_sorted_idx = current_pred_scores.argsort(descending=True)

            current_pred_boxes_sorted = current_pred_boxes[current_pred_sorted_idx]
            current_pred_classes_sorted = current_pred_classes[current_pred_sorted_idx]
            current_pred_scores_sorted = current_pred_scores[current_pred_sorted_idx]

            unique_sem_classes = torch.cat(tensors=(current_pred_classes_sorted, current_target_classes), dim=0).unique()

            for sem_class in unique_sem_classes:

                class_pred_mask = current_pred_classes_sorted == sem_class
                class_target_mask = current_target_classes == sem_class

                self.__dict__[f"total_{sem_class}"] += torch.count_nonzero(class_target_mask)

                current_ious = get_aabb_per_pair_ious(
                    boxes_1_bound=current_pred_boxes_sorted[class_pred_mask], boxes_2_bound=current_target_boxes[class_target_mask]
                )

                if current_ious.shape[1] == 0:
                    # create a dummy GT with zero iou
                    current_ious = torch.zeros(
                        size=(current_ious.shape[0], 1), dtype=current_ious.dtype, device=current_ious.device
                    )

                iou_max, iou_max_idx = current_ious.max(dim=1)

                for iou_threshold in self.iou_thresholds:
                    tp = torch.zeros(size=(iou_max.shape[0], ), dtype=torch.bool, device=iou_max.device)
                    matched_gt = {}

                    for i, (one_iou, one_iou_max_idx) in enumerate(zip(iou_max, iou_max_idx)):
                        if one_iou > iou_threshold:
                            if one_iou_max_idx.item() not in matched_gt:
                                tp[i] = True
                                matched_gt[one_iou_max_idx.item()] = True


                    self.__dict__[f"tp_{sem_class}_{iou_threshold}"].extend(tp)
                self.__dict__[f"score_{sem_class}"].extend(current_pred_scores_sorted[class_pred_mask])


    def compute(self) -> Dict[str, torch.Tensor]:
        results = {}

        for iou_threshold in self.iou_thresholds:
            mean_ap_macro_avg = np.empty(shape=len(self.semantic_classes), dtype=np.float64)
            for sem_idx, sem_class in enumerate(self.semantic_classes):
                if getattr(self, f"total_{sem_class}") == 0:
                    ap = np.nan
                elif len(self.__dict__[f"tp_{sem_class}_{iou_threshold}"]) == 0:
                    ap = 0.
                else:
                    tp = dim_zero_cat(self.__dict__[f"tp_{sem_class}_{iou_threshold}"])
                    score = dim_zero_cat(self.__dict__[f"score_{sem_class}"])

                    score_sorted_idx = score.argsort(descending=True)

                    tp_sorted = tp[score_sorted_idx]

                    fp_sorted = ~tp_sorted

                    tp_sorted = tp_sorted.cumsum(dim=0)
                    fp_sorted = fp_sorted.cumsum(dim=0)

                    recall = tp_sorted / getattr(self, f"total_{sem_class}")

                    precision = tp_sorted / (tp_sorted + fp_sorted)

                    zero = torch.zeros(size=(1, ), device=recall.device, dtype=recall.dtype)
                    one = torch.ones_like(zero)
                    mrec = torch.hstack(tensors=(zero, recall, one))
                    mpre = torch.hstack(tensors=(zero, precision, zero))

                    for i in range(mpre.shape[0] - 1, 0, -1):
                        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
                    idx = torch.where(mrec[1:] != mrec[:-1])[0]
                    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()

                results[f"mean_ap_{sem_class}_{iou_threshold}"] = ap
                mean_ap_macro_avg[sem_idx] = ap
            results[f"mean_ap_macro_avg_{iou_threshold}"] = np.nanmean(mean_ap_macro_avg)
        return results
