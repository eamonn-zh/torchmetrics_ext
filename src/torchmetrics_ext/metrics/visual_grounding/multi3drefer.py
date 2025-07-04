import torch
import numpy as np
from tqdm import tqdm
from typing import Dict
from torchmetrics import Metric
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from scipy.optimize import linear_sum_assignment
from torchmetrics_ext.tools import get_aabb_per_pair_ious


class Multi3DReferMetric(Metric):
    r"""
    Computes F1-scores at multiple IoU thresholds (F1@kIoU) for the Multi3DRefer 3D visual grounding benchmark.
    Predicted and ground truth axis-aligned bounding boxes are compared using the Hungarian matching algorithm based on IoUs.

    Note:
        - the GT box coordinates are axis-aligned by applying the 4x4 transformation matrix provided in <scene_id>.txt from the ScanNet dataset.
        - final metrics are computed as averages across the submitted predictions, rather than across the entire dataset.

    References:
        - Multi3DRefer: https://3dlg-hcvc.github.io/multi3drefer/

    Example 1:
        - evaluate all predictions once
        >>> import torch
        >>> from torchmetrics_ext.metrics.visual_grounding import Multi3DReferMetric
        >>> metric = Multi3DReferMetric(split="validation")
        >>> # preds is a dictionary mapping each unique description identifier (formatted as "{scene_id}_{ann_id}")
        >>> # to a variable number of predicted axis-aligned bounding boxes in shape (N, 2, 3)
        >>> preds = {
        ...     "scene0011_00_0": torch.tensor([[[0., 0., 0.], [0.5, 0.5, 0.5]]]),  # 1 predicted box
        ...     "scene0011_01_1": torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [2., 2., 2.]]]),  # 2 predicted boxes
        ...     ...
        ... }
        >>> result = metric(preds)
    Example 2:
        - evaluate predictions in batches with automatic accumulation over batches and synchronization between multiple devices
        >>> import torch
        >>> from torchmetrics_ext.metrics.visual_grounding import Multi3DReferMetric
        >>> metric = Multi3DReferMetric(split="validation")
        >>> preds_batch_1 = {
        ...     "scene0011_00_0": torch.tensor([[[0., 0., 0.], [0.5, 0.5, 0.5]]]),  # 1 predicted box
        ...     "scene0011_01_1": torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [2., 2., 2.]]]),  # 2 predicted boxes
        ...     ...
        ... }
        >>> metric.update(preds_batch_1)  # can be called from different devices
        >>> preds_batch_2 = {
        ...     "scene0012_00_0": torch.tensor([[[0.5, 0.1, 0.], [1.5, 0.5, 0.5]]]),  # 1 predicted box
        ...     ...
        ... }
        >>> metric.update(preds_batch_2)  # can be called from different devices
        >>> result = metric.compute()
        >>> metric.reset()  # reset metric state for next evaluation round
    """
    iou_thresholds = (0.25, 0.5)
    eval_types = ("zt_wo_d", "zt_w_d", "st_wo_d", "st_w_d", "mt")

    def __init__(self, split="validation"):
        super().__init__()

        # initialize metrics
        for i, eval_type in enumerate(self.eval_types):
            self.add_state(name=f"{eval_type}_total", default=torch.tensor(0), dist_reduce_fx="sum")

            for iou_threshold in self.iou_thresholds:
                self.add_state(
                    name=f"{eval_type}_f1_thresh_{iou_threshold}", default=torch.tensor(0), dist_reduce_fx="sum"
                )
        for iou_threshold in self.iou_thresholds:
            self.add_state(name=f"all_f1_thresh_{iou_threshold}", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state(name="all_total", default=torch.tensor(0), dist_reduce_fx="sum")

        # initialize dataset
        self._load_gt_data(split=split)

    def get_all_data_ids(self):
        return list(self.gt_data.keys())

    def _load_gt_data(self, split):
        self.gt_data = {}
        raw_dataset = load_dataset("3dlg-hcvc/Multi3DRefer", split=split)
        scene_metadata_path = hf_hub_download(
            repo_id="torchmetrics-ext/metadata", filename=f"scannetv2/obj_aabbs_{split}.npz", repo_type="dataset"
        )
        scene_metadata = np.load(scene_metadata_path)
        for row in tqdm(raw_dataset, desc="Preparing evaluation dataset"):
            data_id = f'{row["scene_id"]}_{row["ann_id"]}'
            gt_aabbs = []
            for object_id in row["object_ids"]:
                gt_aabbs.append(scene_metadata[f"{row['scene_id']}_{object_id}"])
            gt_aabbs = np.stack(gt_aabbs) if len(gt_aabbs) > 0 else None
            self.gt_data[data_id] = {
                "gt_aabbs": gt_aabbs, "eval_type": row["eval_type"]
            }

    def _eval_zt(self, pred):
        f1_score = 1 if len(pred) == 0 else 0
        f1_scores = {iou_threshold: f1_score for iou_threshold in self.iou_thresholds}
        return f1_scores

    def _eval_st_or_mt(self, pred, target):

        # initialize the cost matrix
        square_matrix_len = max(len(target), len(pred))
        iou_matrix = torch.zeros(size=(square_matrix_len, square_matrix_len), dtype=pred.dtype, device=pred.device)

        # calculate ious for all combinations
        ious = get_aabb_per_pair_ious(target, pred)
        iou_matrix[:ious.shape[0], :ious.shape[1]] = ious

        # apply matching algorithm
        iou_matrix = iou_matrix.cpu().numpy()  # TODO: only have the numpy version now
        row_idx, col_idx = linear_sum_assignment(iou_matrix * -1)

        matched_ious = iou_matrix[row_idx, col_idx]

        iou_thresholds = np.array(self.iou_thresholds, dtype=np.float32)[:, None]

        tp = (matched_ious >= iou_thresholds).sum(axis=1)

        # calculate f1-scores for each iou threshold
        f1_scores = 2 * tp / (len(pred) + len(target))

        f1_scores = {iou_threshold: f1_scores[i] for i, iou_threshold in enumerate(self.iou_thresholds)}
        return f1_scores

    def update(self, preds: Dict[str, torch.Tensor]) -> None:
        """
        Processes a batch of predicted results, evaluates them against ground truth, and updates
        internal F1-score statistics for all IoU thresholds.

        Args:
            preds (dict):
            A dictionary mapping each unique description identifier (formatted as "{scene_id}_{ann_id}")
            to its predicted axis-aligned bounding boxes. Each value is a tensor of shape (N, 2, 3),
            representing the min and max 3D coordinates for N predicted boxes.

        Example Input:
            preds = {
                "scene0011_00_0": torch.tensor([[[0., 0., 0.], [0.5, 0.5, 0.5]]]),
                "scene0011_01_1": torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [2., 2., 2.]]]),
                ...
            }
        """
        for key, pred_aabbs in preds.items():
            assert key in self.gt_data, f"id {key} is not in the ground truth dataset"
            eval_type = self.gt_data[key]["eval_type"]
            gt_aabbs = self.gt_data[key]["gt_aabbs"]

            if "zt" in eval_type:
                f1_scores = self._eval_zt(pred_aabbs)
            elif "st" in eval_type or "mt" in eval_type:
                f1_scores = self._eval_st_or_mt(pred_aabbs, torch.from_numpy(gt_aabbs))
            else:
                raise NotImplementedError

            self.__dict__[f"{eval_type}_total"] += 1
            self.all_total += 1

            for iou_threshold in self.iou_thresholds:
                self.__dict__[f"{eval_type}_f1_thresh_{iou_threshold}"] += f1_scores[iou_threshold]
                self.__dict__[f"all_f1_thresh_{iou_threshold}"] += f1_scores[iou_threshold]

    def compute(self) -> Dict[str, torch.Tensor]:
        output_dict = {}
        for iou_threshold in self.iou_thresholds:
            for eval_type in self.eval_type:
                output_dict[f"{eval_type}_{iou_threshold}"] = self.__dict__[f"{eval_type}_f1_thresh_{iou_threshold}"] / self.__dict__[f"{eval_type}_total"]
            output_dict[f"all_{iou_threshold}"] = self.__dict__[f"all_f1_thresh_{iou_threshold}"] / self.all_total

        # clean the zt case since it doesn't have thresholds
        output_dict[f"zt_w_d_f1"] = output_dict["zt_w_d_f1_thresh_0.25"]
        del output_dict["zt_w_d_f1_thresh_0.5"]
        del output_dict["zt_w_d_f1_thresh_0.25"]
        return output_dict
