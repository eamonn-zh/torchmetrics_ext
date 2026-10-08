import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from ._multi_target_grounding import MultiTargetGroundingMetric


class Multi3DReferMetric(MultiTargetGroundingMetric):
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

    eval_types = ("zt_wo_d", "zt_w_d", "st_wo_d", "st_w_d", "mt")

    def __init__(self, split: str = "validation", strict: bool = False):
        super().__init__(split=split, strict=strict)

    def _load_gt_data(self, split):
        raw_dataset = load_dataset("3dlg-hcvc/Multi3DRefer", split=split)
        scene_metadata_path = hf_hub_download(
            repo_id="torchmetrics-ext/metadata", filename=f"scannetv2/obj_aabbs_{split}.npz", repo_type="dataset"
        )
        scene_metadata = np.load(scene_metadata_path)
        for row in tqdm(raw_dataset, desc="Preparing evaluation dataset"):
            data_id = f'{row["scene_id"]}_{row["ann_id"]}'
            gt_aabbs = [scene_metadata[f"{row['scene_id']}_{object_id}"] for object_id in row["object_ids"]]
            gt_aabbs = np.stack(gt_aabbs) if len(gt_aabbs) > 0 else None
            self.gt_data[data_id] = {"gt_aabbs": gt_aabbs, "eval_type": row["eval_type"]}
