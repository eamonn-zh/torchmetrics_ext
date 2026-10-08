import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from ._multi_target_grounding import MultiTargetGroundingMetric


class ViGiL3DMetric(MultiTargetGroundingMetric):
    r"""
    Computes F1-scores at multiple IoU thresholds (F1@kIoU) for the ViGiL3D 3D visual grounding benchmark.
    Predicted and ground truth axis-aligned bounding boxes are compared using the Hungarian matching algorithm based on IoUs.

    Note:
        - ScanNet: the GT box coordinates are axis-aligned by applying the 4x4 transformation matrix provided in <scene_id>.txt from the dataset
        - ScanNet++: the GT box coordinates are directly from the dataset without applying any additional transformations
        - final metrics are computed as averages across the submitted predictions, rather than across the entire dataset.

    References:
        - ViGiL3DMetric: https://3dlg-hcvc.github.io/vigil3d/

    Example 1:
        - evaluate all predictions once
        >>> import torch
        >>> from torchmetrics_ext.metrics.visual_grounding import ViGiL3DMetric
        >>> metric = ViGiL3DMetric(split="validation")
        >>> # preds is a dictionary mapping each unique description identifier (formatted as "{scene_id}_{ann_id}")
        >>> # to a variable number of predicted axis-aligned bounding boxes in shape (N, 2, 3)
        >>> preds = {
        ...     "cf49717d-a751-417e-be93-32fa6a4aa1e4": torch.tensor([[[0., 0., 0.], [0.5, 0.5, 0.5]]]),  # 1 predicted box
        ...     "aec1e11f-43c6-4596-8f3a-161880201ef9": torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [2., 2., 2.]]]),  # 2 predicted boxes
        ...     ...
        ... }
        >>> result = metric(preds)
    Example 2:
        - evaluate predictions in batches with automatic accumulation over batches and synchronization between multiple devices
        >>> import torch
        >>> from torchmetrics_ext.metrics.visual_grounding import ViGiL3DMetric
        >>> metric = ViGiL3DMetric(split="validation")
        >>> preds_batch_1 = {
        ...     "cf49717d-a751-417e-be93-32fa6a4aa1e4": torch.tensor([[[0., 0., 0.], [0.5, 0.5, 0.5]]]),  # 1 predicted box
        ...     "aec1e11f-43c6-4596-8f3a-161880201ef9": torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [2., 2., 2.]]]),  # 2 predicted boxes
        ...     ...
        ... }
        >>> metric.update(preds_batch_1)  # can be called from different devices
        >>> preds_batch_2 = {
        ...     "9914bb89-4346-489b-9c0b-65fb3815f06a": torch.tensor([[[0.5, 0.1, 0.], [1.5, 0.5, 0.5]]]),  # 1 predicted box
        ...     ...
        ... }
        >>> metric.update(preds_batch_2)  # can be called from different devices
        >>> result = metric.compute()
        >>> metric.reset()  # reset metric state for next evaluation round
    """

    eval_types = ("zt", "st", "mt")
    scene_obj_id_parser = re.compile(r"^(?P<scene_id>.+)_(?P<ann_id>\d+)$")

    def __init__(self, split: str = "validation", strict: bool = False):
        super().__init__(split=split, strict=strict)

    def _load_gt_data(self, split):
        raw_dataset = load_dataset("3dlg-hcvc/vigil3d", split=split)

        # GT boxes of a scene may come from either ScanNet or ScanNet++, and from either the train or validation split
        for metadata_split in ("train", "validation"):
            for scene_dataset in ("scannetv2", "scannetpp"):
                self._load_gt_aabbs(raw_dataset, f"{scene_dataset}/obj_aabbs_{metadata_split}.npz")

        # check that all data ids are present
        for row in raw_dataset:
            data_id = f"{row['scene_id']}_{row['ann_id']}"
            if data_id not in self.gt_data:
                raise ValueError(f"Data ID {data_id} not found in the ground truth dataset.")

    def _load_gt_aabbs(self, raw_dataset, filename, repo_id="torchmetrics-ext/metadata"):
        scene_metadata_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        scene_metadata = np.load(scene_metadata_path)
        scene_ids = {self.scene_obj_id_parser.search(key).group("scene_id") for key in scene_metadata.keys()}

        num_processed = 0
        for row in tqdm(raw_dataset, desc=f"Preparing evaluation dataset (checking against {repo_id}:{filename})"):
            if row["scene_id"] not in scene_ids:
                continue

            data_id = f"{row['scene_id']}_{row['ann_id']}"
            gt_aabbs = [scene_metadata[f"{row['scene_id']}_{object_id}"] for object_id in row["object_ids"]]
            gt_aabbs = np.stack(gt_aabbs) if len(gt_aabbs) > 0 else None

            # 0 objects -> zt, 1 object -> st, 2+ objects -> mt
            eval_type = self.eval_types[min(len(row["object_ids"]), 2)]

            self.gt_data[data_id] = {"gt_aabbs": gt_aabbs, "eval_type": eval_type}
            num_processed += 1

        print(f"  Processed {num_processed} rows from {repo_id}:{filename}")
