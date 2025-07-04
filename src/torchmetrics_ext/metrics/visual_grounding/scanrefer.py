import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchmetrics import Metric
from datasets import load_dataset
from typing import Dict, Sequence
from huggingface_hub import hf_hub_download
from torchmetrics_ext.tools import get_batch_aabb_ious


class ScanReferMetric(Metric):
    r"""
    Compute the Acc@kIoU for the ScanRefer 3D visual grounding task.

    Note:
        - the GT box coordinates are axis-aligned by applying the 4x4 transformation matrix provided in <scene_id>.txt from the ScanNet dataset.
        - final metrics are computed as averages across the submitted predictions, rather than across the entire dataset.
        - the metric requires the ScanRefer dataset to be downloaded manually from https://github.com/daveredrum/ScanRefer?tab=readme-ov-file#dataset.

    References:
        - ScanRefer: https://daveredrum.github.io/ScanRefer/

    Example:
        >>> import torch
        >>> from torchmetrics_ext.metrics.visual_grounding import ScanReferMetric
        >>> metric = ScanReferMetric(dataset_file_path="path to the ScanRefer_filtered_val.json file", split="validation")
        >>> # preds is a dictionary mapping each unique description identifier (formatted as "{scene_id}_{object_id}_{ann_id}")
        >>> # to the predicted axis-aligned bounding boxes in shape (2, 3)
        >>> preds = {
        >>>     "scene0011_00_0_0": torch.tensor([[0., 0., 0.], [0.5, 0.5, 0.5]]),
        >>>     "scene0011_01_0_1": torch.tensor([[0., 0., 0.], [1., 1., 1.]]),
        >>>     ...
        >>> }
        >>> metric(preds)
    """

    iou_thresholds = (0.25, 0.5)
    eval_types = ("unique", "multiple")

    def __init__(self, dataset_file_path, split="validation"):
        super().__init__()

        # initialize metrics
        self.eval_types_mapping = {}
        for i, eval_type in enumerate(self.eval_types):
            self.eval_types_mapping[eval_type] = i
            self.add_state(name=f"{eval_type}_total", default=torch.tensor(0), dist_reduce_fx="sum")

            for iou_threshold in self.iou_thresholds:
                self.add_state(
                    name=f"{eval_type}_tp_thresh_{iou_threshold}", default=torch.tensor(0), dist_reduce_fx="sum"
                )

        for iou_threshold in self.iou_thresholds:
            self.add_state(name=f"all_tp_thresh_{iou_threshold}", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state(name="all_total", default=torch.tensor(0), dist_reduce_fx="sum")

        # initialize dataset
        self._load_gt_data(dataset_file_path=dataset_file_path, split=split)

    def get_all_data_ids(self):
        return list(self.gt_ids_to_idx.keys())

    def _load_gt_data(self, dataset_file_path, split):

        with open(dataset_file_path, "r") as f:
            raw_dataset = json.load(f)

        raw_dataset = pd.DataFrame(raw_dataset, columns=["scene_id", "object_id", "ann_id"])
        raw_dataset = raw_dataset.astype({"object_id": int, "ann_id": int})

        language_metadata = load_dataset("torchmetrics-ext/metadata", "ScanRefer", split=split).to_pandas()
        raw_dataset = pd.merge(raw_dataset, language_metadata, on=["scene_id", "object_id", "ann_id"])

        scene_metadata_path = hf_hub_download(
            repo_id="torchmetrics-ext/metadata", filename=f"scannetv2/obj_aabbs_{split}.npz", repo_type="dataset"
        )
        scene_metadata = np.load(scene_metadata_path)

        self.gt_ids_to_idx = {}
        gt_eval_types = []
        gt_aabbs = []
        for i, row in enumerate(
            tqdm(raw_dataset.itertuples(index=False), desc="Preparing evaluation dataset", total=len(raw_dataset))
        ):
            data_id = f"{row.scene_id}_{row.object_id}_{row.ann_id}"
            self.gt_ids_to_idx[data_id] = i
            gt_eval_types.append(self.eval_types_mapping[row.eval_type])
            gt_aabbs.append(scene_metadata[f"{row.scene_id}_{row.object_id}"])
        self.gt_eval_types = torch.from_numpy(np.stack(gt_eval_types))
        self.gt_aabbs = torch.from_numpy(np.stack(gt_aabbs))

    def update(self, preds: Dict[str, torch.Tensor]) -> None:
        """
        Processes a batch of predicted results, evaluates them against ground truth, and updates
        internal true positives statistics for all IoU thresholds.

        Args:
            preds (dict):
            A dictionary mapping each unique description identifier (formatted as "{scene_id}_{object_id}_{ann_id}")
            to its predicted axis-aligned bounding box, where each value is a tensor of shape (2, 3),
            representing the min and max 3D coordinates for the predicted box.

        Example Input:
            preds = {
                "scene0011_00_0_0": torch.tensor([[0., 0., 0.], [0.5, 0.5, 0.5]]),
                "scene0011_01_0_1": torch.tensor([[0., 0., 0.], [1., 1., 1.]]),
                ...
            }
        """
        gt_data_idx = []
        predicted_box_batch = []

        for key, value in preds.items():
            assert key in self.gt_ids_to_idx, f"id {key} is not in the ground truth dataset"
            gt_data_idx.append(self.gt_ids_to_idx[key])
            predicted_box_batch.append(value)

        predicted_box_batch = torch.stack(predicted_box_batch)

        gt_eval_type_batch = self.gt_eval_types[gt_data_idx].to(predicted_box_batch.device)
        gt_box_batch = self.gt_aabbs[gt_data_idx].to(predicted_box_batch.device)

        # calculate axis-aligned bounding boxes between predictions and GTs
        ious = get_batch_aabb_ious(predicted_box_batch, gt_box_batch)

        # count true positives above the IoU thresholds
        tp_thresh_masks = {}
        for iou_threshold in self.iou_thresholds:
            tp_thresh_masks[f"tp_thresh_{iou_threshold}_mask"] = ious >= iou_threshold

        eval_type_masks = {}
        for eval_type in self.eval_types_mapping.keys():
            eval_type_masks[eval_type] = gt_eval_type_batch == self.eval_types_mapping[eval_type]

        # update metrics
        self.all_total += len(preds)

        for eval_type in self.eval_types_mapping.keys():
            self.__dict__[f"{eval_type}_total"] += torch.count_nonzero(eval_type_masks[eval_type])

        for iou_threshold in self.iou_thresholds:
            for eval_type in self.eval_types_mapping.keys():
                tp_thresh_mask = tp_thresh_masks[f"tp_thresh_{iou_threshold}_mask"]
                self.__dict__[f"{eval_type}_tp_thresh_{iou_threshold}"] += torch.count_nonzero(tp_thresh_mask & eval_type_masks[eval_type])
            self.__dict__[f"all_tp_thresh_{iou_threshold}"] += torch.count_nonzero(tp_thresh_mask)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute Acc@kIoU based on inputs passed in to ``update`` previously."""
        output_dict = {}
        for iou_threshold in self.iou_thresholds:
            for eval_type in self.eval_types_mapping.keys():
                output_dict[f"{eval_type}_{iou_threshold}"] = self.__dict__[f"{eval_type}_tp_thresh_{iou_threshold}"] / self.__dict__[f"{eval_type}_total"]
            output_dict[f"all_{iou_threshold}"] = self.__dict__[f"all_tp_thresh_{iou_threshold}"] / self.all_total
        return output_dict
