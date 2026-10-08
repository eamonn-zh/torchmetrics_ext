import os
import ast
import torch
import gdown
import pandas as pd
from tqdm import tqdm
from datasets import config
from torchmetrics import Metric
from typing import Dict


class Nr3DMetric(Metric):
    r"""
    Compute the Accuracy for the Nr3D 3D visual grounding task.

    Note:
        - final metrics are computed as averages across the submitted predictions, rather than across the entire dataset.

    References:
        - ReferIt3D: https://referit3d.github.io/

    Example:
        >>> import torch
        >>> from torchmetrics_ext.metrics.visual_grounding import Nr3DMetric
        >>> metric = Nr3DMetric(split="test")
        >>> # preds is a dictionary mapping each unique description identifier (stimulus_id)
        >>> # to the predicted object_id
        >>> preds = {
        ...     "scene0565_00-chair-4-25-0-1-24": 25,
        ...     "scene0653_00-desk-6-16-13-14-15-17-18": 16,
        ...     ...
        ... }
        >>> metric(preds)
    """

    eval_types = ("easy", "hard", "view_dep", "view_indep")
    dataset_google_drive_file_ids = {
        "train": "1ZHWSUOU1VeTmv3fRw6sW1geNKCyHs2El",
        "test": "1ighHYVX6CzmMYS-FAbRg8liGkaHlM14s"
    }

    def __init__(self, split="test"):
        super().__init__()

        # initialize metrics
        for eval_type in (*self.eval_types, "all"):
            self.add_state(name=f"{eval_type}_total", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(name=f"{eval_type}_tp", default=torch.tensor(0), dist_reduce_fx="sum")

        # initialize dataset
        self._load_gt_data(split=split)

    def get_all_data_ids(self):
        return list(self.gt_data.keys())

    def _load_gt_data(self, split):
        cache_path = os.path.join(config.HF_DATASETS_CACHE, "nr3d")
        cache_path = gdown.download(id=self.dataset_google_drive_file_ids[split], output=f"{cache_path}/", resume=True)

        raw_dataset = pd.read_csv(cache_path, usecols=["stimulus_id", "target_id", "tokens"])

        # add the easy or hard label
        raw_dataset["is_easy"] = raw_dataset.stimulus_id.str.split('-', n=4).str[2].astype(int) <= 2
        target_words = {
            'front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost', 'looking', 'across'
        }
        # add the view_dep or view_indep label
        raw_dataset["is_view_dep"] = raw_dataset.tokens.apply(
            lambda x: not set(ast.literal_eval(x)).isdisjoint(target_words)
        )
        raw_dataset = raw_dataset.astype({"target_id": int})

        self.gt_data = {
            row.stimulus_id: {
                "gt_obj_id": row.target_id, "is_easy": "easy" if row.is_easy else "hard",
                "is_view_dep": "view_dep" if row.is_view_dep else "view_indep"
            }
            for row in tqdm(
                raw_dataset.itertuples(index=False), desc="Preparing evaluation dataset", total=len(raw_dataset)
            )
        }

    def update(self, preds: Dict[str, int]) -> None:
        """
        Processes a batch of predicted results, evaluates them against ground truth, and updates
        internal true positives statistics.

        Args:
            preds (dict):
            A dictionary mapping each unique description identifier (stimulus_id)
            to its predicted object_id, where each value is an integer.
        Example Input:
            preds = {
                "scene0565_00-chair-4-25-0-1-24": 25,
                "scene0653_00-desk-6-16-13-14-15-17-18": 16,
                ...
            }
        """
        totals = dict.fromkeys((*self.eval_types, "all"), 0)
        tps = dict.fromkeys((*self.eval_types, "all"), 0)
        for key, pred_obj_id in preds.items():
            if key not in self.gt_data:
                raise KeyError(f"id {key} is not in the ground truth dataset")
            gt = self.gt_data[key]
            is_tp = int(pred_obj_id) == gt["gt_obj_id"]
            for eval_type in (gt["is_easy"], gt["is_view_dep"], "all"):
                totals[eval_type] += 1
                tps[eval_type] += is_tp

        # update metrics
        for eval_type in (*self.eval_types, "all"):
            self.__dict__[f"{eval_type}_total"] += totals[eval_type]
            self.__dict__[f"{eval_type}_tp"] += tps[eval_type]

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute Acc based on inputs passed in to ``update`` previously."""
        return {
            eval_type: self.__dict__[f"{eval_type}_tp"] / self.__dict__[f"{eval_type}_total"]
            for eval_type in (*self.eval_types, "all")
        }
