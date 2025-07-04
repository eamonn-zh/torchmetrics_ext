import os
import ast
import torch
import gdown
import pandas as pd
from tqdm import tqdm
from datasets import config
from torchmetrics import Metric
from typing import Dict, Sequence


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
        >>>     "scene0565_00-chair-4-25-0-1-24": 25,
                "scene0653_00-desk-6-16-13-14-15-17-18": 16,
        >>>     ...
        >>> }
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
        self.eval_types_mapping = {}
        for i, eval_type in enumerate(self.eval_types):
            self.eval_types_mapping[eval_type] = i
            self.add_state(name=f"{eval_type}_total", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(name=f"{eval_type}_tp", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state(name="all_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(name="all_tp", default=torch.tensor(0), dist_reduce_fx="sum")

        # initialize dataset
        self._load_gt_data(split=split)

    def get_all_data_ids(self):
        return list(self.gt_data.keys())

    def _load_gt_data(self, split):
        self.gt_data = {}

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

    def _convert_eval_types_to_idx(self, eval_types: Sequence[str], device) -> torch.Tensor:
        eval_types_tensor = torch.empty(size=(len(eval_types), 2), dtype=torch.uint8, device=device)
        for i, eval_type_tuple in enumerate(eval_types):
            for j, eval_type in enumerate(eval_type_tuple):
                eval_types_tensor[i][j] = self.eval_types_mapping[eval_type]
        return eval_types_tensor

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
        eval_types = []
        predicted_obj_id_batch = []
        gt_obj_id_batch = []
        for key, value in preds.items():
            assert key in self.gt_data, f"id {key} is not in the ground truth dataset"
            eval_types.append((self.gt_data[key]["is_easy"], self.gt_data[key]["is_view_dep"]))
            gt_obj_id_batch.append(self.gt_data[key]["gt_obj_id"])
            predicted_obj_id_batch.append(value)

        gt_obj_id_batch = torch.tensor(gt_obj_id_batch, dtype=torch.uint8)
        predicted_obj_id_batch = torch.tensor(predicted_obj_id_batch, dtype=torch.uint8)

        # convert evaluation types to numerical values for convenience
        eval_types_tensor = self._convert_eval_types_to_idx(eval_types, predicted_obj_id_batch.device)

        eval_type_masks = {}
        for eval_type in self.eval_types_mapping.keys():
            eval_type_masks[eval_type] = (eval_types_tensor == self.eval_types_mapping[eval_type]).any(dim=-1)

        # update metrics
        self.all_total += len(preds)

        for eval_type in self.eval_types_mapping.keys():
            self.__dict__[f"{eval_type}_total"] += torch.count_nonzero(eval_type_masks[eval_type])

        tps = gt_obj_id_batch == predicted_obj_id_batch
        for eval_type in self.eval_types_mapping.keys():
            self.__dict__[f"{eval_type}_tp"] += torch.count_nonzero(eval_type_masks[eval_type] & tps)

        self.all_tp += torch.count_nonzero(tps)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute Acc based on inputs passed in to ``update`` previously."""
        output_dict = {}
        for eval_type in self.eval_types_mapping.keys():
            output_dict[f"{eval_type}"] = self.__dict__[f"{eval_type}_tp"] / self.__dict__[f"{eval_type}_total"]
        output_dict[f"all"] = self.__dict__[f"all_tp"] / self.all_total
        return output_dict
