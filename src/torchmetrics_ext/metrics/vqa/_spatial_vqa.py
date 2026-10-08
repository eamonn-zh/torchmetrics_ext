import torch
from typing import Dict, Optional, Sequence
from torchmetrics import Metric
from datasets import load_dataset


class SpatialVQAMetric(Metric):
    r"""
    Base class for VSI-Bench style spatial VQA benchmarks.

    Multiple-choice questions are scored by exact match (case-insensitive), numerical questions are scored by the
    Mean Relative Accuracy (MRA) over confidence thresholds from 0.5 to 0.95. Per-question-type accuracies (in %) are
    reported, with the sub-types listed in ``merged_question_types`` averaged into a single entry, followed by the
    overall accuracy as the mean over all reported question types.
    """

    mcq_question_types: Sequence[str] = ()
    numeric_question_types: Sequence[str] = ()
    # maps a reported question type to the sub-types averaged into it
    merged_question_types: Dict[str, Sequence[str]] = {}

    def __init__(self, split: str, dataset_path: str, dataset_name: Optional[str] = None):
        super().__init__()

        self.dataset_path = dataset_path

        # initialize metrics
        for question_type in self.question_types:
            self.add_state(
                name=f"{question_type}_acc", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
            )
            self.add_state(
                name=f"{question_type}_total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
            )

        # initialize dataset
        self._load_gt_data(split=split, dataset_name=dataset_name)

    @property
    def question_types(self) -> Sequence[str]:
        return (*self.mcq_question_types, *self.numeric_question_types)

    def _load_gt_data(self, split: str, dataset_name: Optional[str]) -> None:
        raw_dataset = load_dataset(self.dataset_path, dataset_name, split=split)
        # exclude question_id in the value
        self.gt_data = {row["id"]: {key: value for key, value in row.items() if key != "id"} for row in raw_dataset}

    def get_all_data_ids(self):
        return list(self.gt_data.keys())

    @staticmethod
    def _mean_relative_accuracy(pred, target, start, end, interval):
        # follows the official VSI-Bench implementation
        num_pts = int((end - start) / interval + 2)
        conf_intervals = torch.linspace(start, end, steps=num_pts, dtype=torch.float64)
        accuracy = (abs(pred - target) / target) <= (1 - conf_intervals)
        return accuracy.to(conf_intervals.dtype).mean()

    def _score(self, question_type: str, pred_answer: str, gt_answer: str) -> float:
        if question_type in self.mcq_question_types:
            return 1.0 if pred_answer.lower() == gt_answer.lower() else 0.0
        if question_type in self.numeric_question_types:
            try:
                return self._mean_relative_accuracy(float(pred_answer), float(gt_answer), 0.5, 0.95, 0.05)
            except (ValueError, ZeroDivisionError):
                return 0.0
        raise ValueError(f"Unknown question type: {question_type}")

    def update(self, preds: Dict[str, str]) -> None:
        """
        Args:
            preds (dict): A dictionary mapping each unique question identifier ("id") to the predicted answer.
        """
        for question_id, pred_answer in preds.items():
            if question_id not in self.gt_data:
                raise KeyError(f"id {question_id} is not in the ground truth dataset")
            gt = self.gt_data[question_id]
            question_type = gt["question_type"]
            pred_answer = str(pred_answer).strip().split(" ")[0].rstrip(".").strip()
            accuracy = self._score(question_type, pred_answer, gt["ground_truth"])

            self.__dict__[f"{question_type}_total"] += 1
            self.__dict__[f"{question_type}_acc"] += accuracy

    def compute(self) -> Dict[str, torch.Tensor]:
        output_dict = {
            f"{question_type}_acc": self.__dict__[f"{question_type}_acc"] / self.__dict__[f"{question_type}_total"] * 100
            for question_type in self.question_types
        }

        for merged_type, sub_types in self.merged_question_types.items():
            sub_keys = [f"{sub_type}_acc" for sub_type in sub_types]
            output_dict[f"{merged_type}_acc"] = torch.stack([output_dict.pop(k) for k in sub_keys]).mean()

        output_dict["overall_acc"] = torch.stack(list(output_dict.values())).nanmean()
        return output_dict
