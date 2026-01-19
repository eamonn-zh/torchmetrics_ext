import torch
from typing import Dict
from torchmetrics import Metric
from datasets import load_dataset


class ReVSIBenchMetric(Metric):

    mcq_question_types = [
        "object_rel_distance_min",
        "object_rel_distance_max",
        "object_rel_direction_easy",
        "object_rel_direction_easy_reverse",
        "object_rel_direction_hard",
        "object_rel_direction_hard_reverse",
        "route_planning",
    ]

    numeric_question_types = [
        "object_counting_easy",
        "object_counting_hard",
        "object_abs_distance",
        "object_size_estimation",
        "room_size_estimation_single",
        "room_size_estimation_all"
    ]

    def __init__(self, split="test", dataset_path="3dlg-hcvc/ReVSI-Bench", dir_name=None):
        super().__init__()

        self.dataset_path = dataset_path
        self.dir_name = dir_name

        # initialize metrics
        for question_type in (self.mcq_question_types + self.numeric_question_types):
            self.add_state(name=f"{question_type}_acc", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
            self.add_state(name=f"{question_type}_total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")

        # initialize dataset
        self._load_gt_data(split=split)

    def _load_gt_data(self, split):
        self.gt_data = {}
        raw_dataset = load_dataset(self.dataset_path, data_dir=self.dir_name, split=split)
        for row in raw_dataset:
            # exclude question_id in the value
            self.gt_data[row["id"]] = {key: value for key, value in row.items() if key != "id"}

    def get_all_data_ids(self):
        return list(self.gt_data.keys())

    @staticmethod
    def _mean_relative_accuracy(pred, target, start, end, interval):
        num_pts = int((end - start) / interval + 2)
        conf_intervals = torch.linspace(start, end, steps=num_pts, dtype=torch.float64)
        accuracy = (abs(pred - target) / target) <= (1 - conf_intervals)
        return accuracy.to(conf_intervals.dtype).mean()

    def update(self, preds: Dict[int, str]) -> None:
        for question_id, pred_answer in preds.items():
            assert question_id in self.gt_data, f"id {question_id} is not in the ground truth dataset"
            gt_question_type = self.gt_data[question_id]["question_type"]
            self.__dict__[f"{gt_question_type}_total"] += 1
            pred_answer = str(pred_answer).strip().split(" ")[0].rstrip(".").strip()
            gt_answer = self.gt_data[question_id]["ground_truth"]
            if gt_question_type.startswith("object_counting") and float(gt_answer) <= 5:
                accuracy = 1.0 if int(float(pred_answer)) == int(float(gt_answer)) else 0.0
            elif gt_question_type.startswith("object_counting"):
                try:
                    accuracy = self._mean_relative_accuracy(
                        float(pred_answer), float(gt_answer), 0.5, 0.95, 0.05
                    )
                except:
                    accuracy = 0.0
            elif gt_question_type in self.mcq_question_types:
                accuracy = 1.0 if pred_answer.lower() == gt_answer.lower() else 0.0
            elif gt_question_type in self.numeric_question_types:
                try:
                    accuracy = self._mean_relative_accuracy(
                        float(pred_answer), float(gt_answer), 0.5, 0.95, 0.05
                    )
                except:
                    accuracy = 0.0
            self.__dict__[f"{gt_question_type}_acc"] += accuracy

    def compute(self) -> Dict[str, torch.Tensor]:
        output_dict = {}
        for question_type in (self.mcq_question_types + self.numeric_question_types):
            output_dict[f"{question_type}_acc"] = self.__dict__[f"{question_type}_acc"] / self.__dict__[f"{question_type}_total"] * 100

        rel_dir_levels = ("easy", "easy_reverse", "hard", "hard_reverse")
        rel_dir_keys = [f"object_rel_direction_{lvl}_acc" for lvl in rel_dir_levels]
        output_dict["object_rel_direction_acc"] = torch.stack([output_dict[k] for k in rel_dir_keys]).mean()
        for k in rel_dir_keys:
            output_dict.pop(k, None)

        obj_count_levels = ("easy", "hard")
        obj_count_keys = [f"object_counting_{lvl}_acc" for lvl in obj_count_levels]
        output_dict["object_counting_acc"] = torch.stack([output_dict[k] for k in obj_count_keys]).mean()
        for k in obj_count_keys:
            output_dict.pop(k, None)

        rel_dist_levels = ("min", "max")
        rel_dist_keys = [f"object_rel_distance_{lvl}_acc" for lvl in rel_dist_levels]
        output_dict["object_rel_distance_acc"] = torch.stack([output_dict[k] for k in rel_dist_keys]).mean()
        for k in rel_dist_keys:
            output_dict.pop(k, None)

        room_size_levels = ("single", "all")
        room_size_keys = [f"room_size_estimation_{lvl}_acc" for lvl in room_size_levels]
        output_dict["room_size_estimation_acc"] = torch.stack([output_dict[k] for k in room_size_keys]).mean()
        for k in room_size_keys:
            output_dict.pop(k, None)

        output_dict["overall_acc"] = torch.stack(list(output_dict.values())).nanmean()
        return output_dict
