import torch
from torchmetrics import Metric
from typing import Dict, Sequence


class ReferIt3DMetric(Metric):
    r"""
    Compute the Accuracy for the ReferIt3D 3D visual grounding task.
    Please refer to https://referit3d.github.io/ for more details.

    Example:
        >>> import torch
        >>> from torchmetrics_ext.visual_grounding import ReferIt3DMetric
        >>> metric = ReferIt3DMetric()
        >>> # indices of predicted and ground truth objects (B, )
        >>> pred_indices = torch.tensor([5, 2, 0, 0], dtype=torch.uint8)
        >>> gt_indices = torch.tensor([5, 5, 1, 0], dtype=torch.uint8)
        >>> gt_eval_types = (("easy", "view_dep"), ("easy", "view_indep"), ("hard", "view_dep"), ("hard", "view_dep"))
        >>> metric(pred_indices, gt_indices, gt_eval_types)
        {'easy': tensor(0.5),
         'hard': tensor(0.5),
         'view_dep': tensor(0.6667),
         'view_indep': tensor(0.),
         'all': tensor(0.5)}
    """
    def __init__(self, eval_types: Sequence[str] = ("easy", "hard", "view_dep", "view_indep")):
        super().__init__()

        self.eval_types_mapping = {}
        for i, eval_type in enumerate(eval_types):
            self.eval_types_mapping[eval_type] = i
            self.add_state(name=f"{eval_type}_total", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(name=f"{eval_type}_tp", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state(name="all_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(name="all_tp", default=torch.tensor(0), dist_reduce_fx="sum")

    def _convert_eval_types_to_idx(self, eval_types: Sequence[str], device) -> torch.Tensor:
        eval_types_tensor = torch.empty(size=(len(eval_types), 2), dtype=torch.uint8, device=device)
        for i, eval_type_tuple in enumerate(eval_types):
            for j, eval_type in enumerate(eval_type_tuple):
                eval_types_tensor[i][j] = self.eval_types_mapping[eval_type]
        return eval_types_tensor

    def update(self, preds: torch.Tensor, targets: torch.Tensor, eval_types: Sequence[Sequence[str]]) -> None:
        """
        :param preds: predicted index (B, )
        :param targets: ground truth index (B, )
        :param eval_types: a sequence of evaluation type labels (B, 2)
        """

        # check input sizes
        if preds.shape != targets.shape or preds.shape[0] != len(eval_types):
            raise ValueError("preds, targets and eval_types must have the same length")

        # convert evaluation types to numerical values for convenience
        eval_types_tensor = self._convert_eval_types_to_idx(eval_types, preds.device)

        eval_type_masks = {}
        for eval_type in self.eval_types_mapping.keys():
            eval_type_masks[eval_type] = (eval_types_tensor == self.eval_types_mapping[eval_type]).any(dim=-1)

        # update metrics
        self.all_total += targets.shape[0]

        for eval_type in self.eval_types_mapping.keys():
            self.__dict__[f"{eval_type}_total"] += torch.count_nonzero(eval_type_masks[eval_type])

        tps = preds == targets
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
