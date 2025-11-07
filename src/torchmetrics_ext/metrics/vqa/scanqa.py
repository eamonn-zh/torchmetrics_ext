import os
import gdown
import pandas as pd
from typing import Dict
from datasets import config
from torchmetrics import Metric
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


class ScanQAMetric(Metric):

    dataset_google_drive_file_ids = {
        "train": "1-EmSpD_PMX-W4f2xX-zclB5TPrLemi-U",
        "validation": "1-FLPGEAiMTEPaKSS7VlJYCi-qY97wH4y"
    }

    def __init__(self, split="validation"):
        super().__init__()
        self.tokenizer = PTBTokenizer()
        self.scorers = {
            "BLEU_1": Bleu(1),
            "BLEU_4": Bleu(4),
            "METEOR": Meteor(),
            "ROUGE_L": Rouge(),
            "CIDEr": Cider(),
        }

        # initialize metrics
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("gts", default=[], dist_reduce_fx="cat")
        self.add_state("ids", default=[], dist_reduce_fx="cat")

        # initialize dataset
        self._load_gt_data(split=split)

    def _load_gt_data(self, split):
        self.gt_data = {}
        cache_path = os.path.join(config.HF_DATASETS_CACHE, "scanqa")
        cache_path = gdown.download(id=self.dataset_google_drive_file_ids[split], output=f"{cache_path}/", resume=True)
        raw_dataset = pd.read_json(cache_path)[["answers", "question_id", "question"]]

        for row in raw_dataset.itertuples(index=False):
            # exclude question_id in the value
            self.gt_data[row.question_id] = {
                key: getattr(row, key) for key in raw_dataset.columns if key != "question_id"
            }

    def get_all_data_ids(self):
        return list(self.gt_data.keys())

    def update(self, preds: Dict[str, str]) -> None:
        for question_id, pred_answer in preds.items():
            assert question_id in self.gt_data, f"id {question_id} is not in the ground truth dataset"
            self.preds.append(str(pred_answer))
            self.gts.append(self.gt_data[question_id]["answers"])
            self.ids.append(question_id)

    def compute(self) -> Dict[str, float]:
        gts = {}
        preds = {}
        for question_id, pred, gt in zip(self.ids, self.preds, self.gts):
            gts[question_id] = [{"caption": str(one_gt)} for one_gt in gt]
            preds[question_id] = [{"caption": pred}]

        gts = self.tokenizer.tokenize(gts)
        preds = self.tokenizer.tokenize(preds)
        output_dict = {}
        for metric_name, scorer in self.scorers.items():
            score, _ = scorer.compute_score(gts, preds)
            if isinstance(score, list):
                score = score[-1]
            output_dict[metric_name] = score * 100
        return output_dict
