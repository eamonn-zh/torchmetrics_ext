import pytest
from torchmetrics_ext.metrics.vqa import ScanQAMetric


@pytest.fixture(scope="module")
def scanqa_metric():
    return ScanQAMetric(split="validation")


def test_scanqa_metric_perfect_answers(scanqa_metric):
    scanqa_metric.reset()
    question_ids = scanqa_metric.get_all_data_ids()[:100]
    preds = {question_id: scanqa_metric.gt_data[question_id]["answers"][0] for question_id in question_ids}
    result = scanqa_metric(preds)
    assert set(result.keys()) == {"BLEU_1", "BLEU_4", "METEOR", "ROUGE_L", "CIDEr"}
    assert result["BLEU_1"] == pytest.approx(100.0, abs=1e-3)


def test_scanqa_metric_unknown_id(scanqa_metric):
    with pytest.raises(KeyError):
        scanqa_metric.update({"not-a-question-id": "white"})
