import pytest
import torch
from torchmetrics_ext.metrics.vqa import VSIBenchMetric, ReVSIMetric


@pytest.fixture(scope="module", params=[VSIBenchMetric, ReVSIMetric])
def metric(request):
    return request.param()


def test_perfect_answers(metric):
    metric.reset()
    preds = {question_id: gt["ground_truth"] for question_id, gt in metric.gt_data.items()}
    result = metric(preds)
    assert torch.allclose(result["overall_acc"], torch.tensor(100.0, dtype=torch.float64))
    # sub-types are merged into a single entry
    assert "object_rel_direction_acc" in result
    assert not any(key.startswith("object_rel_direction_") and key != "object_rel_direction_acc" for key in result)


def test_wrong_answers(metric):
    metric.reset()
    preds = {question_id: "not an answer" for question_id in metric.gt_data}
    result = metric(preds)
    assert torch.allclose(result["overall_acc"], torch.tensor(0.0, dtype=torch.float64))


def test_mean_relative_accuracy():
    # relative error 0.1 passes the thresholds 0.5 ... 0.9 (rel. error <= 1 - threshold)
    accuracy = VSIBenchMetric._mean_relative_accuracy(11.0, 10.0, 0.5, 0.95, 0.05)
    assert 0.0 < accuracy < 1.0
    assert VSIBenchMetric._mean_relative_accuracy(10.0, 10.0, 0.5, 0.95, 0.05) == 1.0
