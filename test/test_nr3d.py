import pytest
import torch
from torchmetrics_ext.metrics.visual_grounding import Nr3DMetric


@pytest.fixture(scope="module")
def nr3d_metric():
    return Nr3DMetric(split="test")


def test_nr3d_metric_perfect_predictions(nr3d_metric):
    nr3d_metric.reset()
    preds = {key: gt["gt_obj_id"] for key, gt in nr3d_metric.gt_data.items()}
    result = nr3d_metric(preds)
    for value in result.values():
        assert torch.allclose(value, torch.tensor(1.0))


def test_nr3d_metric_eval_types(nr3d_metric):
    nr3d_metric.reset()
    key, gt = next(iter(nr3d_metric.gt_data.items()))
    nr3d_metric.update({key: gt["gt_obj_id"] + 1000})  # wrong prediction with a large object id
    nr3d_metric.update({key: gt["gt_obj_id"]})
    result = nr3d_metric.compute()
    assert torch.allclose(result["all"], torch.tensor(0.5))
    assert torch.allclose(result[gt["is_easy"]], torch.tensor(0.5))
    assert torch.allclose(result[gt["is_view_dep"]], torch.tensor(0.5))
