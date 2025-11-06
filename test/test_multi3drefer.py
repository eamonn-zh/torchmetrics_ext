import pytest
import torch
from torchmetrics_ext.metrics.visual_grounding import Multi3DReferMetric


# Test for Multi3DReferMetric
@pytest.fixture
def multi3drefer_metric():
    return Multi3DReferMetric(split="validation", strict=False)


@pytest.fixture
def multi3drefer_metric_strict():
    return Multi3DReferMetric(split="validation", strict=True)


def test_multi3drefer_metric_low_iou(multi3drefer_metric):
    multi3drefer_metric.reset()  # Reset the metric before testing

    # Test update functionality with dummy data
    preds = {
        "scene0406_00_6": torch.tensor([[[0.83, 0.04, 1.26], [0.88, 0.47, 1.78]]]),  # 0 predicted boxes (w)
        "scene0578_00_69": torch.tensor([[[0.83, 0.04, 1.26], [0.88, 0.47, 1.78]]]),  # 0 predicted boxes (wo)
        "scene0406_00_0": torch.tensor([[[3.83, 3.04, 4.26], [3.88, 3.47, 4.78]]]),  # 1 predicted box
        "scene0406_00_84": torch.tensor([[[3.40, 3.89, 3.31], [3.71, 3.96, 3.58]]]),  # 1 predicted box
        "scene0406_00_10": torch.tensor(
            [[[-3.01, -3.26, 3.034], [-2.54, -3.03, 4.91]], [[-3.43, -3.32, 4.23], [-2.96, -2.89, 4.71]]]
        ),
    }
    result = multi3drefer_metric(preds)
    assert torch.allclose(result["zt_w_d"], torch.tensor(0.0))
    assert torch.allclose(result["zt_wo_d"], torch.tensor(0.0))
    assert torch.allclose(result["st_w_d_0.25"], torch.tensor(0.0))
    assert torch.allclose(result["st_wo_d_0.25"], torch.tensor(0.0))
    assert torch.allclose(result["st_w_d_0.5"], torch.tensor(0.0))
    assert torch.allclose(result["st_wo_d_0.5"], torch.tensor(0.0))
    assert torch.allclose(result["mt_0.25"], torch.tensor(0.0))
    assert torch.allclose(result["mt_0.5"], torch.tensor(0.0))
    assert torch.allclose(result["all_0.25"], torch.tensor(0.0))
    assert torch.allclose(result["all_0.5"], torch.tensor(0.0))


def test_multi3drefer_metric_medium_iou(multi3drefer_metric):
    multi3drefer_metric.reset()  # Reset the metric before testing

    # Test update functionality with dummy data
    preds = {
        "scene0406_00_6": torch.tensor([]),  # 0 predicted boxes (w)
        "scene0578_00_69": torch.tensor([]),  # 0 predicted boxes (wo)
        "scene0406_00_0": torch.tensor([[[0.855, 0.0380, 1.2593], [0.88, 0.47, 1.78]]]),  # 1 predicted box
        "scene0406_00_84": torch.tensor([[[0.0827, 0.8602, 0.3036], [0.706, 0.964, 0.575]]]),  # 1 predicted box
        "scene0406_00_10": torch.tensor(
            [[[-0.2018, -0.3002, 0.0238], [0.46, 0.03, 1.91]], [[-0.43, -0.32, 1.525], [0.04, -0.11, 1.71]]]
        ),
    }
    result = multi3drefer_metric(preds)
    assert torch.allclose(result["zt_w_d"], torch.tensor(1.0))
    assert torch.allclose(result["zt_wo_d"], torch.tensor(1.0))
    assert torch.allclose(result["st_w_d_0.25"], torch.tensor(1.0))
    assert torch.allclose(result["st_wo_d_0.25"], torch.tensor(1.0))
    assert torch.allclose(result["st_w_d_0.5"], torch.tensor(0.0))
    assert torch.allclose(result["st_wo_d_0.5"], torch.tensor(0.0))
    assert torch.allclose(result["mt_0.25"], torch.tensor(1.0))
    assert torch.allclose(result["mt_0.5"], torch.tensor(0.0))
    assert torch.allclose(result["all_0.25"], torch.tensor(1.0))
    assert torch.allclose(result["all_0.5"], torch.tensor(0.40))  # because zt counts as 1


def test_multi3drefer_metric_high_iou(multi3drefer_metric):
    multi3drefer_metric.reset()  # Reset the metric before testing

    # Test update functionality with dummy data
    preds = {
        "scene0406_00_6": torch.tensor([]),  # 0 predicted boxes (w)
        "scene0578_00_69": torch.tensor([]),  # 0 predicted boxes (wo)
        "scene0406_00_0": torch.tensor([[[0.83, 0.04, 1.26], [0.88, 0.47, 1.78]]]),  # 1 predicted box
        "scene0406_00_84": torch.tensor([[[0.40, 0.89, 0.31], [0.71, 0.96, 0.58]]]),  # 1 predicted box
        "scene0406_00_10": torch.tensor(
            [[[-0.01, -0.26, 0.034], [0.46, -0.03, 1.91]], [[-0.43, -0.32, 1.23], [0.04, -0.11, 1.71]]]
        ),
    }
    result = multi3drefer_metric(preds)
    assert torch.allclose(result["zt_w_d"], torch.tensor(1.0))
    assert torch.allclose(result["zt_wo_d"], torch.tensor(1.0))
    assert torch.allclose(result["st_w_d_0.25"], torch.tensor(1.0))
    assert torch.allclose(result["st_wo_d_0.25"], torch.tensor(1.0))
    assert torch.allclose(result["st_w_d_0.5"], torch.tensor(1.0))
    assert torch.allclose(result["st_wo_d_0.5"], torch.tensor(1.0))
    assert torch.allclose(result["mt_0.25"], torch.tensor(1.0))
    assert torch.allclose(result["mt_0.5"], torch.tensor(1.0))
    assert torch.allclose(result["all_0.25"], torch.tensor(1.0))
    assert torch.allclose(result["all_0.5"], torch.tensor(1.0))


def test_multi3drefer_metric_strict_mode(multi3drefer_metric_strict: Multi3DReferMetric):
    # Test with missing predictions
    gt_data = multi3drefer_metric_strict.gt_data
    preds = {key: torch.from_numpy(data["gt_aabbs"]) if data["gt_aabbs"] is not None else torch.tensor([]) for key, data in gt_data.items()}

    result = multi3drefer_metric_strict(preds)
    assert torch.allclose(result["zt_w_d"], torch.tensor(1.0))
    assert torch.allclose(result["zt_wo_d"], torch.tensor(1.0))
    assert torch.allclose(result["st_w_d_0.25"], torch.tensor(1.0))
    assert torch.allclose(result["st_wo_d_0.25"], torch.tensor(1.0))
    assert torch.allclose(result["st_w_d_0.5"], torch.tensor(1.0))
    assert torch.allclose(result["st_wo_d_0.5"], torch.tensor(1.0))
    assert torch.allclose(result["mt_0.25"], torch.tensor(1.0))
    assert torch.allclose(result["mt_0.5"], torch.tensor(1.0))
    assert torch.allclose(result["all_0.25"], torch.tensor(1.0))
    assert torch.allclose(result["all_0.5"], torch.tensor(1.0))


def test_multi3drefer_metric_strict_mode_missing_samples(multi3drefer_metric_strict: Multi3DReferMetric):
    preds = {
        "scene0406_00_6": torch.tensor([]),  # 0 predicted boxes (w)
        "scene0578_00_69": torch.tensor([]),  # 0 predicted boxes (wo)
        "scene0406_00_0": torch.tensor([[[0.83, 0.04, 1.26], [0.88, 0.47, 1.78]]]),  # 1 predicted box
        "scene0406_00_84": torch.tensor([[[0.40, 0.89, 0.31], [0.71, 0.96, 0.58]]]),  # 1 predicted box
    }
    with pytest.raises(ValueError, match="Mismatched IDs between predictions and dataset"):
        multi3drefer_metric_strict(preds)
