import pytest
import torch
from torchmetrics_ext.metrics.visual_grounding import ViGiL3DMetric


# Test for Multi3DReferMetric
@pytest.fixture
def vigil3d_metric():
    return ViGiL3DMetric(split="validation")


def test_vigil3d_metric_low_iou(vigil3d_metric):
    vigil3d_metric.reset()  # Reset the metric before testing

    # Test update functionality with dummy data
    preds = {
        "ebbbcde5-a9ef-4ab7-9be3-e9f997c3f914": torch.tensor([[[0.83, 0.04, 1.26], [0.88, 0.47, 1.78]]]),  # 0 predicted boxes
        "cf49717d-a751-417e-be93-32fa6a4aa1e4": torch.tensor([[[4.41, 4.13, 3.016], [4.54, 5.31, 4.60]]]),  # 1 predicted box
        "239b1e39-7080-49bf-8663-9728613ea770": torch.tensor([
            [
                [3.02576269, 4.627958  , 4.19777474],
                [3.50918661, 5.06877917, 4.49572316]
            ],
            [
                [5.10415515, 4.19511597, 3.43203823],
                [5.54735444, 4.75973447, 3.76923387]
            ]
        ])
    }
    result = vigil3d_metric(preds)
    assert result["zt"].item() == 0.0
    assert result["st_0.25"].item() == 0.0
    assert result["st_0.5"].item() == 0.0
    assert result["mt_0.25"].item() == 0.0
    assert result["mt_0.5"].item() == 0.0
    assert result["all_0.25"].item() == 0.0
    assert result["all_0.5"].item() == 0.0


def test_vigil3d_metric_medium_iou(vigil3d_metric):
    vigil3d_metric.reset()  # Reset the metric before testing

    # Test update functionality with dummy data
    preds = {
        "ebbbcde5-a9ef-4ab7-9be3-e9f997c3f914": torch.tensor([]),  # 0 predicted boxes
        "cf49717d-a751-417e-be93-32fa6a4aa1e4": torch.tensor([[[1.48, 1.13, 0.016], [1.61, 2.31, 1.60]]]),  # 1 predicted box
        "239b1e39-7080-49bf-8663-9728613ea770": torch.tensor([
            [
                [0.03576269, 1.627958  , 1.19777474],
                [0.51918661, 2.06877917, 1.49572316]
            ],
            [
                [2.32415515, 1.19511597, 0.43203823],
                [2.74735444, 1.75973447, 0.76923387]
            ]
        ])
    }
    result = vigil3d_metric(preds)
    assert result["zt"].item() == 1.0
    assert result["st_0.25"].item() == 1.0
    assert result["st_0.5"].item() == 0.0
    assert result["mt_0.25"].item() == 1.0
    assert result["mt_0.5"].item() == 0.5
    assert result["all_0.25"].item() == 1.0
    assert result["all_0.5"].item() == 0.5


def test_multi3drefer_metric_high_iou(vigil3d_metric):
    vigil3d_metric.reset()  # Reset the metric before testing

    # Test update functionality with dummy data
    preds = {
        "ebbbcde5-a9ef-4ab7-9be3-e9f997c3f914": torch.tensor([]),  # 0 predicted boxes
        "cf49717d-a751-417e-be93-32fa6a4aa1e4": torch.tensor([[[1.41, 1.13, 0.016], [1.54, 2.31, 1.60]]]),  # 1 predicted box
        "239b1e39-7080-49bf-8663-9728613ea770": torch.tensor([
            [
                [0.02576269, 1.627958  , 1.19777474],
                [0.50918661, 2.06877917, 1.49572316]
            ],
            [
                [2.10415515, 1.19511597, 0.43203823],
                [2.54735444, 1.75973447, 0.76923387]
            ]
        ])
    }
    result = vigil3d_metric(preds)
    assert result["zt"].item() == 1.0
    assert result["st_0.25"].item() == 1.0
    assert result["st_0.5"].item() == 1.0
    assert result["mt_0.25"].item() == 1.0
    assert result["mt_0.5"].item() == 1.0
    assert result["all_0.25"].item() == 1.0
    assert result["all_0.5"].item() == 1.0
