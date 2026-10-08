import torch
from torchmetrics_ext.metrics.object_detection import MeanAveragePrecisionMetric


def test_mean_average_precision_perfect_predictions():
    metric = MeanAveragePrecisionMetric(semantic_classes=[1, 2])
    boxes = torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[2., 2., 2.], [3., 3., 3.]], [[0., 0., 0.], [2., 2., 2.]]])
    classes = torch.tensor([1, 2, 1])
    batch_idx = torch.tensor([0, 0, 1])
    metric.update(
        pred_boxes=boxes, pred_classes=classes, pred_scores=torch.tensor([0.9, 0.8, 0.7]), pred_batch_idx=batch_idx,
        target_boxes=boxes, target_classes=classes, target_batch_idx=batch_idx
    )
    result = metric.compute()
    assert result["mean_ap_macro_avg_0.25"] == 1.0
    assert result["mean_ap_macro_avg_0.5"] == 1.0
