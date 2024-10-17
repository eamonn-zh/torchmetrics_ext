# TorchMetrics Extension
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchmetrics)](https://pypi.org/project/torchmetrics-ext/)
[![PyPI version](https://badge.fury.io/py/torchmetrics-ext.svg)](https://badge.fury.io/py/torchmetrics-ext)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/eamonn-zh/torchmetrics_ext/blob/master/LICENSE)

## Installation
Simple installation from PyPI
```bash
pip install torchmetrics-ext
```

## What is TorchMetrics Extension
It is an extension of [torchmetrics](https://lightning.ai/docs/torchmetrics/) containing more metrics for machine learning tasks. Currently, it offers metrics for:
- 3D Visual Grounding
  - [ScanRefer](https://daveredrum.github.io/ScanRefer/)
  - [ReferIt3D](https://referit3d.github.io/)

## Using TorchMetrics Extension
Here are examples for using the metrics in TorchMetrics Extension:

### ScanRefer
It measures the thresholded accuracy Acc@kIoU, where the positive predictions have higher intersection over union (IoU) with the ground truths than the thresholds. The metric is based on the [ScanRefer](https://daveredrum.github.io/ScanRefer/).
```python
import torch
from torchmetrics_ext.visual_grounding import ScanReferMetric
metric = ScanReferMetric()

# min max bounds of 3D axis-aligned bounding boxes (B, 2, 3)
pred_aabbs = torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [2., 2., 2.]]], dtype=torch.float32)
gt_aabbs = torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [1.5, 1.5, 1.5]]], dtype=torch.float32)

gt_eval_types = ("unique", "multiple")
results = metric(pred_aabbs, gt_aabbs, gt_eval_types)
```

### ReferIt3D
It measures the accuracy of selecting the target object from the candidates. The metric is based on the [ReferIt3D](https://referit3d.github.io/).
```python
import torch
from torchmetrics_ext.visual_grounding import ReferIt3DMetric
metric = ReferIt3DMetric()

# indices of predicted and ground truth objects (B, )
pred_indices = torch.tensor([5, 2, 0, 0], dtype=torch.uint8)
gt_indices = torch.tensor([5, 5, 1, 0], dtype=torch.uint8)

gt_eval_types = (("easy", "view_dep"), ("easy", "view_indep"), ("hard", "view_dep"), ("hard", "view_dep"))
results = metric(pred_indices, gt_indices, gt_eval_types)
```
