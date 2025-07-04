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
It is an extension of [torchmetrics](https://lightning.ai/docs/torchmetrics/) containing more metrics for machine learning tasks. It offers:
* A standardized interface to increase reproducibility
* Reduces Boilerplate
* Distributed-training compatible
* Rigorously tested
* Automatic accumulation over batches
* Automatic synchronization between multiple devices

Currently, it offers metrics for:
- 3D Visual Grounding
  - [ScanRefer](https://daveredrum.github.io/ScanRefer/)
  - [Nr3D](https://referit3d.github.io/)
  - [Multi3DRefer](https://3dlg-hcvc.github.io/multi3drefer/)
- 3D Object Detection
  - [ScanNet](http://www.scan-net.org/) (Under development)

## Using TorchMetrics Extension
Here are examples for using the metrics in TorchMetrics Extension:

### ScanRefer
> Please download the [ScanRefer dataset](https://github.com/daveredrum/ScanRefer?tab=readme-ov-file#dataset) first, which will be required by the evaluator.

It measures the thresholded accuracy Acc@kIoU, where the positive predictions have higher intersection over union (IoU) with the ground truths than the thresholds. The metric is based on the [ScanRefer](https://daveredrum.github.io/ScanRefer/) task.

```python
import torch
from torchmetrics_ext.metrics.visual_grounding import ScanReferMetric
metric = ScanReferMetric(dataset_file_path="path to the ScanRefer_filtered_val.json file", split="validation")

# preds is a dictionary mapping each unique description identifier (formatted as "{scene_id}_{object_id}_{ann_id}")
# to the predicted axis-aligned bounding boxes in shape (2, 3)
preds = {
    "scene0011_00_0_0": torch.tensor([[0., 0., 0.], [0.5, 0.5, 0.5]]),
    "scene0011_01_0_1": torch.tensor([[0., 0., 0.], [1., 1., 1.]]),
}
metric(preds)
```

### Nr3D
> The dataset will be automatically downloaded from the official [Nr3D Google Drive](https://referit3d.github.io/benchmarks.html).

It measures the accuracy of selecting the target object from the candidates. The metric is based on the [Nr3D](https://referit3d.github.io/) task.

```python
import torch
from torchmetrics_ext.metrics.visual_grounding import Nr3DMetric

metric = Nr3DMetric()

# indices of predicted and ground truth objects (B, )
pred_indices = torch.tensor([5, 2, 0, 0], dtype=torch.uint8)
gt_indices = torch.tensor([5, 5, 1, 0], dtype=torch.uint8)

gt_eval_types = (("easy", "view_dep"), ("easy", "view_indep"), ("hard", "view_dep"), ("hard", "view_dep"))
results = metric(pred_indices, gt_indices, gt_eval_types)
```

### Multi3DRefer
> The dataset will be automatically downloaded from the official [Multi3DRefer Hugging Face repo](https://huggingface.co/datasets/3dlg-hcvc/Multi3DRefer).

It measures the F1-scores at multiple IoU thresholds (F1@kIoU), where the positive predictions have higher intersection over union (IoU) with the ground truths than the thresholds. The metric is based on the [Multi3DRefer](https://3dlg-hcvc.github.io/multi3drefer/) task.
Note: This metric automatically loads ground truths from [Hugging Face](https://huggingface.co/datasets/3dlg-hcvc/Multi3DRefer).

```python
import torch
from torchmetrics_ext.metrics.visual_grounding import Multi3DReferMetric
metric = Multi3DReferMetric(split="validation")

# preds is a dictionary mapping each unique description identifier (formatted as "{scene_id}_{ann_id}")
# to a variable number of predicted axis-aligned bounding boxes in shape (N, 2, 3)
preds = {
    "scene0011_00_0": torch.tensor([[[0., 0., 0.], [0.5, 0.5, 0.5]]]),  # 1 predicted box
    "scene0011_01_1": torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [2., 2., 2.]]])  # 2 predicted boxes
}
result = metric(preds)
```
