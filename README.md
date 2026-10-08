# TorchMetrics Extension

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchmetrics-ext)](https://pypi.org/project/torchmetrics-ext/)
[![PyPI version](https://badge.fury.io/py/torchmetrics-ext.svg)](https://badge.fury.io/py/torchmetrics-ext)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/eamonn-zh/torchmetrics_ext/blob/main/LICENSE)

Ready-to-use evaluators for **3D vision-language benchmarks**, built on top of [TorchMetrics](https://lightning.ai/docs/torchmetrics/).

Each metric downloads its ground truth automatically. You pass in your predictions keyed by sample ID and get the
benchmark's official numbers back. Because every evaluator is a TorchMetrics `Metric`, you also get:

* a standardized interface that makes results reproducible across projects
* automatic accumulation over batches (`update()` many times, `compute()` once)
* automatic synchronization across devices in distributed training

## Installation

```bash
pip install torchmetrics-ext
```

## Supported Benchmarks

| Task | Benchmark | Metric class | Reported metrics | Ground truth |
| --- | --- | --- | --- | --- |
| 3D Visual Grounding | [ScanRefer](https://daveredrum.github.io/ScanRefer/) | `ScanReferMetric` | Acc@0.25IoU, Acc@0.5IoU | manual download\* |
| | [Nr3D](https://referit3d.github.io/) | `Nr3DMetric` | Accuracy | automatic |
| | [Multi3DRefer](https://3dlg-hcvc.github.io/multi3drefer/) | `Multi3DReferMetric` | F1@0.25IoU, F1@0.5IoU | automatic |
| | [ViGiL3D](https://3dlg-hcvc.github.io/vigil3d/) | `ViGiL3DMetric` | F1@0.25IoU, F1@0.5IoU | automatic |
| 3D Visual Question Answering | [ScanQA](https://github.com/ATR-DBI/ScanQA) | `ScanQAMetric` | BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr | automatic |
| | [VSI-Bench](https://vision-x-nyu.github.io/thinking-in-space.github.io/) | `VSIBenchMetric` | Accuracy / MRA | automatic |
| | [ReVSI](https://3dlg-hcvc.github.io/revsi/) | `ReVSIMetric` | Accuracy / MRA | automatic |
| 3D Object Detection | [ScanNet](http://www.scan-net.org/) | `MeanAveragePrecisionMetric` | mAP@0.25IoU, mAP@0.5IoU | — (under development) |

\* The ScanRefer license requires you to [request the dataset](https://github.com/daveredrum/ScanRefer?tab=readme-ov-file#dataset) yourself.

## Quick Start

Every benchmark metric works the same way:

```python
from torchmetrics_ext.metrics.visual_grounding import Multi3DReferMetric

metric = Multi3DReferMetric(split="validation")  # downloads and caches the ground truth

ids = metric.get_all_data_ids()  # all sample IDs of this split

# evaluate everything at once ...
results = metric(preds)          # preds: {sample_id: prediction}

# ... or accumulate over batches (works across DDP processes)
for batch_preds in loader:
    metric.update(batch_preds)
results = metric.compute()
metric.reset()
```

All results are averaged over the predictions you submit, not over the whole split. To get official numbers, submit a
prediction for every ID in `get_all_data_ids()`. Grounding metrics are returned in `[0, 1]`, VQA metrics in `%`.
Passing an ID that is not in the ground truth raises a `KeyError`.

## Usage

### 3D Visual Grounding

Bounding boxes are axis-aligned and given as `[[x_min, y_min, z_min], [x_max, y_max, z_max]]`. ScanNet ground truth
boxes are axis-aligned with the transformation matrix from each scene's `<scene_id>.txt`. Your predictions must use
the same coordinate frame.

#### ScanRefer

Predict **one box** per description, with shape `(2, 3)`.

```python
import torch
from torchmetrics_ext.metrics.visual_grounding import ScanReferMetric

metric = ScanReferMetric(dataset_file_path="./ScanRefer_filtered_val.json", split="validation")

# key: "{scene_id}_{object_id}_{ann_id}"
preds = {
    "scene0011_00_0_0": torch.tensor([[0., 0., 0.], [0.5, 0.5, 0.5]]),
    "scene0011_00_0_1": torch.tensor([[0., 0., 0.], [1., 1., 1.]]),
    ...
}
results = metric(preds)
# {"unique_0.25", "multiple_0.25", "all_0.25", "unique_0.5", "multiple_0.5", "all_0.5"}
```

#### Nr3D

Predict the **object ID** of the target among the scene's candidate objects.

```python
from torchmetrics_ext.metrics.visual_grounding import Nr3DMetric

metric = Nr3DMetric(split="test")

# key: stimulus_id, value: predicted object ID
preds = {
    "scene0565_00-chair-4-25-0-1-24": 25,
    "scene0653_00-desk-6-16-13-14-15-17-18": 16,
    ...
}
results = metric(preds)
# {"easy", "hard", "view_dep", "view_indep", "all"}
```

#### Multi3DRefer

Predict **any number of boxes** per description, with shape `(N, 2, 3)`. Use an empty tensor to predict that nothing
matches. Predictions are matched to the ground truth with the Hungarian algorithm.

```python
import torch
from torchmetrics_ext.metrics.visual_grounding import Multi3DReferMetric

metric = Multi3DReferMetric(split="validation")

# key: "{scene_id}_{ann_id}"
preds = {
    "scene0011_00_0": torch.tensor([[[0., 0., 0.], [0.5, 0.5, 0.5]]]),                           # 1 box
    "scene0011_00_1": torch.tensor([[[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [2., 2., 2.]]]),  # 2 boxes
    "scene0011_00_2": torch.tensor([]),                                                           # no box
    ...
}
results = metric(preds)
# zero-target: "zt_wo_d", "zt_w_d"
# single-target: "st_wo_d_{t}", "st_w_d_{t}"; multi-target: "mt_{t}"; overall: "all_{t}", for t in (0.25, 0.5)
```

Set `strict=True` to require that each `update()` call covers exactly the full split.

#### ViGiL3D

Same input format and matching as Multi3DRefer. Scenes come from both ScanNet and ScanNet++.

```python
import torch
from torchmetrics_ext.metrics.visual_grounding import ViGiL3DMetric

metric = ViGiL3DMetric(split="validation")

# key: "{scene_id}_{ann_id}"
preds = {
    "scene0012_00_cf49717d-a751-417e-be93-32fa6a4aa1e4": torch.tensor([[[1.41, 1.13, 0.02], [1.54, 2.31, 1.60]]]),
    ...
}
results = metric(preds)
# {"zt", "st_0.25", "st_0.5", "mt_0.25", "mt_0.5", "all_0.25", "all_0.5"}
```

### 3D Visual Question Answering

Predictions are answer strings keyed by question ID.

#### VSI-Bench / ReVSI

Multiple-choice questions are scored by exact match. Numerical questions are scored by Mean Relative Accuracy (MRA),
following the [official implementation](https://github.com/vision-x-nyu/thinking-in-space). Only the first word of
each prediction is used, so `"3 chairs."` is read as `"3"`.

```python
from torchmetrics_ext.metrics.vqa import VSIBenchMetric, ReVSIMetric

metric = VSIBenchMetric(split="test")
# or: metric = ReVSIMetric(split="test", subset="all_frame")

# key: question "id", value: predicted answer
preds = {
    0: "3",
    1: "A",
    ...
}
results = metric(preds)
# {"<question_type>_acc", ..., "overall_acc"}
```

#### ScanQA

Each answer is scored against all reference answers with the standard captioning metrics from `pycocoevalcap`.

> METEOR and the PTB tokenizer need **Java** installed. Predictions are stored as strings, so this metric does not
> synchronize across devices. Gather predictions on one process before calling `compute()`.

```python
from torchmetrics_ext.metrics.vqa import ScanQAMetric

metric = ScanQAMetric(split="validation")

# key: "question_id"
preds = {
    "val-scene0441-17": "white rectangular",
    ...
}
results = metric(preds)
# {"BLEU_1", "BLEU_4", "METEOR", "ROUGE_L", "CIDEr"}
```

## Development

```bash
pip install -e ".[dev]"
pytest test
```

The tests download the benchmark ground truth on their first run. Downloads are cached under the Hugging Face
`datasets` cache directory.

## License

[Apache 2.0](LICENSE)
