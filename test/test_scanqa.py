import pytest
import torch
from torchmetrics_ext.metrics.vqa import ScanQAMetric

dataset = ScanQAMetric(split="validation")

pred = {
    "val-scene0441-17": "white rectangular"
}

score = dataset(pred)