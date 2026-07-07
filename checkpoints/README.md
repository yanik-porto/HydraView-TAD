---
license: mit
language:
- en
base_model:
- firework8/ProtoGCN
pipeline_tag: video-classification
tags:
- human
- action
- detection
---

# HydraView - Temporal Action Detection

PyTorch implementation of the paper "Improving Viewpoint-Invariance and Temporal Consistency for Action Detection".

Released on [Github](https://github.com/yanik-porto/HydraView-TAD.git).

## Pretrained Models

All the checkpoints are provided in : [huggingface](https://huggingface.co/yaniknocigar/hydraview-tad)

## Experimental Results

| Dataset | PKUMMD-v1 X-Sub | PKUMMD-v1 X-Sub X-View | BABEL Split1 | BABEL Split2 | BABEL Split3 | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| | 93.15 | 96.10 | 70.68 | 84.04 | 78.61 |