<div align="center" style="display:flex;flex-direction:column;align-items:center;gap:8px;">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:center;">
        <h1 style="margin:0;text-align:center;">HydraView - Temporal Action Detection</h1>
        <a href="https://arxiv.org/pdf/2605.22695" target="_blank">
            <img src="https://img.shields.io/badge/arXiv-2605.22695-B31B1B?style=flat-square" alt="arXiv">
        </a>
    </div>
</div>

<div align="center">
<img src='./static/images/overview.png' align="center" width="1280"/> <br>
Overview of our temporal action detection method with multiple viewpoints. For each input video viewpoint, an untrimmed sequence is encoded with a spatio-temporal encoder to generate features with improved view invariance. These features are then refined by our multi-view and multi-scale temporal encoder (HydraView) for localizing each action over time. <br><br>
</div>


## Introduction

This is the repository that contains the source code for the paper "Improving Viewpoint-Invariance and Temporal Consistency for Action Detection".

The paper is accepted to ICIP 2026.

**Abstract**:
> Viewpoint change invariance and action temporal consistency are critical aspects for the effective deployment of human action detection of untrimmed videos. Existing appearance-based video detection methods often struggle with limited viewpoint diversity during training, while motion-based detection approaches frequently fail to model fine-grained temporal relationships across consecutive motion windows. This paper introduces a novel two-stage action detection approach designed to improve both view-invariance and global temporal coherence properties. In the first stage, we extract motion features from augmented virtual viewpoints, solely used at training. Then, the second stage introduces a new view-invariant, multi-scale temporal encoder based on selective state-space sequence modelling to aggregate information across viewpoints and time scales. Experiments on PKU-MMD and BABEL benchmarks demonstrate that this approach significantly outperforms state-of-the-art methods in all considered splits.

## Installation

### Downloads
Download CUDA (12.6)

git clone https://github.com/yanik-porto/Vim_mamba-1p1p1

### Python Instals
All experiments were made with python 3.12.7.
```shell
export CUDA_HOME=<PATH_TO_YOUR_CUDA_FOLDER>
pip install -r requirements.txt
pip install -e libs/Vim_mamba-1p1p1 --no-build-isolation
pip install -e git+https://github.com/Dao-AILab/causal-conv1d.git@v1.0.0#egg=causal-conv1d --no-build-isolation
```
## Usage

### Checkpoints

### Data

### Test

`python test.py checkpoints/babel/babel_hydra_view_swgcn.yaml checkpoints/babel/split1_1view.pth --name=testset_split1_window_1view --gt_name=tad_labels`

## Citation
If you find this code useful for your research, please cite the paper:

```bibtex
@article{porto2026vitad,
  title={Improving Viewpoint-Invariance and Temporal Consistency for Action Detection},
  author={Porto, Martins and Chalumeau, Demonceaux},
  journal={ICIP},
  year={2026}
}
```

## Acknowledgements
This work was partially supported by grants from projects ANER MOVIS from ``Conseil Regional de Bourgogne-Franche-Comte'' and ANR MANYVIS (ANR-23-CE23-0003-01), to whom we are grateful.

**ICB:** [Laboratoire Interdisciplinaire Carnot de Bourgogne](https://icb.u-bourgogne.fr/)
