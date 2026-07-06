# LARI-Mamba

This repository contains the LARI-Mamba online signature verification code and the reproducibility files prepared for manuscript revision.

It includes the length-adaptive signature verification model, dataset loaders, training and verification scripts, fixed trial-definition rules, released raw score files, and a small score-level metric recalculation utility.

## Repository Layout

```text
LARI-Mamba/
|-- README.md
|-- requirements.txt
|-- train.py
|-- main_DsDTW_length_adaptive.py
|-- test_verification.py
|-- dsdtw.py
|-- length_adaptive_module.py
|-- rare_stable_stage1.py
|-- tsc_mamba.py
|-- soft_dtw_cuda.py
|-- dtw_cuda.py
|-- dist.py
|-- dataset/
|   |-- datasetTrainAll_SF.py
|   |-- datasetTest_SF.py
|   |-- load_dataset.py
|   |-- load_from_txt.py
|   `-- utils.py
|-- rules/
|   `-- fixed trial-definition files
|-- rule/
|   `-- compatibility copy used by legacy verification scripts
|-- scores/
|   `-- released raw comparison-score CSV files
|-- evaluation/
|   |-- metrics.py
|   `-- evaluate_scores.py
|-- configs/
|   |-- lari_mamba_default.yaml
|   `-- evaluation_scores.yaml
|-- docs/
|   |-- reproducibility.md
|   |-- data_layout.md
|   `-- checkpoints.md
|-- checkpoints/
|   `-- README.md
|-- results/
|   `-- README.md
`-- supplementary_material.zip
```

## What Is Included

- LARI-Mamba model code and training entry point.
- Dataset-loading and feature-extraction utilities for online signature datasets.
- Fixed verification rules in `rules/`.
- A legacy-compatible copy of the rules in `rule/`, because `test_verification.py` uses `./rule/...`.
- Released raw comparison scores in `scores/`.
- `supplementary_material.zip`, a compressed copy of the submitted supplementary rules/scores package.
- Empty `checkpoints/` directory for server-side model weights.

## Installation

```bash
pip install -r requirements.txt
```

CUDA-enabled SoftDTW/DTW requires compatible PyTorch, CUDA, `numba`, and `llvmlite` versions.

## Data

Raw public datasets are not redistributed in this repository. Put them under a local data root, or set:

```bash
export LARI_MAMBA_DATA_ROOT=/path/to/data
```

On Windows PowerShell:

```powershell
$env:LARI_MAMBA_DATA_ROOT = "E:\path\to\data"
```

See `docs/data_layout.md` for the expected layout.

## Training

The main training script is `main_DsDTW_length_adaptive.py`. The `train.py` wrapper is provided for convenience:

```bash
python train.py --dataset all --epochs 30 --seed 111
```

Examples:

```bash
python train.py --dataset MCYT --epochs 30 --seed 111
python train.py --dataset all --stage 6 --length-threshold 150 --smooth-weight 0.001
```

Checkpoints are saved to `models/` by default during training. For the GitHub release, final manuscript checkpoints should be copied into `checkpoints/` after transfer from the server.

## Verification With Fixed Rules

`test_verification.py` reads fixed rules from `./rule/...` and can be used for protocol-based evaluation when data and checkpoints are available.

```bash
python test_verification.py --help
```

## Recalculate Metrics From Released Scores

The score-level check does not require datasets or checkpoints:

```bash
python evaluation/evaluate_scores.py --scores-dir scores --output results/score_metrics_summary.csv
```

## Release Check

Before uploading to GitHub:

```bash
python scripts/check_release.py
```

This verifies that the required code files, rules, scores, docs, and checkpoint placeholders are present.
