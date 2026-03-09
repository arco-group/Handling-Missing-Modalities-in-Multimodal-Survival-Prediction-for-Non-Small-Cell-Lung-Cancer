# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
# Load required module and activate environment
module load Python/3.10.4-GCCcore-11.3.0
source AIDA_MM/bin/activate
```

For Merlin (Vision Language Model for CT), use the separate environment:
```bash
module load Python/3.12.3-GCCcore-13.3.0
source merlin_env/bin/activate
```

## Running Experiments

```bash
# Run with default config (survival_experiment)
python main.py

# Override config values via Hydra
python main.py experiment=AIDA_tabular_Cix-regression_tabnet
python main.py device=cpu experiment/paths/system=alvis

# Launch ML/DL experiments in batch
bash launch_ML.sh
bash launch_DL.sh
bash launch_MM_DL.sh
```

## Architecture Overview

### Configuration System (Hydra + OmegaConf)
All experiment parameters live in `confs/`. The main config (`confs/config.yaml`) composes sub-configs from:
- `confs/experiment/` — 30+ predefined experiment configs (model, dataset, CV strategy, metrics)
- `confs/experiment/paths/` — system-specific paths (local, alvis, SSD)
- `confs/experiment/model/`, `databases/`, `metric/`, `loss/`, `preprocessing/`, etc.

### Main Entry Point (`main.py`)
Selects one of five pipeline types based on `cfg.experiment.pipeline`:
- `simple` — single-modality supervised learning with cross-validation
- `missing` — robustness testing under synthetic missing data
- `multimodal_early_fusion` — concatenate modalities before model
- `multimodal_joint_fusion` — shared representation learned jointly
- `multimodal_late_fusion` — combine predictions from separate modality models

### Pipeline Flow (all pipelines)
```
Hydra config → Pipeline → Dataset instantiation → CV fold setup →
Preprocessing (per-fold to prevent leakage) → Model assembly →
PyTorch Lightning training → Evaluation (C-index, IBS, KM) → Results saved
```

### `CMC_utils/` Library
The core reusable library:
- `datasets/` — `SupervisedTaskDataset`, `SurvivalDataset`, `MultimodalDataset` (tabular + imaging)
- `models/tabular/` — TabNet, TabTransformer, FTTransformer, NAIM, custom MLP
- `models/imaging/` — SOTA CNN/ViT wrappers, custom CNNs, MONAI-based models
- `models/generic/multimodal_model.py` — `MultimodalSurvivalLearner`: fuses per-modality encoders via a shared network + survival prediction head
- `pipelines/` — pipeline implementations (simple, missing, multimodal variants)
- `preprocessing/tabular/` — per-fold normalization, categorical encoding, imputation
- `preprocessing/imaging/` — medical image preprocessing
- `metrics/survival_metrics.py` — C-index (Ct, UNO), time-dependent AUC, integrated Brier score
- `losses/survival_losses.py` — Cox, log-likelihood, ranking losses
- `cross_validation/` — stratified CV with provided train/test splits
- `save_load/` — checkpointing and result persistence

### Supported Modalities & Models
- **Tabular**: TabNet, TabTransformer, FTTransformer, NAIM, MLP, classical survival models (Cox PH, RSF, SGB)
- **Imaging (CT/WSI)**: ResNet, ViT (timm), TorchXRayVision models, MONAI, custom CNNs
- **Multimodal**: Early/joint/late fusion of any combination of the above

### Data
- Tabular clinical data: `data/tabular/survival/AIDA/`
- CT embeddings: `data/tabular/survival/AIDA/imaging/embeddings_2/`
- WSI embeddings: `data/tabular/survival/AIDA/wsi/embeddings/`
- Cross-validation splits defined in Excel files alongside the tabular data

### Analysis Scripts (top-level)
- `dataset_preparation.py` — attach imaging embedding paths to patient records
- `new_analysis.py` — log-rank tests, Kaplan-Meier curves, optimal cutoff finding
- `aggregate_results_into_tables.py` — collect fold results into LaTeX-ready tables
- `plot_over_missing.py` — performance vs. missing data percentage curves
- `compute_kaplan_meyer_curve_and_stats.py` / `generate_plot_survival_curves.py` — KM visualizations

### Key Design Decisions
- **Preprocessing is fold-aware**: fitting happens on train split only (`set_fold_preprocessing` / `apply_preprocessing`), preventing data leakage
- **Hydra overrides** are the primary mechanism for hyperparameter sweeps — avoid hardcoding values
- **PyTorch Lightning** handles training loops, GPU placement, and checkpointing
- **Competing risks** support is built into survival models and metrics
- Time is converted from days to years via `time_divisor: 365` in configs
