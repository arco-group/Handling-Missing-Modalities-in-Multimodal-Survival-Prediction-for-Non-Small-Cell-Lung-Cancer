# Launch Scripts — Usage Guide

All scripts must be run from the **project root** (`AIDA_multimodal_F&C/`).

## Environment Setup

```bash
module load Python/3.10.4-GCCcore-11.3.0
source AIDA_MM/bin/activate
```

For scripts using CTFM (Merlin) embeddings, use the alternative environment:

```bash
module load Python/3.12.3-GCCcore-13.3.0
source merlin_env/bin/activate
```

---

## Log Convention

Save logs with the current date and time to the `/logs/` directory:

```bash
mkdir -p logs
bash bashnew/launch_bash/<script>.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_<script_name>.log
```

Example:

```bash
bash bashnew/launch_bash/ML/launch_ML.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_launch_ML.log
```

---

## Directory Structure

```
bashnew/launch_bash/
├── CT_run.sh                  # Legacy CT unimodal sweep (old-style)
├── WSI_run.sh                 # Legacy WSI unimodal sweep (old-style)
├── single_work_MM.sh          # Quick single multimodal run (WSI+CT+Tab, survival_maria)
├── ML/
│   ├── launch_ML.sh           # Master launcher for all ML (classical) unimodal experiments
│   └── label-cox/
│       ├── ct_reg_run.sh          # CT embeddings — classical models (SGB, RSF, CPH), cox label
│       ├── ct_ctclip_reg_run.sh   # CT-CLIP embeddings — classical models, cox label
│       ├── ct_ctfm_reg_run.sh     # CTFM embeddings — classical models, cox label
│       ├── tabular_reg_run.sh     # Tabular only — classical models, cox label
│       └── wsi_reg_run.sh         # WSI embeddings — classical models, cox label
├── DL/
│   ├── launch_DL.sh           # Master launcher for all DL unimodal experiments
│   └── label-cox/
│       ├── ct_reg_run.sh          # CT embeddings — DL models (NAIM/ODST), cox label
│       ├── ct_ctclip_reg_run.sh   # CT-CLIP embeddings — DL models, cox label
│       ├── ct_ctfm_reg_run.sh     # CTFM embeddings — DL models, cox label
│       ├── tabular_reg_run.sh     # Tabular only — DL models, cox label
│       └── wsi_reg_run.sh         # WSI embeddings — DL models, cox label
├── MM-DL/
│   ├── launch_MM_DL.sh        # Master launcher for all multimodal DL experiments
│   ├── cox-label-freeze/      # Frozen unimodal encoders (joint fusion)
│   │   ├── mm_tab_ct.sh           # Tabular + CT (standard embeddings)
│   │   ├── mm_tab_pat.sh          # Tabular + WSI (pathology)
│   │   ├── mm_pat_ct.sh           # WSI + CT (standard embeddings)
│   │   ├── mm_tab_pat_ct.sh       # Tabular + WSI + CT (standard embeddings)
│   │   ├── mm_tab_ct_ctclip.sh    # Tabular + CT-CLIP embeddings
│   │   ├── mm_tab_ct_ctfm.sh      # Tabular + CTFM embeddings
│   │   ├── mm_pat_ct_ctclip.sh    # WSI + CT-CLIP embeddings
│   │   ├── mm_pat_ct_ctfm.sh      # WSI + CTFM embeddings
│   │   ├── mm_tab_pat_ct_ctclip.sh  # Tabular + WSI + CT-CLIP
│   │   └── mm_tab_pat_ct_ctfm.sh    # Tabular + WSI + CTFM
│   └── cox-label-unfreeze/    # Unfrozen encoders (end-to-end fine-tuning)
│       ├── mm_tab_ct.sh
│       ├── mm_tab_pat.sh
│       ├── mm_pat_ct.sh
│       ├── mm_tab_pat_ct.sh
│       ├── mm_tab_ct_ctclip.sh
│       ├── mm_tab_ct_ctfm.sh
│       ├── mm_pat_ct_ctclip.sh
│       ├── mm_pat_ct_ctfm.sh
│       ├── mm_tab_pat_ct_ctclip.sh
│       └── mm_tab_pat_ct_ctfm.sh
└── late_fusion/
    ├── late_fusion.sh         # Late fusion with standard CT embeddings (WSI+Tab only active)
    ├── late_fusion_ctclip.sh  # Late fusion with CT-CLIP embeddings
    └── late_fusion_ctfm.sh    # Late fusion with CTFM (Merlin) embeddings
```

---

## Scripts Reference

### `CT_run.sh` / `WSI_run.sh`
Legacy scripts for unimodal sweeps (CT and WSI). Loop over tasks, models, and targets using old config naming. Mostly superseded by the `ML/` and `DL/` directories.

```bash
bash bashnew/launch_bash/CT_run.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_CT_run.log
bash bashnew/launch_bash/WSI_run.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_WSI_run.log
```

---

### `single_work_MM.sh`
Launches a single multimodal experiment (WSI + CT + Tabular, `survival_maria` model, cox label). Useful for quick testing of a specific configuration.

> **Note:** Requires the environment to be loaded and the working directory set to the project root (the script sets it internally via `cd`).

```bash
bash bashnew/launch_bash/single_work_MM.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_single_work_MM.log
```

---

### `ML/launch_ML.sh`
Master launcher for classical (ML) unimodal experiments. Currently active:
- `ct_ctclip_reg_run.sh` — CT-CLIP + classical survival models
- `ct_ctfm_reg_run.sh` — CTFM + classical survival models

Other scripts (tabular, WSI, standard CT) are commented out and can be activated in the file.

```bash
bash bashnew/launch_bash/ML/launch_ML.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_launch_ML.log
```

Individual scripts can also be run directly:

```bash
bash bashnew/launch_bash/ML/label-cox/ct_reg_run.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_ML_ct_reg.log
bash bashnew/launch_bash/ML/label-cox/ct_ctclip_reg_run.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_ML_ct_ctclip.log
bash bashnew/launch_bash/ML/label-cox/ct_ctfm_reg_run.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_ML_ct_ctfm.log
bash bashnew/launch_bash/ML/label-cox/tabular_reg_run.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_ML_tabular.log
bash bashnew/launch_bash/ML/label-cox/wsi_reg_run.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_ML_wsi.log
```

**Models:** `SGB`, `RSF`, `CPH`
**Target:** `OS_1d_regression` (others configurable in the script)
**Parallelism:** up to `MAX_JOBS=4` concurrent jobs

---

### `DL/launch_DL.sh`
Master launcher for deep learning unimodal experiments. Currently active:
- `ct_ctclip_reg_run.sh` — CT-CLIP + DL survival models
- `ct_ctfm_reg_run.sh` — CTFM + DL survival models

```bash
bash bashnew/launch_bash/DL/launch_DL.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_launch_DL.log
```

Individual scripts:

```bash
bash bashnew/launch_bash/DL/label-cox/ct_reg_run.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_DL_ct_reg.log
bash bashnew/launch_bash/DL/label-cox/ct_ctclip_reg_run.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_DL_ct_ctclip.log
bash bashnew/launch_bash/DL/label-cox/ct_ctfm_reg_run.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_DL_ct_ctfm.log
bash bashnew/launch_bash/DL/label-cox/tabular_reg_run.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_DL_tabular.log
bash bashnew/launch_bash/DL/label-cox/wsi_reg_run.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_DL_wsi.log
```

**Models:** `Survival_NN_ODST_NAIM` (others configurable)
**Target:** `OS_1d_regression`
**Parallelism:** up to `MAX_JOBS=4`

---

### `MM-DL/launch_MM_DL.sh`
Master launcher for all multimodal DL experiments (joint fusion, both freeze and unfreeze). Currently active combinations use CT-CLIP and CTFM embeddings. Standard embedding variants are commented out.

```bash
bash bashnew/launch_bash/MM-DL/launch_MM_DL.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_launch_MM_DL.log
```

Individual scripts (freeze = encoders frozen during training):

```bash
# Frozen encoders
bash bashnew/launch_bash/MM-DL/cox-label-freeze/mm_tab_ct_ctclip.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_freeze_tab_ct_ctclip.log
bash bashnew/launch_bash/MM-DL/cox-label-freeze/mm_tab_ct_ctfm.sh   2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_freeze_tab_ct_ctfm.log
bash bashnew/launch_bash/MM-DL/cox-label-freeze/mm_pat_ct_ctclip.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_freeze_pat_ct_ctclip.log
bash bashnew/launch_bash/MM-DL/cox-label-freeze/mm_pat_ct_ctfm.sh   2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_freeze_pat_ct_ctfm.log
bash bashnew/launch_bash/MM-DL/cox-label-freeze/mm_tab_pat_ct_ctclip.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_freeze_tab_pat_ct_ctclip.log
bash bashnew/launch_bash/MM-DL/cox-label-freeze/mm_tab_pat_ct_ctfm.sh   2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_freeze_tab_pat_ct_ctfm.log

# Unfrozen encoders (end-to-end)
bash bashnew/launch_bash/MM-DL/cox-label-unfreeze/mm_tab_ct_ctclip.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_unfreeze_tab_ct_ctclip.log
bash bashnew/launch_bash/MM-DL/cox-label-unfreeze/mm_tab_ct_ctfm.sh   2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_unfreeze_tab_ct_ctfm.log
bash bashnew/launch_bash/MM-DL/cox-label-unfreeze/mm_pat_ct_ctclip.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_unfreeze_pat_ct_ctclip.log
bash bashnew/launch_bash/MM-DL/cox-label-unfreeze/mm_pat_ct_ctfm.sh   2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_unfreeze_pat_ct_ctfm.log
bash bashnew/launch_bash/MM-DL/cox-label-unfreeze/mm_tab_pat_ct_ctclip.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_unfreeze_tab_pat_ct_ctclip.log
bash bashnew/launch_bash/MM-DL/cox-label-unfreeze/mm_tab_pat_ct_ctfm.sh   2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_unfreeze_tab_pat_ct_ctfm.log
```

**Model:** `concat_model_FC`
**Target:** `OS_1d_regression`
**Parallelism:** `MAX_JOBS=1` (sequential by default)

---

### `late_fusion/`

Late fusion combines independently trained unimodal models at prediction time.

| Script | CT Embeddings | Active modality combinations |
|---|---|---|
| `late_fusion.sh` | Standard CT | WSI + Tabular only (others commented out) |
| `late_fusion_ctclip.sh` | CT-CLIP | CT+Tab, WSI+CT+Tab, WSI+CT |
| `late_fusion_ctfm.sh` | CTFM (Merlin) | CT+Tab, WSI+CT+Tab, WSI+CT |

```bash
bash bashnew/launch_bash/late_fusion/late_fusion.sh       2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_late_fusion.log
bash bashnew/launch_bash/late_fusion/late_fusion_ctclip.sh 2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_late_fusion_ctclip.log
bash bashnew/launch_bash/late_fusion/late_fusion_ctfm.sh   2>&1 | tee logs/$(date +%Y-%m-%d_%H-%M-%S)_late_fusion_ctfm.log
```

**Model:** `survival_NN_ODST_NAIM`
**Target:** `OS_1d_regression`
**Parallelism:** up to `MAX_JOBS=5`, runs 3 parallel jobs per loop iteration

> **Note:** `late_fusion_ctfm.sh` requires the Merlin environment (Python 3.12 + `merlin_env`).

---

## CT Embedding Variants

| Suffix | Embedding source | Config key |
|---|---|---|
| _(none)_ | Standard CNN CT embeddings | `CT_AIDA_*` |
| `_ctclip` | CT-CLIP (vision-language model) | `CT_ctclip_AIDA_*` |
| `_ctfm` | CTFM / Merlin (foundation model) | `CT_ctfm_AIDA_*` |

---

## Common Customizations

**Change target outcome** (edit the `targets` array inside the script):
```bash
targets=("OS_1d_regression" "PFS_1d_regression" "LocalProgression_1d_regression" "M+_1d_regression")
```

**Resume an interrupted experiment:**
```bash
continue_experiment=true   # already set in most scripts
```

**Start fresh:**
```bash
continue_experiment=false
```

**Change parallelism** (edit `MAX_JOBS` inside the script):
```bash
MAX_JOBS=4   # number of concurrent python processes
```
