# Feature extraction — CT & WSI preprocessing

This directory turns **raw imaging data** (chest CT NIfTI volumes, WSI slides) into the **fixed-dimensional embeddings** consumed by the multimodal survival training pipeline.

> Outputs land under `data/tabular/survival/<COHORT>/imaging/embeddings_*/` (CT) and `data/tabular/survival/<COHORT>/wsi/embeddings/` (WSI). Paths follow the layout described in the [root README](../README.md#data-layout).

---

## 0. Prerequisites

- Foundation-model weights downloaded as described in the [main setup](../README.md#3-foundation-model-weights).
- Submodules initialised (`git submodule update --init --recursive`).
- Two Python environments active depending on the step:
  - **`AIDA_MM`** (Python 3.10.4) — CT-FM, CT-CLIP, WSI, segmentation, bounding-box steps.
  - **`merlin_env`** (Python 3.12.3) — **only** for Merlin VLM extraction.
- A Hugging Face token with accepted model licences, exported as `HF_TOKEN` or set via `huggingface-cli login`. **Do not hard-code it into the scripts.**

```bash
export HF_TOKEN=hf_...        # do not commit
module load Python/3.10.4-GCCcore-11.3.0
source ../AIDA_MM/bin/activate
```

---

## 1. Expected raw data layout

Mirror your cohort under `data/tabular/survival/<COHORT>/imaging/`:

```
data/tabular/survival/<COHORT>/
├── imaging/
│   ├── volumes_raw/               # one folder per patient, containing one *.nii.gz
│   │   ├── <patient_id_1>/<patient_id_1>.nii.gz
│   │   └── ...
│   ├── segmentations/             # produced by step 2  (lung_nodules masks)
│   ├── volumes_I/                 # produced by step 3  (lung-bbox cropped volumes)
│   ├── embeddings_ctfm/           # produced by step 4a (CT-FM, 512-d)
│   ├── embeddings_ctclip/         # produced by step 4b (CT-CLIP, 512-d L2-normed)
│   └── embeddings_merlin/         # produced by step 4c (Merlin VLM)
└── wsi/
    ├── raw/                        # one *.pt patch-embedding bag per slide
    └── embeddings/                 # produced by step 5  (aggregated, per-patient)
```

CT files **must** be valid NIfTI (`.nii.gz`), with one volume per patient folder named after the patient ID. WSI files are expected as PyTorch tensors of pre-extracted patch embeddings (see step 5 for the format).

---

## 2. Lung & lesion segmentation — [`extract_lungs.py`](extract_lungs.py)

Runs [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (`task="lung_nodules"`) on every raw CT volume and writes a multi-label NIfTI mask covering the lung lobes and detected nodules. This mask defines the region used to crop the CT in the next step.

```bash
pip install totalsegmentator
python features_extraction/extract_lungs.py --index 0
```

**Edit before running** — the input/output directories are hard-coded near the bottom of the script:

```python
input_dir  = "data/tabular/survival/<COHORT>/imaging/volumes_raw"
output_dir = "data/tabular/survival/<COHORT>/imaging/segmentations"
```

`--index` selects a batch when sharding across SLURM jobs (one GPU per shard). The environment variables `nnUNet_n_proc_DA=1` and `nnUNet_n_proc_export=1` are set automatically to prevent multiprocessing crashes on HPC.

**Expected runtime:** ~30–90 s/volume on a single A100 (`fast=False`).

---

## 3. Lung bounding-box crop — [`bboxex_by_segmentations.py`](bboxex_by_segmentations.py)

Reads each segmentation mask, derives a 3D bounding box around the lungs with **10 % padding**, and crops the corresponding raw CT volume to that bbox. This keeps the field of view consistent across patients and removes table / arms / abdomen that distract CT foundation models.

```bash
python features_extraction/bboxex_by_segmentations.py
```

Output goes to `imaging/volumes_I/<patient_id>/<patient_id>.nii.gz`. The script is thread-parallel (`ThreadPoolExecutor`); tune the pool size in-script if you are I/O-bound.

---

## 4. CT embedding extraction

All three CT scripts read from `imaging/volumes_I/` and write **one `.npy` per patient** under their respective output directory. They auto-skip patients whose embedding already exists, so re-runs are cheap.

### 4a. CT-FM — [`CTFM_embeddings.py`](CTFM_embeddings.py)

Foundation model: **SegResEncoder** from [project-lighter/ct_fm_feature_extractor](https://huggingface.co/project-lighter/ct_fm_feature_extractor). Produces **512-dim** feature vectors.

```bash
pip install lighter_zoo monai
python features_extraction/CTFM_embeddings.py
```

Default I/O:

```python
data_dir = "data/tabular/survival/<COHORT>/imaging/volumes_I"
emb_dir  = "data/tabular/survival/<COHORT>/imaging/embeddings_ctfm"
```

Preprocessing applied (MONAI `Compose`): `LoadImage → Orientation(RAS) → ScaleIntensityRange(-1024..1024 → 0..1) → CropForeground → EnsureType`.

### 4b. CT-CLIP — [`CTCLIP_embeddings.py`](CTCLIP_embeddings.py)

Foundation model: **CT-CLIP** ([ibrahimethemhamamci/CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP)). Produces **512-dim L2-normalised** image embeddings.

```bash
# One-off install of the upstream packages
cd CT-CLIP/transformer_maskgit && pip install -e . && cd -
cd CT-CLIP/CT_CLIP            && pip install -e . && cd -

# Place the zero-shot checkpoint
mkdir -p CT-CLIP/CT_CLIP/clip_weights
huggingface-cli download hamamci-suite/ct-clip-extended CT_CLIP_zeroshot.pt \
    --local-dir CT-CLIP/CT_CLIP/clip_weights
# (or set CTCLIP_CKPT to point elsewhere)

python features_extraction/CTCLIP_embeddings.py
```

Volumes are resampled to the CT-CLIP input grid (HU-clipped, isotropic resize) before forward pass.

### 4c. Merlin VLM — [`CT_embeddings.py`](CT_embeddings.py)

Foundation model: **Merlin** ([StanfordMIMI/Merlin](https://github.com/StanfordMIMI/Merlin)). Runs in the **Python 3.12** environment.

```bash
module load Python/3.12.3-GCCcore-13.3.0
source ../merlin_env/bin/activate
pip install -e ../Merlin/

python features_extraction/CT_embeddings.py
```

Default output: `imaging/embeddings_2/CT_embedding_<patient_id>.npy`. The Merlin DataLoader handles its own MONAI preprocessing (windowing, resampling, normalisation) — no extra preprocessing needed beyond the lung crop from step 3. Set `cache_dir` if you want a different MONAI cache location.

---

## 5. WSI embedding aggregation — [`WSI_embeddings.py`](WSI_embeddings.py)

Whole-Slide Images are **not** processed from raw `.svs` here — we assume patch-level features have already been extracted upstream (e.g. with [HIPT](https://github.com/mahmoodlab/HIPT), [CLAM](https://github.com/mahmoodlab/CLAM), or a similar pipeline) and saved as PyTorch tensors `wsi/raw/<slide_id>.pt`, where the tensor's first slot (`[0]`) is the slide-level aggregated embedding.

```bash
python features_extraction/WSI_embeddings.py
```

The script:

1. Loads every `.pt` under `wsi/raw/`.
2. Cross-references each slide ID against `cross_validation/all_data.xlsx` (column `WSI == 1`) to keep only patients flagged as having usable WSI.
3. Writes one `.npy` per kept patient at `wsi/embeddings/WSI_embedding_<patient_id>.npy`.

If your upstream WSI pipeline produces a different tensor format, adapt the `[0]` indexing accordingly.

---

## 6. Sanity check — [`open_embeddings.py`](open_embeddings.py)

Loads a handful of `.npy` files from each modality and prints shape, dtype, mean, and std — a quick way to confirm that everything lines up dimensionally before kicking off training.

```bash
python features_extraction/open_embeddings.py
```

---

## 7. Wiring extracted embeddings into training

Once steps 2–5 have run, the resulting paths line up with the defaults in:

- `confs/experiment/paths/{local,alvis_snic,SSD}.yaml` — system path roots
- `confs/experiment/databases/AIDA*.yaml` — manifest references for tabular + imaging + WSI

The `AIDA_*` experiment configs (e.g. `AIDA_multimodal_WSI+CT+tabular_Cixregression_freeze_ms_cox_label.yaml`) consume these embeddings directly through the `MultimodalDataset` loader. Switch to your own cohort by adding a new path profile and a new database YAML pointing at your `<COHORT>` directory.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `assert len(nii_files) > 0` fails | All patients already embedded — delete the output directory or skip this modality |
| OOM during CT-FM / Merlin | Reduce `roi_size` / batch size; for Merlin, drop `cache_dir` to disable MONAI cache |
| TotalSegmentator hangs on HPC | Make sure `nnUNet_n_proc_DA=1` and `nnUNet_n_proc_export=1` are set (script does it for you) |
| HF 401 / 403 | Token missing or licence not accepted — `huggingface-cli login` + accept on the model page |
| CT-CLIP `ModuleNotFoundError` | The two upstream packages weren't pip-installed editably — see step 4b |
| Merlin checkpoint not found | Run `python -c "from merlin import Merlin; Merlin()"` once to trigger the auto-download |
