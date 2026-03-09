"""
Extract CT-FM embeddings from NIfTI CT volumes.
Uses SegResEncoder from project-lighter/ct_fm_feature_extractor on HuggingFace.
Outputs 512-dim feature vectors per patient saved as .npy files.

Requirements: pip install lighter_zoo monai
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForeground,
    EnsureType,
    LoadImage,
    Orientation,
    ScaleIntensityRange,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "CT-FM" / "notebooks"))
from utils import IterableDataset

os.environ["HF_TOKEN"] = "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"

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = Path("data/tabular/survival/AIDA/imaging/volumes_I")
emb_dir = Path("data/tabular/survival/AIDA/imaging/embeddings_ctfm")
emb_dir.mkdir(parents=True, exist_ok=True)

# Skip already-processed patients
done_ids = {p.stem.split("_")[-1].replace(".npy", "") for p in emb_dir.glob("*.npy")}
nii_files = [p for p in data_dir.rglob("*.nii.gz") if p.parent.name not in done_ids]
print(f"Patients to process: {len(nii_files)}")
assert len(nii_files) > 0, "All patients already processed!"

# Preprocessing (same as CT-FM repo)
preprocess = Compose([
    LoadImage(ensure_channel_first=True),
    EnsureType(),
    Orientation(axcodes="SPL"),
    ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
    CropForeground(),
])

# Load model from HuggingFace
from lighter_zoo import SegResEncoder  # noqa: E402  (installed separately)

model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
model.eval()
model.to(device)

import monai.inferers  # noqa: E402

patch_size = (24, 128, 128)

for nii_path in nii_files:
    patient_id = nii_path.parent.name
    save_path = emb_dir / f"CTFM_embedding_{patient_id}.npy"

    try:
        img = preprocess(str(nii_path))  # (C, D, H, W)

        splitter = monai.inferers.SlidingWindowSplitter(patch_size, overlap=0.0)
        dataset = IterableDataset(splitter(img.unsqueeze(0)))
        patch_loader = torch.utils.data.DataLoader(dataset, batch_size=16)

        patch_features = []
        with torch.no_grad():
            for batch, _ in patch_loader:
                out = model(batch.squeeze(1).to(device))[-1]  # last encoder level
                feat = torch.nn.functional.adaptive_avg_pool3d(out, 1).flatten(start_dim=1)
                patch_features.append(feat.cpu())

        # Aggregate across patches: mean pooling -> 512-dim vector
        all_feats = torch.cat(patch_features, dim=0)  # (N_patches, 512)
        embedding = all_feats.mean(dim=0).numpy()     # (512,)

        np.save(save_path, embedding)
        print(f"✅ {patient_id} -> {embedding.shape}")

    except Exception as e:
        print(f"❌ {patient_id}: {e}")
