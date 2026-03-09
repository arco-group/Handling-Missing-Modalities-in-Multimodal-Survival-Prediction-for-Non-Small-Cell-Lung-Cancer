"""
Extract ColiPri (microsoft/colipri) embeddings from NIfTI CT volumes.
ColiPri is a 3D vision-language model for chest CT scans.
Outputs 768-dim pooled feature vectors per patient saved as .npy files.

Requirements: pip install colipri
Model: https://huggingface.co/microsoft/colipri
"""

import os
from pathlib import Path

import numpy as np
import torch

os.environ["HF_TOKEN"] = "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"

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = Path("data/tabular/survival/AIDA/imaging/volumes_I")
emb_dir = Path("data/tabular/survival/AIDA/imaging/embeddings_colipri")
emb_dir.mkdir(parents=True, exist_ok=True)

# Skip already-processed patients
done_ids = {p.stem.split("_")[-1].replace(".npy", "") for p in emb_dir.glob("*.npy")}
nii_files = [p for p in data_dir.rglob("*.nii.gz") if p.parent.name not in done_ids]
print(f"Patients to process: {len(nii_files)}")
assert len(nii_files) > 0, "All patients already processed!"

# Load ColiPri model onto the correct device.
# get_model() internally uses init_empty_weights (meta device) + load_checkpoint_and_dispatch,
# which auto-dispatches layers and may leave them mismatched with input tensors.
# Fix: instantiate the model normally (on CPU with real tensors), load safetensors weights
# in-place, then move the whole model to the target device.
from hydra.utils import instantiate  # noqa: E402
from safetensors.torch import load_model as safetensors_load_model  # noqa: E402
from colipri.checkpoint import download_weights, load_model_config  # noqa: E402
from colipri import get_processor  # noqa: E402

checkpoint_path = download_weights()
config = load_model_config()
model = instantiate(config)                           # real CPU tensors, no meta device
safetensors_load_model(model, str(checkpoint_path))   # load weights in-place on CPU
model = model.to(device)                              # move entire model to GPU/CPU
model.eval()
processor = get_processor()

for nii_path in nii_files:
    patient_id = nii_path.parent.name
    save_path = emb_dir / f"ColiPri_embedding_{patient_id}.npy"

    try:
        # ColiPri processor.process_images expects a file path (str/Path)
        preprocessed = processor.process_images(str(nii_path))
        images_batch = processor.to_images_batch(preprocessed)

        # images_batch stays on CPU (model is on CPU)
        if isinstance(images_batch, torch.Tensor):
            images_batch = images_batch.to(device)
        elif isinstance(images_batch, dict):
            images_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in images_batch.items()}

        with torch.no_grad():
            # pool=True, project=True -> (1, 768) projected pooled embedding
            embedding = model.encode_image(images_batch, pool=True, project=True)

        embedding = embedding.squeeze(0).cpu().numpy()  # (768,)
        np.save(save_path, embedding)
        print(f"✅ {patient_id} -> {embedding.shape}")

    except Exception as e:
        print(f"❌ {patient_id}: {e}")
