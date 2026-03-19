"""
Extract CT-CLIP embeddings from NIfTI CT volumes.
CT-CLIP is a contrastive vision-language model for chest CT scans.
Outputs 512-dim L2-normalised image embeddings per patient saved as .npy files.

Repository: https://github.com/ibrahimethemhamamci/CT-CLIP
Requirements:
    cd transformer_maskgit && pip install -e .
    cd CT_CLIP && pip install -e .
Pretrained weights: download from HuggingFace (hamamci-suite/ct-clip-extended)
    and place at CT_CLIP/clip_weights/CT_CLIP_zeroshot.pt  (or set CTCLIP_CKPT env var)
"""

import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
# Add CT_CLIP source to path (installed editably or from repo clone)
sys.path.insert(0, str(REPO_ROOT / "CT-CLIP" / "CT_CLIP" / "ct_clip"))
sys.path.insert(0, str(REPO_ROOT / "CT-CLIP" / "transformer_maskgit"))

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = Path("data/tabular/survival/AIDA/imaging/volumes_I")
emb_dir  = Path("data/tabular/survival/AIDA/imaging/embeddings_ctclip")
emb_dir.mkdir(parents=True, exist_ok=True)

# Skip already-processed patients
done_ids  = {p.stem.split("_")[-1] for p in emb_dir.glob("*.npy")}
nii_files = [p for p in data_dir.rglob("*.nii.gz") if p.parent.name not in done_ids]
print(f"Patients to process: {len(nii_files)}")
assert len(nii_files) > 0, "All patients already processed!"

# ---------------------------------------------------------------------------
# Preprocessing — mirrors the official CT-CLIP data pipeline:
#   HU clipping → resample to 0.75×0.75×1.5 mm → crop/pad to 480×480×240
#   → scale to [-1, 1]
# ---------------------------------------------------------------------------
TARGET_SHAPE = (480, 480, 240)   # (H, W, D) before permuting to model input
TARGET_SPACING_XY = 0.75        # mm
TARGET_SPACING_Z  = 1.5         # mm
HU_MIN, HU_MAX = -1000.0, 1000.0


def preprocess_nifti(nii_path: Path) -> torch.Tensor:
    """Return a (1, 1, D, H, W) float32 tensor ready for CT-CLIP."""
    img  = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)   # (H, W, D) or similar

    # --- slope / intercept (already applied by get_fdata for most scanners,
    #     but clip to valid HU range) ---
    data = np.clip(data, HU_MIN, HU_MAX)

    # --- resample to target voxel spacing ---
    header = img.header
    voxel_dims = np.abs(np.array(header.get_zooms()[:3], dtype=np.float32))
    # guard against zero-spacing
    voxel_dims = np.where(voxel_dims > 0, voxel_dims, 1.0)
    zoom_factors = (
        voxel_dims[0] / TARGET_SPACING_XY,
        voxel_dims[1] / TARGET_SPACING_XY,
        voxel_dims[2] / TARGET_SPACING_Z,
    )
    data = zoom(data, zoom_factors, order=1, prefilter=False)

    # --- crop / pad to TARGET_SHAPE (H, W, D) ---
    out = np.full(TARGET_SHAPE, fill_value=-1.0, dtype=np.float32)
    for ax, (src, tgt) in enumerate(zip(data.shape, TARGET_SHAPE)):
        if src >= tgt:
            # centre-crop
            start = (src - tgt) // 2
            data  = np.take(data, range(start, start + tgt), axis=ax)
        # if src < tgt: padding is already in `out`
    slices = tuple(slice(0, min(s, t)) for s, t in zip(data.shape, TARGET_SHAPE))
    out[slices] = data[slices]

    # --- scale HU to [-1, 1] ---
    out = out / 1000.0

    # data is (H, W, D); model expects (D, H, W) → add batch + channel dims
    out = out.transpose(2, 0, 1)                          # (D, H, W)
    tensor = torch.from_numpy(out).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    return tensor.to(device)


# ---------------------------------------------------------------------------
# Load CT-CLIP model
# ---------------------------------------------------------------------------
from ct_clip import CTCLIP  # noqa: E402
from transformer_maskgit import CTViT  # noqa: E402
from transformers import BertModel, BertTokenizer  # noqa: E402

ckpt_path = Path(
    os.environ.get(
        "CTCLIP_CKPT",
        str(REPO_ROOT / "CT-CLIP" / "clip_weights" / "CT-CLIP_v2.pt"),
    )
)
assert ckpt_path.exists(), (
    f"CT-CLIP checkpoint not found at {ckpt_path}.\n"
    "Download from HuggingFace (hamamci-suite/ct-clip-extended) and set "
    "CTCLIP_CKPT env var or place at CT_CLIP/clip_weights/CT_CLIP_zeroshot.pt"
)

# Build the 3D image encoder (CTViT) — matches the pretrained checkpoint
image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8,
)

# BiomedVLP-CXR-BERT text encoder
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

model = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 294912,
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,
    use_mlm = False,
    downsample_image_embeds = False,
    use_all_token_embeds = False,
)
state = torch.load(str(ckpt_path), map_location="cuda")
# Checkpoints may be wrapped (trainer state) or raw state dicts
if isinstance(state, dict) and "model" in state:
    state = state["model"]
model.load_state_dict(state, strict=False)
model.to(device)
model.eval()

# Dummy text tokens — we only need image embeddings
tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)
dummy_text = tokenizer(
    [""],
    return_tensors="pt",
    padding="max_length",
    max_length=512,
    truncation=True,
)
dummy_text = dummy_text.to(device)

# ---------------------------------------------------------------------------
# Extraction loop
# ---------------------------------------------------------------------------
for nii_path in nii_files:
    patient_id = nii_path.parent.name
    save_path  = emb_dir / f"CTCLIP_embedding_{patient_id}.npy"

    try:
        ct_tensor = preprocess_nifti(nii_path)          # (1, 1, D, H, W)

        with torch.no_grad():
            # return_latents=True → (text_latents, image_latents, enc_image)
            _, image_latents, _ = model(
                dummy_text,
                ct_tensor,
                device=device,
                return_latents=True,
            )

        # image_latents: (1, 512) L2-normalised
        embedding = image_latents.squeeze(0).cpu().numpy()   # (512,)
        np.save(save_path, embedding)
        print(f"✅ {patient_id} -> {embedding.shape}")

    except Exception as e:
        print(f"❌ {patient_id}: {e}")
