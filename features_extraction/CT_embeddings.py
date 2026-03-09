"""
Download Merlin and test the model on sample data that is downloaded from huggingface
"""

import os
import warnings
from pathlib import Path

import torch
import sys
sys.path.extend(['.', '..', '.Merlin'])
import numpy as np
from Merlin.merlin.data import DataLoader
from Merlin.merlin import Merlin


os.environ["HF_TOKEN"] = "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"

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"


data_dir = "data/tabular/survival/AIDA/imaging/volumes_I"

root_dir = Path(data_dir)
emb_dir = Path("data/tabular/survival/AIDA/imaging/embeddings_2")

embeddings =  list(emb_dir.rglob("*.npy"))
ids = [os.path.basename(emb).split('.npy')[0].split('_')[-1] for emb in embeddings]
nii_files = list(root_dir.rglob("*.nii.gz"))

nii_files = [i for i in nii_files if os.path.basename(i.parent) not in ids]
print(nii_files)
assert len(nii_files) > 0, "Yes processed, May the force be with you!"
cache_dir = "data_cache"


datalist = [
    {
        "image": str(path_nii), "text": "", "ID":  os.path.basename(path_nii.parent)
    }
        for path_nii in nii_files
]


dataloader = DataLoader(
    datalist=datalist,
    cache_dir=cache_dir,
    batchsize=8,
    shuffle=False,
    num_workers=0,
)


## Get the Image Embeddings
model = Merlin(ImageEmbedding=True)
model.eval()
model.cuda()

# Create output directory if it doesn't exist

emb_dir.mkdir(parents=True, exist_ok=True)

for i, batch in enumerate(dataloader):
    with torch.no_grad():
        outputs = model(batch["image"].to(device))
  # remove the first dimension if it's 1
    # Extract embeddings (assuming outputs[0] are the embeddings)
    embeddings = outputs[0].cpu().numpy()   # shape (1, 8, 2048)
    for i in range(embeddings.shape[0]):
        ID_patient = batch["ID"][i]
        print(f"Batch {i} - Embeddings shape: {embeddings[i].shape}")
        # Save as .npy file, one per batch

        save_path = emb_dir / f"CT_embedding_{ID_patient}.npy"
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir)
        np.save(save_path, embeddings[i])
        print(f"✅ Saved {embeddings.shape} to {save_path}")