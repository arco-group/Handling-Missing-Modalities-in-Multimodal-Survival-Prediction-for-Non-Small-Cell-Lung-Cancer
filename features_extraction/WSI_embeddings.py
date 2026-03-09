import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# cartella dove hai le slices
input_dir = "data/tabular/survival/AIDA/wsi/raw"

root_dir = Path(input_dir)
emb_dir = "data/tabular/survival/AIDA/wsi/embeddings"

os.makedirs(emb_dir, exist_ok=True)


embeddings =  list(root_dir.rglob("*.pt"))


ids = [os.path.basename(emb).split('.pt')[0] for emb in embeddings]


vectors = [torch.load(str(emb), map_location=torch.device('cpu'))[0].cpu().numpy() for emb in embeddings]


# Open Data All
data_path = "data/tabular/survival/AIDA/cross_validation/all_data.xlsx"
data = pd.read_excel(data_path, index_col=0)


ids_present = [int(id_) in data[data.WSI == 1].index for id_ in ids ]

for id_, vector, present in zip(ids, vectors, ids_present):
    if present:
        np.save(os.path.join(emb_dir, f"WSI_embedding_{str(id_)}.npy"), vector)



