import os
import numpy as np
import nibabel as nib
from natsort import natsorted  # per ordinare i file numericamente

# cartella dove hai le slices
input_dir = "103515"
output_file = "103515_volume.nii.gz"

# lista dei file .npy
files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

# ordinamento naturale (slice_1.npy, slice_2.npy, ..., slice_10.npy)
files = natsorted(files)

# carica tutte le slices in un array 3D
slices = [np.load(os.path.join(input_dir, f)) for f in files]
volume = np.stack(slices, axis=-1)  # asse finale = direzione z

print("Volume shape:", volume.shape)

# crea immagine NIfTI
# affine = matrice di trasformazione, se non hai info metti identità
affine = np.eye(4)
nifti_img = nib.Nifti1Image(volume, affine)

# salva
nib.save(nifti_img, output_file)
print(f"✅ Salvato in {output_file}")