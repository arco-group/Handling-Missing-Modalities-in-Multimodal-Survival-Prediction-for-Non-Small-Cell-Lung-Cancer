import os
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
from tqdm import tqdm
import argparse


def seg_lungs_and_lesions(input_path, output_path):
    # forza nnUNet a usare 1 solo processo → evita crash HPC
    os.environ["nnUNet_n_proc_DA"] = "1"
    os.environ["nnUNet_n_proc_export"] = "1"

    # API    # API
    seg_img = totalsegmentator(
            input_path,
            output_path,
            task="lung_nodules",
            device="gpu",
            fast=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Index of the batch to process")
    args = parser.parse_args()
    index = int(args.index)
    from pathlib import Path

    input_dir = "/mimer/NOBACKUP/groups/naiss2023-6-336/AIDA_multimodal_F&C/data/tabular/survival/NLST/100158"
    output_dir = "/mimer/NOBACKUP/groups/naiss2023-6-336/AIDA_multimodal_F&C/data/tabular/survival/NLST/predictions"


    # Ensure your base directories are Path objects
    input_dir_path = Path(input_dir)

    # 1. Search recursively for all 'volume.nii.gz' files
    # 'rglob' searches recursively. Use '*.nii.gz' if filenames vary.
    nii_files = list(input_dir_path.rglob("*nii.gz"))


    all_id_list = [
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
    ]

    id_list = all_id_list

    for input_path in tqdm(nii_files):
        patient_out = os.path.join(output_dir, input_path.parts[11])

        if os.path.exists(os.path.join(patient_out, "lung_nodules.nii.gz")) and os.path.exists(os.path.join(patient_out, "lung.nii.gz")):
            continue
        os.makedirs(patient_out, exist_ok=True)
        seg_lungs_and_lesions(input_path, patient_out)


