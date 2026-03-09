import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from threading import Lock


def compute_bbox_with_padding(mask, padding_percent=0.1):
    """
    Estrae il bounding box 3D dalla maschera e aggiunge padding.
    
    Args:
        mask: array numpy della maschera binaria
        padding_percent: percentuale di padding (default 10%)
    
    Returns:
        dict con le coordinate del bbox con padding
    """
    # Trova i voxel non-zero
    indices = np.where(mask > 0)
    
    if len(indices[0]) == 0:
        return None  # Maschera vuota
    
    # Calcola i bounds per ogni dimensione
    mins = [indices[i].min() for i in range(3)]
    maxs = [indices[i].max() for i in range(3)]
    
    # Calcola le dimensioni originali del bbox
    dims = [maxs[i] - mins[i] for i in range(3)]
    
    # Calcola il padding (10% di ogni dimensione)
    paddings = [int(np.ceil(dims[i] * padding_percent)) for i in range(3)]
    
    # Applica il padding rispettando i bounds del volume
    bbox = {}
    shape = mask.shape
    for i in range(3):
        bbox[f'dim{i}_min'] = max(0, mins[i] - paddings[i])
        bbox[f'dim{i}_max'] = min(shape[i] - 1, maxs[i] + paddings[i])
        bbox[f'dim{i}_size'] = bbox[f'dim{i}_max'] - bbox[f'dim{i}_min'] + 1
    
    return bbox


def process_patient(pid, seg_dir):
    """
    Processa un singolo paziente.
    Ritorna un dict con i risultati o None se errore.
    """
    patient_seg_dir = os.path.join(seg_dir, pid)
    
    # Possibili nomi dei file (cerca variazioni)
    lung_file = None
    nodules_file = None
    
    # Cerca lung
    try:
        for fname in os.listdir(patient_seg_dir):
            if 'lung' in fname.lower() and 'nodule' not in fname.lower() and fname.endswith('.nii.gz'):
                lung_file = os.path.join(patient_seg_dir, fname)
            elif 'nodule' in fname.lower() and fname.endswith('.nii.gz'):
                nodules_file = os.path.join(patient_seg_dir, fname)
    except Exception as e:
        return {
            'patient_id': pid,
            'error': f"Error listing files: {str(e)}"
        }
    
    # Verifica che entrambi i file esistano
    if lung_file is None or nodules_file is None:
        return {
            'patient_id': pid,
            'error': f"Missing files - Lung: {lung_file is not None}, Nodules: {nodules_file is not None}"
        }
    
    # Carica le segmentazioni con nibabel
    try:
        lung_img = nib.load(lung_file)
        nodules_img = nib.load(nodules_file)
        
        lung_mask = np.asarray(lung_img.dataobj)
        nodules_mask = np.asarray(nodules_img.dataobj)
        
    except Exception as e:
        return {
            'patient_id': pid,
            'error': f"Error loading files: {str(e)}"
        }
    
    # Assicurati che siano binarie (thresholding a > 0)
    lung_mask = (lung_mask > 0).astype(np.uint8)
    nodules_mask = (nodules_mask > 0).astype(np.uint8)
    
    # Fa OR logico tra le maschere
    combined_mask = np.logical_or(lung_mask, nodules_mask).astype(np.uint8)
    
    # Estrai il bbox con padding del 10%
    bbox = compute_bbox_with_padding(combined_mask, padding_percent=0.1)
    
    if bbox is None:
        return {
            'patient_id': pid,
            'error': "Empty combined mask"
        }
    
    # Aggiungi il pid ai risultati
    bbox['patient_id'] = pid
    return bbox


def process_segmentations_multithread(seg_dir, csv_output, num_threads=8):
    """
    Processa le segmentazioni con multithreading,
    fa OR tra lungs e nodules, estrae bbox 3D con padding e salva in CSV.
    """
    results = []
    errors = []
    results_lock = Lock()
    
    all_id_list = sorted([
        d for d in os.listdir(seg_dir)
        if os.path.isdir(os.path.join(seg_dir, d))
    ])
    
    print(f"Processing {len(all_id_list)} patients with {num_threads} threads...")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Sottometti tutti i job
        futures = {
            executor.submit(process_patient, pid, seg_dir): pid 
            for pid in all_id_list
        }
        
        # Processa i risultati man mano che si completano
        with tqdm(total=len(all_id_list), desc="Processing patients") as pbar:
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    result = future.result()
                    
                    with results_lock:
                        if 'error' in result:
                            errors.append((pid, result['error']))
                        else:
                            results.append(result)
                
                except Exception as e:
                    with results_lock:
                        errors.append((pid, f"Future exception: {str(e)}"))
                
                pbar.update(1)
    
    # Stampa errori
    if errors:
        print(f"\n⚠ Errors processing {len(errors)} patients:")
        for pid, error in errors[:10]:  # Mostra primi 10 errori
            print(f"  {pid}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    # Crea DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("ERROR: No patients processed successfully!")
        return None
    
    # Riordina le colonne in modo logico
    cols_order = ['patient_id']
    for i in range(3):
        cols_order.extend([f'dim{i}_min', f'dim{i}_max', f'dim{i}_size'])
    
    df = df[cols_order]
    
    # Salva in CSV
    os.makedirs(os.path.dirname(csv_output), exist_ok=True)
    df.to_csv(csv_output, index=False)
    print(f"\n✓ Saved bounding box coordinates to {csv_output}")
    print(f"✓ Total patients processed successfully: {len(df)}")
    print(f"✓ Total errors: {len(errors)}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 3D bounding boxes from lung segmentations (multithreaded)")
    parser.add_argument(
        "--seg_dir", 
        type=str, 
        default="/mimer/NOBACKUP/groups/naiss2023-6-336/AIDA_multimodal_F&C/data/tabular/survival/AIDA/imaging/segmentations",
        help="Directory with segmentations"
    )
    parser.add_argument(
        "--output_csv", 
        type=str,
        default="/mimer/NOBACKUP/groups/naiss2023-6-336/AIDA_multimodal_F&C/data/tabular/survival/AIDA/imaging/bbox_coordinates.csv",
        help="Output CSV file with bbox coordinates"
    )
    parser.add_argument(
        "--threads", 
        type=int,
        default=8,
        help="Number of worker threads (default: 8)"
    )
    args = parser.parse_args()
    
    df_results = process_segmentations_multithread(args.seg_dir, args.output_csv, num_threads=args.threads)
    if df_results is not None:
        print("\nFirst few rows:")
        print(df_results.head())
        