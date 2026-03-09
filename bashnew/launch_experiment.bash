#!/usr/bin/env bash
#SBATCH --job-name=exp1-expName
#SBATCH -A NAISS2024-5-577 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:2
#SBATCH -t 0-12:00:00
# Output files
#SBATCH --error=logs/expName_%J.err
#SBATCH --output=logs/expName_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=camillocaruso952@gmail.com

# Load modules
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1

# Activate venv
source /mimer/NOBACKUP/groups/naiss2023-6-336/ccaruso/venvGPU/bin/activate

# Executes the code
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ccaruso/CMC_utils_project
# Train
python ./main.py experiment="MIMIC_multimodal"

# Deactivate venv
deactivate
