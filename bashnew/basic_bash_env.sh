#!/usr/bin/env bash

cd "/mimer/NOBACKUP/groups/naiss2023-6-336/AIDA_multimodal_F&C"

module load Python/3.10.4-GCCcore-11.3.0

source AIDA_MM/bin/activate

python ./main.py experiment="AIDA_tabular_CTIx_SGB_6mOS"
