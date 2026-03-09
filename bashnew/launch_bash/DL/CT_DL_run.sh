#!/bin/bash

cd "/mimer/NOBACKUP/groups/naiss2023-6-336/AIDA_multimodal_F&C"
# Define arrays
tasks=( "CTix" ) # TODO "Cix-regression"
models=("NAIM")
targets=("1mPFS" "1mOS" "LocalProgression_1m" "LocalProgression_6m" "M+_1m" "M+_6m" "6mPFS" ) #"6mPFS"
mode=("CT") # "tabular" "WSI" "WSI"



# Loop over all combinations
for task in "${tasks[@]}"; do
  for md in "${mode[@]}"; do
    for model in "${models[@]}"; do
      for target in "${targets[@]}"; do
        experiment="AIDA_${md}_${task}_${model}_${target}"
        echo "🚀 Running experiment: $experiment"
        python main.py experiment="$experiment"
      done
    done
  done
done

