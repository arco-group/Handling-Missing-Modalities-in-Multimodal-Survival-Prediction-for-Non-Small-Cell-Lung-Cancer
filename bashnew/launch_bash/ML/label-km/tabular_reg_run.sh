#!/bin/bash

cd ../../../..
# Define arrays
tasks=("Cix-regression")
models=("SGB" "RSF" "CPH")
targets=("1dOS" "1dPFS" "LocalProgression_1d" "M+_1d")

# Loop over all combinations
for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    for target in "${targets[@]}"; do
      experiment="AIDA_tabular_${task}_${model}_${target}"
      echo "🚀 Running experiment: $experiment"
      python ./main.py experiment="$experiment" continue_experiment=false experiment/preprocessing/label@preprocessing.label=km mode=tabular_km-label_cix
    done
  done
done


cd ../../..