#!/bin/bash

cd ../../..
# Define arrays
tasks=("CTix")
models=("SGB" "RSF" "CPH")
targets=("1mOS" "1mPFS" "6mOS" "6mPFS" "LocalProgression_1m" "LocalProgression_6m" "M+_1m" "M+_6m")

# Loop over all combinations
for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    for target in "${targets[@]}"; do
      experiment="AIDA_WSI_${task}_${model}_${target}"
      echo "🚀 Running experiment: $experiment"
      python ./main.py experiment="$experiment"
    done
  done
done


cd ..
cd ..
cd ..
