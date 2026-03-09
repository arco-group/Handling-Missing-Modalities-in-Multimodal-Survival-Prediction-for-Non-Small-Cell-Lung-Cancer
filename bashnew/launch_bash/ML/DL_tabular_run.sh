#!/bin/bash

cd ../..
# Define arrays
tasks=("CTix") # TODO "Cix-regression"
models=("NAIM")
targets=("1mOS" "1mPFS" "6mPFS" "LocalProgression_1m" "LocalProgression_6m" "M+_1m" "M+_6m") #"6mOS"

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