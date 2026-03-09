#!/bin/bash

# Define arrays
tasks=("Cix-regression")
models=("SGB" "RSF" "CPH")
targets=("OS_1d_regression") #  "PFS_1d_regression" "LocalProgression_1d_regression" "M+_1d_regression"
COUNT=0

# Loop over all combinations
for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    for target in "${targets[@]}"; do
      experiment="AIDA_tabular_${task}_${model}"
      echo "🚀 Running experiment: $experiment"
      python ./main.py experiment="$experiment" continue_experiment=true experiment/preprocessing/label@preprocessing.label=none mode=tabular_cix experiment/databases@db=AIDA_${target} experiment/paths/system@_global_=alvis_cix &

      COUNT=$((COUNT + 1))
      if (( COUNT % MAX_JOBS == 0 )); then
        wait
      fi
    done
  done
done
