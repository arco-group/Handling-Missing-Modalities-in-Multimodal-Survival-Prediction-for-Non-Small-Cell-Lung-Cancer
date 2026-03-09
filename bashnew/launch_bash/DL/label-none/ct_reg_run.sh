#!/bin/bash

echo pwd # Print the current working directory
# Define arrays
tasks=("Cix-regression")
models=("Survival_NN" "Survival_NN_ODST" "Survival_NN_ODST_NAIM" "Survival_NN_NAIM")
targets=("OS_1d_regression") #  "PFS_1d_regression" "LocalProgression_1d_regression" "M+_1d_regression"
MAX_JOBS=4
COUNT=0


# Loop over all combinations
for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    for target in "${targets[@]}"; do
      experiment="AIDA_CT_${task}_${model}"
      echo "🚀 Running experiment: $experiment"
      python ./main.py experiment="$experiment" continue_experiment=true experiment/preprocessing/label@preprocessing.label=none experiment/databases@db="CT2_AIDA_${target}" mode=ct_cix experiment/paths/system@_global_=alvis_cix &
      COUNT=$((COUNT + 1))
      if (( COUNT % MAX_JOBS == 0 )); then
        wait
      fi
    done
  done
done


