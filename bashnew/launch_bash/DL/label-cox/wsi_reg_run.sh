#!/bin/bash

echo pwd # Print the current working directory
# Define arrays
# Define arrays
tasks=("Cix-regression")
models=("Survival_NN" "Survival_NN_ODST" ) # "Survival_NN_ODST_NAIM" "Survival_NN_NAIM"
targets=("OS_1d_regression")
MAX_JOBS=4
COUNT=0
PIDS=()
# Trap Ctrl+C (SIGINT) to kill all child processes
trap 'echo "🛑 Caught Ctrl+C, killing all running jobs..."; kill ${PIDS[@]} 2>/dev/null; exit 1' INT

# Loop over all combinations
for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    for target in "${targets[@]}"; do
      experiment="AIDA_WSI_${task}_${model}"
      echo "🚀 Running experiment: $experiment"
      python ./main.py experiment="$experiment" continue_experiment=false experiment/preprocessing/label@preprocessing.label=cox experiment/databases@db=WSI_AIDA_${target} mode=wsi_cox-label_cix &
      PIDS+=($pid)
      COUNT=$((COUNT + 1))

      # Wait when reaching the maximum concurrent jobs
      if (( COUNT % MAX_JOBS == 0 )); then
        wait
        PIDS=()  # reset tracked PIDs after waiting
      fi
    done
  done
done


