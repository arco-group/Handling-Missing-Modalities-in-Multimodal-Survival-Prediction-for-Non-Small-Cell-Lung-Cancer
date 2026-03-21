#!/bin/bash

# Define arrays
tasks=("Cix-regression")
models=("SGB" "RSF" "CPH")
targets=("OS_1d_regression") 
MAX_JOBS=4
COUNT=0
PIDS=()
# Trap Ctrl+C (SIGINT) to kill all child processes
trap 'echo "🛑 Caught Ctrl+C, killing all running jobs..."; kill ${PIDS[@]} 2>/dev/null; exit 1' INT

for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    for target in "${targets[@]}"; do
      experiment="AIDA_CT_${task}_${model}"
      echo "🚀 Running experiment: $experiment ($target)"

      python ./main.py experiment="$experiment" \
        experiment/paths/system@_global_=local_ctfm \
        continue_experiment=false \
        experiment/preprocessing/label@preprocessing.label=cox \
        experiment/databases@db=CT_ctfm_AIDA_${target} \
        mode=ct_cox-label_cix &
      pid=$!
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

# Final wait for any remaining background jobs
wait
echo "✅ All experiments completed."