#!/bin/bash

echo pwd # Print the current working directory
# Define arrays
tasks=("Cixregression")
models=("concat_model_FC" ) # "concat_model_FC" 
targets=("OS_1d_regression") #  "PFS_1d_regression" "LocalProgression_1d_regression" "M+_1d_regression"
# Loop over all combinations
MAX_JOBS=1
COUNT=0
for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    for target in "${targets[@]}"; do
      experiment="AIDA_multimodal_CT+tabular_${task}_freeze_ms_cox_label"
      echo "🚀 Running experiment: $experiment"
      python ./main.py experiment="$experiment" experiment/model@shared_net=${model} continue_experiment=true mode=ct_tab_cox-label_cix experiment/databases@dbs.0=CT_AIDA_${target} experiment/databases@dbs.1=AIDA_${target} db_name=CT+tabular_multimodal_${target} &
      COUNT=$((COUNT + 1))
      if (( COUNT % MAX_JOBS == 0 )); then
        wait
      fi
    done
  done
done


