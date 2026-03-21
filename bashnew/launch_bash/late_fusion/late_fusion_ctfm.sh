#!/bin/bash

echo pwd # Print the current working directory
# Define arrays
tasks=("Cixregression")
models=("survival_NN_ODST_NAIM" ) # "CrossAttentionMissingModalityMasking" "CrossAttentionMissingModality"  "survival_maria"
targets=("OS_1d_regression") #  "PFS_1d_regression" "LocalProgression_1d_regression" "M+_1d_regression"

MAX_JOBS=5
COUNT=0
PIDS=()
# Trap Ctrl+C (SIGINT) to kill all child processes
trap 'echo "🛑 Caught Ctrl+C, killing all running jobs..."; kill ${PIDS[@]} 2>/dev/null; exit 1' INT


for task in "${tasks[@]}"; do
  for model in "${models[@]}"; do
    for target in "${targets[@]}"; do

      experiment="AIDA_multimodal_CT+Tab_${task}_cox_label"
      python ./main.py experiment=late_AIDA_multimodal_CT+tabular_Cixregression_cox_label_.yaml \
      experiment/paths/system@_global_=local_ctfm \
      experiment/model@model.0=${model} \
      experiment/model@model.1=${model} \
      experiment/databases@dbs.0=CT_ctfm_AIDA_${target} \
      experiment/databases@dbs.1=AIDA_${target} \
      continue_experiment=true \
      model_name=late_fusion_${model} &

      echo "Started late fusion for model: ${model}, task: ${task}, target: ${target}"
      python ./main.py experiment=late_AIDA_multimodal_WSI+CT+tabular_Cixregression_cox_label.yaml \
      experiment/paths/system@_global_=local_ctfm \
      experiment/model@model.0=${model} \
      experiment/model@model.1=${model} \
      experiment/model@model.2=${model} \
      experiment/databases@dbs.0=WSI_AIDA_${target} \
      experiment/databases@dbs.1=CT_ctfm_AIDA_${target} \
      experiment/databases@dbs.2=AIDA_${target} \
      continue_experiment=true \
      model_name=late_fusion_${model} &

      echo "Started late fusion for model: ${model}, task: ${task}, target: ${target}"
      python ./main.py experiment=late_AIDA_multimodal_WSI+CT_Cixregression_cox_label.yaml \
      experiment/paths/system@_global_=local_ctfm \
      experiment/model@model.0=${model} \
      experiment/model@model.1=${model} \
      experiment/databases@dbs.0=WSI_AIDA_${target} \
      experiment/databases@dbs.1=CT_ctfm_AIDA_${target} \
      continue_experiment=true \
      model_name=late_fusion_${model} &

      #echo "Started late fusion for model: ${model}, task: ${task}, target: ${target}"
      #python ./main.py experiment=late_AIDA_multimodal_WSI+tabular_Cixregression_cox_label.yaml \
      #experiment/paths/system@_global_=local_ctfm \
      #experiment/model@model.0=${model} \
      #experiment/model@model.1=${model} \
      #experiment/databases@dbs.0=WSI_AIDA_${target} \
      #experiment/databases@dbs.1=AIDA_${target} \
      #continue_experiment=true \
      #model_name=late_fusion_${model} &


      pid=$!
      PIDS+=($pid)
      COUNT=$((COUNT + 4))

      # Wait when reaching the maximum concurrent jobs
      if (( COUNT % MAX_JOBS == 0 )); then
        wait
        PIDS=()  # reset tracked PIDs after waiting
      fi

    done
  done
done




