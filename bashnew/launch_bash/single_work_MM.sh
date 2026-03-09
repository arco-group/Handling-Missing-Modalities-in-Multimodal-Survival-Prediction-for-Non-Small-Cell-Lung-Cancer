
module load Python/3.10.4-GCCcore-11.3.0
cd "/mimer/NOBACKUP/groups/naiss2023-6-336/AIDA_multimodal_F&C"
source AIDA_MM/bin/activate



target="OS_1d_regression"
model="survival_maria"
task="cix"
experiment="AIDA_multimodal_WSI+CT+tabular_${task}_cox_label"

python ./main.py \
    experiment="${experiment}" \
    experiment/model@shared_net="${model}" \
    continue_experiment=true \
    mode="wsi_ct_tab_cox-label_${task}" \
    experiment/databases@dbs.0="WSI_AIDA_${target}" \
    experiment/databases@dbs.1="CT_AIDA_${target}" \
    experiment/databases@dbs.2="AIDA_${target}" \
    db_name="WSI+CT+tabular_multimodal_${target}" &