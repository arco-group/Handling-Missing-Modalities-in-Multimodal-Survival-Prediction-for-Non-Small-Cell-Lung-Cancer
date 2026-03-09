import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
name_m='C-index'
for m in ['c_index']:

    # Base directory containing subfolders like "0-0-0", "0-0-25", etc.
    results_path = "outputs/fold_specific"
    output_path = "tests/fold_specific"
    os.makedirs(output_path, exist_ok=True)

    targets = ("OS_1d_regression", ) # , "PFS_1d_regression", "LocalProgression_1d_regression", "M+_1d_regression"
    colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}  # blue, green, red
    modalityID_to_name = {1: "WSI", 2: "CT", 3: "Tabular"}

    experiments = {
            "wsi_ct_tab_cox-label_cix": [1, 2, 3],
            "wsi_ct_cox-label_cix"    : [1, 2],
            "wsi_tab_cox-label_cix"   : [1, 3],
            "ct_tab_cox-label_cix"    : [2, 3],

    }
    Unimodal_experiments = {
            "wsi_cox-label_cix"    : [1],
            "ct_cox-label_cix"     : [2],
            "tabular_cox-label_cix": [3],
    }

    experiments_mm_name = {
            "wsi_ct_tab_cox-label_cix": "WSI+CT+Tabular",
            "wsi_ct_cox-label_cix"    : "WSI+CT",
            "wsi_tab_cox-label_cix"   : "WSI+Tabular",
            "ct_tab_cox-label_cix"    : "CT+Tabular",
    }

    # Folder configurations (each represents missing modality levels)
    folders = {
            "wsi_ct_tab_cox-label_cix": [
                    "0-0-0", "0-0-25", "0-0-50", "0-0-75", "0-0-100",
                    "0-25-0", "0-50-0", "0-75-0", "0-100-0",
                    "25-0-0", "50-0-0", "75-0-0", "100-0-0"
            ],
            "wsi_ct_cox-label_cix"    : [
                    "0-0", "0-25", "0-50", "0-75", "0-100",
                    "25-0", "50-0", "75-0", "100-0"
            ],
            "wsi_tab_cox-label_cix"   : [
                    "0-0", "0-25", "0-50", "0-75", "0-100",
                    "25-0", "50-0", "75-0", "100-0"
            ],
            "ct_tab_cox-label_cix"    : [
                    "0-0", "0-25", "0-50", "0-75", "0-100",
                    "25-0", "50-0", "75-0", "100-0"
            ]
    }

    # first of all, saving the modality results for the following models
    models_unimodal = ['CPH', 'RSF', 'survivalGradientBoosting', 'survivalNN_ODST_NAIM']
    override = True
    # Define color palette for different missing modality combinations
    colors_map = {
            'missing_wsi' : '#d62728',  # red
            'missing_tab' : '#ff7f0e',  # orange
            'missing_both': '#8B4513',  # brown
            'unimodal_wsi': '#1f77b4',  # blue
            'unimodal_tab': '#2ca02c',  # green
    }
    for unimodal_exp, mod_list in Unimodal_experiments.items():
        results = []
        for target in targets:
            # map target LocalProgression to LP

            if target == "LocalProgression_1d_regression":
                source = "LocalProgression"
                target_short = "LP"

                target = target.replace(source, target_short)

            for model in models_unimodal:

                # find the experiment with the specific model
                import re

                all_exp = os.listdir(os.path.join(results_path, unimodal_exp))
                exps = [exp for exp in all_exp if re.findall(rf"{model}*", exp) and target in exp]

                base_dir = os.path.join(results_path, unimodal_exp, exps[-1])
                # Loop through folders and read CSV files
                sub_name = os.listdir(os.path.join(base_dir, "provided_missing_provided_missing" if not 'tabular' in unimodal_exp else "provided_provided", "results"))[0]

                csv_path = os.path.join(base_dir, "provided_missing_provided_missing" if not 'tabular' in unimodal_exp else "provided_provided", "results", sub_name, "unbalanced/test", "averages_classes_average_performance.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
                    # Adjust column names if necessary (check if 'mean'/'std' exist)
                    for _, row in df.iterrows():

                        results.append({
                                "modality": modalityID_to_name[mod_list[0]],
                                "metric"  : m,
                                "model"   : model,
                                "target"  : target,
                                "mean"    : row[(m, 'mean')],
                                "std"     : row[(m, 'std')]
                        })
                else:
                    pass


        df_all = pd.DataFrame(results)
        unimodal_path = os.path.join(output_path, f"unimodal_{modalityID_to_name[mod_list[0]]}")
        os.makedirs(unimodal_path, exist_ok=True)
        df_all.to_csv(os.path.join(unimodal_path, f"unimodal_{modalityID_to_name[mod_list[0]]}_{m}_test.csv"), index=False)

    # first of all, saving the modality results for the following models
    models_unimodal = ['survivalNN', 'survivalNN_NAIM', 'survivalNN_ODST_NAIM']
    modality_results_best = []
    for unimodal_exp, mod_list in Unimodal_experiments.items():
        results = []
        for target in targets:
            # map target LocalProgression to LP

            if target == "LocalProgression_1d_regression":
                source = "LocalProgression"
                target_short = "LP"

                target = target.replace(source, target_short)

            for model in models_unimodal:

                # find the experiment with the specific model
                import re

                all_exp = os.listdir(os.path.join(results_path, unimodal_exp))
                if model == 'survivalNN':
                    all_exp = [exp for exp in all_exp if not 'ODST' in exp and not 'NAIM' in exp]

                exps = [exp for exp in all_exp if re.findall(rf"{model}*", exp) and target in exp]
                exps = [exp for exp in exps if 'CT2' not in exp]
                base_dir = os.path.join(results_path, unimodal_exp, exps[-1])
                # Loop through folders and read CSV files
                sub_name = os.listdir(os.path.join(base_dir, "provided_missing_provided_missing" if not 'tabular' in unimodal_exp else "provided_provided", "results"))[0]

                csv_path = os.path.join(base_dir, "provided_missing_provided_missing" if not 'tabular' in unimodal_exp else "provided_provided", "results", sub_name, "unbalanced/test", "averages_classes_average_performance.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
                    # Adjust column names if necessary (check if 'mean'/'std' exist)
                    for _, row in df.iterrows():    

                        results.append({
                                "modality": modalityID_to_name[mod_list[0]],
                                "metric"  : m,
                                "model"   : model,
                                "target"  : target,

                                "mean"    : row[(m, 'mean')],
                                "std"     : row[(m, 'std')]
                        })
                        if model == 'survivalNN_ODST_NAIM':
                            modality_results_best.append({
                                    "Modalities": modalityID_to_name[mod_list[0]],
                                    "metric"  : m,
                                    "model"   : model,
                                    "target"  : target,
                                    "mean"    : row[(m, 'mean')],
                                    "std"     : row[(m, 'std')],
                                    "freeze": False
                            })

                else:
                    pass
        df_all = pd.DataFrame(results)
        unimodal_path = os.path.join(output_path, f"ablation_deep_unimodal_{modalityID_to_name[mod_list[0]]}")
        os.makedirs(unimodal_path, exist_ok=True)
        df_all.to_csv(os.path.join(unimodal_path, f"ablation_deep_unimodal_{modalityID_to_name[mod_list[0]]}_{m}_test.csv"), index=False)

    df_best_modality = pd.DataFrame(modality_results_best)
    ablation_deep_best_modality_path = os.path.join(output_path, f"ablation_deep_best")
    os.makedirs(ablation_deep_best_modality_path, exist_ok=True)
    df_best_modality.to_csv(os.path.join(ablation_deep_best_modality_path, f"ablation_deep_best_modality_test_{m}.csv"), index=False)
    symbols = {'ConcatODST': "circle", 'CrossAttentionMissingModality': "square"}
    models_multimodal = ['ConcatODST'] # , 'CrossAttentionMissingModality'
    for freeze in ['unfrozen', 'ms-frozen']:

        for multimodal_name, mod_list in experiments.items():
            multimodal = os.path.join(output_path, f"multimodal_{experiments_mm_name[multimodal_name]}")
            os.makedirs(multimodal, exist_ok=True)
            exp = os.path.join(multimodal, f"multimodal_{experiments_mm_name[multimodal_name]}_{freeze}_test.csv")
            if os.path.exists(exp) and not override:
                df_all = pd.read_csv(exp)
            else:
        
                results = []
                for target in targets:
                    # map target LocalProgression to LP
                    num_of_modalities = len(mod_list)
                    for model in models_multimodal:

                        # find the experiment with the specific model
                        import re

                        all_exp = os.listdir(os.path.join(results_path, multimodal_name))
                        if model == 'survivalNN':
                            all_exp = [exp for exp in all_exp if not 'ODST' in exp and not 'NAIM' in exp]

                        exps = [exp for exp in all_exp if re.findall(rf"{model}*", exp) and target in exp]

                        if freeze == 'ms-frozen':
                            exps = [exp for exp in all_exp if re.findall(rf"{model}*", exp) and target in exp and 'ms-frozen' in exp]
                        else:
                            exps = [exp for exp in all_exp if re.findall(rf"{model}*", exp) and target in exp and not 'ms-frozen' in exp]

                        if "Masking" not in model:
                            exps = [exp for exp in exps if not 'Masking' in exp]
                        else:
                            exps = [exp for exp in exps if 'Masking' in exp]
                        exps = [exp for exp in exps if 'CT2' not in exp]
                        # if multimodal_name == "wsi_tab_cox-label_cix" and model == "CrossAttentionMissingModality" and freeze == "ms-frozen":
                        #     print("Skipping invalid combination, to be done later")
                        #     continue

                        base_dir = os.path.join(results_path, multimodal_name, exps[-1])
                        # Loop through folders and read CSV files
                        sub_name = os.listdir(os.path.join(base_dir, "provided_provided", "results"))[0]

                        subfolders = folders[multimodal_name]

                        for subfolder in subfolders:

                            csv_path = os.path.join(base_dir, "provided_provided", "results", sub_name, "0", subfolder, "unbalanced/test", "averages_classes_average_performance.csv")
                            if os.path.exists(csv_path):
                                df = pd.read_csv(csv_path, index_col=0, header=[0, 1])
                                # Adjust column names if necessary (check if 'mean'/'std' exist)
                                for _, row in df.iterrows():
                                    partial = {
                                            "modality": modalityID_to_name[mod_list[0]],
                                            "metric"  : m,
                                            "model"   : model,

                                            "target"  : target,
                                            "mean"    : row[(m, 'mean')],
                                            "std"     : row[(m, 'std')]
                                    }
                                    for num, i in enumerate(mod_list):
                                        partial[f"modality_{modalityID_to_name[i]}_missing%"] = int(subfolder.split('-')[num])

                                    if subfolder == ("0-0-0" if num_of_modalities == 3 else "0-0"):
                                        from pathlib import Path
                                        import os

                                        prediction_folder = os.path.join(base_dir, "provided_provided", "predictions", sub_name, "0", subfolder)

                                        # Create a Path object
                                        # Paths
                                        prediction_path = Path(prediction_folder)
                                        prediction_path_rest = Path(os.path.join(base_dir, "provided_provided", "predictions", sub_name, "0"))

                                        # ---- VAL FILES ----
                                        csv_files_val = list(prediction_path_rest.glob("*val.csv"))
                                        files_opened_val = (
                                            pd.concat([pd.read_csv(file) for file in csv_files_val], ignore_index=True)
                                            .drop(columns=['label', 'prediction'])
                                            .rename(columns={'probability': 'score'})
                                        )

                                        # ---- OTHER FILES ----
                                        csv_files = list(prediction_path.rglob("*.csv"))
                                        files_opened = (
                                            pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
                                            .drop(columns=['label', 'prediction'])
                                            .rename(columns={'probability': 'score'})
                                        )

                                        # ---- KEEP ONLY VAL ROWS WITH NEW IDs ----
                                        files_unique_val = files_opened_val[~files_opened_val['ID'].isin(files_opened['ID'])].iloc[:-1,:]

                                        final_file = pd.concat([files_opened, files_unique_val])

                                        print(f"Unique validation samples: {len(files_unique_val)}")
                                        final_file.to_csv(os.path.join(os.path.join(f'{"_".join([modalityID_to_name[mod] for mod in mod_list])}_{freeze}_pred_aggregated_score.csv')))
                                        local_partial = {
                                            "metric"  : m,
                                            "model"   : model,

                                            "target"  : target,
                                            "mean"    : row[(m, 'mean')],
                                            "std"     : row[(m, 'std')]
                                        }
                                        local_partial['freeze'] = freeze == "ms-frozen"
                                        local_partial['Modalities'] = '_'.join([modalityID_to_name[mod] for mod in mod_list])
                                        modality_results_best.append(local_partial)

                                    results.append(partial)
                            else:
                                pass

                df_all = pd.DataFrame(results)

                multimodal = os.path.join(output_path, f"multimodal_{experiments_mm_name[multimodal_name]}")
                os.makedirs(multimodal, exist_ok=True)
                df_all.to_csv(os.path.join(multimodal, f"multimodal_{experiments_mm_name[multimodal_name]}_{freeze}_{m}_test.csv"), index=False)
            print(f"Saved multimodal_{experiments_mm_name[multimodal_name]}_{freeze}_test.csv")

            df_all=df_all[df_all['model'] == 'ConcatODST']

            for target in targets:

                df_metric = df_all[df_all['target'] == target]

                df_metric["setting"] = df_metric.apply(lambda row: "_".join([str(row[f"modality_{modalityID_to_name[i]}_missing%"]) for num, i in enumerate(mod_list)]), axis=1)

                #df_all = pd.DataFrame(results)

                num_of_modalities = len(mod_list)
                df_zero = df_metric[df_metric["setting"] == ("0_0_0" if num_of_modalities == 3 else "0_0")].copy()
                df_metric = df_metric.drop(
                        df_metric[df_metric["setting"] == ("0_0_0" if num_of_modalities == 3 else "0_0")].index
                )

                # Extract which modality was masked and by how much
                df_metric["missing_modality"] = [experiments[multimodal_name][np.argmax([int(n) + 1 for n in i.split('_')])] if i != ("0_0_0" if num_of_modalities == 3 else "0_0") else min(experiments[multimodal_name]) for i in df_metric["setting"]]
                # Repeat the line with setting 0-0-0 for each modality

                for modality in experiments[multimodal_name]:
                    df_zero_mod = df_zero.copy()
                    df_zero_mod["missing_modality"] = modality
                    df_metric = pd.concat([df_metric, df_zero_mod], ignore_index=True)

                mapper = {250: 25, 500: 50, 750: 75, 1000: 100, 2500: 25, 5000: 50, 7500: 75, 10000: 100, 0: 0, 25: 25, 50: 50, 75: 75, 100: 100}
                df_metric["missing_percentage"] = df_metric["setting"].apply(
                        lambda x: mapper[np.max([int(i) for i in x.split("-")])]
                )

                # Sort for clean plotting
                df_metric = df_metric.sort_values(["missing_modality", "missing_percentage"])

                # Define colors for each modality
                colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}  # blue, green, red

                df_best_modality = df_best_modality[df_best_modality['target'] == target]


                # Assuming df is your dataframe
                df = df_metric
                # --- Existing rules ---
                # Remove rows for WSI (missing_modality == 1) where percentage is 25 or 50
                df = df[~((df["missing_modality"] == 1) & (df["missing_percentage"].isin([25, 50])))]

                # Replace 0 -> 64 for WSI (missing_modality == 1)
                df.loc[(df["missing_modality"] == 1) & (df["missing_percentage"] == 0), "missing_percentage"] = 64

                # Replace 0 -> 5 for CT (missing_modality == 2)
                df.loc[(df["missing_modality"] == 2) & (df["missing_percentage"] == 0), "missing_percentage"] = 5

                # --- New rule ---
                # For modality_WSI_missing% column: replace all 0 with 64
                if 1 in df["missing_modality"].unique():
                    df.loc[df["modality_WSI_missing%"] == 0, "modality_WSI_missing%"] = 64
                if 2 in df["missing_modality"].unique():
                    df.loc[df["modality_CT_missing%"] == 0, "modality_CT_missing%"] = 5

                df_metric = df


                num_of_modalities = len(mod_list)
                # Create Plotly figure
                plot=True
                if plot:
                    fig = go.Figure()

                    for modality, plus in zip(sorted(df_metric["missing_modality"].unique()), [0, 0, 0]):
                        for model in models_multimodal:
                            subset = df_metric[(df_metric["missing_modality"] == modality) & (df_metric["model"] == model)]
                            import plotly.colors as pc
                            n = 5   # e.g., 5 or 10
                            sem = subset["std"] / np.sqrt(n)
                            fig.add_trace(go.Scatter(
                                    x=subset["missing_percentage"],
                                    y=subset["mean"],
                                    error_y=dict(
                                            type="data",
                                            array=sem,
                                            visible=True,
                                            color=colors[modality],
                                            thickness=1.2,
                                            width=2
                                    ),
                                    mode="markers+lines",
                                    marker=dict(size=10, color=colors[modality], line=dict(width=1, color="black"), symbol=symbols[model]),  # Set shape here
                                    name=f"{modalityID_to_name[modality]}",
                                    line=dict(
                                            width=2.5,
                                            color=colors[modality],
                                            dash="solid" if model not in  ('CrossAttentionMissingModality') else "dash"
                                    ),
                            ))
                            """"""
                    # Customize layout
                    fig.update_layout(
                        xaxis=dict(
                            title="<b>% Missing</b>",
                            tickmode="array",
                            tickvals=[0, 25, 50, 75, 100],
                            showgrid=True,
                            gridcolor="lightgray",
                            title_font=dict(size=28, family="Arial", color="black"),
                            tickfont=dict(size=22, family="Arial")
                        ),
                        yaxis=dict(
                            title=f"<b>{name_m} (Mean ± SE)</b>",
                            showgrid=True,
                            gridcolor="lightgray",
                            title_font=dict(size=28, family="Arial", color="black"),
                            tickfont=dict(size=22, family="Arial")
                        ),

                        plot_bgcolor="white",

                        legend=dict(
                            title="<b>Masked Modality</b>",
                            orientation="h",
                            yanchor="bottom",
                            y=1.10,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=22, family="Arial", color="black"),
                            title_font=dict(size=24, family="Arial", color="black")
                        ),

                        # 🔥 Global font for anything not overridden
                        font=dict(size=24, family="Arial", color="black"),

                        width=950,
                        height=600,

                        margin=dict(l=80, r=30, t=90, b=80)
                    )
                    # Save the figure
                    fig.write_image(os.path.join(multimodal, f"{target}-{freeze}_{m}_missing_modality_performance_plot_test.png"), scale=2)
                    print(f"Saved missing_modality_performance_plot_test.png for {multimodal_name} {freeze} {target}")
    df_all_best = pd.DataFrame(modality_results_best)

    multimodal_overall = os.path.join(output_path, f"multimodal_overall")
    os.makedirs(multimodal_overall, exist_ok=True)
    df_all_best.to_csv(os.path.join(multimodal_overall, f"multimodal_{experiments_mm_name[multimodal_name]}{m}_{freeze}_test.csv"), index=False)
    print(f"Saved multimodal_{experiments_mm_name[multimodal_name]}_{freeze}_test.csv")
