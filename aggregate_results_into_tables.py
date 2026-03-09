import os
import pandas as pd

# Directory principale dove ci sono le sottocartelle
base_dir = "outputs/fold_specific"
output_path = "tests/fold_specific"


# Lista per accumulare i dataframe
all_results = []

# mapping nomi modello
model_map = {
    "RSF": "RSF",
    "CPH": "CPH",
    "survivalGradientBoosting": "GB"  # abbreviato
}



for modality in os.listdir(base_dir):
    modality_subpath = os.path.join(base_dir, modality)
    all_results = []
    if not os.path.isdir(modality_subpath) or modality.startswith("."):
        continue

    exps = os.listdir(modality_subpath)
    for exp in exps:
        if exp.startswith(".") or 'hyp' in exp:
            continue
        path_exp = os.path.join(modality_subpath, exp)
        name_exp = os.listdir(path_exp)[0]
        results_path = os.path.join(path_exp, name_exp,"results")
        if not os.path.exists(results_path):
            print(f"⚠️  Percorso non trovato: {results_path}")
            continue
        try:
            results_path = os.path.join(results_path, os.listdir(results_path)[0])
        except IndexError:
            print('PATH EMPTY?', results_path, os.listdir(results_path))
            continue
        csv_path = os.path.join(results_path, "unbalanced/test/averages_classes_average_performance.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            df.columns = [f'{i.replace(".1", "")}_{x}' if x==x else f'{i}' for i, x in zip(df.iloc[0].index, df.iloc[0].to_list())]
            df = df.drop(df.index[0])

            # parsing nome cartella
            model_name = exp.split("_noimputation" if "_noimputation" in exp else "_knn")[0]
            parts = exp.split("_")
            raw_model = model_name
            model = model_map.get(raw_model, raw_model)
            df = df.iloc[0, [1,2]].to_frame().T
            try:
                start = parts.index("AIDA")
                end = parts.index("time2event")
                task = "_".join(parts[start+1:end-1])
            except ValueError:
                task = exp

            df.loc["test","model"] = model
            df.loc["test", "task"] = task
            all_results.append(df)
            metrics_names = df.columns.tolist()[:-2]


    # concatena tutti i risultati
    if not all_results:
        print("❌ Nessun risultato trovato")
        continue

    df = pd.concat(all_results, ignore_index=True)
    df[metrics_names] = df[metrics_names].astype(float)
    df.groupby(["model", "task"]).mean()
    output_path_mod = os.path.join(output_path, modality)
    os.makedirs(output_path_mod, exist_ok=True)
    df.sort_values(by=metrics_names[0], ascending=False).to_csv(os.path.join(output_path_mod,f"{metrics_names[0]}_all_results.csv"), index=False)
    best_m_index = df.sort_values(by=metrics_names[0], ascending=False).groupby("task").first().reset_index().iloc[:, [0,1,-1]].sort_values(by=f"{metrics_names[0]}", ascending=False)

    best_m_index.to_csv(os.path.join(output_path_mod,f"best_{metrics_names[0]}_by_task.csv"), index=False)

print("May the Force be with you!")