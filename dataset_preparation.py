import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def compute_cardinality_n_missing(column, remove_modalities=False):
    if remove_modalities:
        column = column.str.replace("\A\d_\d_\d_", "", regex=True)
    print("Computing cardinality of missing values in %s" % column.name)
    print(column.value_counts())
    print(f"{len(column)} total patients")
    print(f"{column.isna().sum()} missing values")



def CT_embeddings(output_path: str) -> None:
    emb_dir = "data/tabular/survival/AIDA/imaging/embeddings_2"
    emb_dir = Path(emb_dir)


    data_path = "data/tabular/survival/AIDA/tabular/clinical_data_III_stage.csv"
    data_clinical = pd.read_csv(data_path, index_col=0)



    data_path = "data/tabular/survival/AIDA/cross_validation/all_data.xlsx"
    data = pd.read_excel(data_path, index_col=0)


    for clinical_patient_id in data_clinical.index:
        data.loc[data.index == int(clinical_patient_id), 'OS'] = data_clinical.loc[data_clinical.index == int(clinical_patient_id), "OS"]
        data.loc[data.index == int(clinical_patient_id), 'PFS'] = data_clinical.loc[data_clinical.index == int(clinical_patient_id), "PFS"]


    for emb_file in list(emb_dir.rglob("*.npy")):
        patient_id = emb_file.stem.split('_')[-1]
        data.loc[data.index == int(patient_id), 'CT_embedding_path'] = str(emb_file)


    # DROP ROWS WITHOUT EMBEDDING
    data.to_excel(os.path.join(output_path, "CT_embeddings_2.xlsx"))



def WSI_embeddings(output_path: str) -> None:
    emb_dir = "data/tabular/survival/AIDA/wsi/embeddings"
    emb_dir = Path(emb_dir)

    data_path = "data/tabular/survival/AIDA/tabular/clinical_data_III_stage.csv"
    data_clinical = pd.read_csv(data_path, index_col=0)


    data_path = "data/tabular/survival/AIDA/cross_validation/all_data.xlsx"
    data = pd.read_excel(data_path, index_col=0)

    for clinical_patient_id in data_clinical.index:
        data.loc[data.index == int(clinical_patient_id), 'OS'] = data_clinical.loc[data_clinical.index == int(clinical_patient_id), "OS"]
        data.loc[data.index == int(clinical_patient_id), 'PFS'] = data_clinical.loc[data_clinical.index == int(clinical_patient_id), "PFS"]


    for emb_file in list(emb_dir.rglob("*.npy")):
        patient_id = emb_file.stem.split('_')[-1]
        data.loc[data.index == int(patient_id), 'WSI_embedding_path'] = str(emb_file)

    # DROP ROWS WITHOUT EMBEDDING
    data.to_excel(os.path.join(output_path, "WSI_embeddings.xlsx"))












def clinical_data_III_stage(output_path: str) -> None:
    data_path = "./data/RACCOLTA_DATI_III_UCBM.xlsx"
    data = pd.read_excel(data_path, header=[0, 1], sheet_name="DATI CLINICI", na_values=("NON TROVATO", "0 = non eseguito", "0= non eseguito"))
    data = data.drop(columns="IV STADIO")
    data.columns = data.columns.droplevel(0)
    data.set_index("ID paziente", inplace=True)

    labels_path = './data/cross_validation/all_data.xlsx'
    labels = pd.read_excel(labels_path, index_col=0)

    data = data.loc[labels.index]

    data["Data di nascita"] = pd.to_datetime(data["Data di nascita"], format="%d/%m/%Y")
    data["Data diagnosi"] = pd.to_datetime(data["Data diagnosi"], format="%d/%m/%Y")
    data["eta"] = (data["Data diagnosi"] - data["Data di nascita"]).dt.days // 365

    data["Data inizio RT"] = pd.to_datetime(data["Data inizio RT"], format="%d/%m/%Y")
    data["Data fine RT"] = pd.to_datetime(data["Data fine RT"], format="%d/%m/%Y")
    data["Durata RT"] = (data["Data fine RT"] - data["Data inizio RT"]).dt.days

    categorical_features = ["Sesso", "Stadio alla diagnosi", "Smoking habitus", "Familiarità", "Comorbidità1", "Comorbidità2", "Comorbidità3", "Diagnosi", "EGFR", "ALK", "MET", "Tecnica", "CT induzione", "CT concomitante",  "Immuno post radiochemio (adj)"]
    # "Schema CT concomitante",  todo da chiedere a claudia
    # features removed for too few values: "BRAF",  "NTRK", "RET", "ROS1"

    numerical_features = ["eta", "Peso (kg)", "Altezza (cm)", "cT", "cN", "NRS", "Sig/die", "PS ECOG basale", "PDL1(%)",
                          "Durata RT", "CTV iniziale", "Dose erogata [Gy]", "Numero frazioni", "Giorni di Sospensione",
                          "Numero cicli induzione", "Tox esofagea", "Tox polmonare", "Tox Hb", "Tox Neu", "Tox PLT"]

    data = data[categorical_features + numerical_features]
    data = pd.concat([data, labels], axis=1)

    # Aggiustamenti alle features
    data.loc[:, "Stadio alla diagnosi"] = data["Stadio alla diagnosi"].replace({"III A": "IIIA", "III B": "IIIB", "III C": "IIIC", "II B": "IIB"})
    data.loc[:, "Diagnosi"] = data["Diagnosi"].str.lower().str.strip().str.replace("4= adenosquamoso", "6= altro")
    data.loc[:, "EGFR"] = data["EGFR"].apply(lambda x: "2 = positivo" if pd.notna(x) and x != "1 = negativo" else x)
    data.loc[:, "PDL1(%)"] = data["PDL1(%)"].replace("<1%", "0.009").astype(float)
    for col in ["Tox esofagea", "Tox polmonare", "Tox Hb", "Tox Neu", "Tox PLT"]:
        data.loc[:, col] = data.loc[:, col].replace("G", "", regex=True).astype(float)
    data.loc[:, "cT"] = data["cT"].replace({"1c": "1", "2b": "2", "2a": "2", "x": np.nan})
    data.loc[:, "CT induzione"] = data["CT induzione"].fillna("0 = NO")
    data.loc[:, "CT concomitante"] = data["CT concomitante"].fillna("NO")
    data.loc[:, "Immuno post radiochemio (adj)"] = data["Immuno post radiochemio (adj)"].fillna("NO")

    # Aggiustamento alle labels
    data.loc[:, "OS"] = data.status.apply(lambda x: "uncensored" if x in ("DOD", "DOC") else "censored")
    data.loc[:, "PFS"] = data.PFSevent.apply(lambda x: "uncensored" if x != "censored" else "censored")
    data.loc[data.status == "DOC", "PFS"] = "censored"

    # Salvataggio del file
    save_path = os.path.join(output_path, "tabular", "survival", "AIDA")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, "clinical_data_III_stage.csv")
    data.to_csv(file_path, index=True)

    cat_features_info = pd.DataFrame(categorical_features, columns=["feature"]).assign(type="category")
    num_features_info = pd.DataFrame(numerical_features, columns=["feature"]).assign(type="float")
    features_info = pd.concat([cat_features_info, num_features_info], axis=0)
    features_info_path = os.path.join(save_path, "features_info.csv")
    features_info.to_csv(features_info_path, index=False)

    for col in categorical_features:
        print(data[col].value_counts(), data[col].dtype)
    for col in ["OS", "PFS"]:
        print(data[col].value_counts(), data[col].dtype)

def split_train_val_test(data_path, output_path: str, val_size: float = 0.1, test_size: float = 0.2, random_state: int = 42) -> None:

    def map_event_pfs(row):
        if (str(row["PFSevent"]).lower() == "censored") or (str(row["status"]).upper() == "DOC") or (str(row["status"]).upper() == "NED") or (str(row["status"]).upper() == "AWD"):
            return "censored"
        else:
            return "uncensored"


    import os
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, train_test_split

    # === CONFIG ===
    columns_to_stratify = ["CT", "WSI"]
    possible_labels = [("PFSevent", 'days2PFS'), ("status", 'days2OS'),
                          ("recidiva", 'days2recidiva'), ("metastasi", 'days2metastasi')
                        ]

    # === LOAD DATA ===
    data = pd.read_excel(data_path)

    data["PFSevent"] = data.apply(map_event_pfs, axis=1)

    def assign_time_quartile(df, time_col="days2PFS"):
        # Calcolo quartili solo all’interno di ciascun gruppo
        df["quartile"] = pd.qcut(df[time_col].rank(method="first"), q=2, labels=["Q1", "Q2"])
        return df


    data = data.rename_axis("idx", axis=0).reset_index().set_index("ID paziente")
    data.rename_axis("ID", axis=0, inplace=True)

    n_cols = data.shape[1]

    # === LOOP OVER TASKS ===
    for (label, time_col) in possible_labels:

        data = data.groupby(label, group_keys=False).apply(assign_time_quartile, time_col=time_col)



        print(f"\nProcessing task: {label}")
        task_path = os.path.join(output_path, time_col + '_task')
        os.makedirs(task_path, exist_ok=True)

        # ---- label selection logic ----
        label_to_stratify = ["quartile"]

        # ---- build stratification column ----
        data['stratify_col'] = data[columns_to_stratify + label_to_stratify].apply(
                lambda row: '_'.join(row.astype(str)), axis=1
        )

        data = data.loc[data["stratify_col"] != "_".join(["0"] * len(columns_to_stratify + label_to_stratify))]

        print(np.unique(data["stratify_col"], return_counts=True))

        # ============================================================
        # 4. GESTIONE DELLE CLASSI SINGLETON
        # ============================================================
        counts = data["stratify_col"].value_counts()
        singleton_classes = counts[(counts == 1) | (counts == 2)].index
        singletons = data[data["stratify_col"].isin(singleton_classes)]
        data_no_singletons = data[~data["stratify_col"].isin(singleton_classes)]




        # ---- cross-validation folds ----
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_index, test_index) in enumerate(kf.split(data_no_singletons, data_no_singletons['stratify_col'])):
            print(f"  Fold {fold}")

            train_data = data_no_singletons.iloc[train_index]
            test_data = data_no_singletons.iloc[test_index]

            # ---- split train into train/val ----
            train_split, val_split = train_test_split(
                    train_data,
                    test_size=0.15,  # you can adjust (e.g., 0.2)
                    stratify=train_data['stratify_col'],
                    random_state=fold
            )

            # ---- randomly assign singletons to train or val ----
            for _, row in singletons.iterrows():
                if random.random() < 0.8:  # 80% chance → train
                    train_split = pd.concat([train_split, row.to_frame().T], ignore_index=False)
                else:  # 20% chance → val
                    val_split = pd.concat([val_split, row.to_frame().T], ignore_index=False)

            # ---- drop helper column ----
            for df_ in [train_split, val_split, test_data]:
                df_.drop(columns=["stratify_col", "quartile"], inplace=True)
                df_.rename_axis("ID", axis=0, inplace=True)
            # ---- save fold ----
            fold_dir = os.path.join(task_path, f"{fold}_0")
            os.makedirs(fold_dir, exist_ok=True)
            train_split.to_csv(os.path.join(fold_dir, "train.csv"))
            val_split.to_csv(os.path.join(fold_dir, "val.csv"))
            test_data.to_csv(os.path.join(fold_dir, "test.csv"))

    print("\n✅ All folds successfully created with train/val/test splits.")


def prepare_cv(output_path: str) -> None:
    path = "./data/cross_validation/folds"
    sets = ["train", "test"]

    data_path = os.path.join(output_path, "tabular", "survival", "AIDA", "tabular", "clinical_data_III_stage.csv")




    data = pd.read_csv(data_path).rename_axis("idx", axis=0).reset_index().set_index("ID paziente")


    for i in range(5):
        for set in sets:
            fold_path = os.path.join(path, f"fold_{i}", set+".csv")
            fold_data = pd.read_csv(fold_path)
            fold_data.loc[:, "idx"] = data.loc[fold_data["ID paziente"], "idx"].values
            os.makedirs(os.path.join(output_path, "tabular", "survival", "AIDA", "cross_validation", f"{i}_0"), exist_ok=True)
            fold_data.to_csv(os.path.join(output_path, "tabular", "survival", "AIDA", "cross_validation", f"{i}_0", f"{set}.csv"), index=False)


if __name__ == "__main__":
    output_path = "./data"
    #clinical_data_III_stage(output_path)
    CT_embeddings("data/tabular/survival/AIDA/imaging")
    #WSI_embeddings("data/tabular/survival/AIDA/wsi")
    #split_train_val_test(data_path="data/tabular/survival/AIDA/cross_validation/all_data.xlsx", output_path="data/tabular/survival/AIDA/cross_validation")
    #prepare_cv(output_path)