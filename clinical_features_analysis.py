import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. Load data
# ------------------------------
file_path = "/Users/filruff/Desktop/PHD/PROGETTI/Multimodal_MARIA_AIDA/data/tabular/survival/AIDA/tabular/clinical_data_III_stage.csv"
df = pd.read_csv(file_path, header=0)

# ------------------------------
# 2. Basic cleaning & Outcome Splitting
# ------------------------------
df.columns = df.columns.str.replace(' ', '_')

# Define Outcome Column
outcome_col = "status"


# Create the Grouping Column
def classify_outcome(status):
    # Adjust these strings if your CSV uses lowercase or different spacing
    if status in ['AWD', 'NED']:
        return 'Survivor'
    elif status in ['DOC', 'DOD']:
        return 'Non-Survivor'
    else:
        return np.nan  # Exclude unknown status


df['Group'] = df[outcome_col].apply(classify_outcome)

# Filter out patients with unknown outcome for the analysis
df_analysis = df.dropna(subset=['Group']).copy()

# Count per group
group_counts = df_analysis['Group'].value_counts()
n_surv = group_counts.get('Survivor', 0)
n_dead = group_counts.get('Non-Survivor', 0)

print(f"Total Analysis Cohort: {len(df_analysis)}")
print(f"Survivors (NED/AWD): {n_surv}")
print(f"Non-Survivors (DOC/DOD): {n_dead}")

# ------------------------------
# 3. Define Variables of Interest
# ------------------------------
# Using your exact variable mappings
variables = {
        "Age (years)"               : ("eta", "continuous"),
        "Sex"                       : ("Sesso", "categorical"),
        "Stage at diagnosis"        : ("Stadio_alla_diagnosi", "categorical"),
        "Histology"                 : ("Diagnosi", "categorical"),
        "Smoking habitus"           : ("Smoking_habitus", "categorical"),
        "Adjuvant immunotherapy"    : ("Immuno_post_radiochemio_(adj)", "categorical"),
        "PD-L1 (%)"                 : ("PDL1(%)", "continuous"),
        "Radiation dose (Gy)"       : ("Dose_erogata_[Gy]", "continuous"),
        "Number of fractions"       : ("Numero_frazioni", "continuous"),
        "Esophageal toxicity"       : ("Tox_esofagea", "categorical"),
        "Pulmonary toxicity"        : ("Tox_polmonare", "categorical"),
        "Comorbidity 1"             : ("Comorbidità1", "categorical"),
        "Comorbidity 2"             : ("Comorbidità2", "categorical"),
        "Comorbidity 3"             : ("Comorbidità3", "categorical"),
        "EGFR mutation"             : ("EGFR", "categorical"),
        "ALK rearrangement"         : ("ALK", "categorical"),
        "MET alteration"            : ("MET", "categorical"),
        "Technique"                 : ("Tecnica", "categorical"),
        "Induction chemotherapy"    : ("Tipo_CT_induzione", "categorical"),
        "Concomitant chemotherapy"  : ("Schema_CT_concomitante", "categorical"),
        "Weight (kg)"               : ("Peso_(kg)", "continuous"),
        "Height (cm)"               : ("Altezza_(cm)", "continuous"),
        "NRS"                       : ("NRS", "continuous"),
        "Cigarettes per day"        : ("Sig/die", "continuous"),
        "ECOG PS"                   : ("PS_ECOG_basale", "categorical"),
        "Radiation therapy duration": ("Durata_RT", "continuous"),
        "Suspension days"           : ("Giorni_di_Sospensione", "continuous"),
        "Hemoglobin toxicity"       : ("Tox_Hb", "categorical"),
        "Neutrophil toxicity"       : ("Tox_Neu", "categorical"),
        "Platelet toxicity"         : ("Tox_PLT", "categorical"),
        "PFS"                       : ("PFS", "categorical"),
        "metastasis"                : ("metastasi", "categorical"),
        "local_recurrence"         : ("recidiva", "categorical"),
}


# ------------------------------
# 4. Statistical Functions
# ------------------------------

def analyze_continuous(df, col, group_col='Group'):
    """Calculates Mean +/- SD per group and T-test p-value"""
    # Convert to numeric, handle errors
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Split
    g_surv = df[df[group_col] == 'Survivor'][col].dropna()
    g_dead = df[df[group_col] == 'Non-Survivor'][col].dropna()

    # Stats
    mean_overall = f"{df[col].mean():.1f} ± {df[col].std():.1f}" if not df[col].empty else "N/A"
    mean_surv = f"{g_surv.mean():.1f} ± {g_surv.std():.1f}" if not g_surv.empty else "N/A"
    mean_dead = f"{g_dead.mean():.1f} ± {g_dead.std():.1f}" if not g_dead.empty else "N/A"

    # T-test (ind)
    if len(g_surv) > 1 and len(g_dead) > 1:
        stat, p_val = stats.ttest_ind(g_surv, g_dead, equal_var=False)
    else:
        p_val = np.nan

    return mean_overall, mean_dead, mean_surv, p_val, None  # None = no sub-categories


def analyze_categorical(df, col, group_col='Group'):
    """Calculates Counts (%) per group and Chi-square p-value"""
    # Create contingency table
    ct = pd.crosstab(df[col], df[group_col])

    # Check if empty
    if ct.empty:
        return "N/A", "N/A", np.nan, []

    # Ensure both columns exist (in case one group has no data for this var)
    if 'Non-Survivor' not in ct.columns:
        ct['Non-Survivor'] = 0
    if 'Survivor' not in ct.columns:
        ct['Survivor'] = 0




    # Calculate Percentages
    # Note: Percentages are column-wise (within the group)
    n_dead_total = ct['Non-Survivor'].sum()
    n_surv_total = ct['Survivor'].sum()

    results = []
    for category in ct.index:
        c_dead = ct.at[category, 'Non-Survivor']
        c_surv = ct.at[category, 'Survivor']
        pct_overall = (c_dead + c_surv) / (n_dead_total + n_surv_total) * 100 if (n_dead_total + n_surv_total) > 0 else 0
        pct_dead = (c_dead / n_dead_total * 100) if n_dead_total > 0 else 0
        pct_surv = (c_surv / n_surv_total * 100) if n_surv_total > 0 else 0


        str_overall = f"{(c_dead + c_surv)} ({pct_overall:.2f}%)"
        str_dead = f"{c_dead} ({pct_dead:.1f}%)"
        str_surv = f"{c_surv} ({pct_surv:.1f}%)"
        results.append((category, str_overall, str_dead, str_surv))

    # Chi-square test
    # Chi2 requires counts, not percentages
    if ct.size > 0 and n_dead_total > 0 and n_surv_total > 0:
        chi2, p_val, dof, expected = stats.chi2_contingency(ct)
    else:
        p_val = np.nan

    return "", "", p_val, results


# ------------------------------
# 5. Run Analysis Loop
# ------------------------------

table_rows = []

print(f"{'Variable':<30} | {'Non-Survivor (n=' + str(n_dead) + ')':<25} | {'Survivor (n=' + str(n_surv) + ')':<25} | {'P-Value'}")
print("-" * 95)

for label, (col_name, data_type) in variables.items():
    if col_name not in df_analysis.columns:
        print(f"Skipping {label}: Column '{col_name}' not found.")
        continue

    if data_type == 'continuous':
        val_overall, val_dead, val_surv, p, _ = analyze_continuous(df_analysis, col_name)
        p_str = f"{p:.3f}" if pd.notna(p) and p >= 0.001 else ("<0.001" if pd.notna(p) else "N/A")


        print(f"{label:<30} | {val_overall:<25} | {val_dead:<25} | {val_surv:<25} | {p_str}")
        table_rows.append({
                "Variable"    : label, "Type": "Continuous", "Category": "", "Overall": val_overall,
                "Non-Survivor": val_dead, "Survivor": val_surv, "P-value": p_str
        })

    elif data_type == 'categorical':
        _, _, p, sub_cats = analyze_categorical(df_analysis, col_name)
        p_str = f"{p:.3f}" if pd.notna(p) and p >= 0.001 else ("<0.001" if pd.notna(p) else "N/A")

        # Print Main Header with P-value
        print(f"{label:<30} | {'':<25} | {'':<25} | {'':<25} | {p_str}")
        table_rows.append({
                "Variable"    : label, "Type": "Categorical Header", "Category": "", "Overall": "",
                "Non-Survivor": "", "Survivor": "", "P-value": p_str
        })

        # Print Sub-categories
        for cat, v_overall, v_dead, v_surv in sub_cats:
            print(f"  {str(cat):<28} | {v_overall:<25} | {v_dead:<25} | {v_surv:<25} |")
            table_rows.append({
                    "Variable"    : "", "Type": "Categorical Level", "Category": cat, "Overall": v_overall,
                    "Non-Survivor": v_dead, "Survivor": v_surv, "P-value": ""
            })

print("-" * 95)

# ------------------------------
# 6. Generate LaTeX Table
# ------------------------------
latex_lines = [
        "\\begin{table*}[!ht]",
        "\\centering",
        "\\small",
        f"\\caption{{Baseline characteristics stratified by survival status (n={len(df_analysis)}).}}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        f"Variable & & Non-Survivor (n={n_dead}) & Survivor (n={n_surv}) & P-value \\\\",
        "\\midrule"
]

for row in table_rows:
    if row['Type'] == 'Continuous':
        latex_lines.append(f"{row['Variable']} & {row['Overall']} & {row['Non-Survivor']} & {row['Survivor']} & {row['P-value']} \\\\")
    elif row['Type'] == 'Categorical Header':
        latex_lines.append(f"{row['Variable']} & & & & {row['P-value']} \\\\")
    elif row['Type'] == 'Categorical Level':
        # Escape special chars in category names for LaTeX
        cat_clean = str(row['Category']).replace('%', '\\%').replace('_', '\\_')
        latex_lines.append(f"\\hspace{{3mm}} {cat_clean} & {row['Overall']} & {row['Non-Survivor']} & {row['Survivor']} & \\\\")

latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:baseline_characteristics}",
        "\\end{table*}"
])

with open("stratified_table1.tex", "w") as f:
    f.write("\n".join(latex_lines))

print("\n✅ Stratified LaTeX table saved to 'stratified_table1.tex'")