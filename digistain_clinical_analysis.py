import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr, mannwhitneyu, kruskal
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

# Optional survival analysis
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("lifelines not installed: Kaplan-Meier plots will be skipped.")
    print("Install with: pip install lifelines")


# =========================
# 1. SETTINGS
# =========================
CSV_FILE = "/Users/nana/Desktop/clinical_metadata(Sheet1).csv"
OUTPUT_DIR = "digistain_clinical_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

DI_COLS = [
    "DIv1 - AmidePhosphateRatios",
    "DIv2 - DigistainIndices",
    "DIv3 - DigistainIndicesTrimmed",
    "DIv4 - NormalisedAmideHeights",
    "DIv5 - NormalisedPhosphateHeights",
    "DIv6 - UnnormalisedAmidePhosphateRatios"
]

ORDINAL_COLS = [
    "Grade",
    "Size",
    "DFI",
    "Survival "
]

CATEGORICAL_COLS = [
    "Stage",
    "ER Status",
    "PR status",
    "HER2 status",
    "Triple Negative",
    "Recurrence",
    "Dead or Alive",
    "Vascular Invasion",
    "NPI (3 groups)",
    "Tumour Type (groups)"
]

OPTIONAL_NUMERIC_COLS = [
    "Tubule formation",
    "Pleomorphism",
    "Mitosis",
    "<50%",
    "?50%"
]


# =========================
# 2. LOAD + CLEAN
# =========================
def clean_dataframe(df):
    df = df.copy()

    # Strip spaces from column names
    df.columns = [c.strip() for c in df.columns]

    # Fix known awkward column name
    if "Survival" not in df.columns and "Survival " in df.columns:
        df = df.rename(columns={"Survival ": "Survival"})

    # Convert DI columns to numeric
    for col in DI_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert numeric clinical columns
    numeric_candidates = ["Age at diagnosis", "Size", "Grade", "DFI", "Survival"] + OPTIONAL_NUMERIC_COLS
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardise common categorical values
    def clean_text(x):
        if pd.isna(x):
            return np.nan
        return str(x).strip()

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Binary encodings for analysis
    if "Recurrence" in df.columns:
        df["Recurrence_binary"] = df["Recurrence"].map({
            "Yes": 1, "No": 0
        })

    if "Triple Negative" in df.columns:
        df["TripleNegative_binary"] = df["Triple Negative"].map({
            "Yes": 1, "No": 0
        })

    if "ER Status" in df.columns:
        df["ER_binary"] = df["ER Status"].map({
            "Positive": 1, "Negative": 0
        })

    if "PR status" in df.columns:
        df["PR_binary"] = df["PR status"].map({
            "Positive": 1, "Negative": 0
        })

    if "HER2 status" in df.columns:
        df["HER2_binary"] = df["HER2 status"].map({
            "Positive": 1, "Negative": 0
        })

    if "Dead or Alive" in df.columns:
        def encode_event(x):
            if pd.isna(x):
                return np.nan
            x = str(x).strip().lower()
            if x == "alive":
                return 0
            return 1
        df["Death_event"] = df["Dead or Alive"].apply(encode_event)

    return df


df = pd.read_csv(CSV_FILE)
df = clean_dataframe(df)

# Keep rows with at least one DI value
df = df.dropna(subset=DI_COLS, how="all").copy()

print(f"Loaded dataset with {len(df)} rows")
print("Columns:")
print(df.columns.tolist())


# =========================
# 3. HELPER FUNCTIONS
# =========================
def safe_spearman(x, y):
    mask = (~pd.isna(x)) & (~pd.isna(y))
    if mask.sum() < 3:
        return np.nan, np.nan, mask.sum()
    r, p = spearmanr(x[mask], y[mask])
    return r, p, mask.sum()

def binary_group_test(df, di_col, group_col, positive_values=None):
    """
    Mann-Whitney U test for 2-group comparisons.
    """
    tmp = df[[di_col, group_col]].dropna().copy()
    if len(tmp) < 3:
        return np.nan, np.nan, 0, 0

    groups = tmp[group_col].unique()
    if len(groups) != 2:
        return np.nan, np.nan, 0, 0

    g1, g2 = groups[0], groups[1]
    x1 = tmp[tmp[group_col] == g1][di_col]
    x2 = tmp[tmp[group_col] == g2][di_col]

    if len(x1) < 2 or len(x2) < 2:
        return np.nan, np.nan, len(x1), len(x2)

    stat, p = mannwhitneyu(x1, x2, alternative="two-sided")
    return stat, p, len(x1), len(x2)

def multi_group_test(df, di_col, group_col):
    """
    Kruskal-Wallis test for >2 groups.
    """
    tmp = df[[di_col, group_col]].dropna().copy()
    if len(tmp) < 3:
        return np.nan, np.nan, 0

    grouped = [g[di_col].values for _, g in tmp.groupby(group_col)]
    if len(grouped) < 2:
        return np.nan, np.nan, len(grouped)

    if any(len(g) < 2 for g in grouped):
        return np.nan, np.nan, len(grouped)

    stat, p = kruskal(*grouped)
    return stat, p, len(grouped)

def plot_boxplot(df, xcol, ycol, filename, title=None, order=None):
    tmp = df[[xcol, ycol]].dropna().copy()
    if len(tmp) == 0:
        return
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=tmp, x=xcol, y=ycol, order=order)
    sns.stripplot(data=tmp, x=xcol, y=ycol, order=order, alpha=0.55, size=5)
    plt.title(title if title else f"{ycol} vs {xcol}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

def plot_violin(df, xcol, ycol, filename, title=None, order=None):
    tmp = df[[xcol, ycol]].dropna().copy()
    if len(tmp) == 0:
        return
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=tmp, x=xcol, y=ycol, order=order, inner="box")
    plt.title(title if title else f"{ycol} vs {xcol}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

def plot_scatter_with_trend(df, xcol, ycol, filename, title=None):
    tmp = df[[xcol, ycol]].dropna().copy()
    if len(tmp) < 3:
        return
    plt.figure(figsize=(8, 6))
    sns.regplot(data=tmp, x=xcol, y=ycol, scatter_kws={"alpha": 0.7})
    plt.title(title if title else f"{ycol} vs {xcol}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

def make_heatmap(df, cols, filename, title):
    tmp = df[cols].copy()
    corr = tmp.corr(method="spearman", numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

def kaplan_meier_by_di(df, di_col):
    if not LIFELINES_AVAILABLE:
        return None

    if "Survival" not in df.columns or "Death_event" not in df.columns:
        return None

    tmp = df[[di_col, "Survival", "Death_event"]].dropna().copy()
    if len(tmp) < 10:
        return None

    median_di = tmp[di_col].median()
    tmp["DI_group"] = np.where(tmp[di_col] >= median_di, "High DI", "Low DI")

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    for group in ["Low DI", "High DI"]:
        group_df = tmp[tmp["DI_group"] == group]
        kmf.fit(group_df["Survival"], event_observed=group_df["Death_event"], label=group)
        kmf.plot_survival_function(ci_show=True)

    low = tmp[tmp["DI_group"] == "Low DI"]
    high = tmp[tmp["DI_group"] == "High DI"]

    result = logrank_test(
        low["Survival"], high["Survival"],
        event_observed_A=low["Death_event"],
        event_observed_B=high["Death_event"]
    )

    plt.title(f"Kaplan-Meier Survival: {di_col}\nlog-rank p = {result.p_value:.4g}")
    plt.xlabel("Survival time")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    fname = f"KM_{safe_filename(di_col)}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
    plt.close()

    return {
        "DI": di_col,
        "median_split": median_di,
        "logrank_p": result.p_value,
        "n": len(tmp)
    }

def safe_filename(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s)

def roc_for_recurrence(df, di_cols):
    if "Recurrence_binary" not in df.columns:
        return None

    tmp = df[di_cols + ["Recurrence_binary"]].copy()
    tmp = tmp.dropna(subset=["Recurrence_binary"])
    if tmp["Recurrence_binary"].nunique() < 2:
        return None
    if len(tmp) < 10:
        return None

    X = tmp[di_cols]
    y = tmp["Recurrence_binary"].astype(int)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Recurrence prediction using all DI variants")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ROC_recurrence_all_DI.png"), dpi=300)
    plt.close()

    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    return {
        "auc": roc_auc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "n": len(tmp)
    }


# =========================
# 4. SUMMARY PLOTS
# =========================
# Histograms for all DIs
for di in DI_COLS:
    if di not in df.columns:
        continue
    plt.figure(figsize=(8, 6))
    sns.histplot(df[di].dropna(), kde=True)
    plt.title(f"Distribution of {di}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{safe_filename(di)}.png"), dpi=300)
    plt.close()

# Correlation heatmap
numeric_cols_for_heatmap = [c for c in DI_COLS if c in df.columns]
for c in ["Age at diagnosis", "Size", "Grade", "DFI", "Survival", "Tubule formation", "Pleomorphism", "Mitosis"]:
    if c in df.columns:
        numeric_cols_for_heatmap.append(c)

make_heatmap(
    df,
    numeric_cols_for_heatmap,
    filename="spearman_heatmap.png",
    title="Spearman correlation heatmap: DI metrics and clinical variables"
)


# =========================
# 5. SPEARMAN CORRELATIONS
# =========================
correlation_results = []

candidate_corr_targets = ["Grade", "Size", "Age at diagnosis", "DFI", "Survival", "Tubule formation", "Pleomorphism", "Mitosis"]
candidate_corr_targets = [c for c in candidate_corr_targets if c in df.columns]

for di in DI_COLS:
    if di not in df.columns:
        continue
    for target in candidate_corr_targets:
        r, p, n = safe_spearman(df[di], df[target])
        correlation_results.append({
            "DI": di,
            "Clinical_variable": target,
            "Spearman_r": r,
            "p_value": p,
            "n": n
        })

corr_df = pd.DataFrame(correlation_results).sort_values(["Clinical_variable", "p_value"], na_position="last")
corr_df.to_csv(os.path.join(OUTPUT_DIR, "spearman_correlations.csv"), index=False)


# =========================
# 6. GROUP COMPARISONS
# =========================
group_results = []

binary_targets = [
    "ER Status",
    "PR status",
    "HER2 status",
    "Triple Negative",
    "Recurrence",
    "Dead or Alive",
    "Vascular Invasion"
]
binary_targets = [c for c in binary_targets if c in df.columns]

for di in DI_COLS:
    if di not in df.columns:
        continue

    for target in binary_targets:
        stat, p, n1, n2 = binary_group_test(df, di, target)
        group_results.append({
            "DI": di,
            "Clinical_variable": target,
            "Test": "Mann-Whitney U",
            "Statistic": stat,
            "p_value": p,
            "Group1_n": n1,
            "Group2_n": n2
        })

multi_targets = ["Stage", "NPI (3 groups)", "Tumour Type (groups)", "Grade"]
multi_targets = [c for c in multi_targets if c in df.columns]

for di in DI_COLS:
    if di not in df.columns:
        continue

    for target in multi_targets:
        stat, p, ngroups = multi_group_test(df, di, target)
        group_results.append({
            "DI": di,
            "Clinical_variable": target,
            "Test": "Kruskal-Wallis",
            "Statistic": stat,
            "p_value": p,
            "Number_of_groups": ngroups
        })

group_df = pd.DataFrame(group_results).sort_values("p_value", na_position="last")
group_df.to_csv(os.path.join(OUTPUT_DIR, "group_comparisons.csv"), index=False)


# =========================
# 7. KEY VISUALISATIONS
# =========================
# Grade boxplots
if "Grade" in df.columns:
    for di in DI_COLS:
        if di in df.columns:
            plot_boxplot(
                df, "Grade", di,
                filename=f"boxplot_grade_{safe_filename(di)}.png",
                title=f"{di} vs tumour grade",
                order=sorted(df["Grade"].dropna().unique())
            )

# Stage boxplots
if "Stage" in df.columns:
    for di in DI_COLS:
        if di in df.columns:
            stage_order = sorted(df["Stage"].dropna().unique(), key=lambda x: str(x))
            plot_boxplot(
                df, "Stage", di,
                filename=f"boxplot_stage_{safe_filename(di)}.png",
                title=f"{di} vs stage",
                order=stage_order
            )

# Recurrence
if "Recurrence" in df.columns:
    for di in DI_COLS:
        if di in df.columns:
            plot_violin(
                df, "Recurrence", di,
                filename=f"violin_recurrence_{safe_filename(di)}.png",
                title=f"{di} vs recurrence",
                order=["No", "Yes"] if set(df["Recurrence"].dropna().unique()).issubset({"No", "Yes"}) else None
            )

# ER / PR / HER2 / Triple Negative
for biomarker in ["ER Status", "PR status", "HER2 status", "Triple Negative"]:
    if biomarker in df.columns:
        for di in DI_COLS:
            if di in df.columns:
                plot_boxplot(
                    df, biomarker, di,
                    filename=f"boxplot_{safe_filename(biomarker)}_{safe_filename(di)}.png",
                    title=f"{di} vs {biomarker}"
                )

# Scatter plots against continuous/ordinal variables
for xvar in ["Grade", "Mitosis", "Pleomorphism", "Tubule formation", "DFI", "Survival", "Size"]:
    if xvar in df.columns:
        for di in DI_COLS:
            if di in df.columns:
                plot_scatter_with_trend(
                    df, xvar, di,
                    filename=f"scatter_{safe_filename(xvar)}_{safe_filename(di)}.png",
                    title=f"{di} vs {xvar}"
                )


# =========================
# 8. KAPLAN-MEIER SURVIVAL
# =========================
km_results = []
if LIFELINES_AVAILABLE:
    for di in DI_COLS:
        if di in df.columns:
            res = kaplan_meier_by_di(df, di)
            if res is not None:
                km_results.append(res)

if len(km_results) > 0:
    km_df = pd.DataFrame(km_results).sort_values("logrank_p")
    km_df.to_csv(os.path.join(OUTPUT_DIR, "kaplan_meier_results.csv"), index=False)


# =========================
# 9. RECURRENCE PREDICTION
# =========================
roc_results = roc_for_recurrence(df, [c for c in DI_COLS if c in df.columns])

if roc_results is not None:
    pd.DataFrame([{
        "AUC": roc_results["auc"],
        "n": roc_results["n"]
    }]).to_csv(os.path.join(OUTPUT_DIR, "recurrence_roc_summary.csv"), index=False)

    pd.DataFrame(roc_results["classification_report"]).T.to_csv(
        os.path.join(OUTPUT_DIR, "recurrence_classification_report.csv")
    )

    pd.DataFrame(roc_results["confusion_matrix"]).to_csv(
        os.path.join(OUTPUT_DIR, "recurrence_confusion_matrix.csv"),
        index=False
    )


# =========================
# 10. RANK WHICH DI IS BEST
# =========================
ranking_rows = []

# Best by grade correlation
if "Grade" in corr_df["Clinical_variable"].values:
    grade_corr = corr_df[corr_df["Clinical_variable"] == "Grade"].copy()
    for _, row in grade_corr.iterrows():
        ranking_rows.append({
            "DI": row["DI"],
            "Metric": "Grade Spearman |r|",
            "Value": abs(row["Spearman_r"]) if pd.notna(row["Spearman_r"]) else np.nan,
            "p_value": row["p_value"]
        })

# Best by recurrence separation
if "Recurrence" in group_df["Clinical_variable"].values:
    recurrence_tests = group_df[group_df["Clinical_variable"] == "Recurrence"].copy()
    for _, row in recurrence_tests.iterrows():
        ranking_rows.append({
            "DI": row["DI"],
            "Metric": "Recurrence separation (1/p)",
            "Value": 1 / row["p_value"] if pd.notna(row["p_value"]) and row["p_value"] > 0 else np.nan,
            "p_value": row["p_value"]
        })

# Best by survival split
if len(km_results) > 0:
    for _, row in pd.DataFrame(km_results).iterrows():
        ranking_rows.append({
            "DI": row["DI"],
            "Metric": "Survival separation (1/p)",
            "Value": 1 / row["logrank_p"] if row["logrank_p"] > 0 else np.nan,
            "p_value": row["logrank_p"]
        })

ranking_df = pd.DataFrame(ranking_rows)
if len(ranking_df) > 0:
    ranking_df.to_csv(os.path.join(OUTPUT_DIR, "DI_ranking_summary.csv"), index=False)


# =========================
# 11. TEXT SUMMARY
# =========================
summary_lines = []
summary_lines.append(f"Total samples analysed: {len(df)}")
summary_lines.append("")

summary_lines.append("Top grade correlations:")
if "Grade" in corr_df["Clinical_variable"].values:
    top_grade = corr_df[corr_df["Clinical_variable"] == "Grade"].sort_values("p_value").head(6)
    summary_lines.append(top_grade.to_string(index=False))
    summary_lines.append("")

summary_lines.append("Top recurrence group differences:")
if "Recurrence" in group_df["Clinical_variable"].values:
    top_rec = group_df[group_df["Clinical_variable"] == "Recurrence"].sort_values("p_value").head(6)
    summary_lines.append(top_rec.to_string(index=False))
    summary_lines.append("")

if len(km_results) > 0:
    summary_lines.append("Kaplan-Meier results:")
    summary_lines.append(pd.DataFrame(km_results).sort_values("logrank_p").to_string(index=False))
    summary_lines.append("")

if roc_results is not None:
    summary_lines.append(f"Recurrence ROC AUC using all DI variants: {roc_results['auc']:.3f}")
    summary_lines.append("")

with open(os.path.join(OUTPUT_DIR, "analysis_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))

print("\nAnalysis complete.")
print(f"All results saved in: {OUTPUT_DIR}")