import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu, kruskal
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")

# Optional survival analysis
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


# ============================================================
# 1. USER SETTINGS
# ============================================================

# Folder containing all .dpt spectra files
SPECTRA_FOLDER = "spectra"

# Clinical metadata CSV
CLINICAL_CSV = "clinical_metadata(Sheet1).csv"

# Column used to merge spectra-derived DI with clinical metadata
# Change this if your metadata uses a different sample ID column
METADATA_ID_COL = "sample"

# Output folder
OUTPUT_DIR = "digistain_pipeline_results"

# Spectral windows (cm^-1)
PHOSPHATE_WINDOW = (1080, 1140)
PHOSPHATE_TRIM_WINDOW = (1100, 1130)

AMIDE_WINDOW = (1620, 1680)
AMIDE_TRIM_WINDOW = (1630, 1660)

# Baseline anchors for linear baseline inside each band
PHOSPHATE_BASELINE_ANCHORS = (1080, 1140)
AMIDE_BASELINE_ANCHORS = (1620, 1680)

# Smoothing parameters
USE_SAVGOL = True
SAVGOL_WINDOW = 11   # must be odd
SAVGOL_POLYORDER = 3

# Minimum epsilon to avoid divide-by-zero
EPS = 1e-12

Path(OUTPUT_DIR).mkdir(exist_ok=True)


# ============================================================
# 2. HELPERS
# ============================================================

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)

def gaussian(x, A, mu, sigma, c):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + c

def ensure_ascending(wn: np.ndarray, y: np.ndarray):
    """
    Ensure wavenumber is ascending for integration / interpolation.
    """
    if len(wn) < 2:
        return wn, y
    if wn[0] > wn[-1]:
        return wn[::-1], y[::-1]
    return wn, y

def linear_baseline(x: np.ndarray, y: np.ndarray):
    """
    Build a linear baseline using first and last point in selected band.
    """
    if len(x) < 2:
        return np.zeros_like(y)
    x1, x2 = x[0], x[-1]
    y1, y2 = y[0], y[-1]
    if abs(x2 - x1) < EPS:
        return np.full_like(y, y1)
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

def normalise_max(y: np.ndarray):
    ymax = np.nanmax(np.abs(y))
    if ymax < EPS:
        return y.copy()
    return y / ymax

def area_under_curve(x: np.ndarray, y: np.ndarray):
    if len(x) < 2:
        return np.nan
    return np.trapz(y, x)

def peak_height(y: np.ndarray):
    if len(y) == 0:
        return np.nan
    return np.nanmax(y)

def get_window(x: np.ndarray, y: np.ndarray, start: float, end: float):
    lo, hi = min(start, end), max(start, end)
    mask = (x >= lo) & (x <= hi)
    return x[mask], y[mask]

def smooth_spectrum(y: np.ndarray):
    if not USE_SAVGOL:
        return y.copy()
    if len(y) < SAVGOL_WINDOW or SAVGOL_WINDOW % 2 == 0:
        return y.copy()
    return savgol_filter(y, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)

def load_dpt_file(filepath: str):
    """
    Robust .dpt loader.
    Assumes 2 columns: wavenumber, absorbance
    """
    data = np.loadtxt(filepath)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected format in {filepath}")
    wn = data[:, 0].astype(float)
    absorb = data[:, 1].astype(float)
    wn, absorb = ensure_ascending(wn, absorb)
    return wn, absorb

def gaussian_peak_metrics(x: np.ndarray, y: np.ndarray):
    """
    Fit a Gaussian and return amplitude and fitted area.
    """
    if len(x) < 5 or np.allclose(y, y[0]):
        return np.nan, np.nan, np.nan, np.nan

    A0 = np.max(y) - np.min(y)
    mu0 = x[np.argmax(y)]
    sigma0 = max((x[-1] - x[0]) / 6, EPS)
    c0 = np.min(y)

    try:
        popt, _ = curve_fit(
            gaussian, x, y,
            p0=[A0, mu0, sigma0, c0],
            maxfev=10000
        )
        A, mu, sigma, c = popt
        fitted = gaussian(x, A, mu, sigma, c)
        area = np.trapz(fitted - c, x)
        return A, mu, abs(sigma), area
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

def safe_ratio(num, den):
    if pd.isna(num) or pd.isna(den) or abs(den) < EPS:
        return np.nan
    return num / den

def encode_binary(series: pd.Series, positive_map: dict):
    return series.map(positive_map)

def save_plot(fig, filename: str):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3. DI CALCULATION FROM ONE SPECTRUM
# ============================================================

def compute_di_set(wn: np.ndarray, absorb: np.ndarray):
    """
    Computes multiple DI variants from one FTIR spectrum.
    Returns dict of features.
    """
    result = {}

    # Raw + smoothed
    absorb_smooth = smooth_spectrum(absorb)

    # Whole-spectrum max normalisation
    absorb_norm = normalise_max(absorb_smooth)

    # ----------------------------
    # Raw windows
    # ----------------------------
    x_p, y_p = get_window(wn, absorb_smooth, *PHOSPHATE_WINDOW)
    x_a, y_a = get_window(wn, absorb_smooth, *AMIDE_WINDOW)

    x_pt, y_pt = get_window(wn, absorb_smooth, *PHOSPHATE_TRIM_WINDOW)
    x_at, y_at = get_window(wn, absorb_smooth, *AMIDE_TRIM_WINDOW)

    # ----------------------------
    # Baseline-corrected windows
    # ----------------------------
    if len(x_p) >= 2:
        y_p_base = linear_baseline(x_p, y_p)
        y_p_corr = y_p - y_p_base
    else:
        y_p_corr = np.array([])

    if len(x_a) >= 2:
        y_a_base = linear_baseline(x_a, y_a)
        y_a_corr = y_a - y_a_base
    else:
        y_a_corr = np.array([])

    # ----------------------------
    # Normalised windows
    # ----------------------------
    x_pn, y_pn = get_window(wn, absorb_norm, *PHOSPHATE_WINDOW)
    x_an, y_an = get_window(wn, absorb_norm, *AMIDE_WINDOW)

    # ----------------------------
    # Metrics
    # ----------------------------
    phosphate_auc = area_under_curve(x_p, y_p)
    amide_auc = area_under_curve(x_a, y_a)

    phosphate_trim_auc = area_under_curve(x_pt, y_pt)
    amide_trim_auc = area_under_curve(x_at, y_at)

    phosphate_peak = peak_height(y_p)
    amide_peak = peak_height(y_a)

    phosphate_corr_auc = area_under_curve(x_p, y_p_corr) if len(y_p_corr) > 1 else np.nan
    amide_corr_auc = area_under_curve(x_a, y_a_corr) if len(y_a_corr) > 1 else np.nan

    phosphate_norm_auc = area_under_curve(x_pn, y_pn)
    amide_norm_auc = area_under_curve(x_an, y_an)

    # Gaussian fits on baseline-corrected windows
    gp_A, gp_mu, gp_sigma, gp_area = gaussian_peak_metrics(x_p, y_p_corr) if len(x_p) >= 5 else (np.nan,)*4
    ga_A, ga_mu, ga_sigma, ga_area = gaussian_peak_metrics(x_a, y_a_corr) if len(x_a) >= 5 else (np.nan,)*4

    # ----------------------------
    # DI variants
    # ----------------------------
    result["DIv1_raw_auc_ratio"] = safe_ratio(phosphate_auc, amide_auc)
    result["DIv2_peak_height_ratio"] = safe_ratio(phosphate_peak, amide_peak)
    result["DIv3_trimmed_auc_ratio"] = safe_ratio(phosphate_trim_auc, amide_trim_auc)
    result["DIv4_baseline_corrected_auc_ratio"] = safe_ratio(phosphate_corr_auc, amide_corr_auc)
    result["DIv5_normalised_auc_ratio"] = safe_ratio(phosphate_norm_auc, amide_norm_auc)
    result["DIv6_gaussian_area_ratio"] = safe_ratio(gp_area, ga_area)
    result["DIv7_gaussian_amplitude_ratio"] = safe_ratio(gp_A, ga_A)

    # Save raw components too
    result["phosphate_auc"] = phosphate_auc
    result["amide_auc"] = amide_auc
    result["phosphate_trim_auc"] = phosphate_trim_auc
    result["amide_trim_auc"] = amide_trim_auc
    result["phosphate_peak"] = phosphate_peak
    result["amide_peak"] = amide_peak
    result["phosphate_corr_auc"] = phosphate_corr_auc
    result["amide_corr_auc"] = amide_corr_auc
    result["phosphate_norm_auc"] = phosphate_norm_auc
    result["amide_norm_auc"] = amide_norm_auc
    result["phosphate_gauss_area"] = gp_area
    result["amide_gauss_area"] = ga_area
    result["phosphate_gauss_amp"] = gp_A
    result["amide_gauss_amp"] = ga_A
    result["phosphate_gauss_mu"] = gp_mu
    result["amide_gauss_mu"] = ga_mu

    return result


# ============================================================
# 4. PROCESS ALL SPECTRA
# ============================================================

def spectrum_id_from_filename(filepath: str):
    return Path(filepath).stem

def process_all_spectra(folder: str):
    rows = []
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".dpt")])

    if not files:
        raise FileNotFoundError(f"No .dpt files found in folder: {folder}")

    for i, fname in enumerate(files, 1):
        fpath = os.path.join(folder, fname)
        try:
            wn, absorb = load_dpt_file(fpath)
            features = compute_di_set(wn, absorb)
            features["sample"] = spectrum_id_from_filename(fname)
            features["source_file"] = fname
            features["n_points"] = len(wn)
            rows.append(features)

            if i <= 5:
                # Save first few spectrum plots as QC
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(wn, absorb, label="Raw", alpha=0.7)
                ax.plot(wn, smooth_spectrum(absorb), label="Smoothed", alpha=0.9)
                ax.axvspan(*PHOSPHATE_WINDOW, alpha=0.2, label="Phosphate")
                ax.axvspan(*AMIDE_WINDOW, alpha=0.2, label="Amide")
                ax.invert_xaxis()
                ax.set_xlabel("Wavenumber (cm$^{-1}$)")
                ax.set_ylabel("Absorbance")
                ax.set_title(f"Spectrum QC: {Path(fname).stem}")
                ax.legend()
                save_plot(fig, f"QC_spectrum_{safe_filename(Path(fname).stem)}.png")

        except Exception as e:
            print(f"[WARNING] Failed to process {fname}: {e}")

    df = pd.DataFrame(rows)
    return df


# ============================================================
# 5. LOAD AND CLEAN CLINICAL METADATA
# ============================================================

def clean_clinical_metadata(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Fix common awkward names
    if "Survival " in df.columns and "Survival" not in df.columns:
        df = df.rename(columns={"Survival ": "Survival"})

    # Convert likely numeric columns
    for col in ["Grade", "Size", "DFI", "Survival", "Tubule formation", "Pleomorphism", "Mitosis", "Age at diagnosis"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strip text columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "": np.nan})

    # Binary encodings
    if "Recurrence" in df.columns:
        df["Recurrence_binary"] = df["Recurrence"].map({"Yes": 1, "No": 0})
    if "Dead or Alive" in df.columns:
        df["Death_event"] = df["Dead or Alive"].str.lower().map({"dead": 1, "alive": 0})
    if "ER Status" in df.columns:
        df["ER_binary"] = df["ER Status"].map({"Positive": 1, "Negative": 0})
    if "PR status" in df.columns:
        df["PR_binary"] = df["PR status"].map({"Positive": 1, "Negative": 0})
    if "HER2 status" in df.columns:
        df["HER2_binary"] = df["HER2 status"].map({"Positive": 1, "Negative": 0})
    if "Triple Negative" in df.columns:
        df["TripleNegative_binary"] = df["Triple Negative"].map({"Yes": 1, "No": 0})

    return df


# ============================================================
# 6. STATISTICS
# ============================================================

def safe_spearman(x, y):
    mask = (~pd.isna(x)) & (~pd.isna(y))
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())
    r, p = spearmanr(x[mask], y[mask])
    return r, p, int(mask.sum())

def binary_group_test(df, feature_col, group_col):
    tmp = df[[feature_col, group_col]].dropna().copy()
    if tmp.empty or tmp[group_col].nunique() != 2:
        return np.nan, np.nan, None

    groups = list(tmp[group_col].dropna().unique())
    g1, g2 = groups[0], groups[1]
    x1 = tmp.loc[tmp[group_col] == g1, feature_col].dropna()
    x2 = tmp.loc[tmp[group_col] == g2, feature_col].dropna()

    if len(x1) < 2 or len(x2) < 2:
        return np.nan, np.nan, None

    stat, p = mannwhitneyu(x1, x2, alternative="two-sided")
    return stat, p, groups

def multi_group_test(df, feature_col, group_col):
    tmp = df[[feature_col, group_col]].dropna().copy()
    if tmp.empty or tmp[group_col].nunique() < 2:
        return np.nan, np.nan

    arrays = [g[feature_col].values for _, g in tmp.groupby(group_col)]
    arrays = [a for a in arrays if len(a) >= 2]

    if len(arrays) < 2:
        return np.nan, np.nan

    stat, p = kruskal(*arrays)
    return stat, p

def plot_box(df, xcol, ycol, fname, title=None, order=None):
    tmp = df[[xcol, ycol]].dropna()
    if len(tmp) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    tmp.boxplot(column=ycol, by=xcol, ax=ax, grid=False)
    plt.suptitle("")
    ax.set_title(title or f"{ycol} vs {xcol}")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    save_plot(fig, fname)

def plot_scatter(df, xcol, ycol, fname, title=None):
    tmp = df[[xcol, ycol]].dropna()
    if len(tmp) < 3:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tmp[xcol], tmp[ycol], alpha=0.75)
    z = np.polyfit(tmp[xcol], tmp[ycol], 1)
    p = np.poly1d(z)
    xs = np.linspace(tmp[xcol].min(), tmp[xcol].max(), 100)
    ax.plot(xs, p(xs))
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title or f"{ycol} vs {xcol}")
    save_plot(fig, fname)

def plot_hist(df, col, fname):
    s = df[col].dropna()
    if len(s) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(s, bins=20)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {col}")
    save_plot(fig, fname)

def plot_roc_recurrence(df, di_cols):
    if "Recurrence_binary" not in df.columns:
        return None

    tmp = df[di_cols + ["Recurrence_binary"]].copy().dropna(subset=["Recurrence_binary"])
    if len(tmp) < 10 or tmp["Recurrence_binary"].nunique() < 2:
        return None

    X = tmp[di_cols]
    y = tmp["Recurrence_binary"].astype(int)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC for Recurrence Prediction")
    ax.legend()
    save_plot(fig, "ROC_recurrence.png")

    return roc_auc

def kaplan_meier_for_feature(df, feature_col):
    if not LIFELINES_AVAILABLE:
        return None
    if "Survival" not in df.columns or "Death_event" not in df.columns:
        return None

    tmp = df[[feature_col, "Survival", "Death_event"]].dropna().copy()
    if len(tmp) < 10:
        return None

    median_value = tmp[feature_col].median()
    tmp["group"] = np.where(tmp[feature_col] >= median_value, "High", "Low")

    g_low = tmp[tmp["group"] == "Low"]
    g_high = tmp[tmp["group"] == "High"]

    if len(g_low) < 3 or len(g_high) < 3:
        return None

    kmf = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(7, 5))
    kmf.fit(g_low["Survival"], event_observed=g_low["Death_event"], label="Low")
    kmf.plot_survival_function(ax=ax)
    kmf.fit(g_high["Survival"], event_observed=g_high["Death_event"], label="High")
    kmf.plot_survival_function(ax=ax)

    res = logrank_test(
        g_low["Survival"], g_high["Survival"],
        event_observed_A=g_low["Death_event"],
        event_observed_B=g_high["Death_event"]
    )

    ax.set_title(f"Kaplan-Meier: {feature_col}\nlog-rank p = {res.p_value:.4g}")
    ax.set_xlabel("Survival time")
    ax.set_ylabel("Survival probability")
    save_plot(fig, f"KM_{safe_filename(feature_col)}.png")

    return {"feature": feature_col, "median_split": median_value, "logrank_p": res.p_value, "n": len(tmp)}


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():
    print("Processing spectra...")
    spectra_df = process_all_spectra(SPECTRA_FOLDER)
    spectra_df.to_csv(os.path.join(OUTPUT_DIR, "spectra_DI_features.csv"), index=False)
    print(f"Computed DI features for {len(spectra_df)} spectra")

    print("Loading clinical metadata...")
    clinical_df = pd.read_csv(CLINICAL_CSV)
    clinical_df = clean_clinical_metadata(clinical_df)

    if METADATA_ID_COL not in clinical_df.columns:
        raise KeyError(
            f"Column '{METADATA_ID_COL}' not found in clinical metadata.\n"
            f"Available columns: {clinical_df.columns.tolist()}"
        )

    print("Merging spectra features with clinical metadata...")
    merged = spectra_df.merge(
        clinical_df,
        left_on="sample",
        right_on=METADATA_ID_COL,
        how="inner"
    )
    merged.to_csv(os.path.join(OUTPUT_DIR, "merged_spectra_clinical.csv"), index=False)
    print(f"Merged dataset has {len(merged)} matched samples")

    di_cols = [c for c in spectra_df.columns if c.startswith("DIv")]
    component_cols = [
        "phosphate_auc", "amide_auc",
        "phosphate_trim_auc", "amide_trim_auc",
        "phosphate_peak", "amide_peak",
        "phosphate_corr_auc", "amide_corr_auc",
        "phosphate_norm_auc", "amide_norm_auc",
        "phosphate_gauss_area", "amide_gauss_area",
        "phosphate_gauss_amp", "amide_gauss_amp"
    ]

    # -------------------------
    # QC plots
    # -------------------------
    for col in di_cols:
        plot_hist(merged, col, f"hist_{safe_filename(col)}.png")

    # -------------------------
    # Spearman correlations
    # -------------------------
    corr_targets = [c for c in ["Grade", "Size", "DFI", "Survival", "Tubule formation", "Pleomorphism", "Mitosis"] if c in merged.columns]
    corr_rows = []

    for feature in di_cols + component_cols:
        if feature not in merged.columns:
            continue
        for target in corr_targets:
            r, p, n = safe_spearman(merged[feature], merged[target])
            corr_rows.append({
                "feature": feature,
                "clinical_variable": target,
                "spearman_r": r,
                "p_value": p,
                "n": n
            })

            if target in ["Grade", "Mitosis", "Pleomorphism", "Tubule formation", "Size"]:
                plot_scatter(
                    merged, target, feature,
                    f"scatter_{safe_filename(target)}_{safe_filename(feature)}.png",
                    title=f"{feature} vs {target}"
                )

    corr_df = pd.DataFrame(corr_rows).sort_values(["clinical_variable", "p_value"], na_position="last")
    corr_df.to_csv(os.path.join(OUTPUT_DIR, "spearman_correlations.csv"), index=False)

    # -------------------------
    # Group comparisons
    # -------------------------
    group_rows = []

    binary_targets = [c for c in ["Recurrence", "Dead or Alive", "ER Status", "PR status", "HER2 status", "Triple Negative", "Vascular Invasion"] if c in merged.columns]
    multi_targets = [c for c in ["Stage", "NPI (3 groups)", "Tumour Type (groups)", "Grade"] if c in merged.columns]

    for feature in di_cols:
        if feature not in merged.columns:
            continue

        for target in binary_targets:
            stat, p, groups = binary_group_test(merged, feature, target)
            group_rows.append({
                "feature": feature,
                "clinical_variable": target,
                "test": "Mann-Whitney U",
                "statistic": stat,
                "p_value": p,
                "groups": str(groups)
            })
            plot_box(
                merged, target, feature,
                f"box_{safe_filename(target)}_{safe_filename(feature)}.png",
                title=f"{feature} vs {target}"
            )

        for target in multi_targets:
            stat, p = multi_group_test(merged, feature, target)
            group_rows.append({
                "feature": feature,
                "clinical_variable": target,
                "test": "Kruskal-Wallis",
                "statistic": stat,
                "p_value": p,
                "groups": np.nan
            })
            plot_box(
                merged, target, feature,
                f"box_{safe_filename(target)}_{safe_filename(feature)}.png",
                title=f"{feature} vs {target}"
            )

    group_df = pd.DataFrame(group_rows).sort_values("p_value", na_position="last")
    group_df.to_csv(os.path.join(OUTPUT_DIR, "group_comparisons.csv"), index=False)

    # -------------------------
    # ROC for recurrence
    # -------------------------
    roc_auc = plot_roc_recurrence(merged, di_cols)
    roc_df = pd.DataFrame([{"AUC": roc_auc}]) if roc_auc is not None else pd.DataFrame()
    if not roc_df.empty:
        roc_df.to_csv(os.path.join(OUTPUT_DIR, "roc_summary.csv"), index=False)

    # -------------------------
    # Kaplan-Meier
    # -------------------------
    km_rows = []
    if LIFELINES_AVAILABLE:
        for feature in di_cols:
            res = kaplan_meier_for_feature(merged, feature)
            if res is not None:
                km_rows.append(res)

    if km_rows:
        pd.DataFrame(km_rows).sort_values("logrank_p").to_csv(
            os.path.join(OUTPUT_DIR, "kaplan_meier_results.csv"),
            index=False
        )

    # -------------------------
    # Ranking summary
    # -------------------------
    ranking_rows = []

    if not corr_df.empty and "Grade" in corr_df["clinical_variable"].values:
        tmp = corr_df[corr_df["clinical_variable"] == "Grade"].copy()
        for _, row in tmp.iterrows():
            ranking_rows.append({
                "feature": row["feature"],
                "metric": "Grade correlation |rho|",
                "value": abs(row["spearman_r"]) if pd.notna(row["spearman_r"]) else np.nan,
                "p_value": row["p_value"]
            })

    if not group_df.empty and "Recurrence" in group_df["clinical_variable"].values:
        tmp = group_df[group_df["clinical_variable"] == "Recurrence"].copy()
        for _, row in tmp.iterrows():
            ranking_rows.append({
                "feature": row["feature"],
                "metric": "Recurrence separation 1/p",
                "value": 1 / row["p_value"] if pd.notna(row["p_value"]) and row["p_value"] > 0 else np.nan,
                "p_value": row["p_value"]
            })

    ranking_df = pd.DataFrame(ranking_rows)
    if not ranking_df.empty:
        ranking_df.to_csv(os.path.join(OUTPUT_DIR, "feature_ranking.csv"), index=False)

    # -------------------------
    # Text summary
    # -------------------------
    summary_lines = []
    summary_lines.append(f"Number of spectra processed: {len(spectra_df)}")
    summary_lines.append(f"Number of merged clinical samples: {len(merged)}")
    summary_lines.append("")

    if not corr_df.empty:
        summary_lines.append("Top correlations:")
        summary_lines.append(corr_df.head(15).to_string(index=False))
        summary_lines.append("")

    if not group_df.empty:
        summary_lines.append("Top group comparisons:")
        summary_lines.append(group_df.head(15).to_string(index=False))
        summary_lines.append("")

    if roc_auc is not None:
        summary_lines.append(f"Recurrence ROC AUC using all DI features: {roc_auc:.3f}")
        summary_lines.append("")

    if km_rows:
        summary_lines.append("Kaplan-Meier results:")
        summary_lines.append(pd.DataFrame(km_rows).sort_values("logrank_p").to_string(index=False))

    with open(os.path.join(OUTPUT_DIR, "analysis_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    print("Done.")
    print(f"All results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()