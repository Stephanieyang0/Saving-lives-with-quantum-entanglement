import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.stats import spearmanr, pearsonr, linregress

warnings.filterwarnings("ignore")

# ============================================================
# USER SETTINGS
# ============================================================

DPT_FOLDER = "/Users/nana/Desktop/project_metadata/dpt_files"
METADATA_CSV = "/Users/nana/Desktop/project_metadata/clinical_metadata(Sheet1).csv"
METADATA_ID_COL = "Anonymised Identifier"

WINDOW_FILES = {
    1: "/Users/nana/Desktop/Digistain/matlab/Window1.csv",   # amide peak
    2: "/Users/nana/Desktop/Digistain/matlab/Window2.csv",   # phosphate peak
    3: "/Users/nana/Desktop/Digistain/matlab/Window3.csv",   # phosphate base
    4: "/Users/nana/Desktop/Digistain/matlab/Window4.csv"    # amide base
}

LEARNED_FILTER_FILE = "/Users/nana/Desktop/ML_entanglecam_results/spectral_filters_and_weights.csv"

OUTPUT_DIR = "/Users/nana/Desktop/compare_original_gaussian_ml_results"

WN_MIN = 1000.0
WN_MAX = 1800.0
N_INTERP = 10001

DI_SCALE = 0.6798
EPS = 1e-12

# Use one common clinical reference for fair side-by-side plots
COMMON_CLINICAL_DI = "DIv3 - DigistainIndicesTrimmed"

GAUSSIAN_PARAMS = {
    1: {"mu": 1667.8,            "sigma": 25.347194649652200, "A": 0.8210912330100240},  # amide peak
    2: {"mu": 1221.0,            "sigma": 16.04200580217160,  "A": 0.7979553136112290},  # phosphate peak
    3: {"mu": 1168.2818403969,   "sigma": 7.986078378916150,  "A": 0.7678725860948360},  # phosphate base
    4: {"mu": 1604.97351330716,  "sigma": 16.412169228106600, "A": 0.9584470888388100},  # amide base
}

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def clean_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return s

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)

def extract_numeric_sample_id(filename: str) -> str:
    stem = Path(filename).stem
    m = re.search(r"\d+", stem)
    return m.group(0) if m else stem

def ensure_ascending(x, y):
    if len(x) < 2:
        return x, y
    if x[0] > x[-1]:
        return x[::-1], y[::-1]
    return x, y

def crop_xy(x, y, xmin=WN_MIN, xmax=WN_MAX):
    mask = (x >= xmin) & (x <= xmax)
    return x[mask], y[mask]

def load_dpt(path):
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unexpected .dpt format in {path}")
    wn = arr[:, 0].astype(float)
    ab = arr[:, 1].astype(float)
    wn, ab = ensure_ascending(wn, ab)
    wn, ab = crop_xy(wn, ab)
    return wn, ab

def load_window_csv(path):
    df = pd.read_csv(path).dropna()
    df.columns = [str(c).strip() for c in df.columns]
    wn = pd.to_numeric(df["wavenumber"], errors="coerce").values
    ab = pd.to_numeric(df["ABS"], errors="coerce").values
    mask = np.isfinite(wn) & np.isfinite(ab)
    wn, ab = wn[mask], ab[mask]
    wn, ab = ensure_ascending(wn, ab)
    wn, ab = crop_xy(wn, ab)
    return wn, ab

def load_all_windows():
    windows = {}
    for k, path in WINDOW_FILES.items():
        wn, ab = load_window_csv(path)
        windows[k] = {
            "wn": wn,
            "abs": ab,
            "trn": 10 ** (-ab)
        }
    return windows

def interp_common_grid(x1, y1, x2, y2, n=N_INTERP):
    min_w = max(np.min(x1), np.min(x2))
    max_w = min(np.max(x1), np.max(x2))
    if max_w <= min_w:
        return None, None, None
    xq = np.linspace(min_w, max_w, n)
    f1 = interp1d(x1, y1, kind="linear", bounds_error=False, fill_value=np.nan)
    f2 = interp1d(x2, y2, kind="linear", bounds_error=False, fill_value=np.nan)
    y1q = f1(xq)
    y2q = f2(xq)
    mask = np.isfinite(y1q) & np.isfinite(y2q)
    return xq[mask], y1q[mask], y2q[mask]

def integrate_curve(x, y):
    if len(x) < 2:
        return np.nan
    return np.trapz(y, x)

def gaussian_filter(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def safe_raw_di(P_peak, P_base, A_peak, A_base, scale=DI_SCALE):
    numer = P_peak - P_base
    denom = A_peak - A_base
    if (not np.isfinite(numer)) or (not np.isfinite(denom)) or abs(denom) < EPS:
        return np.nan
    return scale * (numer / denom)

def safe_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[mask] = a[mask] / b[mask]
    return out

def raw_to_bounded(raw):
    if np.ndim(raw) == 0:
        if not np.isfinite(raw):
            return np.nan
        raw = max(raw, 0.0)
        return 1.0 / (1.0 + raw)
    raw = np.asarray(raw, dtype=float)
    out = np.full_like(raw, np.nan, dtype=float)
    mask = np.isfinite(raw)
    tmp = raw.copy()
    tmp[mask] = np.maximum(tmp[mask], 0.0)
    out[mask] = 1.0 / (1.0 + tmp[mask])
    return out

def running_median(x, y, bins=20):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    if len(tmp) < bins:
        return None, None
    tmp["bin"] = pd.qcut(tmp["x"], q=bins, labels=False, duplicates="drop")
    g = tmp.groupby("bin").agg({"x": "median", "y": "median"}).dropna()
    return g["x"].values, g["y"].values

def metrics(x, y):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3:
        return None
    x = tmp["x"].values
    y = tmp["y"].values
    rho_s, p_s = spearmanr(x, y)
    rho_p, p_p = pearsonr(x, y)
    fit = linregress(x, y)
    return {
        "n": len(tmp),
        "spearman_rho": rho_s,
        "spearman_p": p_s,
        "pearson_r": rho_p,
        "pearson_p": p_p,
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r_squared": fit.rvalue ** 2,
    }

def plot_hexbin_panel(ax, x, y, xlab, ylab, title, bins=35):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    x = tmp["x"].values
    y = tmp["y"].values

    hb = ax.hexbin(x, y, gridsize=bins, mincnt=1)
    xm, ym = running_median(x, y, bins=18)
    if xm is not None:
        ax.plot(xm, ym, linewidth=2)

    fit = linregress(x, y)
    xx = np.linspace(np.min(x), np.max(x), 200)
    yy = fit.intercept + fit.slope * xx
    ax.plot(xx, yy, linewidth=1.7, alpha=0.8)

    rho_s, _ = spearmanr(x, y)
    rho_p, _ = pearsonr(x, y)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    txt = (
        f"n = {len(x)}\n"
        f"ρ = {rho_s:.3f}\n"
        f"r = {rho_p:.3f}\n"
        f"R² = {fit.rvalue**2:.3f}"
    )
    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

# ============================================================
# LOAD FILTERS
# ============================================================

print("Loading measured windows...")
windows = load_all_windows()

print("Loading learned ML filters...")
fdf = pd.read_csv(LEARNED_FILTER_FILE)
fdf.columns = [str(c).strip() for c in fdf.columns]

required_filter_cols = ["wavenumber", "learned_P_filter", "learned_A_filter"]
missing = [c for c in required_filter_cols if c not in fdf.columns]
if missing:
    raise KeyError(f"Missing columns in learned filter file: {missing}")

ml_wn = fdf["wavenumber"].values.astype(float)
ml_wP = fdf["learned_P_filter"].values.astype(float)
ml_wA = fdf["learned_A_filter"].values.astype(float)

# split into positive/negative lobes for physically meaningful response
ml_wP_pos = np.maximum(ml_wP, 0.0)
ml_wP_neg = np.maximum(-ml_wP, 0.0)
ml_wA_pos = np.maximum(ml_wA, 0.0)
ml_wA_neg = np.maximum(-ml_wA, 0.0)

# ============================================================
# PROCESS DPT FILES
# ============================================================

print("Processing .dpt files...")

rows = []
dpt_files = sorted([f for f in os.listdir(DPT_FOLDER) if f.lower().endswith(".dpt")])

for i, fname in enumerate(dpt_files, start=1):
    fpath = os.path.join(DPT_FOLDER, fname)
    sample_id = clean_id(extract_numeric_sample_id(fname))

    try:
        wn, ab = load_dpt(fpath)
        trn = 10 ** (-ab)

        # ---------------- Original DI ----------------
        ints_orig = {}
        for k in [1, 2, 3, 4]:
            xq, s_trn, w_trn = interp_common_grid(wn, trn, windows[k]["wn"], windows[k]["trn"])
            ints_orig[k] = np.nan if xq is None or len(xq) < 2 else integrate_curve(xq, s_trn * w_trn)

        raw_orig = safe_raw_di(
            P_peak=ints_orig[2],
            P_base=ints_orig[3],
            A_peak=ints_orig[1],
            A_base=ints_orig[4]
        )
        bounded_orig = raw_to_bounded(raw_orig)

        # ---------------- Gaussian DI ----------------
        ints_gauss = {}
        for k in [1, 2, 3, 4]:
            p = GAUSSIAN_PARAMS[k]
            g = gaussian_filter(wn, p["A"], p["mu"], p["sigma"])
            ints_gauss[k] = integrate_curve(wn, trn * g)

        raw_gauss = safe_raw_di(
            P_peak=ints_gauss[2],
            P_base=ints_gauss[3],
            A_peak=ints_gauss[1],
            A_base=ints_gauss[4]
        )
        bounded_gauss = raw_to_bounded(raw_gauss)

        # ---------------- ML DI with positive-minus-negative lobes ----------------
        fPp = interp1d(ml_wn, ml_wP_pos, bounds_error=False, fill_value=0.0)
        fPn = interp1d(ml_wn, ml_wP_neg, bounds_error=False, fill_value=0.0)
        fAp = interp1d(ml_wn, ml_wA_pos, bounds_error=False, fill_value=0.0)
        fAn = interp1d(ml_wn, ml_wA_neg, bounds_error=False, fill_value=0.0)

        wPp = fPp(wn)
        wPn = fPn(wn)
        wAp = fAp(wn)
        wAn = fAn(wn)

        # absorbance-weighted positive-minus-negative responses
        P_ml_resp = integrate_curve(wn, ab * wPp) - integrate_curve(wn, ab * wPn)
        A_ml_resp = integrate_curve(wn, ab * wAp) - integrate_curve(wn, ab * wAn)

        # store BOTH possible orientations for ML
        raw_ml_PA = np.nan
        raw_ml_AP = np.nan

        if np.isfinite(P_ml_resp) and np.isfinite(A_ml_resp):
            if abs(A_ml_resp) > EPS:
                raw_ml_PA = P_ml_resp / A_ml_resp   # P/A
            if abs(P_ml_resp) > EPS:
                raw_ml_AP = A_ml_resp / P_ml_resp   # A/P

        rows.append({
            "sample": sample_id,
            "source_file": fname,

            "DI_original_raw": raw_orig,
            "DI_original_bounded": bounded_orig,

            "DI_gaussian_raw": raw_gauss,
            "DI_gaussian_bounded": bounded_gauss,

            "ML_P_resp": P_ml_resp,
            "ML_A_resp": A_ml_resp,
            "DI_ML_raw_PA": raw_ml_PA,
            "DI_ML_raw_AP": raw_ml_AP,
        })

        if i % 100 == 0:
            print(f"Processed {i}/{len(dpt_files)} .dpt files")

    except Exception as e:
        print(f"[WARNING] Failed on {fname}: {e}")

df = pd.DataFrame(rows)

if len(df) == 0:
    raise RuntimeError("No .dpt samples were processed successfully.")

if "sample" not in df.columns:
    raise RuntimeError("Column 'sample' was not created. Check rows.append().")

# ------------------------------------------------------------
# Choose ML orientation globally so it matches Original DI
# ------------------------------------------------------------
tmp_pa = df[["DI_original_raw", "DI_ML_raw_PA"]].dropna()
tmp_ap = df[["DI_original_raw", "DI_ML_raw_AP"]].dropna()

rho_pa = np.nan
rho_ap = np.nan

if len(tmp_pa) >= 20:
    rho_pa, _ = spearmanr(tmp_pa["DI_original_raw"], tmp_pa["DI_ML_raw_PA"])

if len(tmp_ap) >= 20:
    rho_ap, _ = spearmanr(tmp_ap["DI_original_raw"], tmp_ap["DI_ML_raw_AP"])

if np.isfinite(rho_ap) and (not np.isfinite(rho_pa) or rho_ap > rho_pa):
    df["DI_ML_raw"] = df["DI_ML_raw_AP"]
    df["ML_orientation"] = "A/P (flipped from P/A)"
    chosen_rho = rho_ap
else:
    df["DI_ML_raw"] = df["DI_ML_raw_PA"]
    df["ML_orientation"] = "P/A"
    chosen_rho = rho_pa

df["DI_ML_bounded"] = raw_to_bounded(df["DI_ML_raw"].values)
df["ML_orientation_rho_vs_original_raw"] = chosen_rho


df.to_csv(os.path.join(OUTPUT_DIR, "01_per_sample_DI_from_dpt_v2.csv"), index=False)

# ============================================================
# MERGE WITH CLINICAL
# ============================================================

meta = pd.read_csv(METADATA_CSV)
meta.columns = [str(c).strip() for c in meta.columns]
meta[METADATA_ID_COL] = meta[METADATA_ID_COL].apply(clean_id)
df["sample"] = df["sample"].apply(clean_id)

merged = df.merge(meta, left_on="sample", right_on=METADATA_ID_COL, how="left")
merged.to_csv(os.path.join(OUTPUT_DIR, "02_merged_DI_clinical_v2.csv"), index=False)

# ============================================================
# METRICS
# ============================================================

method_cols = ["DI_original_bounded", "DI_gaussian_bounded", "DI_ML_bounded"]
clinical_col = COMMON_CLINICAL_DI

rows_clin = []
for mcol in method_cols:
    res = metrics(merged[clinical_col], merged[mcol])
    if res is not None:
        rows_clin.append({"method": mcol, "clinical_DI": clinical_col, **res})
clin_df = pd.DataFrame(rows_clin)
clin_df.to_csv(os.path.join(OUTPUT_DIR, "03_method_vs_common_clinical_metrics_v2.csv"), index=False)

pair_rows = []
pairs = [
    ("DI_original_bounded", "DI_gaussian_bounded"),
    ("DI_original_bounded", "DI_ML_bounded"),
    ("DI_gaussian_bounded", "DI_ML_bounded"),
]
for a, b in pairs:
    res = metrics(merged[a], merged[b])
    if res is not None:
        pair_rows.append({"method_x": a, "method_y": b, **res})
pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(os.path.join(OUTPUT_DIR, "04_method_vs_method_metrics_v2.csv"), index=False)

# ============================================================
# BETTER LOOKING PLOTS
# ============================================================

# 1. Common clinical comparison panels
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, mcol, title in zip(
    axes,
    method_cols,
    ["Original DI vs clinical", "Gaussian DI vs clinical", "ML DI vs clinical"]
):
    plot_hexbin_panel(
        ax,
        merged[clinical_col],
        merged[mcol],
        xlab=clinical_col,
        ylab=mcol,
        title=title,
        bins=30
    )
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_common_clinical_hexbin_panels.png"), dpi=300)
plt.close()

# 2. Pairwise method comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (a, b), title in zip(
    axes,
    pairs,
    ["Original vs Gaussian", "Original vs ML", "Gaussian vs ML"]
):
    plot_hexbin_panel(
        ax,
        merged[a],
        merged[b],
        xlab=a,
        ylab=b,
        title=title,
        bins=30
    )
    # identity line only if ranges overlap reasonably
    tmp = merged[[a, b]].dropna()
    if len(tmp) > 0:
        lo = min(tmp[a].min(), tmp[b].min())
        hi = max(tmp[a].max(), tmp[b].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, alpha=0.8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_pairwise_method_hexbin_panels.png"), dpi=300)
plt.close()

# 3. Cleaner distributions
fig, ax = plt.subplots(figsize=(8.5, 5.5))
data = [merged[c].dropna().values for c in method_cols]
parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
for pc in parts["bodies"]:
    pc.set_alpha(0.35)
ax.boxplot(data, labels=method_cols, widths=0.2, showfliers=False)
ax.set_ylabel("DI value")
ax.set_title("Distribution of DI methods on .dpt cohort")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_method_distributions_clean.png"), dpi=300)
plt.close()

# 4. Correlation summary bar chart
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.bar(clin_df["method"], clin_df["spearman_rho"])
ax.set_ylabel(f"Spearman ρ vs {clinical_col}")
ax.set_title("Clinical agreement by method")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_clinical_agreement_bar.png"), dpi=300)
plt.close()

# 5. Optional filter overlay reference
if {"mean_map_spectrum", "gaussian_P_filter", "gaussian_A_filter", "learned_P_filter", "learned_A_filter"}.issubset(fdf.columns):
    spec = fdf["mean_map_spectrum"].values.astype(float)
    spec = (spec - np.nanmin(spec)) / (np.nanmax(spec) - np.nanmin(spec) + EPS)

    gP = fdf["gaussian_P_filter"].values.astype(float)
    gA = fdf["gaussian_A_filter"].values.astype(float)
    lP = fdf["learned_P_filter"].values.astype(float)
    lA = fdf["learned_A_filter"].values.astype(float)

    gP = gP / (np.nanmax(np.abs(gP)) + EPS)
    gA = gA / (np.nanmax(np.abs(gA)) + EPS)
    lP = lP / (np.nanmax(np.abs(lP)) + EPS)
    lA = lA / (np.nanmax(np.abs(lA)) + EPS)

    plt.figure(figsize=(11, 5.5))
    plt.plot(ml_wn, spec, label="Mean map FTIR spectrum", linewidth=2)
    plt.plot(ml_wn, gP, label="Gaussian P filter", linewidth=2)
    plt.plot(ml_wn, gA, label="Gaussian A filter", linewidth=2)
    plt.plot(ml_wn, lP, label="Learned P filter", linewidth=2)
    plt.plot(ml_wn, lA, label="Learned A filter", linewidth=2)
    plt.axvline(1225, linestyle="--", linewidth=1, color="grey")
    plt.axvline(1650, linestyle="--", linewidth=1, color="grey")
    plt.text(1225, 1.02, "Phosphate", ha="center", va="bottom")
    plt.text(1650, 1.02, "Amide", ha="center", va="bottom")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Normalised amplitude")
    plt.title("Learned and Gaussian filters over FTIR spectrum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_filter_overlay_reference_v2.png"), dpi=300)
    plt.close()

# ============================================================
# SUMMARY
# ============================================================

with open(os.path.join(OUTPUT_DIR, "summary_v2.txt"), "w") as f:
    f.write("Improved comparison of Original, Gaussian, and ML DI methods\n\n")
    f.write(f"Clinical comparison column: {clinical_col}\n\n")
    f.write("Method vs clinical metrics:\n")
    if len(clin_df) > 0:
        f.write(clin_df.to_string(index=False))
    f.write("\n\nMethod vs method metrics:\n")
    if len(pair_df) > 0:
        f.write(pair_df.to_string(index=False))
    f.write("\n\nNotes:\n")
    f.write("- ML DI is recomputed using positive-minus-negative learned filter lobes.\n")
    f.write("- ML DI orientation is auto-flipped if needed to align with Original DI.\n")
    f.write("- Common clinical DI column is used for fair visual comparison.\n")

print("Done.")
print(f"Saved improved results into: {OUTPUT_DIR}")

import matplotlib.pyplot as plt
import numpy as np

# normalise spectrum
spec = fdf["mean_map_spectrum"].values.astype(float)
spec = (spec - np.nanmin(spec)) / (np.nanmax(spec) - np.nanmin(spec))

wn = fdf["wavenumber"].values

# gaussian filters
gP = fdf["gaussian_P_filter"].values
gA = fdf["gaussian_A_filter"].values

# ML filters
lP = fdf["learned_P_filter"].values
lA = fdf["learned_A_filter"].values

# normalise filters
gP = gP / np.max(np.abs(gP))
gA = gA / np.max(np.abs(gA))
lP = lP / np.max(np.abs(lP))
lA = lA / np.max(np.abs(lA))

plt.figure(figsize=(12,6))

# mean spectrum
plt.plot(wn, spec, linewidth=3, label="Mean FTIR spectrum")

# gaussian filters
plt.plot(wn, gP, linewidth=2, label="Gaussian P filter")
plt.plot(wn, gA, linewidth=2, label="Gaussian A filter")

# ML filters
plt.plot(wn, lP, linewidth=2, label="ML P filter")
plt.plot(wn, lA, linewidth=2, label="ML A filter")

# clinical DigiStain bands
plt.axvline(1220, linestyle="--", color="grey")
plt.axvline(1650, linestyle="--", color="grey")

plt.text(1220,1.02,"Phosphate",ha="center")
plt.text(1650,1.02,"Amide I",ha="center")

plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Normalised amplitude")
plt.title("FTIR spectrum with DigiStain, Gaussian and ML filters")
plt.legend()
plt.tight_layout()

plt.savefig("clinical_filter_overlap.png",dpi=300)
plt.show()