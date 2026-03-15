# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # ============================================================
# # USER SETTINGS
# # ============================================================

# # one example spectrum file
# SPECTRUM_FILE = "/Users/nana/Desktop/project_metadata/dpt_files/sample5073.dpt"

# # output
# OUTPUT_DIR = "/Users/nana/Desktop/linear_bg_plots"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # spectral crop for full display
# FULL_MIN = 1000.0
# FULL_MAX = 1800.0

# # band windows for linear background subtraction
# # edit these if you want slightly different ranges
# PHOSPHATE_MIN = 1180.0
# PHOSPHATE_MAX = 1260.0

# AMIDE_MIN = 1605.0
# AMIDE_MAX = 1685.0

# # figure style
# DPI = 300

# # ============================================================
# # HELPERS
# # ============================================================

# def load_dpt(path):
#     arr = np.loadtxt(path)
#     if arr.ndim != 2 or arr.shape[1] < 2:
#         raise ValueError(f"Unexpected format in {path}")
#     wn = arr[:, 0].astype(float)
#     ab = arr[:, 1].astype(float)
#     if wn[0] > wn[-1]:
#         wn = wn[::-1]
#         ab = ab[::-1]
#     return wn, ab

# def crop_band(wn, ab, wmin, wmax):
#     mask = (wn >= wmin) & (wn <= wmax)
#     return wn[mask], ab[mask]

# def linear_baseline(x, y):
#     """
#     Straight line joining first and last point in the selected band.
#     """
#     x1, x2 = x[0], x[-1]
#     y1, y2 = y[0], y[-1]
#     m = (y2 - y1) / (x2 - x1)
#     c = y1 - m * x1
#     return m * x + c

# def band_area_after_baseline(x, y):
#     base = linear_baseline(x, y)
#     ycorr = y - base
#     auc = np.trapz(ycorr, x)
#     return base, ycorr, auc

# def annotate_peak(ax, x, y, label):
#     idx = np.argmax(y)
#     xp = x[idx]
#     yp = y[idx]
#     ax.axvline(xp, linestyle="--", linewidth=1.5)
#     ax.text(xp, yp + 0.02 * (np.max(y) - np.min(y) + 1e-9), label, ha="center", va="bottom")

# # ============================================================
# # LOAD SPECTRUM
# # ============================================================

# wn, ab = load_dpt(SPECTRUM_FILE)

# # crop for full-spectrum view
# mask_full = (wn >= FULL_MIN) & (wn <= FULL_MAX)
# wn_full = wn[mask_full]
# ab_full = ab[mask_full]

# # phosphate band
# wn_p, ab_p = crop_band(wn, ab, PHOSPHATE_MIN, PHOSPHATE_MAX)
# base_p, ycorr_p, auc_p = band_area_after_baseline(wn_p, ab_p)

# # amide band
# wn_a, ab_a = crop_band(wn, ab, AMIDE_MIN, AMIDE_MAX)
# base_a, ycorr_a, auc_a = band_area_after_baseline(wn_a, ab_a)

# # ============================================================
# # PLOT 1: FULL SPECTRUM WITH BAND WINDOWS
# # ============================================================

# fig, ax = plt.subplots(figsize=(11, 5))
# ax.plot(wn_full, ab_full, linewidth=2)
# ax.axvspan(PHOSPHATE_MIN, PHOSPHATE_MAX, alpha=0.15, label="Phosphate window")
# ax.axvspan(AMIDE_MIN, AMIDE_MAX, alpha=0.15, label="Amide window")

# # FTIR convention: high -> low wavenumber left to right
# ax.invert_xaxis()

# ax.set_xlabel("Wavenumber (cm$^{-1}$)")
# ax.set_ylabel("Absorbance")
# ax.set_title("Example FTIR spectrum with phosphate and amide windows")
# ax.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "01_full_spectrum_windows.png"), dpi=DPI)
# plt.close()

# # ============================================================
# # PLOT 2: PHOSPHATE + AMIDE LINEAR BASELINE SUBTRACTION
# # ============================================================

# fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# # -------- phosphate --------
# ax = axes[0]
# ax.plot(wn_p, ab_p, linewidth=2, label="Spectrum")
# ax.plot(wn_p, base_p, linewidth=2, linestyle="--", label="Linear baseline")
# ax.fill_between(wn_p, ab_p, base_p, where=(ab_p >= base_p), alpha=0.3, label="Baseline-subtracted area")
# annotate_peak(ax, wn_p, ab_p, "Phosphate peak")

# ax.invert_xaxis()
# ax.set_xlabel("Wavenumber (cm$^{-1}$)")
# ax.set_ylabel("Absorbance")
# ax.set_title(f"Phosphate region\nAUC after baseline subtraction = {auc_p:.3f}")
# ax.legend()

# # -------- amide --------
# ax = axes[1]
# ax.plot(wn_a, ab_a, linewidth=2, label="Spectrum")
# ax.plot(wn_a, base_a, linewidth=2, linestyle="--", label="Linear baseline")
# ax.fill_between(wn_a, ab_a, base_a, where=(ab_a >= base_a), alpha=0.3, label="Baseline-subtracted area")
# annotate_peak(ax, wn_a, ab_a, "Amide peak")

# ax.invert_xaxis()
# ax.set_xlabel("Wavenumber (cm$^{-1}$)")
# ax.set_ylabel("Absorbance")
# ax.set_title(f"Amide region\nAUC after baseline subtraction = {auc_a:.3f}")
# ax.legend()

# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "02_phosphate_amide_linear_bg_subtraction.png"), dpi=DPI)
# plt.close()

# # ============================================================
# # PLOT 3: SINGLE PRESENTATION-STYLE FIGURE LIKE YOUR SKETCH
# # ============================================================

# fig, ax = plt.subplots(figsize=(12, 6))

# # plot full spectrum
# ax.plot(wn_full, ab_full, linewidth=2, color="black", label="Spectrum")

# # phosphate overlay
# ax.plot(wn_p, base_p, linewidth=3, linestyle="--", color="red")
# ax.fill_between(wn_p, ab_p, base_p, where=(ab_p >= base_p), alpha=0.25, color="tab:blue")
# p_peak = wn_p[np.argmax(ab_p)]
# ax.axvline(p_peak, linestyle="--", linewidth=2, color="tab:blue")
# ax.text(p_peak, np.max(ab_p) + 0.02, "Phosphate", color="tab:blue", ha="center")

# # amide overlay
# ax.plot(wn_a, base_a, linewidth=3, linestyle="--", color="red")
# ax.fill_between(wn_a, ab_a, base_a, where=(ab_a >= base_a), alpha=0.25, color="tab:green")
# a_peak = wn_a[np.argmax(ab_a)]
# ax.axvline(a_peak, linestyle="--", linewidth=2, color="tab:green")
# ax.text(a_peak, np.max(ab_a) + 0.02, "Amide", color="tab:green", ha="center")

# ax.invert_xaxis()
# ax.set_xlabel("Wavenumber (cm$^{-1}$)")
# ax.set_ylabel("Absorbance")
# ax.set_title("Linear background subtraction in phosphate and amide regions")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "03_presentation_style_linear_bg_plot.png"), dpi=DPI)
# plt.close()

# print("Done.")
# print(f"Saved plots to: {OUTPUT_DIR}")
# print(f"Phosphate AUC (baseline-subtracted): {auc_p:.4f}")
# print(f"Amide AUC (baseline-subtracted): {auc_a:.4f}")

# import os
# import re
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # ============================================================
# # USER SETTINGS
# # ============================================================

# DPT_FOLDER = "/Users/nana/Desktop/project_metadata/dpt_files"
# OUTPUT_DIR = "/Users/nana/Desktop/linear_bg_DI_results"

# # full display crop
# FULL_MIN = 1000.0
# FULL_MAX = 1800.0

# # linear-background subtraction windows
# PHOSPHATE_MIN = 1180.0
# PHOSPHATE_MAX = 1260.0

# AMIDE_MIN = 1605.0
# AMIDE_MAX = 1685.0

# # optional metadata
# METADATA_CSV = "/Users/nana/Desktop/project_metadata/clinical_metadata(Sheet1).csv"
# METADATA_ID_COL = "Anonymised Identifier"
# CLINICAL_DI_COL = "DIv3 - DigistainIndicesTrimmed"   # change if needed

# EPS = 1e-12
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ============================================================
# # HELPERS
# # ============================================================

# def clean_id(x):
#     if pd.isna(x):
#         return np.nan
#     s = str(x).strip()
#     try:
#         f = float(s)
#         if f.is_integer():
#             return str(int(f))
#         return str(f)
#     except Exception:
#         return s

# def extract_numeric_sample_id(filename: str) -> str:
#     stem = os.path.splitext(os.path.basename(filename))[0]
#     m = re.search(r"\d+", stem)
#     return m.group(0) if m else stem

# def load_dpt(path):
#     arr = np.loadtxt(path)
#     if arr.ndim != 2 or arr.shape[1] < 2:
#         raise ValueError(f"Unexpected .dpt format in {path}")
#     wn = arr[:, 0].astype(float)
#     ab = arr[:, 1].astype(float)
#     if wn[0] > wn[-1]:
#         wn = wn[::-1]
#         ab = ab[::-1]
#     return wn, ab

# def crop_band(wn, ab, wmin, wmax):
#     mask = (wn >= wmin) & (wn <= wmax)
#     return wn[mask], ab[mask]

# def linear_baseline(x, y):
#     x1, x2 = x[0], x[-1]
#     y1, y2 = y[0], y[-1]
#     m = (y2 - y1) / (x2 - x1)
#     c = y1 - m * x1
#     return m * x + c

# def band_auc_after_baseline(x, y):
#     if len(x) < 2:
#         return np.nan, None, None
#     base = linear_baseline(x, y)
#     ycorr = y - base
#     auc = np.trapz(ycorr, x)
#     return auc, base, ycorr

# def safe_ratio(num, den):
#     if (not np.isfinite(num)) or (not np.isfinite(den)) or abs(den) < EPS:
#         return np.nan
#     return num / den

# def raw_to_bounded(raw):
#     if not np.isfinite(raw):
#         return np.nan
#     raw = max(raw, 0.0)
#     return 1.0 / (1.0 + raw)

# def metrics(x, y):
#     from scipy.stats import spearmanr, pearsonr, linregress
#     tmp = pd.DataFrame({"x": x, "y": y}).dropna()
#     if len(tmp) < 3:
#         return None
#     x = tmp["x"].values
#     y = tmp["y"].values
#     rho_s, p_s = spearmanr(x, y)
#     rho_p, p_p = pearsonr(x, y)
#     fit = linregress(x, y)
#     return {
#         "n": len(tmp),
#         "spearman_rho": rho_s,
#         "spearman_p": p_s,
#         "pearson_r": rho_p,
#         "pearson_p": p_p,
#         "slope": fit.slope,
#         "intercept": fit.intercept,
#         "r_squared": fit.rvalue ** 2,
#     }

# # ============================================================
# # PROCESS ALL .DPT FILES
# # ============================================================

# rows = []
# dpt_files = sorted([f for f in os.listdir(DPT_FOLDER) if f.lower().endswith(".dpt")])

# if not dpt_files:
#     raise FileNotFoundError(f"No .dpt files found in {DPT_FOLDER}")

# for i, fname in enumerate(dpt_files, start=1):
#     fpath = os.path.join(DPT_FOLDER, fname)

#     try:
#         wn, ab = load_dpt(fpath)

#         # crop full display
#         mask_full = (wn >= FULL_MIN) & (wn <= FULL_MAX)
#         wn_full = wn[mask_full]
#         ab_full = ab[mask_full]

#         # phosphate
#         wn_p, ab_p = crop_band(wn_full, ab_full, PHOSPHATE_MIN, PHOSPHATE_MAX)
#         auc_p, base_p, ycorr_p = band_auc_after_baseline(wn_p, ab_p)

#         # amide
#         wn_a, ab_a = crop_band(wn_full, ab_full, AMIDE_MIN, AMIDE_MAX)
#         auc_a, base_a, ycorr_a = band_auc_after_baseline(wn_a, ab_a)

#         di_raw = safe_ratio(auc_p, auc_a)     # P/A
#         di_bounded = raw_to_bounded(di_raw)

#         rows.append({
#             "sample": clean_id(extract_numeric_sample_id(fname)),
#             "source_file": fname,
#             "phosphate_auc_bgsub": auc_p,
#             "amide_auc_bgsub": auc_a,
#             "DI_raw_P_over_A": di_raw,
#             "DI_bounded": di_bounded,
#         })

#         if i % 100 == 0:
#             print(f"Processed {i}/{len(dpt_files)} spectra")

#     except Exception as e:
#         print(f"[WARNING] Failed on {fname}: {e}")

# df = pd.DataFrame(rows)
# df.to_csv(os.path.join(OUTPUT_DIR, "01_linear_bg_DI_scores.csv"), index=False)

# if len(df) == 0:
#     raise RuntimeError("No spectra were processed successfully.")

# # ============================================================
# # OPTIONAL MERGE WITH CLINICAL METADATA
# # ============================================================

# merged = None
# metrics_df = None

# if os.path.exists(METADATA_CSV):
#     meta = pd.read_csv(METADATA_CSV)
#     meta.columns = [str(c).strip() for c in meta.columns]

#     if METADATA_ID_COL in meta.columns:
#         meta[METADATA_ID_COL] = meta[METADATA_ID_COL].apply(clean_id)
#         df["sample"] = df["sample"].apply(clean_id)

#         merged = df.merge(meta, left_on="sample", right_on=METADATA_ID_COL, how="left")
#         merged.to_csv(os.path.join(OUTPUT_DIR, "02_linear_bg_DI_merged_clinical.csv"), index=False)

#         if CLINICAL_DI_COL in merged.columns:
#             res_raw = metrics(merged[CLINICAL_DI_COL], merged["DI_raw_P_over_A"])
#             res_bounded = metrics(merged[CLINICAL_DI_COL], merged["DI_bounded"])

#             rows_metrics = []
#             if res_raw is not None:
#                 rows_metrics.append({"method": "DI_raw_P_over_A", "clinical_DI": CLINICAL_DI_COL, **res_raw})
#             if res_bounded is not None:
#                 rows_metrics.append({"method": "DI_bounded", "clinical_DI": CLINICAL_DI_COL, **res_bounded})

#             metrics_df = pd.DataFrame(rows_metrics)
#             metrics_df.to_csv(os.path.join(OUTPUT_DIR, "03_linear_bg_DI_vs_clinical_metrics.csv"), index=False)

# # ============================================================
# # PLOTS
# # ============================================================

# # 1. Raw DI histogram
# plt.figure(figsize=(7, 5))
# plt.hist(df["DI_raw_P_over_A"].dropna(), bins=40)
# plt.xlabel("Raw DI = P/A")
# plt.ylabel("Count")
# plt.title("Distribution of linear-background-subtracted raw DI")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "04_hist_raw_DI.png"), dpi=300)
# plt.close()

# # 2. Bounded DI histogram
# plt.figure(figsize=(7, 5))
# plt.hist(df["DI_bounded"].dropna(), bins=40)
# plt.xlabel("Bounded DI")
# plt.ylabel("Count")
# plt.title("Distribution of linear-background-subtracted bounded DI")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "05_hist_bounded_DI.png"), dpi=300)
# plt.close()

# # 3. Violin + box of raw and bounded
# fig, ax = plt.subplots(figsize=(7, 5))
# data = [
#     df["DI_raw_P_over_A"].dropna().values,
#     df["DI_bounded"].dropna().values
# ]
# parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
# for pc in parts["bodies"]:
#     pc.set_alpha(0.35)
# ax.boxplot(data, labels=["Raw P/A", "Bounded"], widths=0.2, showfliers=False)
# ax.set_ylabel("DI value")
# ax.set_title("Linear background subtraction DI distributions")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "06_violin_box_DI.png"), dpi=300)
# plt.close()

# # 4. Clinical comparison plots
# if merged is not None and CLINICAL_DI_COL in merged.columns:
#     from scipy.stats import linregress, spearmanr, pearsonr

#     # raw
#     tmp = merged[[CLINICAL_DI_COL, "DI_raw_P_over_A"]].dropna()
#     if len(tmp) >= 3:
#         x = tmp[CLINICAL_DI_COL].values
#         y = tmp["DI_raw_P_over_A"].values
#         fit = linregress(x, y)
#         rho_s, _ = spearmanr(x, y)
#         rho_p, _ = pearsonr(x, y)

#         plt.figure(figsize=(7, 5.5))
#         plt.hexbin(x, y, gridsize=30, mincnt=1)
#         xx = np.linspace(np.min(x), np.max(x), 200)
#         yy = fit.intercept + fit.slope * xx
#         plt.plot(xx, yy, linewidth=2)
#         plt.xlabel(CLINICAL_DI_COL)
#         plt.ylabel("DI_raw_P_over_A")
#         plt.title("Linear background-subtracted raw DI vs clinical DI")
#         txt = f"n = {len(tmp)}\nρ = {rho_s:.3f}\nr = {rho_p:.3f}\nR² = {fit.rvalue**2:.3f}"
#         plt.text(0.03, 0.97, txt, transform=plt.gca().transAxes,
#                  va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTPUT_DIR, "07_raw_DI_vs_clinical.png"), dpi=300)
#         plt.close()

#     # bounded
#     tmp = merged[[CLINICAL_DI_COL, "DI_bounded"]].dropna()
#     if len(tmp) >= 3:
#         x = tmp[CLINICAL_DI_COL].values
#         y = tmp["DI_bounded"].values
#         fit = linregress(x, y)
#         rho_s, _ = spearmanr(x, y)
#         rho_p, _ = pearsonr(x, y)

#         plt.figure(figsize=(7, 5.5))
#         plt.hexbin(x, y, gridsize=30, mincnt=1)
#         xx = np.linspace(np.min(x), np.max(x), 200)
#         yy = fit.intercept + fit.slope * xx
#         plt.plot(xx, yy, linewidth=2)
#         plt.xlabel(CLINICAL_DI_COL)
#         plt.ylabel("DI_bounded")
#         plt.title("Linear background-subtracted bounded DI vs clinical DI")
#         txt = f"n = {len(tmp)}\nρ = {rho_s:.3f}\nr = {rho_p:.3f}\nR² = {fit.rvalue**2:.3f}"
#         plt.text(0.03, 0.97, txt, transform=plt.gca().transAxes,
#                  va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTPUT_DIR, "08_bounded_DI_vs_clinical.png"), dpi=300)
#         plt.close()

# # 5. Example spectrum plot with shaded AUC and DI annotation
# example_file = os.path.join(DPT_FOLDER, dpt_files[0])
# wn, ab = load_dpt(example_file)
# mask_full = (wn >= FULL_MIN) & (wn <= FULL_MAX)
# wn_full = wn[mask_full]
# ab_full = ab[mask_full]

# wn_p, ab_p = crop_band(wn_full, ab_full, PHOSPHATE_MIN, PHOSPHATE_MAX)
# auc_p, base_p, ycorr_p = band_auc_after_baseline(wn_p, ab_p)

# wn_a, ab_a = crop_band(wn_full, ab_full, AMIDE_MIN, AMIDE_MAX)
# auc_a, base_a, ycorr_a = band_auc_after_baseline(wn_a, ab_a)

# di_example = safe_ratio(auc_p, auc_a)

# plt.figure(figsize=(12, 6))
# plt.plot(wn_full, ab_full, linewidth=2, color="black", label="Spectrum")

# plt.plot(wn_p, base_p, linewidth=2.5, linestyle="--", color="red")
# plt.fill_between(wn_p, ab_p, base_p, where=(ab_p >= base_p), alpha=0.25, color="tab:blue", label="Phosphate AUC")
# plt.axvline(wn_p[np.argmax(ab_p)], linestyle="--", linewidth=1.5)

# plt.plot(wn_a, base_a, linewidth=2.5, linestyle="--", color="red")
# plt.fill_between(wn_a, ab_a, base_a, where=(ab_a >= base_a), alpha=0.25, color="tab:green", label="Amide AUC")
# plt.axvline(wn_a[np.argmax(ab_a)], linestyle="--", linewidth=1.5)

# plt.gca().invert_xaxis()
# plt.xlabel("Wavenumber (cm$^{-1}$)")
# plt.ylabel("Absorbance")
# plt.title(f"Example spectrum with linear background subtraction\nDI = P/A = {di_example:.3f}")
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "09_example_spectrum_with_DI.png"), dpi=300)
# plt.close()

# # ============================================================
# # SUMMARY
# # ============================================================

# with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
#     f.write("Linear background subtraction DI results\n\n")
#     f.write(f"Processed spectra: {len(df)}\n")
#     f.write(f"Phosphate window: {PHOSPHATE_MIN}–{PHOSPHATE_MAX} cm^-1\n")
#     f.write(f"Amide window: {AMIDE_MIN}–{AMIDE_MAX} cm^-1\n\n")
#     f.write("DI definition:\n")
#     f.write("DI_raw_P_over_A = phosphate_auc_bgsub / amide_auc_bgsub\n")
#     f.write("DI_bounded = 1 / (1 + max(DI_raw_P_over_A, 0))\n\n")
#     if metrics_df is not None and len(metrics_df) > 0:
#         f.write("Clinical comparison metrics:\n")
#         f.write(metrics_df.to_string(index=False))

# print("Done.")
# print(f"Saved outputs to: {OUTPUT_DIR}")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

CSV_FOLDER = "/Users/nana/Desktop/wavenumber absorption"
OUTPUT_DIR = "/Users/nana/Desktop/DI_csv_results"

PHOS_MIN = 1180
PHOS_MAX = 1260

AMIDE_MIN = 1605
AMIDE_MAX = 1685

EPS = 1e-12
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FUNCTIONS
# ============================================================

def load_csv_spectrum(path):
    df = pd.read_csv(path)

    # try common column names
    cols = [c.lower() for c in df.columns]

    if "wavenumber" in cols:
        wn = df[df.columns[cols.index("wavenumber")]].values
    else:
        wn = df.iloc[:,0].values

    if "absorbance" in cols:
        ab = df[df.columns[cols.index("absorbance")]].values
    else:
        ab = df.iloc[:,1].values

    wn = wn.astype(float)
    ab = ab.astype(float)

    if wn[0] > wn[-1]:
        wn = wn[::-1]
        ab = ab[::-1]

    return wn, ab


def crop_band(wn, ab, wmin, wmax):

    mask = (wn >= wmin) & (wn <= wmax)
    return wn[mask], ab[mask]


def linear_baseline(x,y):

    x1,x2 = x[0],x[-1]
    y1,y2 = y[0],y[-1]

    m = (y2-y1)/(x2-x1)
    c = y1-m*x1

    return m*x+c


def baseline_auc(x,y):

    base = linear_baseline(x,y)
    corrected = y-base
    auc = np.trapz(corrected,x)

    return auc,base,corrected


def safe_ratio(a,b):

    if not np.isfinite(a) or not np.isfinite(b) or abs(b)<EPS:
        return np.nan

    return a/b


# ============================================================
# PROCESS ALL CSV FILES
# ============================================================

rows = []

files = [f for f in os.listdir(CSV_FOLDER) if f.lower().endswith(".csv")]

for i,fname in enumerate(files):

    path = os.path.join(CSV_FOLDER,fname)

    try:

        wn,ab = load_csv_spectrum(path)
        trn = 10 ** (-ab)

        # phosphate band
        wn_p,trn_p = crop_band(wn,trn,PHOS_MIN,PHOS_MAX)
        P_auc,base_p,corr_p = baseline_auc(wn_p,trn_p)

        # amide band
        wn_a,trn_a = crop_band(wn,trn,AMIDE_MIN,AMIDE_MAX)
        A_auc,base_a,corr_a = baseline_auc(wn_a,trn_a)
        DI = safe_ratio(P_auc,A_auc)

        rows.append({
            "file":fname,
            "phosphate_auc":P_auc,
            "amide_auc":A_auc,
            "DI_P_over_A":DI
        })

    except Exception as e:
        print("failed:",fname,e)


df = pd.DataFrame(rows)

df.to_csv(
    os.path.join(OUTPUT_DIR,"DI_scores.csv"),
    index=False
)

print("DI scores saved.")

# ============================================================
# PLOT 1: DI distribution
# ============================================================

plt.figure(figsize=(6,4))

plt.hist(df["DI_P_over_A"].dropna(),bins=30)

plt.xlabel("DI = P/A")
plt.ylabel("count")

plt.title("Distribution of DI across spectra")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR,"DI_distribution.png"),
    dpi=300
)

plt.close()

# ============================================================
# PLOT 2: example spectrum with AUC
# ============================================================

example = files[0]

wn,ab = load_csv_spectrum(os.path.join(CSV_FOLDER,example))

wn_p,ab_p = crop_band(wn,ab,PHOS_MIN,PHOS_MAX)
P_auc,base_p,corr_p = baseline_auc(wn_p,ab_p)

wn_a,ab_a = crop_band(wn,ab,AMIDE_MIN,AMIDE_MAX)
A_auc,base_a,corr_a = baseline_auc(wn_a,ab_a)

plt.figure(figsize=(10,5))

plt.plot(wn,ab,color="black",label="spectrum")

plt.plot(wn_p,base_p,'r--')
plt.fill_between(wn_p,ab_p,base_p,color="blue",alpha=0.3,label="phosphate area")

plt.plot(wn_a,base_a,'r--')
plt.fill_between(wn_a,ab_a,base_a,color="green",alpha=0.3,label="amide area")

plt.gca().invert_xaxis()

plt.xlabel("wavenumber (cm⁻¹)")
plt.ylabel("absorbance")

plt.title(f"Example DI = {safe_ratio(P_auc,A_auc):.3f}")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR,"example_DI_plot.png"),
    dpi=300
)

plt.close()

print("Plots saved.")