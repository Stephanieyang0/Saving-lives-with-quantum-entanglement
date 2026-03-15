#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline-corrected AUC around phosphate band (≈1234 cm^-1), with fixed geometry
============================================================================

Implements your adjusted plan:

1) Load many spectra (e.g., 60 CSV files: 2 columns: wavenumber, absorbance or transmission-like).
2) For each spectrum:
   - Find the peak position near 1234 cm^-1 (within a search window).
   - FIX the window centre to that peak (per-spectrum centring).
   - Use a linear baseline model fitted using two baseline windows placed ±70 cm^-1 away
     from the signal window centre (same width as signal window).
   - Baseline-correct locally and compute AUC = ∫ (y - baseline) dν over the signal window.
3) Compare different window widths (FWHM-like box width) = 34, 30, 20 cm^-1
   and quantify "information lost" vs the largest width (default 34).
4) Visualisations:
   - For a chosen spectrum: overlay of baseline fit and shaded AUC for each width.
   - Across all spectra: histogram of peak centres, boxplots of AUCs per width,
     scatter AUC(20) vs AUC(34), and "loss fraction" distribution.
5) Optional: if you have a legacy DI (e.g. FitSpy DI or old DI*0.1), fit a scaling factor
   to map AUC-based index to match (least squares).

Notes:
- This stays in "absorbance space" by default. If your files are transmission, set INPUT_IS_TRANSMISSION=True.
- Baseline windows: centered at (mu_peak ± 70), with same width as signal window.
- If baseline windows run out of bounds, the code clips and still fits robustly.

Run:
  python auc_baseline_1234.py

Outputs saved in OUTDIR.
"""

# import os
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # =========================
# # USER SETTINGS
# # =========================

# # Option A: a folder of CSV files
# DATA_DIR = "/Users/nana/Desktop/wavenumber absorption"
# GLOB_PATTERN = "*.CSV"   # or "*.csv"

# # Option B: explicit list (leave empty to use folder+glob)
# SPECTRUM_FILES = []

# HEADER = False  # set True if your CSV has a header line

# # If your input column2 is transmission (0..1) and you want absorbance:
# INPUT_IS_TRANSMISSION = False  # True -> convert A = -log10(T)

# # Focus band around phosphate peak
# PEAK_SEARCH_CENTER = 1234.0
# PEAK_SEARCH_HALF_WIDTH = 25.0   # search within [center-25, center+25]
# # You mentioned ~1268 and ~1200 region: this search window covers it if you enlarge it.
# # If your data peak may shift, increase to 40-60.

# # Baseline window geometry
# BASELINE_OFFSET = 70.0  # baseline windows at mu±70 cm^-1 from the signal window centre

# # Window widths to trial (cm^-1) -- your plan
# WINDOW_WIDTHS = [34.0, 30.0, 20.0]

# # Integration method
# # If True: only integrate where (y - baseline) > 0, otherwise signed area
# AUC_POSITIVE_ONLY = True

# # Output
# OUTDIR = "auc_1234_results"
# os.makedirs(OUTDIR, exist_ok=True)

# # Which spectrum to show detailed overlay plots for
# EXAMPLE_INDEX = 0  # 0-based index after sorting filenames

# # =========================
# # Helpers
# # =========================

# def load_2col(path, header=False):
#     df = pd.read_csv(path, sep=None, engine="python", header=0 if header else None)
#     df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
#     wn = df.iloc[:, 0].to_numpy(float)
#     y  = df.iloc[:, 1].to_numpy(float)
#     order = np.argsort(wn)
#     return wn[order], y[order]

# def to_absorbance(y):
#     """Convert transmission to absorbance A=-log10(T)."""
#     y = np.asarray(y, float)
#     y = np.clip(y, 1e-12, 1.0)  # avoid log blow-up
#     return -np.log10(y)

# def find_peak_near(wn, y, center, half_width):
#     """Return mu_peak as argmax(y) within the search region."""
#     m = (wn >= center - half_width) & (wn <= center + half_width)
#     if not np.any(m):
#         return np.nan
#     idx = np.argmax(y[m])
#     return float(wn[m][idx])

# def mask_window(wn, mu, width):
#     """Boolean mask for [mu-width/2, mu+width/2]."""
#     hw = width / 2.0
#     return (wn >= mu - hw) & (wn <= mu + hw)

# def fit_linear_baseline(wn, y, mu, width, offset):
#     """
#     Fit y ≈ a*wn + b using points in two baseline windows:
#       left  centered at mu-offset
#       right centered at mu+offset
#     each with same 'width' as the signal window.

#     Returns (a, b, mask_baseline).
#     """
#     mL = mask_window(wn, mu - offset, width)
#     mR = mask_window(wn, mu + offset, width)
#     mb = mL | mR
#     if np.sum(mb) < 2:
#         # fallback: use nearest points around the signal window edges
#         # (still gives some baseline)
#         m_edge = mask_window(wn, mu, width + 2*offset)
#         mb = m_edge
#         if np.sum(mb) < 2:
#             return np.nan, np.nan, mb

#     X = wn[mb]
#     Y = y[mb]
#     # linear least squares
#     A = np.column_stack([X, np.ones_like(X)])
#     coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
#     a, b = float(coef[0]), float(coef[1])
#     return a, b, mb

# def auc_between_curve_and_baseline(wn, y, a, b, mu, width, positive_only=True):
#     """
#     Compute AUC = ∫ (y - (a*wn+b)) dwn over signal window.
#     Uses trapezoid rule.
#     """
#     ms = mask_window(wn, mu, width)
#     if np.sum(ms) < 2 or not np.isfinite(a) or not np.isfinite(b):
#         return np.nan, ms, None

#     x = wn[ms]
#     yy = y[ms]
#     base = a*x + b
#     diff = yy - base
#     if positive_only:
#         diff = np.maximum(diff, 0.0)
#     auc = float(np.trapz(diff, x))
#     return auc, ms, (x, yy, base, diff)

# # =========================
# # Main analysis
# # =========================

# def main():
#     # Gather files
#     files = []
#     if SPECTRUM_FILES:
#         files = list(SPECTRUM_FILES)
#     else:
#         files = sorted(glob.glob(os.path.join(DATA_DIR, GLOB_PATTERN)))

#     if len(files) < 3:
#         raise SystemExit("No (or too few) spectra found. Check DATA_DIR / GLOB_PATTERN / SPECTRUM_FILES.")

#     print(f"Found {len(files)} spectra.")

#     # Load all; align grids by interpolation onto a common wn grid (optional but useful for plotting mean)
#     wn0, y0 = load_2col(files[0], header=HEADER)
#     if INPUT_IS_TRANSMISSION:
#         y0 = to_absorbance(y0)

#     # use full range of first spectrum
#     wn_grid = wn0
#     Y = []
#     for p in files:
#         wn, y = load_2col(p, header=HEADER)
#         if INPUT_IS_TRANSMISSION:
#             y = to_absorbance(y)
#         yg = np.interp(wn_grid, wn, y)
#         Y.append(yg)
#     Y = np.vstack(Y)

#     mean_spec = np.mean(Y, axis=0)

#     # Per-spectrum results
#     rows = []
#     # Store for plots
#     AUC = {w: [] for w in WINDOW_WIDTHS}
#     MU  = []

#     for i, p in enumerate(files):
#         y = Y[i]
#         mu = find_peak_near(wn_grid, y, PEAK_SEARCH_CENTER, PEAK_SEARCH_HALF_WIDTH)
#         MU.append(mu)

#         record = {"file": os.path.basename(p), "mu_peak": mu}

#         for w in WINDOW_WIDTHS:
#             a, b, mb = fit_linear_baseline(wn_grid, y, mu, w, BASELINE_OFFSET)
#             auc, ms, pack = auc_between_curve_and_baseline(wn_grid, y, a, b, mu, w, positive_only=AUC_POSITIVE_ONLY)
#             record[f"auc_w{int(w)}"] = auc
#             AUC[w].append(auc)

#         rows.append(record)

#     df = pd.DataFrame(rows)
#     df.to_csv(os.path.join(OUTDIR, "auc_table.csv"), index=False)

#     print("Saved:", os.path.join(OUTDIR, "auc_table.csv"))

#     # Information loss relative to widest window (first in list)
#     w_ref = max(WINDOW_WIDTHS)  # typically 34
#     ref = np.array(AUC[w_ref], float)
#     loss = {}
#     for w in WINDOW_WIDTHS:
#         if w == w_ref:
#             continue
#         a = np.array(AUC[w], float)
#         # loss fraction: 1 - AUC(w)/AUC(ref)
#         frac = 1.0 - (a / (ref + 1e-12))
#         loss[w] = frac
#         df[f"loss_frac_w{int(w)}_vs_{int(w_ref)}"] = frac

#     df.to_csv(os.path.join(OUTDIR, "auc_table_with_loss.csv"), index=False)

#     # =========================
#     # VISUALISATIONS (global)
#     # =========================

#     # 1) Histogram of peak centres
#     plt.figure(figsize=(7,4))
#     mu_arr = np.array(MU, float)
#     mu_arr = mu_arr[np.isfinite(mu_arr)]
#     plt.hist(mu_arr, bins=20, edgecolor="black")
#     plt.xlabel("Detected peak centre near 1234 (cm$^{-1}$)")
#     plt.ylabel("Count")
#     plt.title("Per-spectrum peak centre distribution")
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTDIR, "mu_peak_hist.png"), dpi=200)
#     plt.close()

#     # 2) Boxplot of AUC across widths
#     plt.figure(figsize=(8,5))
#     data = [np.array(AUC[w], float) for w in WINDOW_WIDTHS]
#     plt.boxplot(data, labels=[f"{int(w)}" for w in WINDOW_WIDTHS], showfliers=True)
#     plt.ylabel("AUC (arb.·cm$^{-1}$)")
#     plt.xlabel("Window width (cm$^{-1}$)")
#     plt.title("AUC after local linear baseline subtraction (1234 band)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTDIR, "auc_boxplot_by_width.png"), dpi=200)
#     plt.close()

#     # 3) Scatter AUC(20) vs AUC(34) if both present
#     if 20.0 in WINDOW_WIDTHS and w_ref in WINDOW_WIDTHS:
#         a20 = np.array(AUC[20.0], float)
#         a34 = np.array(AUC[w_ref], float)
#         m = np.isfinite(a20) & np.isfinite(a34)
#         if np.sum(m) >= 3:
#             plt.figure(figsize=(6,6))
#             plt.scatter(a34[m], a20[m], s=30)
#             lo = min(np.min(a34[m]), np.min(a20[m]))
#             hi = max(np.max(a34[m]), np.max(a20[m]))
#             plt.plot([lo, hi], [lo, hi])
#             plt.xlabel(f"AUC width {int(w_ref)}")
#             plt.ylabel("AUC width 20")
#             plt.title("Information retained: AUC(20) vs AUC(34)")
#             plt.tight_layout()
#             plt.savefig(os.path.join(OUTDIR, "auc20_vs_auc34.png"), dpi=200)
#             plt.close()

#     # 4) Loss fraction histograms
#     for w, frac in loss.items():
#         frac = np.array(frac, float)
#         frac = frac[np.isfinite(frac)]
#         plt.figure(figsize=(7,4))
#         plt.hist(frac, bins=20, edgecolor="black")
#         plt.xlabel(f"Loss fraction = 1 - AUC({int(w)})/AUC({int(w_ref)})")
#         plt.ylabel("Count")
#         plt.title("Information loss distribution")
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTDIR, f"loss_hist_w{int(w)}_vs_{int(w_ref)}.png"), dpi=200)
#         plt.close()

#     # =========================
#     # VISUALISATIONS (example spectrum)
#     # =========================

#     ex = min(EXAMPLE_INDEX, len(files)-1)
#     ex_file = files[ex]
#     ex_y = Y[ex]
#     ex_mu = MU[ex]

#     plt.figure(figsize=(11,6))
#     plt.plot(wn_grid, ex_y, lw=2, label=f"Spectrum (example)")

#     # draw baseline + AUC shading for each width
#     for w in WINDOW_WIDTHS:
#         a, b, mb = fit_linear_baseline(wn_grid, ex_y, ex_mu, w, BASELINE_OFFSET)
#         auc, ms, pack = auc_between_curve_and_baseline(wn_grid, ex_y, a, b, ex_mu, w, positive_only=AUC_POSITIVE_ONLY)
#         if pack is None:
#             continue
#         x, yy, base, diff = pack

#         # baseline line on the signal region
#         plt.plot(x, base, lw=1, label=f"Baseline fit (w={int(w)})")

#         # shade AUC (difference)
#         plt.fill_between(x, base, base + diff, alpha=0.20, label=f"AUC area (w={int(w)})")

#         # show baseline windows points
#         plt.scatter(wn_grid[mb], ex_y[mb], s=8, alpha=0.35)

#         # window edges
#         hw = w/2
#         plt.axvline(ex_mu - hw, lw=0.8, alpha=0.4)
#         plt.axvline(ex_mu + hw, lw=0.8, alpha=0.4)

#     plt.axvline(ex_mu, lw=1.2, alpha=0.7, linestyle="--", label=f"Peak centre ~ {ex_mu:.2f}")
#     plt.gca().invert_xaxis()
#     plt.xlabel("Wavenumber (cm$^{-1}$)")
#     plt.ylabel("Absorbance (arb.)" + (" (converted from T)" if INPUT_IS_TRANSMISSION else ""))
#     plt.title(f"Example baseline-corrected AUC around 1234 band\n{os.path.basename(ex_file)}")
#     plt.legend(ncol=2, fontsize=9)
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTDIR, "example_auc_overlay.png"), dpi=220)
#     plt.close()

#     # Mean spectrum plot + show geometry (peak centre fixed to mean peak)
#     mean_mu = find_peak_near(wn_grid, mean_spec, PEAK_SEARCH_CENTER, PEAK_SEARCH_HALF_WIDTH)

#     plt.figure(figsize=(11,5))
#     plt.plot(wn_grid, mean_spec, lw=2, label="Mean spectrum")
#     plt.axvline(mean_mu, linestyle="--", lw=1.2, label=f"Mean peak centre ~ {mean_mu:.2f}")

#     for w in WINDOW_WIDTHS:
#         hw = w/2
#         # signal window
#         plt.axvspan(mean_mu-hw, mean_mu+hw, alpha=0.10, label=f"Signal win {int(w)}")
#         # baseline windows (centres ±70)
#         plt.axvspan((mean_mu-BASELINE_OFFSET)-hw, (mean_mu-BASELINE_OFFSET)+hw, alpha=0.05)
#         plt.axvspan((mean_mu+BASELINE_OFFSET)-hw, (mean_mu+BASELINE_OFFSET)+hw, alpha=0.05)

#     plt.gca().invert_xaxis()
#     plt.xlabel("Wavenumber (cm$^{-1}$)")
#     plt.ylabel("Absorbance (arb.)")
#     plt.title("Window geometry on mean spectrum (signal + baseline windows)")
#     plt.legend(ncol=2, fontsize=9)
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTDIR, "mean_geometry.png"), dpi=220)
#     plt.close()

#     print("Saved plots to:", os.path.abspath(OUTDIR))
#     print("Key plots:")
#     print(" - example_auc_overlay.png  (shows baseline fit + shaded AUC for w=34/30/20)")
#     print(" - auc_boxplot_by_width.png (global comparison)")
#     print(" - auc20_vs_auc34.png       (retained information)")
#     print(" - loss_hist_*.png          (information loss distribution)")
#     print(" - mu_peak_hist.png         (peak centring distribution)")
#     print(" - mean_geometry.png        (fixed geometry illustration)")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import os, glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.model_selection import LeaveOneOut
# from sklearn.linear_model import LinearRegression

# # =========================
# # USER SETTINGS
# # =========================
# DATA_DIR = "/Users/nana/Desktop/wavenumber absorption"
# GLOB_PATTERN = "*.CSV"          # adjust if lower-case
# HEADER = False

# # If your second column is transmission: set True
# INPUT_IS_TRANSMISSION = False

# OUTDIR = "next_steps_outputs"
# os.makedirs(OUTDIR, exist_ok=True)

# # ---- Targets (optional but recommended) ----
# # Put your PA_ratio table here (filename -> PA_true)
# PA_TABLE = {"24C 15948 A2_S1_06112025_1350.csv": 0.046127287406064800,
# "24C 15948 A2_S3_06112025_1352.csv": 0.09198127778163410,
# "24C 15948 A2_S4_06112025_1353.csv": 0.055222094074456500,
# "24C 15948 A2_S5_06112025_1354.csv": 0.04672077952683070,
# "24C 15948 A2_S7_06112025_1355.csv": 0.08752096293176850,
# "24C 15948 A2_S8_06112025_1356.csv": 0.09024791637515490,
# "24C 15948 A2_S9_06112025_1356.csv": 0.0829211562274142,
# "24C 17037_S1_06112025_1422.csv": 0.05172891872512750,
# "24C 17037_S2_06112025_1422.csv": 0.05175957174556670,
# "24C 17037_S4_06112025_1424.csv": 0.07134117960007720,
# "24C 17037_S5_06112025_1425.csv": 0.05394396428685540,
# "24H00050619_S10_07112025_1430.csv": 0.0214109091425551,
# "24H00050619_S1_07112025_1417.csv": 0.02214713351071800,
# "24H00050619_S2_07112025_1418.csv": 0.02380632672201190,
# "24H00050619_S3_07112025_1419.csv": 0.00394983938576703,
# "24H00050619_S4_07112025_1420.csv":  0.0276055352094291,
# "24H00050619_S5_07112025_1421.csv":  0.0265879975568126,
# "24H00050619_S6_07112025_1426.csv":  0.024288,
# "24H00050619_S8_07112025_1428.csv": 0.05877514968951400,
# "24H00050619_S9_07112025_1429.csv": 4.99069207984537E-08,
# "PH001224K_S10_07112025_1618.csv":  0.0642527836305066,
# "PH001224K_S2_07112025_1611.csv":  0.260650474136729,
# "PH001224K_S3_07112025_1612.csv":  0.289019630259768,
# "PH001224K_S4_07112025_1614.csv": 0.28406433348973100,
# "PH001224K_S5_07112025_1615.csv": 0.2633896180669370,
# "PH001224K_S6_07112025_1616.csv":  0.0685073091283367,
# "PH001224K_S7_07112025_1617.csv":  0.0690274025891749,
# "PH001224K_S8_07112025_1618.csv":  0.0651879374886166,
# "PH001224K_S9_07112025_1618.csv":  0.0663987824534293,
# "PH011238E_S1_07112025_1624.csv":  0.0681647211705133,
# "PH011238E_S2_07112025_1624.csv":  0.05694456069566790,
# "PH011238E_S3_07112025_1625.csv":  0.048801457714199200,
# "PH011238E_S4_07112025_1625.csv":  0.06413151774100020,
# "PH011238E_S5_07112025_1627.csv":  0.05578472316446760,
# "PH011238E_S6_07112025_1628.csv":  0.0614231394437556,
# "PH022241H_S1_07112025_1359.csv":  0.399898989898989,
# "PH022241H_S2_07112025_1400.csv":  0.0646056371589984,
# "PH022241H_S3_07112025_1400.csv":  0.318207726037786,
# "PH022241H_S5_07112025_1402.csv":  0.316448577733016,
# "PH022241H_S6_07112025_1404.csv":  0.242087088161428,
# "PH022241H_S7_07112025_1404.csv":  0.0692276008242713,
# "PH022241H_S8_07112025_1405.csv":  0.06920962978485050,
# "PH022241H_S9_07112025_1405.csv":  0.012027793755916600
#     # "24C 15948 A2_S5_06112025_1354.csv": 0.0467207795268307,
#     # ...
# }
# # If you don’t have PA for all 60, it will just evaluate on available subset.

# # ---- Phosphate settings (AUC method) ----
# P_SEARCH_CENTER = 1234.0
# P_SEARCH_HALF_WIDTH = 40.0     # widen if needed
# BASELINE_OFFSET = 70.0
# P_WIDTHS = [34.0, 30.0, 20.0]  # windows to compare

# # ---- Amide settings (AUC method) ----
# A_SEARCH_CENTER = 1655.0
# A_SEARCH_HALF_WIDTH = 30.0
# A_WIDTH = 40.0  # choose something peak-sized; later you can scan like phosphate

# # ---- EntangleCam centres (fixed) ----
# MU_A_PEAK   = 1670.0
# MU_A_TROUGH = 1600.0
# MU_P_PEAK   = 1238.0  # your detected mean ~1238
# MU_P_TROUGH = 1290.0

# # Width scan (EntangleCam Gaussian)
# FWHM_A_SCAN = np.arange(10, 61, 2)
# FWHM_P_SCAN = np.arange(20, 141, 4)

# # Combined loss weights
# ALPHA_RMSE = 1.0
# LAMBDA_STAB = 0.5

# EPS = 1e-12
# LN2 = np.log(2)

# # =========================
# # HELPERS
# # =========================
# def load_2col(path, header=False):
#     df = pd.read_csv(path, sep=None, engine="python", header=0 if header else None)
#     df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
#     wn = df.iloc[:, 0].to_numpy(float)
#     y  = df.iloc[:, 1].to_numpy(float)
#     order = np.argsort(wn)
#     return wn[order], y[order]

# def to_absorbance(y):
#     y = np.asarray(y, float)
#     y = np.clip(y, 1e-12, 1.0)
#     return -np.log10(y)

# def mask_window(wn, mu, width):
#     hw = width / 2.0
#     return (wn >= mu - hw) & (wn <= mu + hw)

# def find_peak_near(wn, y, center, half_width):
#     m = (wn >= center - half_width) & (wn <= center + half_width)
#     if not np.any(m):
#         return np.nan
#     idx = np.argmax(y[m])
#     return float(wn[m][idx])

# def fit_linear_baseline(wn, y, mu, width, offset):
#     mL = mask_window(wn, mu - offset, width)
#     mR = mask_window(wn, mu + offset, width)
#     mb = mL | mR
#     if np.sum(mb) < 2:
#         return np.nan, np.nan, mb
#     X = wn[mb]
#     Y = y[mb]
#     A = np.column_stack([X, np.ones_like(X)])
#     coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
#     return float(coef[0]), float(coef[1]), mb

# def auc_local(wn, y, mu, width, offset, positive_only=True):
#     a, b, mb = fit_linear_baseline(wn, y, mu, width, offset)
#     ms = mask_window(wn, mu, width)
#     if np.sum(ms) < 2 or not np.isfinite(a) or not np.isfinite(b):
#         return np.nan
#     x = wn[ms]
#     yy = y[ms]
#     base = a*x + b
#     diff = yy - base
#     if positive_only:
#         diff = np.maximum(diff, 0.0)
#     return float(np.trapz(diff, x))

# def gaussian_window(wn, mu, fwhm):
#     sigma = fwhm / (2.0 * np.sqrt(2.0 * LN2))
#     return np.exp(-0.5 * ((wn - mu) / (sigma + EPS))**2)

# def weighted_avg(y, w):
#     s = np.sum(w)
#     if s <= 0:
#         return np.nan
#     return float(np.sum(y*w) / s)

# def entanglecam_DI(y, wn, fA, fP):
#     # A channel
#     wAp = gaussian_window(wn, MU_A_PEAK, fA)
#     wAt = gaussian_window(wn, MU_A_TROUGH, fA)
#     Aap = weighted_avg(y, wAp)
#     Aat = weighted_avg(y, wAt)
#     dA = Aap - Aat

#     # P channel
#     wPp = gaussian_window(wn, MU_P_PEAK, fP)
#     wPt = gaussian_window(wn, MU_P_TROUGH, fP)
#     App = weighted_avg(y, wPp)
#     Apt = weighted_avg(y, wPt)
#     dP = App - Apt

#     if not np.isfinite(dA) or abs(dA) < 1e-9:
#         return np.nan, dA, dP
#     return dP / dA, dA, dP

# def mad(x):
#     x = np.asarray(x, float)
#     x = x[np.isfinite(x)]
#     if x.size == 0:
#         return np.nan
#     med = np.median(x)
#     return 1.4826 * np.median(np.abs(x - med))

# def loocv_rmse(x, y):
#     """
#     Fit y ~ m*x + c with LOOCV.
#     Returns rmse, preds
#     """
#     x = np.asarray(x, float).reshape(-1,1)
#     y = np.asarray(y, float)
#     loo = LeaveOneOut()
#     preds = np.full_like(y, np.nan, dtype=float)

#     for tr, te in loo.split(x):
#         model = LinearRegression()
#         model.fit(x[tr], y[tr])
#         preds[te] = model.predict(x[te])

#     rmse = np.sqrt(np.mean((preds - y)**2))
#     return float(rmse), preds

# # =========================
# # MAIN
# # =========================
# files = sorted(glob.glob(os.path.join(DATA_DIR, GLOB_PATTERN)))
# if len(files) < 5:
#     raise SystemExit("Too few spectra found. Check DATA_DIR/GLOB_PATTERN.")

# # load and align to a common wn grid (use first file)
# wn0, y0 = load_2col(files[0], header=HEADER)
# if INPUT_IS_TRANSMISSION:
#     y0 = to_absorbance(y0)
# wn_grid = wn0

# Y = []
# names = []
# for p in files:
#     wn, y = load_2col(p, header=HEADER)
#     if INPUT_IS_TRANSMISSION:
#         y = to_absorbance(y)
#     yg = np.interp(wn_grid, wn, y)
#     Y.append(yg)
#     names.append(os.path.basename(p))
# Y = np.vstack(Y)

# mean_spec = np.mean(Y, axis=0)

# # -------------------------
# # A) AUC-based DI
# # -------------------------
# rows = []
# for i, nm in enumerate(names):
#     y = Y[i]

#     muP = find_peak_near(wn_grid, y, P_SEARCH_CENTER, P_SEARCH_HALF_WIDTH)
#     muA = find_peak_near(wn_grid, y, A_SEARCH_CENTER, A_SEARCH_HALF_WIDTH)

#     rec = {"file": nm, "muP": muP, "muA": muA}

#     # phosphate AUCs at multiple widths
#     for w in P_WIDTHS:
#         rec[f"AUC_P_w{int(w)}"] = auc_local(wn_grid, y, muP, w, BASELINE_OFFSET, positive_only=True)

#     # amide AUC (single width)
#     rec["AUC_A"] = auc_local(wn_grid, y, muA, A_WIDTH, BASELINE_OFFSET, positive_only=True)

#     rows.append(rec)

# df_auc = pd.DataFrame(rows)
# df_auc.to_csv(os.path.join(OUTDIR, "auc_channels.csv"), index=False)

# # Choose one phosphate width as default for DI (start with 34)
# W_USE = 34
# df_auc["DI_AUC_raw"] = df_auc[f"AUC_P_w{W_USE}"] / (df_auc["AUC_A"] + EPS)

# # If PA_true available, evaluate
# pa_true = []
# di_raw = []
# used_files = []
# for nm, di in zip(df_auc["file"], df_auc["DI_AUC_raw"]):
#     if nm in PA_TABLE and np.isfinite(di) and np.isfinite(PA_TABLE[nm]):
#         pa_true.append(PA_TABLE[nm])
#         di_raw.append(di)
#         used_files.append(nm)

# pa_true = np.array(pa_true, float)
# di_raw = np.array(di_raw, float)

# if pa_true.size >= 5:
#     rmse, pa_pred = loocv_rmse(di_raw, pa_true)

#     # plots
#     plt.figure(figsize=(6,6))
#     plt.scatter(pa_true, pa_pred, s=35)
#     lo = min(pa_true.min(), pa_pred.min())
#     hi = max(pa_true.max(), pa_pred.max())
#     plt.plot([lo, hi], [lo, hi])
#     plt.xlabel("PA_true")
#     plt.ylabel("PA_pred (LOOCV)")
#     plt.title(f"AUC-based DI (wP={W_USE}) LOOCV | RMSE={rmse:.4f}")
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTDIR, "auc_DI_loocv_scatter.png"), dpi=200)
#     plt.close()

#     resid = pa_pred - pa_true
#     plt.figure(figsize=(5,5))
#     plt.boxplot(resid, vert=True)
#     plt.axhline(0)
#     plt.ylabel("Residual (PA_pred - PA_true)")
#     plt.title("AUC-based DI residuals (LOOCV)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTDIR, "auc_DI_loocv_residual_box.png"), dpi=200)
#     plt.close()

# # -------------------------
# # B) EntangleCam Gaussian optimisation
# # -------------------------
# # Compute DI for each (fA,fP), then evaluate:
# #  - stability = MAD(DI) / median(|dA|)
# #  - if PA available: LOOCV RMSE after linear calibration PA~DI
# #  - combined loss = ALPHA*rmse + LAMBDA*stability

# # mask for PA-known spectra (same as above but using names)
# pa_mask = np.array([nm in PA_TABLE and np.isfinite(PA_TABLE[nm]) for nm in names])
# pa_vals = np.array([PA_TABLE[nm] if nm in PA_TABLE else np.nan for nm in names], float)

# best = None
# H = np.full((len(FWHM_A_SCAN), len(FWHM_P_SCAN)), np.nan)

# for i, fA in enumerate(FWHM_A_SCAN):
#     for j, fP in enumerate(FWHM_P_SCAN):
#         DI_list = []
#         dA_list = []
#         for k in range(Y.shape[0]):
#             di, dA, dP = entanglecam_DI(Y[k], wn_grid, float(fA), float(fP))
#             DI_list.append(di)
#             dA_list.append(dA)

#         DI_arr = np.array(DI_list, float)
#         dA_arr = np.array(dA_list, float)

#         # stability penalty
#         m = np.isfinite(DI_arr) & np.isfinite(dA_arr)
#         if np.sum(m) < 5:
#             continue
#         stab = mad(DI_arr[m]) / (np.median(np.abs(dA_arr[m])) + EPS)

#         # fit-to-PA term (only on PA-known)
#         mk = m & pa_mask
#         rmse = np.nan
#         if np.sum(mk) >= 5:
#             rmse, _ = loocv_rmse(DI_arr[mk], pa_vals[mk])
#         else:
#             # if too few labels, focus only on stability
#             rmse = 0.0

#         loss = ALPHA_RMSE * rmse + LAMBDA_STAB * stab
#         H[i, j] = loss

#         if best is None or loss < best["loss"]:
#             best = dict(loss=float(loss), rmse=float(rmse), stab=float(stab),
#                         fA=float(fA), fP=float(fP))

# print("BEST EntangleCam:", best)
# pd.DataFrame([best]).to_csv(os.path.join(OUTDIR, "best_entanglecam_params.csv"), index=False)

# # Heatmap of loss
# plt.figure(figsize=(8,6))
# plt.imshow(
#     H, origin="lower", aspect="auto",
#     extent=[FWHM_P_SCAN[0], FWHM_P_SCAN[-1], FWHM_A_SCAN[0], FWHM_A_SCAN[-1]]
# )
# plt.colorbar(label="Combined loss (lower better)")
# plt.scatter([best["fP"]], [best["fA"]], s=80)
# plt.xlabel("FWHM_P (cm$^{-1}$)")
# plt.ylabel("FWHM_A (cm$^{-1}$)")
# plt.title("EntangleCam Gaussian optimisation heatmap")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTDIR, "entanglecam_loss_heatmap.png"), dpi=200)
# plt.close()

# # Overlay best Gaussians on mean spectrum
# fA = best["fA"]
# fP = best["fP"]
# wAp = gaussian_window(wn_grid, MU_A_PEAK, fA)
# wAt = gaussian_window(wn_grid, MU_A_TROUGH, fA)
# wPp = gaussian_window(wn_grid, MU_P_PEAK, fP)
# wPt = gaussian_window(wn_grid, MU_P_TROUGH, fP)

# plt.figure(figsize=(11,5))
# plt.plot(wn_grid, mean_spec, lw=2, label="Mean spectrum")
# scale = (np.nanmax(mean_spec) - np.nanmin(mean_spec)) + EPS
# base = np.nanmin(mean_spec)

# plt.plot(wn_grid, base + scale*(wAp/np.max(wAp)), "--", label="A peak")
# plt.plot(wn_grid, base + scale*(wAt/np.max(wAt)), "--", label="A trough")
# plt.plot(wn_grid, base + scale*(wPp/np.max(wPp)), "--", label="P peak")
# plt.plot(wn_grid, base + scale*(wPt/np.max(wPt)), "--", label="P trough")

# plt.gca().invert_xaxis()
# plt.xlabel("Wavenumber (cm$^{-1}$)")
# plt.ylabel("Absorbance (arb.)")
# plt.title(f"Mean spectrum + best EntangleCam Gaussians (fA={fA:.0f}, fP={fP:.0f})")
# plt.legend(ncol=2)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTDIR, "entanglecam_best_overlay.png"), dpi=220)
# plt.close()

# print("Saved outputs to:", os.path.abspath(OUTDIR))
# import numpy as np
# import matplotlib.pyplot as plt

# LN2 = np.log(2)

# def gaussian_window(wn, mu, fwhm):
#     sigma = fwhm / (2*np.sqrt(2*LN2))
#     return np.exp(-0.5*((wn-mu)/sigma)**2)
# windows = [
#     {"mu":1650.29, "fwhm":80.0, "label":"W1 (amide peak)", "color":"tab:orange"},
#     {"mu":1580.29, "fwhm":80.0, "label":"W4 (amide trough)", "color":"tab:red"},
#     {"mu":1237.80, "fwhm":63.7, "label":"W2 (phosphate peak)", "color":"tab:green"},
# ]

# plot_windows_on_absorbance(
#     wn_grid,         # your wavenumber grid
#     A_spec,          # absorbance spectrum (NOT transmission)
#     windows,
#     title="Optimised EntangleCam windows on absorbance spectrum"
# )


# def plot_windows_on_absorbance(
#         wn,
#         absorbance,
#         windows,     # list of dicts: {"mu":..., "fwhm":..., "label":..., "color":...}
#         title=""
#     ):

#     plt.figure(figsize=(10,5))

#     # absorbance
#     plt.plot(wn, absorbance, lw=2, color="black", label="Absorbance")

#     A_min = np.nanmin(absorbance)
#     A_max = np.nanmax(absorbance)
#     scale = 0.9*(A_max - A_min)

#     for w in windows:

#         g = gaussian_window(wn, w["mu"], w["fwhm"])

#         # purely for display
#         g_plot = A_min + scale * g / g.max()

#         plt.plot(
#             wn,
#             g_plot,
#             lw=2,
#             color=w.get("color", None),
#             label=f'{w["label"]}  (μ={w["mu"]:.1f}, FWHM={w["fwhm"]:.1f})'
#         )

#         # show true FWHM limits
#         plt.axvline(
#             w["mu"] - w["fwhm"]/2,
#             color=w.get("color", None),
#             ls=":",
#             alpha=0.6
#         )
#         plt.axvline(
#             w["mu"] + w["fwhm"]/2,
#             color=w.get("color", None),
#             ls=":",
#             alpha=0.6
#         )

#     plt.gca().invert_xaxis()
#     plt.xlabel("Wavenumber (cm$^{-1}$)")
#     plt.ylabel("Absorbance (arb.)")
#     plt.title(title)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# import os, glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.model_selection import LeaveOneOut
# from sklearn.linear_model import LinearRegression

# # =========================
# # USER SETTINGS
# # =========================
# DATA_DIR = "/Users/nana/Desktop/wavenumber absorption"
# GLOB_PATTERN = "*.CSV"          # adjust if lower-case
# HEADER = False

# # If your second column is transmission: set True
# INPUT_IS_TRANSMISSION = False

# OUTDIR = "steps_outputs"
# os.makedirs(OUTDIR, exist_ok=True)

# # ---- Targets (optional but recommended) ----
# # Put your PA_ratio table here (filename -> PA_true)
# PA_TABLE = {"24C 15948 A2_S1_06112025_1350.csv": 0.046127287406064800,
# "24C 15948 A2_S3_06112025_1352.csv": 0.09198127778163410,
# "24C 15948 A2_S4_06112025_1353.csv": 0.055222094074456500,
# "24C 15948 A2_S5_06112025_1354.csv": 0.04672077952683070,
# "24C 15948 A2_S7_06112025_1355.csv": 0.08752096293176850,
# "24C 15948 A2_S8_06112025_1356.csv": 0.09024791637515490,
# "24C 15948 A2_S9_06112025_1356.csv": 0.0829211562274142,
# "24C 17037_S1_06112025_1422.csv": 0.05172891872512750,
# "24C 17037_S2_06112025_1422.csv": 0.05175957174556670,
# "24C 17037_S4_06112025_1424.csv": 0.07134117960007720,
# "24C 17037_S5_06112025_1425.csv": 0.05394396428685540,
# "24H00050619_S10_07112025_1430.csv": 0.0214109091425551,
# "24H00050619_S1_07112025_1417.csv": 0.02214713351071800,
# "24H00050619_S2_07112025_1418.csv": 0.02380632672201190,
# "24H00050619_S3_07112025_1419.csv": 0.00394983938576703,
# "24H00050619_S4_07112025_1420.csv":  0.0276055352094291,
# "24H00050619_S5_07112025_1421.csv":  0.0265879975568126,
# "24H00050619_S6_07112025_1426.csv":  0.024288,
# "24H00050619_S8_07112025_1428.csv": 0.05877514968951400,
# "24H00050619_S9_07112025_1429.csv": 4.99069207984537E-08,
# "PH001224K_S10_07112025_1618.csv":  0.0642527836305066,
# "PH001224K_S2_07112025_1611.csv":  0.260650474136729,
# "PH001224K_S3_07112025_1612.csv":  0.289019630259768,
# "PH001224K_S4_07112025_1614.csv": 0.28406433348973100,
# "PH001224K_S5_07112025_1615.csv": 0.2633896180669370,
# "PH001224K_S6_07112025_1616.csv":  0.0685073091283367,
# "PH001224K_S7_07112025_1617.csv":  0.0690274025891749,
# "PH001224K_S8_07112025_1618.csv":  0.0651879374886166,
# "PH001224K_S9_07112025_1618.csv":  0.0663987824534293,
# "PH011238E_S1_07112025_1624.csv":  0.0681647211705133,
# "PH011238E_S2_07112025_1624.csv":  0.05694456069566790,
# "PH011238E_S3_07112025_1625.csv":  0.048801457714199200,
# "PH011238E_S4_07112025_1625.csv":  0.06413151774100020,
# "PH011238E_S5_07112025_1627.csv":  0.05578472316446760,
# "PH011238E_S6_07112025_1628.csv":  0.0614231394437556,
# "PH022241H_S1_07112025_1359.csv":  0.399898989898989,
# "PH022241H_S2_07112025_1400.csv":  0.0646056371589984,
# "PH022241H_S3_07112025_1400.csv":  0.318207726037786,
# "PH022241H_S5_07112025_1402.csv":  0.316448577733016,
# "PH022241H_S6_07112025_1404.csv":  0.242087088161428,
# "PH022241H_S7_07112025_1404.csv":  0.0692276008242713,
# "PH022241H_S8_07112025_1405.csv":  0.06920962978485050,
# "PH022241H_S9_07112025_1405.csv":  0.012027793755916600
#     # "24C 15948 A2_S5_06112025_1354.csv": 0.0467207795268307,
#     # ...
# }
# # If you don’t have PA for all 60, it will just evaluate on available subset.

# # ---- Phosphate settings (AUC method) ----
# P_SEARCH_CENTER = 1238.0
# P_SEARCH_HALF_WIDTH = 40.0     # widen if needed
# BASELINE_OFFSET = 70.0
# P_WIDTHS = [34.0, 30.0, 20.0]  # windows to compare

# # ---- Amide settings (AUC method) ----
# A_SEARCH_CENTER = 1650.0
# A_SEARCH_HALF_WIDTH = 30.0
# A_WIDTH = 40.0  # choose something peak-sized; later you can scan like phosphate

# # ---- EntangleCam centres (fixed) ----
# MU_A_PEAK   = 1670.0
# MU_A_TROUGH = 1600.0
# MU_P_PEAK   = 1238.0  # your detected mean ~1238
# MU_P_TROUGH = 1290.0

# # Width scan (EntangleCam Gaussian)
# FWHM_A_SCAN = np.arange(10, 61, 2)
# FWHM_P_SCAN = np.arange(20, 141, 4)

# # Combined loss weights
# ALPHA_RMSE = 1.0
# LAMBDA_STAB = 0.5

# EPS = 1e-12
# LN2 = np.log(2)

# # =========================
# # HELPERS
# # =========================
# def load_2col(path, header=False):
#     df = pd.read_csv(path, sep=None, engine="python", header=0 if header else None)
#     df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
#     wn = df.iloc[:, 0].to_numpy(float)
#     y  = df.iloc[:, 1].to_numpy(float)
#     order = np.argsort(wn)
#     return wn[order], y[order]

# def to_absorbance(y):
#     y = np.asarray(y, float)
#     y = np.clip(y, 1e-12, 1.0)
#     return -np.log10(y)

# def mask_window(wn, mu, width):
#     hw = width / 2.0
#     return (wn >= mu - hw) & (wn <= mu + hw)

# def find_peak_near(wn, y, center, half_width):
#     m = (wn >= center - half_width) & (wn <= center + half_width)
#     if not np.any(m):
#         return np.nan
#     idx = np.argmax(y[m])
#     return float(wn[m][idx])

# def fit_linear_baseline(wn, y, mu, width, offset):
#     mL = mask_window(wn, mu - offset, width)
#     mR = mask_window(wn, mu + offset, width)
#     mb = mL | mR
#     if np.sum(mb) < 2:
#         return np.nan, np.nan, mb
#     X = wn[mb]
#     Y = y[mb]
#     A = np.column_stack([X, np.ones_like(X)])
#     coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
#     return float(coef[0]), float(coef[1]), mb

# def auc_local(wn, y, mu, width, offset, positive_only=True):
#     a, b, mb = fit_linear_baseline(wn, y, mu, width, offset)
#     ms = mask_window(wn, mu, width)
#     if np.sum(ms) < 2 or not np.isfinite(a) or not np.isfinite(b):
#         return np.nan
#     x = wn[ms]
#     yy = y[ms]
#     base = a*x + b
#     diff = yy - base
#     if positive_only:
#         diff = np.maximum(diff, 0.0)
#     return float(np.trapz(diff, x))

# def gaussian_window(wn, mu, fwhm):
#     sigma = fwhm / (2.0 * np.sqrt(2.0 * LN2))
#     return np.exp(-0.5 * ((wn - mu) / (sigma + EPS))**2)

# def weighted_avg(y, w):
#     s = np.sum(w)
#     if s <= 0:
#         return np.nan
#     return float(np.sum(y*w) / s)

# def entanglecam_DI(y, wn, fA, fP):
#     # A channel
#     wAp = gaussian_window(wn, MU_A_PEAK, fA)
#     wAt = gaussian_window(wn, MU_A_TROUGH, fA)
#     Aap = weighted_avg(y, wAp)
#     Aat = weighted_avg(y, wAt)
#     dA = Aap - Aat

#     # P channel
#     wPp = gaussian_window(wn, MU_P_PEAK, fP)
#     wPt = gaussian_window(wn, MU_P_TROUGH, fP)
#     App = weighted_avg(y, wPp)
#     Apt = weighted_avg(y, wPt)
#     dP = App - Apt

#     if not np.isfinite(dA) or abs(dA) < 1e-9:
#         return np.nan, dA, dP
#     return dP / dA, dA, dP

# def mad(x):
#     x = np.asarray(x, float)
#     x = x[np.isfinite(x)]
#     if x.size == 0:
#         return np.nan
#     med = np.median(x)
#     return 1.4826 * np.median(np.abs(x - med))

# def loocv_rmse(x, y):
#     """
#     Fit y ~ m*x + c with LOOCV.
#     Returns rmse, preds
#     """
#     x = np.asarray(x, float).reshape(-1,1)
#     y = np.asarray(y, float)
#     loo = LeaveOneOut()
#     preds = np.full_like(y, np.nan, dtype=float)

#     for tr, te in loo.split(x):
#         model = LinearRegression()
#         model.fit(x[tr], y[tr])
#         preds[te] = model.predict(x[te])

#     rmse = np.sqrt(np.mean((preds - y)**2))
#     return float(rmse), preds

# # =========================
# # MAIN
# # =========================
# files = sorted(glob.glob(os.path.join(DATA_DIR, GLOB_PATTERN)))
# if len(files) < 5:
#     raise SystemExit("Too few spectra found. Check DATA_DIR/GLOB_PATTERN.")

# # load and align to a common wn grid (use first file)
# wn0, y0 = load_2col(files[0], header=HEADER)
# if INPUT_IS_TRANSMISSION:
#     y0 = to_absorbance(y0)
# wn_grid = wn0

# Y = []
# names = []
# for p in files:
#     wn, y = load_2col(p, header=HEADER)
#     if INPUT_IS_TRANSMISSION:
#         y = to_absorbance(y)
#     yg = np.interp(wn_grid, wn, y)
#     Y.append(yg)
#     names.append(os.path.basename(p))
# Y = np.vstack(Y)

# mean_spec = np.mean(Y, axis=0)

# # -------------------------
# # A) AUC-based DI
# # -------------------------
# rows = []
# for i, nm in enumerate(names):
#     y = Y[i]

#     muP = find_peak_near(wn_grid, y, P_SEARCH_CENTER, P_SEARCH_HALF_WIDTH)
#     muA = find_peak_near(wn_grid, y, A_SEARCH_CENTER, A_SEARCH_HALF_WIDTH)

#     rec = {"file": nm, "muP": muP, "muA": muA}

#     # phosphate AUCs at multiple widths
#     for w in P_WIDTHS:
#         rec[f"AUC_P_w{int(w)}"] = auc_local(wn_grid, y, muP, w, BASELINE_OFFSET, positive_only=True)

#     # amide AUC (single width)
#     rec["AUC_A"] = auc_local(wn_grid, y, muA, A_WIDTH, BASELINE_OFFSET, positive_only=True)

#     rows.append(rec)

# df_auc = pd.DataFrame(rows)
# df_auc.to_csv(os.path.join(OUTDIR, "auc_channels.csv"), index=False)

# # Choose one phosphate width as default for DI (start with 34)
# W_USE = 34
# df_auc["DI_AUC_raw"] = df_auc[f"AUC_P_w{W_USE}"] / (df_auc["AUC_A"] + EPS)

# # If PA_true available, evaluate
# pa_true = []
# di_raw = []
# used_files = []
# for nm, di in zip(df_auc["file"], df_auc["DI_AUC_raw"]):
#     if nm in PA_TABLE and np.isfinite(di) and np.isfinite(PA_TABLE[nm]):
#         pa_true.append(PA_TABLE[nm])
#         di_raw.append(di)
#         used_files.append(nm)

# pa_true = np.array(pa_true, float)
# di_raw = np.array(di_raw, float)

# if pa_true.size >= 5:
#     rmse, pa_pred = loocv_rmse(di_raw, pa_true)

#     # plots
#     plt.figure(figsize=(6,6))
#     plt.scatter(pa_true, pa_pred, s=35)
#     lo = min(pa_true.min(), pa_pred.min())
#     hi = max(pa_true.max(), pa_pred.max())
#     plt.plot([lo, hi], [lo, hi])
#     plt.xlabel("PA_true")
#     plt.ylabel("PA_pred (LOOCV)")
#     plt.title(f"AUC-based DI (wP={W_USE}) LOOCV | RMSE={rmse:.4f}")
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTDIR, "auc_DI_loocv_scatter.png"), dpi=200)
#     plt.close()

#     resid = pa_pred - pa_true
#     plt.figure(figsize=(5,5))
#     plt.boxplot(resid, vert=True)
#     plt.axhline(0)
#     plt.ylabel("Residual (PA_pred - PA_true)")
#     plt.title("AUC-based DI residuals (LOOCV)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTDIR, "auc_DI_loocv_residual_box.png"), dpi=200)
#     plt.close()

# # -------------------------
# # B) EntangleCam Gaussian optimisation
# # -------------------------
# # Compute DI for each (fA,fP), then evaluate:
# #  - stability = MAD(DI) / median(|dA|)
# #  - if PA available: LOOCV RMSE after linear calibration PA~DI
# #  - combined loss = ALPHA*rmse + LAMBDA*stability

# # mask for PA-known spectra (same as above but using names)
# pa_mask = np.array([nm in PA_TABLE and np.isfinite(PA_TABLE[nm]) for nm in names])
# pa_vals = np.array([PA_TABLE[nm] if nm in PA_TABLE else np.nan for nm in names], float)

# best = None
# H = np.full((len(FWHM_A_SCAN), len(FWHM_P_SCAN)), np.nan)

# for i, fA in enumerate(FWHM_A_SCAN):
#     for j, fP in enumerate(FWHM_P_SCAN):
#         DI_list = []
#         dA_list = []
#         for k in range(Y.shape[0]):
#             di, dA, dP = entanglecam_DI(Y[k], wn_grid, float(fA), float(fP))
#             DI_list.append(di)
#             dA_list.append(dA)

#         DI_arr = np.array(DI_list, float)
#         dA_arr = np.array(dA_list, float)

#         # stability penalty
#         m = np.isfinite(DI_arr) & np.isfinite(dA_arr)
#         if np.sum(m) < 5:
#             continue
#         stab = mad(DI_arr[m]) / (np.median(np.abs(dA_arr[m])) + EPS)

#         # fit-to-PA term (only on PA-known)
#         mk = m & pa_mask
#         rmse = np.nan
#         if np.sum(mk) >= 5:
#             rmse, _ = loocv_rmse(DI_arr[mk], pa_vals[mk])
#         else:
#             # if too few labels, focus only on stability
#             rmse = 0.0

#         loss = ALPHA_RMSE * rmse + LAMBDA_STAB * stab
#         H[i, j] = loss

#         if best is None or loss < best["loss"]:
#             best = dict(loss=float(loss), rmse=float(rmse), stab=float(stab),
#                         fA=float(fA), fP=float(fP))

# print("BEST EntangleCam:", best)
# pd.DataFrame([best]).to_csv(os.path.join(OUTDIR, "best_entanglecam_params.csv"), index=False)

# # Heatmap of loss
# plt.figure(figsize=(8,6))
# plt.imshow(
#     H, origin="lower", aspect="auto",
#     extent=[FWHM_P_SCAN[0], FWHM_P_SCAN[-1], FWHM_A_SCAN[0], FWHM_A_SCAN[-1]]
# )
# plt.colorbar(label="Combined loss (lower better)")
# plt.scatter([best["fP"]], [best["fA"]], s=80)
# plt.xlabel("FWHM_P (cm$^{-1}$)")
# plt.ylabel("FWHM_A (cm$^{-1}$)")
# plt.title("EntangleCam Gaussian optimisation heatmap")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTDIR, "entanglecam_loss_heatmap.png"), dpi=200)
# plt.close()

# # Overlay best Gaussians on mean spectrum
# fA = best["fA"]
# fP = best["fP"]
# wAp = gaussian_window(wn_grid, MU_A_PEAK, fA)
# wAt = gaussian_window(wn_grid, MU_A_TROUGH, fA)
# wPp = gaussian_window(wn_grid, MU_P_PEAK, fP)
# wPt = gaussian_window(wn_grid, MU_P_TROUGH, fP)

# plt.figure(figsize=(11,5))
# plt.plot(wn_grid, mean_spec, lw=2, label="Mean spectrum")
# scale = (np.nanmax(mean_spec) - np.nanmin(mean_spec)) + EPS
# base = np.nanmin(mean_spec)

# plt.plot(wn_grid, base + scale*(wAp/np.max(wAp)), "--", label="A peak")
# plt.plot(wn_grid, base + scale*(wAt/np.max(wAt)), "--", label="A trough")
# plt.plot(wn_grid, base + scale*(wPp/np.max(wPp)), "--", label="P peak")
# plt.plot(wn_grid, base + scale*(wPt/np.max(wPt)), "--", label="P trough")

# plt.gca().invert_xaxis()
# plt.xlabel("Wavenumber (cm$^{-1}$)")
# plt.ylabel("Absorbance (arb.)")
# plt.title(f"Mean spectrum + best EntangleCam Gaussians (fA={fA:.0f}, fP={fP:.0f})")
# plt.legend(ncol=2)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTDIR, "entanglecam_best_overlay.png"), dpi=220)
# plt.close()

# print("Saved outputs to:", os.path.abspath(OUTDIR))
# import os, glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # ==========================
# # USER SETTINGS
# # ==========================
# DATA_GLOB = "/Users/nana/Desktop/wavenumber absorption/*.CSV"  # adjust
# OUTDIR = "auc_1234_baseline_results"
# os.makedirs(OUTDIR, exist_ok=True)

# # Target band settings
# PEAK_SEARCH_RANGE = (1210, 1260)   # where to find the "1234-ish" peak
# BASELINE_OFFSET = 70.0            # ±70 cm^-1 from signal centre
# WINDOW_WIDTHS = [34.0, 30.0, 20.0] # cm^-1

# # Crop for speed/robustness (optional)
# CROP_MIN, CROP_MAX = 1000.0, 1800.0

# # If your CSVs have headers, set True
# HEADER = False

# # Clamp negative baseline-corrected absorbance to 0 before integrating
# CLAMP_NEGATIVE = True


# # ==========================
# # IO helpers
# # ==========================
# def load_2col(path):
#     """
#     Robust 2-col loader handling:
#     - comma-separated "a,b"
#     - tab/space separated
#     - with/without header
#     """
#     if HEADER:
#         df = pd.read_csv(path, sep=None, engine="python")
#         arr = df.values
#     else:
#         # try comma first; if fails, fallback to whitespace
#         try:
#             arr = np.loadtxt(path, delimiter=",")
#             if arr.ndim == 1 or arr.shape[1] < 2:
#                 raise ValueError
#         except Exception:
#             arr = np.loadtxt(path, delimiter=None)

#     if arr.ndim != 2 or arr.shape[1] < 2:
#         raise ValueError(f"Expected 2 columns in {path}, got {arr.shape}")

#     wn = arr[:, 0].astype(float)
#     A  = arr[:, 1].astype(float)

#     m = np.isfinite(wn) & np.isfinite(A)
#     wn, A = wn[m], A[m]

#     # sort ascending for safe interpolation/integration
#     idx = np.argsort(wn)
#     wn, A = wn[idx], A[idx]

#     # optional crop
#     m2 = (wn >= CROP_MIN) & (wn <= CROP_MAX)
#     wn, A = wn[m2], A[m2]

#     return wn, A


# # ==========================
# # Spectral utilities
# # ==========================
# def find_local_peak_mu(wn, A, lo, hi):
#     m = (wn >= lo) & (wn <= hi)
#     if np.sum(m) < 5:
#         return np.nan
#     i = np.argmax(A[m])
#     return float(wn[m][i])

# def window_mask(wn, center, width):
#     half = 0.5 * width
#     return (wn >= center - half) & (wn <= center + half)

# def fit_linear_baseline(wn, A, mu, w, offset=70.0):
#     """
#     Fit baseline line A_base = a*wn + b using points
#     in two baseline windows located at mu±offset with same width w.
#     """
#     left_c  = mu - offset
#     right_c = mu + offset

#     mL = window_mask(wn, left_c,  w)
#     mR = window_mask(wn, right_c, w)
#     m = mL | mR

#     if np.sum(m) < 4:
#         return None  # not enough points

#     x = wn[m]
#     y = A[m]
#     # least squares for y = a x + b
#     X = np.vstack([x, np.ones_like(x)]).T
#     a, b = np.linalg.lstsq(X, y, rcond=None)[0]
#     return float(a), float(b), mL, mR

# def auc_after_baseline(wn, A, mu, w, offset=70.0, clamp_negative=True):
#     """
#     AUC over signal window after subtracting locally fitted baseline line.
#     """
#     fit = fit_linear_baseline(wn, A, mu, w, offset=offset)
#     if fit is None:
#         return np.nan, None

#     a, b, mL, mR = fit
#     mS = window_mask(wn, mu, w)

#     if np.sum(mS) < 4:
#         return np.nan, None

#     xS = wn[mS]
#     yS = A[mS]
#     yBase = a * xS + b
#     yCorr = yS - yBase
#     if clamp_negative:
#         yCorr = np.maximum(yCorr, 0.0)

#     auc = float(np.trapz(yCorr, xS))
#     meta = {
#         "a": a, "b": b,
#         "mS": mS, "mL": mL, "mR": mR,
#         "xS": xS, "yS": yS,
#         "yBase": yBase,
#         "yCorr": yCorr
#     }
#     return auc, meta


# # ==========================
# # Main run
# # ==========================
# paths = sorted(glob.glob(DATA_GLOB))
# if not paths:
#     raise FileNotFoundError(f"No files matched: {DATA_GLOB}")

# rows = []
# peak_centres = []

# # store example for plotting (first valid file)
# example = None

# for p in paths:
#     wn, A = load_2col(p)

#     mu = find_local_peak_mu(wn, A, *PEAK_SEARCH_RANGE)
#     peak_centres.append(mu)

#     rec = {"file": os.path.basename(p), "mu_peak": mu}

#     for w in WINDOW_WIDTHS:
#         auc, meta = auc_after_baseline(
#             wn, A, mu, w,
#             offset=BASELINE_OFFSET,
#             clamp_negative=CLAMP_NEGATIVE
#         )
#         rec[f"AUC_w{int(w)}"] = auc

#         if example is None and np.isfinite(auc):
#             example = {
#                 "file": os.path.basename(p),
#                 "wn": wn,
#                 "A": A,
#                 "mu": mu,
#                 "meta_by_w": {w: meta}
#             }

#     rows.append(rec)

# df = pd.DataFrame(rows)
# df.to_csv(os.path.join(OUTDIR, "AUC_table.csv"), index=False)
# print(f"Saved table: {os.path.join(OUTDIR, 'AUC_table.csv')}")

# # ==========================
# # Derived metrics (information retained / lost)
# # ==========================
# # choose 34 as reference
# df["retained_20_over_34"] = df["AUC_w20"] / df["AUC_w34"]
# df["loss_20_vs_34"] = 1.0 - df["retained_20_over_34"]

# df.to_csv(os.path.join(OUTDIR, "AUC_table_with_loss.csv"), index=False)

# # ==========================
# # Plot 1: Boxplot of AUC by window width
# # ==========================
# vals34 = df["AUC_w34"].values
# vals30 = df["AUC_w30"].values
# vals20 = df["AUC_w20"].values

# data_box = [vals34[np.isfinite(vals34)], vals30[np.isfinite(vals30)], vals20[np.isfinite(vals20)]]

# plt.figure(figsize=(10, 6))
# plt.boxplot(data_box, labels=["34", "30", "20"], showfliers=True)
# plt.xlabel("Window width (cm$^{-1}$)")
# plt.ylabel("AUC (arb.·cm$^{-1}$)")
# plt.title("AUC after local linear baseline subtraction (1234 band)")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTDIR, "auc_boxplot_by_width.png"), dpi=200)
# plt.show()

# # ==========================
# # Plot 2: AUC(20) vs AUC(34) + y=x
# # ==========================
# m = np.isfinite(df["AUC_w34"]) & np.isfinite(df["AUC_w20"])
# x = df.loc[m, "AUC_w34"].values
# y = df.loc[m, "AUC_w20"].values

# plt.figure(figsize=(8, 7))
# plt.scatter(x, y)
# if x.size:
#     lo = min(x.min(), y.min())
#     hi = max(x.max(), y.max())
#     plt.plot([lo, hi], [lo, hi])
# plt.xlabel("AUC width 34")
# plt.ylabel("AUC width 20")
# plt.title("Information retained: AUC(20) vs AUC(34)")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTDIR, "auc20_vs_auc34.png"), dpi=200)
# plt.show()

# # ==========================
# # Plot 3: Information loss histogram
# # ==========================
# loss = df["loss_20_vs_34"].values
# loss = loss[np.isfinite(loss)]

# plt.figure(figsize=(10, 6))
# plt.hist(loss, bins=20, edgecolor="black")
# plt.xlabel("Loss fraction = 1 - AUC(20)/AUC(34)")
# plt.ylabel("Count")
# plt.title("Information loss distribution")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTDIR, "loss_hist_w20_vs_34.png"), dpi=200)
# plt.show()

# # ==========================
# # Plot 4: Peak centre histogram
# # ==========================
# mu_arr = np.array(peak_centres, float)
# mu_arr = mu_arr[np.isfinite(mu_arr)]

# plt.figure(figsize=(10, 6))
# plt.hist(mu_arr, bins=20, edgecolor="black")
# plt.xlabel("Detected peak centre near 1234 (cm$^{-1}$)")
# plt.ylabel("Count")
# plt.title("Per-spectrum peak centre distribution")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTDIR, "mu_peak_hist.png"), dpi=200)
# plt.show()

# # ==========================
# # Plot 5: Example overlay with baselines + AUC areas
# # ==========================
# if example is not None:
#     wn = example["wn"]
#     A  = example["A"]
#     mu = example["mu"]

#     plt.figure(figsize=(12, 6))
#     plt.plot(wn, A, lw=2, label="Spectrum (example)")

#     # show each width's baseline + shaded AUC
#     for w in WINDOW_WIDTHS:
#         meta = example["meta_by_w"].get(w, None)
#         if meta is None:
#             continue

#         # baseline line over signal segment
#         xS = meta["xS"]; yBase = meta["yBase"]; yCorr = meta["yCorr"]

#         plt.plot(xS, yBase, lw=2, label=f"Baseline fit (w={int(w)})")
#         # shade area between baseline and curve where yCorr>0
#         plt.fill_between(xS, yBase, yBase + yCorr, alpha=0.2, label=f"AUC area (w={int(w)})")

#     plt.axvline(mu, ls="--", lw=1.5, label=f"Peak centre ~ {mu:.2f}")

#     # IR convention
#     plt.gca().invert_xaxis()

#     plt.xlabel("Wavenumber (cm$^{-1}$)")
#     plt.ylabel("Absorbance (arb.)")
#     plt.title(f"Example baseline-corrected AUC around 1234 band\n{example['file']}")
#     plt.legend(ncol=2, fontsize=9)
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTDIR, "example_auc_overlay.png"), dpi=200)
#     plt.show()

# # ==========================
# # Plot 6: Window geometry on mean spectrum
# # ==========================
# # Build mean spectrum by interpolating to a common grid:
# # Use first spectrum as reference grid
# wn0, A0 = load_2col(paths[0])
# grid = wn0.copy()
# specs = []
# mus = []

# for p in paths:
#     wn, A = load_2col(p)
#     A_g = np.interp(grid, wn, A)
#     specs.append(A_g)
#     mus.append(find_local_peak_mu(wn, A, *PEAK_SEARCH_RANGE))

# mean_spec = np.mean(np.vstack(specs), axis=0)
# mu_mean = float(np.nanmean(np.array(mus, float)))

# plt.figure(figsize=(12, 6))
# plt.plot(grid, mean_spec, lw=2, label="Mean spectrum")
# plt.axvline(mu_mean, ls="--", lw=1.5, label=f"Mean peak centre ~ {mu_mean:.2f}")

# for w in WINDOW_WIDTHS:
#     # signal band
#     s_lo = mu_mean - 0.5*w
#     s_hi = mu_mean + 0.5*w
#     plt.axvspan(s_lo, s_hi, alpha=0.12, label=f"Signal win {int(w)}")

# # baseline windows for w=34 (geometry illustration)
# w = 34.0
# for c in [mu_mean-BASELINE_OFFSET, mu_mean+BASELINE_OFFSET]:
#     b_lo = c - 0.5*w
#     b_hi = c + 0.5*w
#     plt.axvspan(b_lo, b_hi, alpha=0.08)

# plt.gca().invert_xaxis()
# plt.xlabel("Wavenumber (cm$^{-1}$)")
# plt.ylabel("Absorbance (arb.)")
# plt.title("Window geometry on mean spectrum (signal + baseline windows)")
# plt.legend(ncol=3, fontsize=9)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTDIR, "mean_geometry.png"), dpi=200)
# plt.show()

# print("\nDone. Key outputs:")
# print(" - AUC_table_with_loss.csv")
# print(" - auc_boxplot_by_width.png")
# print(" - auc20_vs_auc34.png")
# print(" - loss_hist_w20_vs_34.png")
# print(" - mu_peak_hist.png")
# print(" - example_auc_overlay.png")
# print(" - mean_geometry.png")
"""
FULL PIPELINE (single script)
-----------------------------
What you get (all in ABSORBANCE space):
1) Robust loading of 60 CSV spectra (comma/tab/space; header/no-header).
2) Optional transmission -> absorbance conversion (A = -log10(T)).
3) AUC method (local linear baseline subtraction) for:
   - phosphate band near ~1234 (peak detected per spectrum)
   - amide band near ~1650 (peak detected per spectrum)
   with window widths you choose (e.g. 34/30/20 for phosphate).
4) "AUC ratio" PA_pred_AUC = AUC_P / AUC_A (absorbance-area ratio).
5) EntangleCam fixed-centre Gaussian window method (Beer-law style):
   - fixed centres: MU_A_PEAK=1670, MU_A_TROUGH=1600, MU_P_PEAK=1238, MU_P_TROUGH=1290
   - scan FWHM_A and FWHM_P
   - stability metric = MAD(DI)/median(|ΔA|)  (lower better)
   - if PA labels exist: LOOCV calibration PA_true ~ m*DI + c; compute RMSE
   - combined loss = ALPHA_RMSE*RMSE + LAMBDA_STAB*stability
6) Visualisations:
   - AUC boxplot vs width (phosphate)
   - AUC(20) vs AUC(34) and info loss histogram
   - peak centre histogram (~1234)
   - example spectrum with baseline lines + shaded AUCs
   - mean spectrum with signal+baseline window geometry
   - EntangleCam heatmap (combined loss)
   - overlay best EntangleCam Gaussians ON MEAN ABSORBANCE (with true FWHM lines)
   - best EntangleCam: LOOCV scatter + residuals boxplot (if enough PA labels)
7) Tables saved:
   - auc_channels.csv (all files)
   - auc_table_with_loss.csv
   - entanglecam_best_params.csv
   - entanglecam_DI_table_best.csv  (DI per spectrum for best params)
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression

# =========================
# USER SETTINGS
# =========================
DATA_DIR = "/Users/nana/Desktop/wavenumber absorption"
GLOB_PATTERN = "*.CSV"          # adjust if needed (e.g. "*.csv")
HEADER = False                  # True if files have a header row

# If your second column is transmission: set True
INPUT_IS_TRANSMISSION = False

OUTDIR = "steps_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ---- Targets (optional) ----
PA_TABLE = {
    "24C 15948 A2_S1_06112025_1350.csv": 0.046127287406064800,
    "24C 15948 A2_S3_06112025_1352.csv": 0.09198127778163410,
    "24C 15948 A2_S4_06112025_1353.csv": 0.055222094074456500,
    "24C 15948 A2_S5_06112025_1354.csv": 0.04672077952683070,
    "24C 15948 A2_S7_06112025_1355.csv": 0.08752096293176850,
    "24C 15948 A2_S8_06112025_1356.csv": 0.09024791637515490,
    "24C 15948 A2_S9_06112025_1356.csv": 0.08292115622741420,
    "24C 17037_S1_06112025_1422.csv": 0.05172891872512750,
    "24C 17037_S2_06112025_1422.csv": 0.05175957174556670,
    "24C 17037_S4_06112025_1424.csv": 0.07134117960007720,
    "24C 17037_S5_06112025_1425.csv": 0.05394396428685540,
    "24H00050619_S10_07112025_1430.csv": 0.0214109091425551,
    "24H00050619_S1_07112025_1417.csv": 0.02214713351071800,
    "24H00050619_S2_07112025_1418.csv": 0.02380632672201190,
    "24H00050619_S3_07112025_1419.csv": 0.00394983938576703,
    "24H00050619_S4_07112025_1420.csv": 0.0276055352094291,
    "24H00050619_S5_07112025_1421.csv": 0.0265879975568126,
    "24H00050619_S6_07112025_1426.csv": 0.024288,
    "24H00050619_S8_07112025_1428.csv": 0.05877514968951400,
    "24H00050619_S9_07112025_1429.csv": 4.99069207984537E-08,
    "PH001224K_S10_07112025_1618.csv": 0.0642527836305066,
    "PH001224K_S2_07112025_1611.csv": 0.260650474136729,
    "PH001224K_S3_07112025_1612.csv": 0.289019630259768,
    "PH001224K_S4_07112025_1614.csv": 0.28406433348973100,
    "PH001224K_S5_07112025_1615.csv": 0.2633896180669370,
    "PH001224K_S6_07112025_1616.csv": 0.0685073091283367,
    "PH001224K_S7_07112025_1617.csv": 0.0690274025891749,
    "PH001224K_S8_07112025_1618.csv": 0.0651879374886166,
    "PH001224K_S9_07112025_1618.csv": 0.0663987824534293,
    "PH011238E_S1_07112025_1624.csv": 0.0681647211705133,
    "PH011238E_S2_07112025_1624.csv": 0.05694456069566790,
    "PH011238E_S3_07112025_1625.csv": 0.048801457714199200,
    "PH011238E_S4_07112025_1625.csv": 0.06413151774100020,
    "PH011238E_S5_07112025_1627.csv": 0.05578472316446760,
    "PH011238E_S6_07112025_1628.csv": 0.0614231394437556,
    "PH022241H_S1_07112025_1359.csv": 0.399898989898989,
    "PH022241H_S2_07112025_1400.csv": 0.0646056371589984,
    "PH022241H_S3_07112025_1400.csv": 0.318207726037786,
    "PH022241H_S5_07112025_1402.csv": 0.316448577733016,
    "PH022241H_S6_07112025_1404.csv": 0.242087088161428,
    "PH022241H_S7_07112025_1404.csv": 0.0692276008242713,
    "PH022241H_S8_07112025_1405.csv": 0.06920962978485050,
    "PH022241H_S9_07112025_1405.csv": 0.012027793755916600,
}

# -------------------------
# AUC method settings
# -------------------------
# phosphate band (1234-ish)
P_SEARCH_CENTER = 1238.0
P_SEARCH_HALF_WIDTH = 40.0
BASELINE_OFFSET = 70.0
P_WIDTHS = [34.0, 30.0, 20.0]  # compare how much area retained

# amide band (1650-ish)
A_SEARCH_CENTER = 1650.0
A_SEARCH_HALF_WIDTH = 40.0
A_WIDTH = 40.0  # you can scan this too if you want

# Crop (optional): keep only the useful region
CROP_MIN, CROP_MAX = 1000.0, 1750.0

# -------------------------
# EntangleCam fixed centres (Beer-law DI)
# -------------------------
MU_A_PEAK   = 1670.0
MU_A_TROUGH = 1600.0
MU_P_PEAK   = 1238.0
MU_P_TROUGH = 1290.0

FWHM_A_SCAN = np.arange(10, 61, 2)
FWHM_P_SCAN = np.arange(20, 141, 4)

# Combined loss weights (for EntangleCam optimisation)
ALPHA_RMSE = 1.0
LAMBDA_STAB = 0.5

# Numerics
EPS = 1e-12
LN2 = np.log(2)


# =========================
# HELPERS
# =========================
def load_2col(path, header=False):
    """
    Robust 2-column numeric reader handling:
      - comma-separated: "3.99e2,1.09e-1"
      - tab/space separated
      - with/without header
    """
    if header:
        df = pd.read_csv(path, sep=None, engine="python")
        arr = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna().values
        wn = arr[:, 0].astype(float)
        y  = arr[:, 1].astype(float)
    else:
        # try comma first; if fails fallback to whitespace
        try:
            arr = np.loadtxt(path, delimiter=",")
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError
        except Exception:
            arr = np.loadtxt(path, delimiter=None)
        wn = arr[:, 0].astype(float)
        y  = arr[:, 1].astype(float)

    m = np.isfinite(wn) & np.isfinite(y)
    wn, y = wn[m], y[m]
    idx = np.argsort(wn)
    wn, y = wn[idx], y[idx]

    # optional crop
    mc = (wn >= CROP_MIN) & (wn <= CROP_MAX)
    wn, y = wn[mc], y[mc]

    return wn, y


def to_absorbance(T):
    T = np.asarray(T, float)
    T = np.clip(T, 1e-12, 1.0)
    return -np.log10(T)


def mask_window(wn, mu, width):
    hw = width / 2.0
    return (wn >= mu - hw) & (wn <= mu + hw)


def find_peak_near(wn, y, center, half_width):
    m = (wn >= center - half_width) & (wn <= center + half_width)
    if np.sum(m) < 5:
        return np.nan
    idx = np.argmax(y[m])
    return float(wn[m][idx])


def fit_linear_baseline(wn, y, mu, width, offset):
    """
    Fit baseline y_base = a*wn + b using baseline windows at mu±offset, same width.
    """
    mL = mask_window(wn, mu - offset, width)
    mR = mask_window(wn, mu + offset, width)
    mb = mL | mR
    if np.sum(mb) < 4:
        return np.nan, np.nan, mb, mL, mR

    X = wn[mb]
    Y = y[mb]
    A = np.column_stack([X, np.ones_like(X)])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b, mb, mL, mR


def auc_local(wn, y, mu, width, offset, positive_only=True):
    """
    AUC of (y - baseline) over the signal window at mu with width.
    Baseline fitted from windows at mu±offset.
    """
    a, b, mb, mL, mR = fit_linear_baseline(wn, y, mu, width, offset)
    ms = mask_window(wn, mu, width)

    if np.sum(ms) < 4 or not np.isfinite(a) or not np.isfinite(b):
        return np.nan, None

    x = wn[ms]
    yy = y[ms]
    base = a*x + b
    diff = yy - base
    if positive_only:
        diff = np.maximum(diff, 0.0)

    auc = float(np.trapz(diff, x))
    meta = dict(a=a, b=b, ms=ms, mL=mL, mR=mR, x=x, yy=yy, base=base, diff=diff)
    return auc, meta


def gaussian_window(wn, mu, fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * LN2))
    return np.exp(-0.5 * ((wn - mu) / (sigma + EPS))**2)


def weighted_avg(y, w):
    s = np.sum(w)
    if s <= 0:
        return np.nan
    return float(np.sum(y*w) / s)


def entanglecam_DI(y, wn, fA, fP):
    # A channel
    wAp = gaussian_window(wn, MU_A_PEAK, fA)
    wAt = gaussian_window(wn, MU_A_TROUGH, fA)
    Aap = weighted_avg(y, wAp)
    Aat = weighted_avg(y, wAt)
    dA = Aap - Aat

    # P channel
    wPp = gaussian_window(wn, MU_P_PEAK, fP)
    wPt = gaussian_window(wn, MU_P_TROUGH, fP)
    App = weighted_avg(y, wPp)
    Apt = weighted_avg(y, wPt)
    dP = App - Apt

    if not np.isfinite(dA) or abs(dA) < 1e-9:
        return np.nan, dA, dP
    return dP / dA, dA, dP


def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def loocv_linear_rmse(x, y):
    """
    Fit y ~ m*x + c with LOOCV.
    Returns rmse, preds, (m_all, c_all) fitted on full data.
    """
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float)
    loo = LeaveOneOut()
    preds = np.full_like(y, np.nan, dtype=float)

    for tr, te in loo.split(x):
        model = LinearRegression()
        model.fit(x[tr], y[tr])
        preds[te] = model.predict(x[te])

    rmse = float(np.sqrt(np.mean((preds - y)**2)))

    model_full = LinearRegression()
    model_full.fit(x, y)
    m_all = float(model_full.coef_[0])
    c_all = float(model_full.intercept_)

    return rmse, preds, m_all, c_all


def plot_entanglecam_windows_on_absorbance(wn, A, params, savepath):
    """
    Plot best EntangleCam Gaussian windows on absorbance spectrum (A),
    and draw true FWHM edges so widths are obvious.
    """
    fA = params["fA"]
    fP = params["fP"]

    windows = [
        ("A peak",   MU_A_PEAK,   fA),
        ("A trough", MU_A_TROUGH, fA),
        ("P peak",   MU_P_PEAK,   fP),
        ("P trough", MU_P_TROUGH, fP),
    ]

    plt.figure(figsize=(11,5))
    plt.plot(wn, A, lw=2, color="black", label="Absorbance (mean)")

    A_min = np.nanmin(A)
    A_max = np.nanmax(A)
    scale = 0.9*(A_max - A_min) + EPS

    for (lab, mu, fwhm) in windows:
        g = gaussian_window(wn, mu, fwhm)
        g_plot = A_min + scale*(g/g.max())
        plt.plot(wn, g_plot, lw=2, ls="--", label=f"{lab} (μ={mu:.1f}, FWHM={fwhm:.1f})")
        # show FWHM edges
        plt.axvline(mu - fwhm/2, ls=":", alpha=0.6)
        plt.axvline(mu + fwhm/2, ls=":", alpha=0.6)

    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Absorbance (arb.)")
    plt.title(f"EntangleCam windows on absorbance (fA={fA:.0f}, fP={fP:.0f})")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(savepath, dpi=220)
    plt.close()


# =========================
# MAIN: Load all spectra + align
# =========================
paths = sorted(glob.glob(os.path.join(DATA_DIR, GLOB_PATTERN)))
if len(paths) < 5:
    raise SystemExit("Too few spectra found. Check DATA_DIR/GLOB_PATTERN.")

# reference grid from first file
wn0, y0 = load_2col(paths[0], header=HEADER)
if INPUT_IS_TRANSMISSION:
    y0 = to_absorbance(y0)
wn_grid = wn0

Y = []
names = []
for p in paths:
    wn, y = load_2col(p, header=HEADER)
    if INPUT_IS_TRANSMISSION:
        y = to_absorbance(y)
    yg = np.interp(wn_grid, wn, y)  # align to common grid
    Y.append(yg)
    names.append(os.path.basename(p))

Y = np.vstack(Y)  # N x Nwn
mean_spec = np.mean(Y, axis=0)

print(f"Loaded {Y.shape[0]} spectra; grid Nwn={Y.shape[1]} [{wn_grid.min():.1f},{wn_grid.max():.1f}]")

# Prepare PA arrays
pa_mask = np.array([nm in PA_TABLE and np.isfinite(PA_TABLE[nm]) for nm in names], dtype=bool)
pa_vals = np.array([PA_TABLE[nm] if nm in PA_TABLE else np.nan for nm in names], float)
print(f"PA_true loaded; finite: {np.sum(pa_mask)} / {len(names)}")


# =========================
# A) AUC-based channels + visuals
# =========================
rows = []
example = None  # for example baseline+area plot

for i, nm in enumerate(names):
    y = Y[i]

    muP = find_peak_near(wn_grid, y, P_SEARCH_CENTER, P_SEARCH_HALF_WIDTH)
    muA = find_peak_near(wn_grid, y, A_SEARCH_CENTER, A_SEARCH_HALF_WIDTH)

    rec = {"file": nm, "muP": muP, "muA": muA}

    meta_by_w = {}

    # phosphate AUCs at multiple widths
    for w in P_WIDTHS:
        aucP, metaP = auc_local(wn_grid, y, muP, float(w), BASELINE_OFFSET, positive_only=True)
        rec[f"AUC_P_w{int(w)}"] = aucP
        meta_by_w[f"P_{int(w)}"] = metaP

    # amide AUC (single width)
    aucA, metaA = auc_local(wn_grid, y, muA, float(A_WIDTH), BASELINE_OFFSET, positive_only=True)
    rec["AUC_A"] = aucA
    meta_by_w["A"] = metaA

    rows.append(rec)

    if example is None and np.isfinite(rec.get("AUC_P_w34", np.nan)) and np.isfinite(rec.get("AUC_A", np.nan)):
        example = dict(file=nm, y=y.copy(), muP=muP, muA=muA, meta_by_w=meta_by_w)

df_auc = pd.DataFrame(rows)
df_auc.to_csv(os.path.join(OUTDIR, "auc_channels.csv"), index=False)

# default AUC ratio uses phosphate width=34
W_USE = 34
df_auc["PA_pred_AUC_raw"] = df_auc[f"AUC_P_w{W_USE}"] / (df_auc["AUC_A"] + EPS)

# info retention and loss (20 vs 34)
df_auc["retained_20_over_34"] = df_auc["AUC_P_w20"] / (df_auc["AUC_P_w34"] + EPS)
df_auc["loss_20_vs_34"] = 1.0 - df_auc["retained_20_over_34"]
df_auc.to_csv(os.path.join(OUTDIR, "auc_table_with_loss.csv"), index=False)

# ---- Plot: boxplot of AUC_P by width
data_box = []
labels = []
for w in P_WIDTHS:
    v = df_auc[f"AUC_P_w{int(w)}"].to_numpy(float)
    v = v[np.isfinite(v)]
    data_box.append(v)
    labels.append(str(int(w)))

plt.figure(figsize=(9,6))
plt.boxplot(data_box, labels=labels, showfliers=True)
plt.xlabel("Phosphate window width (cm$^{-1}$)")
plt.ylabel("AUC_P after baseline subtraction (arb.·cm$^{-1}$)")
plt.title("Phosphate AUC distribution vs window width")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "aucP_boxplot_by_width.png"), dpi=200)
plt.close()

# ---- Plot: AUC(20) vs AUC(34) + y=x
m = np.isfinite(df_auc["AUC_P_w34"]) & np.isfinite(df_auc["AUC_P_w20"])
x = df_auc.loc[m, "AUC_P_w34"].values
y = df_auc.loc[m, "AUC_P_w20"].values

plt.figure(figsize=(7,7))
plt.scatter(x, y, s=30)
if x.size:
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    plt.plot([lo, hi], [lo, hi])
plt.xlabel("AUC_P (width 34)")
plt.ylabel("AUC_P (width 20)")
plt.title("Information retained: AUC(20) vs AUC(34)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "auc20_vs_auc34.png"), dpi=200)
plt.close()

# ---- Plot: info loss histogram
loss = df_auc["loss_20_vs_34"].to_numpy(float)
loss = loss[np.isfinite(loss)]
plt.figure(figsize=(9,6))
plt.hist(loss, bins=20, edgecolor="black")
plt.xlabel("Loss fraction = 1 - AUC(20)/AUC(34)")
plt.ylabel("Count")
plt.title("Information loss distribution (phosphate AUC)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "loss_hist_w20_vs_34.png"), dpi=200)
plt.close()

# ---- Plot: peak centre histogram near 1234
muP_arr = df_auc["muP"].to_numpy(float)
muP_arr = muP_arr[np.isfinite(muP_arr)]
plt.figure(figsize=(9,6))
plt.hist(muP_arr, bins=20, edgecolor="black")
plt.xlabel("Detected phosphate peak centre near 1234 (cm$^{-1}$)")
plt.ylabel("Count")
plt.title("Per-spectrum phosphate peak centre distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "muP_peak_hist.png"), dpi=200)
plt.close()

# ---- Plot: example spectrum with baseline + AUC shading for phosphate widths
if example is not None:
    wn = wn_grid
    y = example["y"]
    mu = example["muP"]

    plt.figure(figsize=(12,6))
    plt.plot(wn, y, lw=2, label="Absorbance spectrum (example)")

    for w in P_WIDTHS:
        meta = example["meta_by_w"].get(f"P_{int(w)}", None)
        if meta is None:
            continue
        xS, base, diff = meta["x"], meta["base"], meta["diff"]
        plt.plot(xS, base, lw=2, label=f"Baseline fit (w={int(w)})")
        plt.fill_between(xS, base, base+diff, alpha=0.18, label=f"AUC area (w={int(w)})")

    plt.axvline(mu, ls="--", lw=1.5, label=f"Peak centre ~ {mu:.2f}")
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Absorbance (arb.)")
    plt.title(f"Example baseline-corrected AUC around ~1234 band\n{example['file']}")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "example_auc_overlay_1234.png"), dpi=200)
    plt.close()

# ---- Plot: geometry on mean spectrum (signal + baseline windows)
mu_mean = float(np.nanmean(muP_arr)) if muP_arr.size else P_SEARCH_CENTER
plt.figure(figsize=(12,6))
plt.plot(wn_grid, mean_spec, lw=2, label="Mean absorbance")
plt.axvline(mu_mean, ls="--", lw=1.5, label=f"Mean peak centre ~ {mu_mean:.2f}")

for w in P_WIDTHS:
    plt.axvspan(mu_mean - 0.5*w, mu_mean + 0.5*w, alpha=0.10, label=f"Signal win {int(w)}")

# show baseline windows for w=34
w = 34.0
for c in [mu_mean-BASELINE_OFFSET, mu_mean+BASELINE_OFFSET]:
    plt.axvspan(c - 0.5*w, c + 0.5*w, alpha=0.06)

plt.gca().invert_xaxis()
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Absorbance (arb.)")
plt.title("Window geometry on mean spectrum (signal + baseline windows)")
plt.legend(ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "mean_geometry_1234.png"), dpi=200)
plt.close()

# ---- If you have PA labels: LOOCV calibration for AUC ratio (optional)
auc_pa_mask = np.array([nm in PA_TABLE and np.isfinite(PA_TABLE[nm]) for nm in df_auc["file"]], dtype=bool)
x_auc = df_auc.loc[auc_pa_mask, "PA_pred_AUC_raw"].to_numpy(float)
y_pa  = np.array([PA_TABLE[nm] for nm in df_auc.loc[auc_pa_mask, "file"]], float)

if x_auc.size >= 5:
    rmse_auc, preds_auc, m_auc, c_auc = loocv_linear_rmse(x_auc, y_pa)
    plt.figure(figsize=(6,6))
    plt.scatter(y_pa, preds_auc, s=35)
    lo = min(y_pa.min(), preds_auc.min())
    hi = max(y_pa.max(), preds_auc.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("PA_true")
    plt.ylabel("PA_pred (LOOCV) from AUC ratio")
    plt.title(f"AUC ratio LOOCV | RMSE={rmse_auc:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "auc_ratio_loocv_scatter.png"), dpi=200)
    plt.close()

    resid = preds_auc - y_pa
    plt.figure(figsize=(5,5))
    plt.boxplot(resid, vert=True)
    plt.axhline(0)
    plt.ylabel("Residual (PA_pred - PA_true)")
    plt.title("AUC ratio residuals (LOOCV)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "auc_ratio_loocv_residual_box.png"), dpi=200)
    plt.close()


# =========================
# B) EntangleCam Gaussian optimisation (fixed centres)
# =========================
best = None
H = np.full((len(FWHM_A_SCAN), len(FWHM_P_SCAN)), np.nan)

for i, fA in enumerate(FWHM_A_SCAN):
    for j, fP in enumerate(FWHM_P_SCAN):
        DI_list = []
        dA_list = []

        for k in range(Y.shape[0]):
            di, dA, dP = entanglecam_DI(Y[k], wn_grid, float(fA), float(fP))
            DI_list.append(di)
            dA_list.append(dA)

        DI_arr = np.array(DI_list, float)
        dA_arr = np.array(dA_list, float)

        mfin = np.isfinite(DI_arr) & np.isfinite(dA_arr)
        if np.sum(mfin) < 5:
            continue

        # stability (lower is better)
        stab = mad(DI_arr[mfin]) / (np.median(np.abs(dA_arr[mfin])) + EPS)

        # rmse term (if labels exist)
        mk = mfin & pa_mask
        if np.sum(mk) >= 5:
            rmse, _, _, _ = loocv_linear_rmse(DI_arr[mk], pa_vals[mk])
        else:
            rmse = 0.0

        loss = ALPHA_RMSE * rmse + LAMBDA_STAB * stab
        H[i, j] = loss

        if best is None or loss < best["loss"]:
            best = dict(loss=float(loss), rmse=float(rmse), stab=float(stab), fA=float(fA), fP=float(fP))

print("\n=== BEST ENTANGLECAM (fixed centres) ===")
print(best)

pd.DataFrame([best]).to_csv(os.path.join(OUTDIR, "entanglecam_best_params.csv"), index=False)

# Heatmap
plt.figure(figsize=(8,6))
plt.imshow(
    H, origin="lower", aspect="auto",
    extent=[FWHM_P_SCAN[0], FWHM_P_SCAN[-1], FWHM_A_SCAN[0], FWHM_A_SCAN[-1]]
)
plt.colorbar(label="Combined loss = α*RMSE + λ*stability (lower better)")
plt.scatter([best["fP"]], [best["fA"]], s=80)
plt.xlabel("FWHM_P (cm$^{-1}$)")
plt.ylabel("FWHM_A (cm$^{-1}$)")
plt.title("EntangleCam Gaussian optimisation heatmap (fixed centres)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "entanglecam_loss_heatmap.png"), dpi=200)
plt.close()

# Overlay best windows on mean absorbance (with FWHM edges)
plot_entanglecam_windows_on_absorbance(
    wn_grid, mean_spec, best,
    savepath=os.path.join(OUTDIR, "entanglecam_best_overlay_on_absorbance.png")
)

# DI per spectrum at best params + optional LOOCV plots
DI_best = []
dA_best = []
dP_best = []
for k in range(Y.shape[0]):
    di, dA, dP = entanglecam_DI(Y[k], wn_grid, best["fA"], best["fP"])
    DI_best.append(di); dA_best.append(dA); dP_best.append(dP)

DI_best = np.array(DI_best, float)
dA_best = np.array(dA_best, float)
dP_best = np.array(dP_best, float)

df_di = pd.DataFrame({
    "file": names,
    "DI": DI_best,
    "dA": dA_best,
    "dP": dP_best,
    "PA_true": pa_vals
})
df_di.to_csv(os.path.join(OUTDIR, "entanglecam_DI_table_best.csv"), index=False)

# If enough PA labels: LOOCV calibration PA_true ~ m*DI + c, scatter + residuals
mk = np.isfinite(DI_best) & pa_mask
if np.sum(mk) >= 5:
    rmse_ec, preds_ec, m_ec, c_ec = loocv_linear_rmse(DI_best[mk], pa_vals[mk])

    y_true = pa_vals[mk]
    y_pred = preds_ec

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=35)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("PA_true")
    plt.ylabel("PA_pred (LOOCV) from EntangleCam DI")
    plt.title(f"EntangleCam fixed-centre LOOCV | RMSE={rmse_ec:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "entanglecam_loocv_scatter.png"), dpi=200)
    plt.close()

    resid = y_pred - y_true
    plt.figure(figsize=(5,5))
    plt.boxplot(resid, vert=True)
    plt.axhline(0)
    plt.ylabel("Residual (PA_pred - PA_true)")
    plt.title("EntangleCam DI residuals (LOOCV)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "entanglecam_loocv_residual_box.png"), dpi=200)
    plt.close()

print("\nSaved outputs to:", os.path.abspath(OUTDIR))
print("Key files:")
print(" - auc_channels.csv")
print(" - auc_table_with_loss.csv")
print(" - aucP_boxplot_by_width.png")
print(" - auc20_vs_auc34.png")
print(" - loss_hist_w20_vs_34.png")
print(" - muP_peak_hist.png")
print(" - example_auc_overlay_1234.png")
print(" - mean_geometry_1234.png")
print(" - entanglecam_best_params.csv")
print(" - entanglecam_loss_heatmap.png")
print(" - entanglecam_best_overlay_on_absorbance.png")
print(" - entanglecam_DI_table_best.csv")
