# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# EntangleCam-style DI map from long-form FTIR map file (map_all_wavenumbers.txt)
# using AUC-matched synthetic Gaussian windows (Beer-law in absorbance space).

# What it does:
# - Load map_all_wavenumbers.txt with columns: X, Y, wavenumber, Absorption
# - Load Window1..Window4.csv (your measured filter windows in absorbance)
# - Convert window absorbance -> transmission; isolate main passband; compute AUC
# - Build synthetic Gaussian transmission windows with fixed mu and AUC-matching
# - Convert synthetic Gaussian transmission -> absorbance windows (Beer’s law)
# - Compute DI per pixel exactly as in DI_map.py (uses log10 sums of filtered transmissions)
# - Save DI map CSV + plots + gaussian_window*.csv (absorbance)

# This is consistent with:
# - DI_map.py method for DI calculation (transmission multiplication + log10 sums) :contentReference[oaicite:2]{index=2}
# - Gaus_to_fit_filter.py “AUC-matched Gaussian” idea, but enforced fixed centres :contentReference[oaicite:3]{index=3}
# """

# from __future__ import annotations

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# from scipy.signal import savgol_filter
# from scipy.optimize import minimize_scalar
# from math import sqrt, pi, erf


# # =========================
# # USER SETTINGS
# # =========================
# MAP_TXT_PATH = "/Users/nana/Desktop/map_all_wavenumbers.txt"          # long-form map file
# WINDOW_FILES = ["/Users/nana/Desktop/Digistain/matlab/Window1.csv", "/Users/nana/Desktop/Digistain/matlab/Window2.csv", "/Users/nana/Desktop/Digistain/matlab/Window3.csv", "/Users/nana/Desktop/Digistain/matlab/Window4.csv"]

# # Wavenumber crop for DI computation
# CROP_MIN = 1000.0
# CROP_MAX = 1800.0

# # Interpolation grid size (DI_map.py uses 10001 for MATLAB parity) :contentReference[oaicite:4]{index=4}
# INTERP_N = 10001

# # --- Fixed centres (cm^-1): set these to what YOU want ---
# # Window meaning (same as DI_map.py):
# #   outputs[0] = amide_peak (Window1)
# #   outputs[1] = phosphate_peak (Window2)
# #   outputs[2] = phosphate_base (Window3)
# #   outputs[3] = amide_base (Window4) :contentReference[oaicite:5]{index=5}
# MU_FIXED = {
#     "Window1": 1667.8,  # amide peak
#     "Window4": 1604.97351330716,  # amide baseline/trough
#     "Window2": 1221,  # phosphate peak (fix to your detected centre)
#     "Window3": 1168.2818403969,  # phosphate baseline/trough
# }

# # --- Passband isolation on measured windows (to remove side ripples) ---
# RELATIVE_THRESHOLD = 0.05     # keep T >= threshold * max(T_smooth)
# SG_WINDOW = 51                # odd
# SG_POLY = 3

# # --- Sigma (width) fit bounds for Gaussian (cm^-1) ---
# SIGMA_MIN = 1.0
# SIGMA_MAX = 300.0

# # --- Convert Gaussian transmission -> absorbance safely ---
# T_EPS = 1e-12                 # avoid log(0)
# ABS_CAP = 5.0                 # cap absorbance tails so CSV doesn’t explode

# # --- Output folder ---
# OUT_DIR = "entanglecam_di_map_outputs"
# os.makedirs(OUT_DIR, exist_ok=True)

# # --- Plot settings ---
# INVERT_X = True               # typical FTIR axis (high->low)
# INVERT_Y = True               # image-style map orientation
# SCATTER_MARKER = "s"
# SCATTER_MARKER_SIZE = 10
# CLIM = (-2, 2)                # adjust to your expected DI range


# # =========================
# # Helpers: interpolation & DI (matches DI_map.py logic)
# # =========================

# def _ensure_sorted_unique_x(x: np.ndarray, y: np.ndarray):
#     x = np.asarray(x, float)
#     y = np.asarray(y, float)
#     m = np.isfinite(x) & np.isfinite(y)
#     x, y = x[m], y[m]
#     if x.size == 0:
#         return x, y
#     order = np.argsort(x)
#     x, y = x[order], y[order]
#     xu, inv = np.unique(x, return_inverse=True)
#     if xu.size != x.size:
#         y_acc = np.zeros_like(xu, float)
#         cnt = np.zeros_like(xu, float)
#         np.add.at(y_acc, inv, y)
#         np.add.at(cnt, inv, 1.0)
#         y = y_acc / np.maximum(cnt, 1.0)
#         x = xu
#     return x, y


# def interpolate_data(cell_wn, cell_abs, win_wn, win_abs, interp_n=10001):
#     cellWN, cellABS = _ensure_sorted_unique_x(cell_wn, cell_abs)
#     winWN, winABS = _ensure_sorted_unique_x(win_wn, win_abs)

#     if cellWN.size < 2 or winWN.size < 2:
#         raise ValueError("Need >=2 points in both cell and window spectra.")

#     minWVNM = max(np.min(winWN), np.min(cellWN))
#     maxWVNM = min(np.max(winWN), np.max(cellWN))
#     if maxWVNM <= minWVNM:
#         raise ValueError("No overlap between cell and window spectra.")

#     wn_i = np.linspace(minWVNM, maxWVNM, int(interp_n))
#     f_win = interp1d(winWN, winABS, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)
#     f_cell = interp1d(cellWN, cellABS, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)

#     win_abs_i = f_win(wn_i)
#     cell_abs_i = f_cell(wn_i)

#     m = np.isfinite(wn_i) & np.isfinite(win_abs_i) & np.isfinite(cell_abs_i)
#     return wn_i[m], cell_abs_i[m], win_abs_i[m]


# def get_di_for_pixel(cellWN, cellABS, windows_abs, interp_n=10001):
#     """
#     Same DI logic as DI_map.py: multiply transmissions, sum, log10, DI = 0.1*(C-D)/(A-B). :contentReference[oaicite:6]{index=6}
#     windows_abs: list of 4 dicts {wavenumber, ABS}
#     """
#     outputs = []
#     for i in range(4):
#         wn_i, cell_i, win_i = interpolate_data(
#             cellWN, cellABS,
#             windows_abs[i]["wavenumber"], windows_abs[i]["ABS"],
#             interp_n=interp_n
#         )
#         cell_trn = 10.0 ** (-cell_i)
#         win_trn  = 10.0 ** (-win_i)
#         outputs.append(cell_trn * win_trn)

#     amide_peak     = np.sum(outputs[0])
#     phosphate_peak = np.sum(outputs[1])
#     phosphate_base = np.sum(outputs[2])
#     amide_base     = np.sum(outputs[3])

#     eps = 1e-300
#     A = np.log10(max(amide_peak, eps))
#     B = np.log10(max(amide_base, eps))
#     C = np.log10(max(phosphate_peak, eps))
#     D = np.log10(max(phosphate_base, eps))

#     denom = (A - B)
#     if not np.isfinite(denom) or abs(denom) < 1e-15:
#         return np.nan
#     return float(0.1 * (C - D) / denom)


# # =========================
# # Helpers: window processing & AUC-matched Gaussians
# # =========================

# def absorbance_to_transmittance(A):
#     return np.power(10.0, -A)

# def transmittance_to_absorbance(T):
#     T_clip = np.clip(T, T_EPS, None)
#     A = -np.log10(T_clip)
#     return np.minimum(A, ABS_CAP)

# def safe_savgol(y, win, poly):
#     y = np.asarray(y, float)
#     n = y.size
#     if n < 7:
#         return y.copy()
#     win = min(win, n if (n % 2 == 1) else (n - 1))
#     win = max(win, 7)
#     if win % 2 == 0:
#         win -= 1
#     poly = min(poly, win - 1)
#     return savgol_filter(y, window_length=win, polyorder=poly)

# def largest_contiguous_true(mask):
#     mask = np.asarray(mask, bool)
#     idx = np.flatnonzero(mask)
#     if idx.size == 0:
#         return np.zeros_like(mask, bool)
#     splits = np.where(np.diff(idx) != 1)[0] + 1
#     groups = np.split(idx, splits)
#     largest = max(groups, key=len)
#     out = np.zeros_like(mask, bool)
#     out[largest] = True
#     return out

# def clean_window_transmission(wn, T):
#     """
#     Keep largest passband lobe using smoothed thresholding.
#     AUC is computed on the cleaned (zeroed-outside) transmission.
#     """
#     T_s = safe_savgol(T, SG_WINDOW, SG_POLY)
#     thr = RELATIVE_THRESHOLD * np.nanmax(T_s) if np.isfinite(T_s).any() else np.inf
#     mask = (T_s >= thr)
#     mask = largest_contiguous_true(mask)
#     T_clean = np.where(mask, T, 0.0)
#     auc = float(np.trapz(T_clean, wn)) if wn.size else np.nan
#     return T_clean, auc, mask

# def gaussian_T(x, Aamp, mu, sigma):
#     return Aamp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# def gaussian_auc_factor(sigma, mu, a, b):
#     """
#     Integral of exp(-0.5((x-mu)/sigma)^2) from a to b:
#     sigma * sqrt(pi/2) * [erf((b-mu)/(sqrt(2)sigma)) - erf((a-mu)/(sqrt(2)sigma))]
#     """
#     if sigma <= 0:
#         return np.nan
#     z1 = (b - mu) / (sqrt(2) * sigma)
#     z0 = (a - mu) / (sqrt(2) * sigma)
#     return sigma * sqrt(pi / 2) * (erf(z1) - erf(z0))

# def amplitude_from_auc(sigma, mu, auc_target, a, b):
#     fac = gaussian_auc_factor(sigma, mu, a, b)
#     if not np.isfinite(fac) or fac <= 0:
#         return np.nan
#     return auc_target / fac

# def fit_sigma_auc_matched(wn, T_clean, mu, auc_target):
#     """
#     Fit only sigma by SSE on passband region; amplitude is fixed by AUC constraint.
#     """
#     mask = (T_clean > 0)
#     x = wn[mask]
#     y = T_clean[mask]
#     if x.size < 5:
#         return np.nan, np.nan

#     a, b = float(np.min(wn)), float(np.max(wn))

#     def obj(log_sigma):
#         sigma = float(np.exp(log_sigma))
#         Aamp = amplitude_from_auc(sigma, mu, auc_target, a, b)
#         yhat = gaussian_T(x, Aamp, mu, sigma)
#         return float(np.sum((yhat - y) ** 2))

#     res = minimize_scalar(
#         obj,
#         bounds=(np.log(SIGMA_MIN), np.log(SIGMA_MAX)),
#         method="bounded",
#     )
#     sigma_star = float(np.exp(res.x))
#     A_star = float(amplitude_from_auc(sigma_star, mu, auc_target, a, b))
#     return A_star, sigma_star


# def load_window_abs_csv(path, crop_min, crop_max):
#     df = pd.read_csv(path)
#     cols = {c.lower(): c for c in df.columns}

#     # accept flexible headers
#     wn_col = None
#     for k in ["wavenumber", "wn", "wavenumbers"]:
#         if k in cols:
#             wn_col = cols[k]
#             break
#     if wn_col is None:
#         wn_col = df.columns[0]

#     abs_col = None
#     for k in ["absorbance", "absorption", "abs", "a"]:
#         if k in cols:
#             abs_col = cols[k]
#             break
#     if abs_col is None:
#         abs_col = df.columns[1]

#     wn = pd.to_numeric(df[wn_col], errors="coerce").to_numpy(float)
#     ab = pd.to_numeric(df[abs_col], errors="coerce").to_numpy(float)

#     m = np.isfinite(wn) & np.isfinite(ab) & (wn >= crop_min) & (wn <= crop_max)
#     wn, ab = wn[m], ab[m]
#     order = np.argsort(wn)
#     return wn[order], ab[order]


# # =========================
# # Map loading
# # =========================

# def load_map_longform(path, crop_min, crop_max):
#     T = pd.read_csv(path, sep="\t", engine="python")
#     for c in ["X", "Y", "wavenumber", "Absorption"]:
#         if c not in T.columns:
#             raise ValueError(f"Missing column '{c}' in {path}. Found: {list(T.columns)}")
#         T[c] = pd.to_numeric(T[c], errors="coerce")

#     T = T.dropna(subset=["X", "Y", "wavenumber", "Absorption"])
#     T = T[(T["wavenumber"] >= crop_min) & (T["wavenumber"] <= crop_max)].copy()
#     return T


# def try_make_grid(x, y, z):
#     xu = np.unique(x)
#     yu = np.unique(y)
#     nx, ny = xu.size, yu.size
#     if nx * ny != x.size:
#         return None
#     x_to_ix = {v: i for i, v in enumerate(xu)}
#     y_to_iy = {v: i for i, v in enumerate(yu)}
#     Z = np.full((ny, nx), np.nan, float)
#     for xi, yi, zi in zip(x, y, z):
#         Z[y_to_iy[yi], x_to_ix[xi]] = zi
#     return xu, yu, Z


# # =========================
# # MAIN
# # =========================

# def main():
#     # ---------- 1) Load map ----------
#     map_df = load_map_longform(MAP_TXT_PATH, CROP_MIN, CROP_MAX)
#     XY = map_df[["X", "Y"]].to_numpy(float)
#     uniqueXY, pixelIdx = np.unique(XY, axis=0, return_inverse=True)
#     nPix = uniqueXY.shape[0]
#     print(f"[INFO] Map rows: {len(map_df):,} | unique pixels: {nPix:,}")

#     wn_all = map_df["wavenumber"].to_numpy(float)
#     ab_all = map_df["Absorption"].to_numpy(float)

#     # Build per-pixel arrays
#     wavenumbers_list = [None] * nPix
#     absorbance_list  = [None] * nPix
#     for k in range(nPix):
#         rows = (pixelIdx == k)
#         wavenumbers_list[k] = wn_all[rows]
#         absorbance_list[k]  = ab_all[rows]
#         if (k + 1) % 5000 == 0:
#             print(f"[INFO] Collected spectra: {k+1:,}/{nPix:,}")

#     X_pix = uniqueXY[:, 0]
#     Y_pix = uniqueXY[:, 1]

#     # ---------- 2) Load measured windows + fit AUC-matched Gaussians ----------
#     win_meas = []
#     win_gauss_abs = []  # list of dicts for DI computation
#     gauss_param_rows = []

#     plt.figure(figsize=(10, 5.6))
#     for wf in WINDOW_FILES:
#         name = os.path.splitext(os.path.basename(wf))[0]  # "Window1"
#         wn_w, A_w = load_window_abs_csv(wf, CROP_MIN, CROP_MAX)
#         T_w = absorbance_to_transmittance(A_w)

#         # clean passband and compute AUC
#         T_clean, auc_target, mask = clean_window_transmission(wn_w, T_w)

#         mu = float(MU_FIXED.get(name, np.nan))
#         if not np.isfinite(mu):
#             # fallback: use peak of cleaned region
#             if np.any(mask):
#                 mu = float(wn_w[np.argmax(T_clean)])
#             else:
#                 mu = float(wn_w[np.argmax(T_w)])

#         # fit sigma only (A comes from AUC constraint)
#         Aamp, sigma = fit_sigma_auc_matched(wn_w, T_clean, mu, auc_target)
#         fwhm = float(2.0 * sqrt(2.0 * np.log(2.0)) * sigma) if np.isfinite(sigma) else np.nan

#         # synthetic gaussian transmission on same wn grid
#         T_syn = gaussian_T(wn_w, Aamp, mu, sigma)

#         # convert to absorbance window for Beer-law DI pipeline
#         A_syn = transmittance_to_absorbance(T_syn)

#         win_meas.append((name, wn_w, T_w, T_clean))
#         win_gauss_abs.append({"wavenumber": wn_w, "ABS": A_syn})

#         gauss_param_rows.append({
#             "window": name,
#             "mu_fixed": mu,
#             "sigma_fit": sigma,
#             "FWHM": fwhm,
#             "A_amplitude": Aamp,
#             "AUC_cleaned_target": auc_target,
#             "AUC_gaussian_numeric": float(np.trapz(T_syn, wn_w)) if np.all(np.isfinite(T_syn)) else np.nan
#         })

#         # Save gaussian window absorbance CSV (for auditability)
#         idx = name.replace("Window", "")
#         out_csv = os.path.join(OUT_DIR, f"gaussian_window{idx}.csv")
#         pd.DataFrame({"Wavenumber": wn_w, "Absorbance": A_syn}).to_csv(out_csv, index=False)

#         # Overlay plot: measured clean vs gaussian (in transmission)
#         plt.plot(wn_w, T_clean, linewidth=2.0, label=f"{name} cleaned (measured)")
#         plt.plot(wn_w, T_syn, linestyle="--", linewidth=2.0, label=f"{name} Gaussian (AUC-matched)")

#     plt.xlabel("Wavenumber (cm$^{-1}$)")
#     plt.ylabel("Transmission")
#     plt.title("Measured windows (cleaned) vs AUC-matched synthetic Gaussians (fixed centres)")
#     plt.grid(True, alpha=0.35)
#     if INVERT_X:
#         plt.gca().invert_xaxis()
#     plt.legend(ncol=2, fontsize=9)
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUT_DIR, "windows_measured_vs_gaussian.png"), dpi=200)
#     plt.show()

#     gauss_df = pd.DataFrame(gauss_param_rows).sort_values("window")
#     gauss_df.to_csv(os.path.join(OUT_DIR, "gaussian_fit_params.csv"), index=False)
#     print("\n[INFO] Gaussian params saved:", os.path.join(OUT_DIR, "gaussian_fit_params.csv"))
#     print(gauss_df.to_string(index=False))

#     # ---------- 3) Compute DI map ----------
#     di = np.full(nPix, np.nan, float)
#     for k in range(nPix):
#         try:
#             di[k] = get_di_for_pixel(
#                 wavenumbers_list[k],
#                 absorbance_list[k],
#                 win_gauss_abs,
#                 interp_n=INTERP_N
#             )
#         except Exception:
#             di[k] = np.nan

#         if (k + 1) % 5000 == 0:
#             print(f"[INFO] DI computed: {k+1:,}/{nPix:,}")

#     out_di_csv = os.path.join(OUT_DIR, "DI_map_entanglecam_gaussian.csv")
#     pd.DataFrame({"X": X_pix, "Y": Y_pix, "DI": di}).to_csv(out_di_csv, index=False)
#     print("\n[SAVED]", out_di_csv)

#     # ---------- 4) Plot DI scatter ----------
#     plt.figure(figsize=(7.6, 6.4))
#     sc = plt.scatter(X_pix, Y_pix, s=SCATTER_MARKER_SIZE, c=di, marker=SCATTER_MARKER, linewidths=0)
#     plt.gca().set_aspect("equal", adjustable="box")
#     if INVERT_Y:
#         plt.gca().invert_yaxis()
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title("DI Map (EntangleCam synthetic Gaussian windows)")
#     cb = plt.colorbar(sc)
#     cb.set_label("DI index")
#     sc.set_clim(*CLIM)
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUT_DIR, "DI_map_scatter.png"), dpi=200)
#     plt.show()

#     # ---------- 5) Optional: gridded image if regular lattice ----------
#     grid = try_make_grid(X_pix, Y_pix, di)
#     if grid is not None:
#         xu, yu, Z = grid
#         plt.figure(figsize=(7.6, 6.4))
#         extent = [xu.min(), xu.max(), yu.max(), yu.min()] if INVERT_Y else [xu.min(), xu.max(), yu.min(), yu.max()]
#         im = plt.imshow(Z, extent=extent, aspect="equal")
#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.title("DI Map (gridded)")
#         cb = plt.colorbar(im)
#         cb.set_label("DI index")
#         im.set_clim(*CLIM)
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUT_DIR, "DI_map_gridded.png"), dpi=200)
#         plt.show()
#     else:
#         print("[INFO] Not a complete regular grid -> skipped gridded image.")

#     print("\n[DONE] Outputs in:", OUT_DIR)

#     # 6) Save (keep raw DI + add 0..1 visualisation scale)
#     di_table = pd.DataFrame({"X": X_pix, "Y": Y_pix, "DI_raw": di})

#     # ---- scale-to-[0,1] for plotting only ----
#     # Robust scaling using percentiles (prevents a few hot pixels dominating the colorbar)
#     SCALE_P_LO = 1.0   # lower percentile
#     SCALE_P_HI = 99.0  # upper percentile

#     finite = np.isfinite(di_table["DI_raw"].to_numpy(float))
#     if finite.sum() < 10:
#         # fallback: simple min/max if too few finite points
#         vmin = np.nanmin(di_table["DI_raw"].to_numpy(float))
#         vmax = np.nanmax(di_table["DI_raw"].to_numpy(float))
#     else:
#         vals = di_table.loc[finite, "DI_raw"].to_numpy(float)
#         vmin, vmax = np.percentile(vals, [SCALE_P_LO, SCALE_P_HI])
    
#     if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= 1e-12:
#         di_table["DI_01"] = np.nan
#     else:
#         di01 = (di_table["DI_raw"] - vmin) / (vmax - vmin)
#         di_table["DI_01"] = np.clip(di01, 0.0, 1.0)
    
#     OUT_CSV = "DigiStainIndex_map.csv"
#     di_table.to_csv(OUT_CSV, index=False)
#     print(f"Saved: {OUT_CSV} (columns: X, Y, DI_raw, DI_01)")
#     print(f"[INFO] Plot scaling: vmin={vmin:.6g}, vmax={vmax:.6g} (percentiles {SCALE_P_LO}/{SCALE_P_HI})")
    
#     # 7) Plot scatter map (0..1 like Digistain)
#     COLORMAP = "turbo"
#     plt.figure()
#     sc = plt.scatter(
#         di_table["X"], di_table["Y"],
#         s=SCATTER_MARKER_SIZE,
#         c=di_table["DI_01"],          # <- plot the 0..1 version
#         marker=SCATTER_MARKER,
#         cmap=COLORMAP,
#         alpha=1.0,
#         linewidths=0
#     )
#     plt.gca().set_aspect("equal", adjustable="box")
#     if INVERT_Y:
#         plt.gca().invert_yaxis()
    
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title("DI Map (0–1 scaled for display)")
#     cb = plt.colorbar(sc)
#     cb.set_label("DI index (0–1)")
#     sc.set_clim(0.0, 1.0)
#     plt.tight_layout()
#     plt.show()
    
#     # Optional: show gridded image if regular grid (uses DI_01)
#     grid = try_make_grid(X_pix, Y_pix, di_table["DI_01"].to_numpy(float))
#     if grid is not None:
#         xu, yu, Z = grid
#         plt.figure()
#         extent = [xu.min(), xu.max(), yu.max(), yu.min()] if INVERT_Y else [xu.min(), xu.max(), yu.min(), yu.max()]
#         im = plt.imshow(Z, extent=extent, aspect="equal", cmap=COLORMAP, vmin=0.0, vmax=1.0)
    
#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.title("DI Map (synthetic filter)")
#         cb = plt.colorbar(im)
#         cb.set_label("DI index (0–1)")
#         plt.tight_layout()
#         plt.show()
        

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML DigiStain map:
Train a regression model (PLSRegression recommended) from full spectra -> PA_ratio (or DI),
then apply to every pixel in map_all_wavenumbers.txt to produce a 2D map.

Outputs:
  - ml_outputs/model_metrics.txt
  - ml_outputs/pred_table.csv            (X,Y,DI_raw,DI_01)
  - ml_outputs/plots: scatter map, hist, calibration scatter, PLS loadings, PCA sanity checks

Assumptions:
  - map_all_wavenumbers.txt columns: X, Y, wavenumber, Absorption (absorbance)
  - training spectra are 2-column files (wavenumber, absorbance) with optional headers
  - labels CSV provides filename -> target (PA_ratio)
"""

from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

# =========================
# USER SETTINGS
# =========================
MAP_TXT_PATH = "/Users/nana/Desktop/map_all_wavenumbers.txt"

TRAIN_SPECTRA_DIR = "/Users/nana/Desktop/wavenumber absorption"  # folder containing your ~60 spectra CSVs
LABELS_CSV = "/Users/nana/Desktop/pa_ratio.csv"                 # filename, PA_ratio

# Spectral range + grid
WN_MIN = 1000.0
WN_MAX = 1800.0
WN_GRID_N = 800          # resample to a fixed grid (smaller => faster)
USE_BASELINE_CORR = True # strongly recommended for robustness

# Transmission-domain baseline correction anchors (edit if you want)
BASE_WN1 = 1184.0
BASE_WN2 = 1290.0
T_EPS = 1e-12

# Model choice
MODEL = "PLS"            # "PLS" or "RIDGE"
PLS_MAX_COMP = 25        # tune components via CV
RIDGE_ALPHAS = np.logspace(-4, 4, 25)

# Map display scaling
SCALE_P_LO = 1.0
SCALE_P_HI = 99.0

OUT_DIR = "ml_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# IO helpers
# =========================
def robust_read_2col(path: str):
    """
    Read 2-column spectra that might be comma-separated OR tab/space-separated.
    """
    # Try pandas with auto sep first
    df = pd.read_csv(path, sep=None, engine="python")
    if df.shape[1] < 2:
        # fallback: try numpy with comma
        arr = np.loadtxt(path, delimiter=",")
        wn, a = arr[:,0], arr[:,1]
        return wn.astype(float), a.astype(float)

    # take first two numeric columns
    cols = df.columns.tolist()
    c0, c1 = cols[0], cols[1]
    wn = pd.to_numeric(df[c0], errors="coerce").to_numpy(float)
    a  = pd.to_numeric(df[c1], errors="coerce").to_numpy(float)
    m = np.isfinite(wn) & np.isfinite(a)
    wn, a = wn[m], a[m]
    order = np.argsort(wn)
    return wn[order], a[order]

def make_wn_grid():
    return np.linspace(WN_MIN, WN_MAX, int(WN_GRID_N))

def interp_to_grid(wn, y, grid):
    wn = np.asarray(wn, float)
    y  = np.asarray(y, float)
    if wn.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    # ensure ascending
    if wn[0] > wn[-1]:
        wn = wn[::-1]; y = y[::-1]
    # clip to overlap
    lo = max(grid.min(), wn.min())
    hi = min(grid.max(), wn.max())
    out = np.full_like(grid, np.nan, dtype=float)
    m = (grid >= lo) & (grid <= hi)
    out[m] = np.interp(grid[m], wn, y)
    return out

# =========================
# Baseline correction in transmission
# =========================
def A_to_T(A):
    return 10.0 ** (-np.asarray(A, float))

def T_to_A(T):
    T = np.clip(np.asarray(T, float), T_EPS, None)
    return -np.log10(T)

def interp_at(x, y, x0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x[0] > x[-1]:
        x = x[::-1]; y = y[::-1]
    x0 = float(np.clip(x0, x.min(), x.max()))
    return float(np.interp(x0, x, y))

def baseline_correct_absorbance(grid_wn, A_grid):
    """
    Convert A->T, subtract linear baseline Tb in T between BASE_WN1/BASE_WN2, return corrected absorbance.
    """
    T = A_to_T(A_grid)
    T1 = interp_at(grid_wn, T, BASE_WN1)
    T2 = interp_at(grid_wn, T, BASE_WN2)

    Tb = T1 + (T2 - T1) * (grid_wn - BASE_WN1) / (BASE_WN2 - BASE_WN1)
    Tcorr = T - Tb
    # clip (baseline-removed signal)
    Tcorr = np.clip(Tcorr, T_EPS, None)
    return T_to_A(Tcorr)

# =========================
# Load training set
# =========================
def load_training():
    labels = pd.read_csv(LABELS_CSV)
    if not {"filename", "PA_ratio"}.issubset(set(labels.columns)):
        raise ValueError("LABELS_CSV must have columns: filename, PA_ratio")

    grid = make_wn_grid()
    X_list, y_list, kept = [], [], []

    for _, row in labels.iterrows():
        fn = str(row["filename"])
        y  = float(row["PA_ratio"])
        path = os.path.join(TRAIN_SPECTRA_DIR, fn)
        if not os.path.exists(path):
            # try case-insensitive match
            cand = [p for p in os.listdir(TRAIN_SPECTRA_DIR) if p.lower() == fn.lower()]
            if cand:
                path = os.path.join(TRAIN_SPECTRA_DIR, cand[0])
            else:
                print(f"[WARN] Missing training spectrum: {fn}")
                continue

        wn, A = robust_read_2col(path)
        A_grid = interp_to_grid(wn, A, grid)
        if np.isfinite(A_grid).sum() < 0.9 * grid.size:
            print(f"[WARN] Too many NaNs after interpolation: {fn}")
            continue

        if USE_BASELINE_CORR:
            A_grid = baseline_correct_absorbance(grid, A_grid)

        X_list.append(A_grid)
        y_list.append(y)
        kept.append(fn)

    X = np.vstack(X_list).astype(float)
    y = np.asarray(y_list, float)
    print(f"[INFO] Training: {X.shape[0]} spectra, grid={X.shape[1]} points")
    return grid, X, y, kept

# =========================
# Model selection
# =========================
def cv_select_pls(X, y, max_comp=25, n_splits=5, seed=0):
    kf = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=seed)
    best = None
    for ncomp in range(1, min(max_comp, X.shape[0], X.shape[1]) + 1):
        yhat_all = np.full_like(y, np.nan, dtype=float)
        for tr, te in kf.split(X):
            model = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("pls", PLSRegression(n_components=ncomp))
            ])
            model.fit(X[tr], y[tr])
            yhat_all[te] = model.predict(X[te]).ravel()

        rmse = np.sqrt(mean_squared_error(y, yhat_all))
        r2   = r2_score(y, yhat_all)
        if best is None or rmse < best["rmse"]:
            best = {"ncomp": ncomp, "rmse": rmse, "r2": r2, "yhat_cv": yhat_all.copy()}
    return best

def cv_select_ridge(X, y, alphas, n_splits=5, seed=0):
    kf = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=seed)
    best = None
    for a in alphas:
        yhat_all = np.full_like(y, np.nan, dtype=float)
        for tr, te in kf.split(X):
            model = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=float(a)))
            ])
            model.fit(X[tr], y[tr])
            yhat_all[te] = model.predict(X[te]).ravel()

        rmse = np.sqrt(mean_squared_error(y, yhat_all))
        r2   = r2_score(y, yhat_all)
        if best is None or rmse < best["rmse"]:
            best = {"alpha": float(a), "rmse": rmse, "r2": r2, "yhat_cv": yhat_all.copy()}
    return best

# =========================
# Load full map and predict
# =========================
def load_map_longform(path):
    df = pd.read_csv(path, sep=None, engine="python")
    need = ["X", "Y", "wavenumber"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in map file. Columns={list(df.columns)}")
    # absorbance column name may vary
    abs_col = None
    for cand in ["Absorption", "Absorbance", "ABS"]:
        if cand in df.columns:
            abs_col = cand
            break
    if abs_col is None:
        raise ValueError("Map file must contain Absorption/Absorbance/ABS column.")
    for c in ["X","Y","wavenumber",abs_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["X","Y","wavenumber",abs_col])
    df = df[(df["wavenumber"]>=WN_MIN) & (df["wavenumber"]<=WN_MAX)].copy()
    df = df.rename(columns={abs_col:"A"})
    return df

def predict_map(grid, model, map_df):
    XY = map_df[["X","Y"]].to_numpy(float)
    uniqueXY, idx = np.unique(XY, axis=0, return_inverse=True)
    nPix = uniqueXY.shape[0]
    print(f"[INFO] Map pixels: {nPix:,} (rows={len(map_df):,})")

    wn_all = map_df["wavenumber"].to_numpy(float)
    A_all  = map_df["A"].to_numpy(float)

    pred = np.full(nPix, np.nan, float)

    # group by pixel index efficiently
    for k in range(nPix):
        rows = (idx == k)
        wn = wn_all[rows]
        A  = A_all[rows]
        order = np.argsort(wn)
        wn = wn[order]; A = A[order]
        A_grid = interp_to_grid(wn, A, grid)
        if np.isfinite(A_grid).sum() < 0.9*grid.size:
            continue
        if USE_BASELINE_CORR:
            A_grid = baseline_correct_absorbance(grid, A_grid)

        pred[k] = model.predict(A_grid.reshape(1,-1)).ravel()[0]

        if (k+1) % 5000 == 0:
            print(f"[INFO] Predicted: {k+1:,}/{nPix:,}")

    out = pd.DataFrame({
        "X": uniqueXY[:,0],
        "Y": uniqueXY[:,1],
        "DI_raw": pred
    })
    return out

def scale_01(vals, p_lo=1.0, p_hi=99.0):
    vals = np.asarray(vals, float)
    m = np.isfinite(vals)
    if m.sum() < 10:
        return np.full_like(vals, np.nan)
    vmin, vmax = np.percentile(vals[m], [p_lo, p_hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
        return np.full_like(vals, np.nan)
    z = (vals - vmin) / (vmax - vmin)
    return np.clip(z, 0.0, 1.0)

def try_make_grid(x, y, z):
    xu = np.unique(x); yu = np.unique(y)
    nx, ny = xu.size, yu.size
    if nx * ny != x.size:
        return None
    x_to_ix = {v:i for i,v in enumerate(xu)}
    y_to_iy = {v:i for i,v in enumerate(yu)}
    Z = np.full((ny,nx), np.nan, float)
    for xi, yi, zi in zip(x,y,z):
        Z[y_to_iy[yi], x_to_ix[xi]] = zi
    return xu, yu, Z

# =========================
# Main
# =========================
def main():
    grid, Xtr, ytr, kept = load_training()

    # ----- select model -----
    if MODEL.upper() == "PLS":
        best = cv_select_pls(Xtr, ytr, max_comp=PLS_MAX_COMP, n_splits=5, seed=0)
        ncomp = best["ncomp"]
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pls", PLSRegression(n_components=ncomp))
        ])
        model.fit(Xtr, ytr)
        name = f"PLS(ncomp={ncomp})"
    else:
        best = cv_select_ridge(Xtr, ytr, RIDGE_ALPHAS, n_splits=5, seed=0)
        a = best["alpha"]
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=a))
        ])
        model.fit(Xtr, ytr)
        name = f"Ridge(alpha={a:g})"

    # ----- save CV metrics + calibration plot -----
    with open(os.path.join(OUT_DIR, "model_metrics.txt"), "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"CV RMSE: {best['rmse']:.6g}\n")
        f.write(f"CV R2:   {best['r2']:.6g}\n")
    print(f"[INFO] Selected {name} | CV RMSE={best['rmse']:.4g} R2={best['r2']:.4g}")

    plt.figure()
    plt.scatter(ytr, best["yhat_cv"], s=40)
    lo = min(np.min(ytr), np.min(best["yhat_cv"]))
    hi = max(np.max(ytr), np.max(best["yhat_cv"]))
    plt.plot([lo,hi],[lo,hi])
    plt.xlabel("Target (PA_ratio or DI)")
    plt.ylabel("CV prediction")
    plt.title(f"Cross-val calibration: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cv_calibration.png"), dpi=200)
    plt.show()

    # ----- interpretability: PLS loadings or PCA sanity -----
    # PLS loadings (only if PLS)
    if MODEL.upper() == "PLS":
        pls = model.named_steps["pls"]
        # X loadings: shape (n_features, n_components)
        W = pls.x_weights_
        for j in range(min(W.shape[1], 3)):
            plt.figure()
            plt.plot(grid, W[:, j])
            plt.gca().invert_xaxis()
            plt.xlabel("Wavenumber (cm$^{-1}$)")
            plt.ylabel(f"PLS weight (comp {j+1})")
            plt.title("Spectral features driving the regression")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"pls_weight_comp{j+1}.png"), dpi=200)
            plt.show()

    # PCA quick check (structure of training spectra)
    pca = PCA(n_components=min(5, Xtr.shape[0]-1, Xtr.shape[1]))
    Xp = pca.fit_transform(Xtr)
    plt.figure()
    plt.scatter(Xp[:,0], Xp[:,1], s=50)
    for i, fn in enumerate(kept):
        plt.text(Xp[i,0], Xp[i,1], str(i+1), fontsize=9)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("Training spectra PCA (sanity check)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "train_pca_scores.png"), dpi=200)
    plt.show()

    # ----- load full map and predict -----
    map_df = load_map_longform(MAP_TXT_PATH)
    pred_table = predict_map(grid, model, map_df)

    pred_table["DI_01"] = scale_01(pred_table["DI_raw"].to_numpy(float), SCALE_P_LO, SCALE_P_HI)
    pred_table.to_csv(os.path.join(OUT_DIR, "pred_table.csv"), index=False)
    print("[SAVED]", os.path.join(OUT_DIR, "pred_table.csv"))

    # ----- plots -----
    # histogram
    plt.figure()
    v = pred_table["DI_raw"].to_numpy(float)
    plt.hist(v[np.isfinite(v)], bins=80)
    plt.xlabel("Predicted value (raw)")
    plt.ylabel("Count")
    plt.title("Distribution of ML-predicted index")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pred_hist.png"), dpi=200)
    plt.show()

    # scatter map (0..1)
    plt.figure(figsize=(7,6))
    sc = plt.scatter(pred_table["X"], pred_table["Y"], s=6, c=pred_table["DI_01"], cmap="turbo", linewidths=0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().invert_yaxis()
    plt.colorbar(sc, label="Index (0–1 scaled)")
    plt.title("ML DigiStain map (display scaled 0–1)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "map_scatter_DI01.png"), dpi=200)
    plt.show()

    # gridded image if possible
    grid_try = try_make_grid(pred_table["X"].to_numpy(float),
                             pred_table["Y"].to_numpy(float),
                             pred_table["DI_01"].to_numpy(float))
    if grid_try is not None:
        xu, yu, Z = grid_try
        plt.figure(figsize=(7,6))
        extent = [xu.min(), xu.max(), yu.max(), yu.min()]
        im = plt.imshow(Z, extent=extent, aspect="equal", cmap="turbo", vmin=0.0, vmax=1.0)
        plt.colorbar(im, label="Index (0–1)")
        plt.title("ML DigiStain map (gridded)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "map_gridded_DI01.png"), dpi=200)
        plt.show()
    else:
        print("[INFO] Not a complete regular grid -> keep scatter map (recommended).")

    print("[DONE] Outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
