"""
entanglecam_filter_design.py

Step 1: Baseline-removed Gaussian filter optimisation (true SNR objective)
Step 2: Non-Gaussian learned spectral weighting (regularised logistic / ridge)
Step 3: Spatial binning sweep (2x2, 3x3, 4x4) -> recompute σ(ν), refit filters
Bonus: Compare against DigiStain 4 window method (if Window1-4.csv provided)

INPUTS:
  - Long-format hyperspectral map TXT/CSV with columns: X, Y, wavenumber, Absorption
    (tab or comma separated is OK)
  - Optional window files: Window1.csv..Window4.csv (absorbance or transmission)

OUTPUTS (per bin size):
  - noise_spectrum.png
  - gaussian_heatmap.png
  - gaussian_best_overlay.png
  - learned_weighting_function.png
  - pareto_resolution_vs_snr.csv
  - optional: digistain_map_vs_mlscore.png, correlation_stats.txt

Install:
  pip install numpy pandas scipy matplotlib scikit-learn

Run:
  python3 entanglecam_filter_design.py --map 24C15948A2_map.txt --out outputs --windows Window1.csv Window2.csv Window3.csv Window4.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


LN2 = np.log(2.0)
EPS = 1e-12


# =========================
# Baseline removal (ALS)
# =========================
def als_baseline(y, lam=3e5, p=0.01, niter=10):
    y = np.asarray(y, dtype=float)
    L = y.size
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = W + lam * (D @ D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


# =========================
# IO + reshape
# =========================
def read_long_map(path, delim=None):
    df = pd.read_csv(path, sep=delim, engine="python")
    # allow flexible headers
    cols = [c.strip().lower() for c in df.columns]
    colmap = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("x",):
            colmap[c] = "X"
        elif lc in ("y",):
            colmap[c] = "Y"
        elif "wavenumber" in lc or lc in ("wn", "nu"):
            colmap[c] = "wavenumber"
        elif "abs" in lc or "absorption" in lc or "intensity" in lc:
            colmap[c] = "Absorption"
    df = df.rename(columns=colmap)
    needed = {"X","Y","wavenumber","Absorption"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Map file must contain columns {needed}. Found {set(df.columns)}")
    return df[["X","Y","wavenumber","Absorption"]].copy()


def pivot_pixels(df):
    # pixel id by unique XY
    df["pix_id"] = df.groupby(["X","Y"]).ngroup()
    pix_xy = df.drop_duplicates("pix_id")[["pix_id","X","Y"]].set_index("pix_id")
    piv = df.pivot_table(index="pix_id", columns="wavenumber", values="Absorption", aggfunc="mean")
    wns = piv.columns.to_numpy(dtype=float)
    Xmat = piv.to_numpy(dtype=float)
    pix_ids = piv.index.to_numpy()
    XY = pix_xy.loc[pix_ids][["X","Y"]].to_numpy(dtype=float)
    return Xmat, wns, XY


def fill_nan_by_col_median(X):
    med = np.nanmedian(X, axis=0)
    ii = np.where(~np.isfinite(X))
    X = X.copy()
    X[ii] = med[ii[1]]
    return X


def make_xy_grid(XY):
    xs = np.sort(np.unique(XY[:, 0]))
    ys = np.sort(np.unique(XY[:, 1]))
    x_to_i = {v: i for i, v in enumerate(xs)}
    y_to_i = {v: i for i, v in enumerate(ys)}
    xi = np.array([x_to_i[v] for v in XY[:, 0]], dtype=int)
    yi = np.array([y_to_i[v] for v in XY[:, 1]], dtype=int)
    return xs, ys, xi, yi


def build_cube(Xmat, XY):
    xs, ys, xi, yi = make_xy_grid(XY)
    H, W = len(ys), len(xs)
    C = Xmat.shape[1]
    cube = np.full((H, W, C), np.nan, dtype=float)
    cube[yi, xi, :] = Xmat
    return cube, xs, ys


def block_bin_cube(cube, bin_size):
    H, W, C = cube.shape
    Hb = H // bin_size
    Wb = W // bin_size
    cube = cube[:Hb*bin_size, :Wb*bin_size, :]
    cube = cube.reshape(Hb, bin_size, Wb, bin_size, C)
    return np.nanmean(cube, axis=(1, 3))


def cube_to_pixels(cube):
    # flatten valid pixels
    H, W, C = cube.shape
    X = cube.reshape(H*W, C)
    valid = np.isfinite(X).all(axis=1)
    return X[valid], valid.reshape(H, W)


# =========================
# Preprocess spectra
# =========================
def preprocess_spectra(
    Xmat, wns,
    lam=3e5, p=0.01, niter=10,
    thickness_norm=True, norm_mode="median",
    smooth=True, sg_win=11, sg_poly=3,
):
    X = fill_nan_by_col_median(Xmat)

    X_clean = np.empty_like(X)
    for i in range(X.shape[0]):
        y = X[i, :]
        base = als_baseline(y, lam=lam, p=p, niter=niter)
        y2 = y - base

        if thickness_norm:
            s = np.median(y2) if norm_mode == "median" else np.mean(y2)
            if not np.isfinite(s) or abs(s) < 1e-9:
                s = 1.0
            y2 = y2 / s

        if smooth and sg_win < len(y2):
            if sg_win % 2 == 0:
                sg_win += 1
            y2 = savgol_filter(y2, window_length=sg_win, polyorder=sg_poly)

        X_clean[i, :] = y2

    return X_clean


# =========================
# Step 1: Gaussian SNR optimisation
# score = ∫ w(ν)*A(ν) dν / sqrt( ∫ w(ν)^2 * σ(ν)^2 dν )
# where A(ν) is mean baseline-removed spectrum in Amide band
# =========================
def gaussian_weight(wns, mu, fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * LN2))
    return np.exp(-0.5 * ((wns - mu) / sigma) ** 2)


def snr_objective(wns, mean_spec, noise_spec, amide_mask, mu, fwhm):
    w = gaussian_weight(wns, mu, fwhm)
    num = np.trapz(w[amide_mask] * mean_spec[amide_mask], wns[amide_mask])
    denom = np.sqrt(np.trapz((w**2) * (noise_spec**2), wns) + EPS)
    return float(num / denom), float(num), float(denom)


def grid_search_gaussian(
    wns, mean_spec, noise_spec,
    amide_range=(1600, 1700),
    mu_grid=np.linspace(1605, 1695, 61),
    fwhm_grid=np.linspace(20, 140, 61),
):
    amide_mask = (wns >= amide_range[0]) & (wns <= amide_range[1])
    scores = np.zeros((len(fwhm_grid), len(mu_grid)), dtype=float)
    nums = np.zeros_like(scores)
    dens = np.zeros_like(scores)

    best = (-np.inf, None, None, None, None)
    for i, fwhm in enumerate(fwhm_grid):
        for j, mu in enumerate(mu_grid):
            s, num, den = snr_objective(wns, mean_spec, noise_spec, amide_mask, mu, fwhm)
            scores[i, j] = s
            nums[i, j] = num
            dens[i, j] = den
            if s > best[0]:
                best = (s, mu, fwhm, num, den)

    return scores, nums, dens, best, mu_grid, fwhm_grid


# =========================
# Step 2: Non-Gaussian learned weighting
# We create proxy labels from the best Gaussian response across pixels,
# then learn a linear spectral weighting w via LogisticRegression (or Ridge).
# =========================
def standardize(X):
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=1)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd, mu, sd


def learn_weighting_function(X_pixels, proxy_score, seed=0):
    # binary labels by median split (balanced)
    thr = np.median(proxy_score)
    y = (proxy_score >= thr).astype(int)

    Xz, mu, sd = standardize(X_pixels)

    Xtr, Xte, ytr, yte = train_test_split(Xz, y, test_size=0.25, random_state=seed, stratify=y)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
    clf.fit(Xtr, ytr)

    p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)

    w = clf.coef_.ravel()  # learned weights in z-space
    return w, auc, thr, (mu, sd)


def apply_weighting(X_pixels, w, scaler):
    mu, sd = scaler
    Xz = (X_pixels - mu) / sd
    return Xz @ w


# =========================
# DigiStain (optional) using windows
# You may want this only for comparison.
# DI = (log10(S1)-log10(S4)) / (log10(S2)-log10(S3))
# where Sk = ∫ T_sample(ν)*T_windowk(ν) dν across available ν.
# Here, T_sample = 10^(-Absorption) BUT if you're using baseline-removed spectra,
# treat them carefully; for Digistain comparison use the *raw absorbance* map or
# a consistent absorbance baseline correction that preserves absolute A.
# =========================
def read_window_csv(path):
    df = pd.read_csv(path, sep=None, engine="python")
    cols = [c.strip().lower() for c in df.columns]
    wn_col = None
    y_col = None
    kind = None
    for c in df.columns:
        lc = c.strip().lower()
        if "wavenumber" in lc or lc in ("wn","x"):
            wn_col = c
        if "abs" in lc or "absorb" in lc:
            y_col = c
            kind = "abs"
        if "trans" in lc or "transmission" in lc or lc in ("t", "tr", "t%"):
            y_col = c
            kind = "trans"
    if wn_col is None:
        wn_col = df.columns[0]
    if y_col is None:
        y_col = df.columns[1]
        kind = "abs"

    wn = pd.to_numeric(df[wn_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    m = np.isfinite(wn) & np.isfinite(y)
    wn, y = wn[m], y[m]
    s = np.argsort(wn)
    wn, y = wn[s], y[s]

    if kind == "abs":
        Tw = 10.0 ** (-y)
    else:
        mx = np.nanmax(y)
        Tw = y if mx <= 1.5 else (y / 100.0)
    return wn, np.clip(Tw, 0.0, 1.0)


def digistain_from_raw_absorbance(Xraw_pixels, wns, windows_paths):
    # Xraw_pixels: absorbance A(ν) (NOT baseline-removed arbitrary units)
    # T_sample = 10^-A
    Ts = 10.0 ** (-Xraw_pixels)  # (Npix, C)

    Tws = []
    for p in windows_paths:
        wn_w, Tw = read_window_csv(p)
        Tw_i = np.interp(wns, wn_w, Tw, left=np.nan, right=np.nan)
        Tw_i = np.where(np.isfinite(Tw_i), Tw_i, 0.0)
        Tws.append(Tw_i)

    # integrate Sk = ∫ Ts(ν)*Twk(ν) dν
    S = []
    for k in range(4):
        weighted = Ts * Tws[k][None, :]
        Sk = np.trapz(weighted, wns, axis=1)
        S.append(Sk)

    # A=Window1, C=Window2, D=Window3, B=Window4 (matching your earlier nomenclature)
    A = np.log10(np.maximum(S[0], EPS))
    C = np.log10(np.maximum(S[1], EPS))
    D = np.log10(np.maximum(S[2], EPS))
    B = np.log10(np.maximum(S[3], EPS))
    DI = (A - B) / np.maximum((C - D), EPS)
    return DI


# =========================
# Main pipeline over bin sizes
# =========================
def run(map_path, outdir, windows=None, wn_min=1100, wn_max=1750,
        lam=3e5, p=0.01, niter=10):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    print("Loading map:", map_path)
    df = read_long_map(map_path, delim=None)

    # Crop
    df = df[(df["wavenumber"] >= wn_min) & (df["wavenumber"] <= wn_max)].copy()

    # Pivot raw absorbance
    Xraw, wns, XY = pivot_pixels(df)
    # Keep only columns within crop
    wns = wns.astype(float)
    order = np.argsort(wns)
    wns = wns[order]
    Xraw = Xraw[:, order]
    Xraw = fill_nan_by_col_median(Xraw)

    # Build raw cube for binning
    raw_cube, xs, ys = build_cube(Xraw, XY)

    # Bin sweep
    bin_sizes = [1, 2, 3, 4]
    pareto_rows = []

    for b in bin_sizes:
        print(f"\n=== BIN {b}x{b} ===")
        cube_b = raw_cube if b == 1 else block_bin_cube(raw_cube, b)
        Xraw_b, valid_mask = cube_to_pixels(cube_b)

        # Baseline remove + thickness norm + smoothing => "chemistry" space
        Xclean_b = preprocess_spectra(
            Xraw_b, wns,
            lam=lam, p=p, niter=niter,
            thickness_norm=True, norm_mode="median",
            smooth=True, sg_win=11, sg_poly=3
        )

        # Empirical noise σ(ν) across pixels (after baseline removal)
        mean_spec = np.mean(Xclean_b, axis=0)
        noise_spec = np.std(Xclean_b, axis=0, ddof=1)

        # Save mean/noise plots
        plt.figure(figsize=(10,4))
        plt.plot(wns, mean_spec)
        plt.gca().invert_xaxis()
        plt.title(f"Mean baseline-removed spectrum (bin {b}x{b})")
        plt.xlabel("Wavenumber (cm⁻1)")
        plt.tight_layout()
        plt.savefig(outdir / f"mean_spectrum_bin{b}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10,4))
        plt.plot(wns, noise_spec)
        plt.gca().invert_xaxis()
        plt.title(f"Empirical spectral noise σ(ν) (bin {b}x{b})")
        plt.xlabel("Wavenumber (cm⁻1)")
        plt.ylabel("Std dev across pixels")
        plt.tight_layout()
        plt.savefig(outdir / f"noise_spectrum_bin{b}.png", dpi=200)
        plt.close()

        # -----------------------
        # STEP 1: Gaussian SNR search
        # -----------------------
        mu_grid = np.linspace(1605, 1695, 61)
        fwhm_grid = np.linspace(20, 160, 71)  # expect 30–80-ish if baseline is fixed
        scores, nums, dens, best, mu_grid, fwhm_grid = grid_search_gaussian(
            wns, mean_spec, noise_spec,
            amide_range=(1600, 1700),
            mu_grid=mu_grid,
            fwhm_grid=fwhm_grid
        )
        best_score, best_mu, best_fwhm, best_num, best_den = best
        print(f"Best Gaussian (bin {b}): mu={best_mu:.2f}, FWHM={best_fwhm:.2f}, SNR-score={best_score:.6g}")

        # Heatmap
        plt.figure(figsize=(10,6))
        plt.imshow(scores, aspect="auto", origin="lower",
                   extent=[mu_grid[0], mu_grid[-1], fwhm_grid[0], fwhm_grid[-1]])
        plt.colorbar(label="SNR score")
        plt.scatter([best_mu], [best_fwhm], s=80)
        plt.xlabel("Gaussian center μ (cm⁻1)")
        plt.ylabel("FWHM (cm⁻1)")
        plt.title(f"Gaussian SNR optimisation (bin {b}x{b})")
        plt.tight_layout()
        plt.savefig(outdir / f"gaussian_heatmap_bin{b}.png", dpi=200)
        plt.close()

        # Overlay with mean spectrum (scaled)
        w_best = gaussian_weight(wns, best_mu, best_fwhm)
        w_best = w_best / (w_best.max() + EPS)
        mean_scaled = (mean_spec - mean_spec.min()) / (mean_spec.max() - mean_spec.min() + EPS)

        plt.figure(figsize=(10,4))
        plt.plot(wns, mean_scaled, label="Mean baseline-removed (scaled)")
        plt.plot(wns, w_best, label=f"Best Gaussian (μ={best_mu:.1f}, FWHM={best_fwhm:.1f})")
        plt.axvspan(1600, 1700, alpha=0.15, label="Amide band")
        plt.gca().invert_xaxis()
        plt.xlabel("Wavenumber (cm⁻1)")
        plt.title(f"Best Gaussian vs mean spectrum (bin {b}x{b})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"gaussian_best_overlay_bin{b}.png", dpi=200)
        plt.close()

        # Proxy score per pixel (Gaussian response)
        proxy = np.trapz(Xclean_b * w_best[None, :], wns, axis=1)

        # -----------------------
        # STEP 2: Learn non-Gaussian weighting
        # -----------------------
        w_ml, auc, thr, scaler = learn_weighting_function(Xclean_b, proxy, seed=0)
        print(f"Proxy AUC (bin {b}): {auc:.4f}")

        plt.figure(figsize=(10,4))
        plt.plot(wns, w_ml)
        plt.axhline(0, linewidth=1)
        plt.axvspan(1600, 1700, alpha=0.15, label="Amide band")
        plt.gca().invert_xaxis()
        plt.title(f"Learned spectral weighting w(ν) (bin {b}x{b})")
        plt.xlabel("Wavenumber (cm⁻1)")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(outdir / f"learned_weighting_function_bin{b}.png", dpi=200)
        plt.close()

        # ML score map
        ml_score = apply_weighting(Xclean_b, w_ml, scaler)

        # define SNR-like figure for the learned filter using same objective:
        # numerator = ∫ w * mean_spec (amide only)
        # denom = sqrt(∫ w^2 * σ^2)
        amide_mask = (wns >= 1600) & (wns <= 1700)
        num_ml = np.trapz(w_ml[amide_mask] * mean_spec[amide_mask], wns[amide_mask])
        den_ml = np.sqrt(np.trapz((w_ml**2) * (noise_spec**2), wns) + EPS)
        snr_ml = float(num_ml / den_ml)

        # Resolution proxy: bin size (larger bin => lower resolution)
        pareto_rows.append({
            "bin_size": b,
            "best_gaussian_mu": best_mu,
            "best_gaussian_fwhm": best_fwhm,
            "gaussian_snr_score": best_score,
            "ml_proxy_auc": auc,
            "ml_snr_score": snr_ml,
            "n_pixels_used": int(Xclean_b.shape[0])
        })

        # Optional Digistain comparison
        if windows is not None and len(windows) == 4:
            DI = digistain_from_raw_absorbance(Xraw_b, wns, windows)

            # simple correlation stats
            corr = np.corrcoef(DI, ml_score)[0, 1]
            txt = (
                f"bin {b}x{b}\n"
                f"Gaussian: mu={best_mu:.2f}, fwhm={best_fwhm:.2f}, score={best_score:.6g}\n"
                f"ML: auc={auc:.4f}, snr={snr_ml:.6g}\n"
                f"Corr(DI, MLscore) = {corr:.4f}\n"
            )
            (outdir / f"compare_DI_ML_bin{b}.txt").write_text(txt)

            # scatter
            plt.figure(figsize=(6,6))
            plt.scatter(DI, ml_score, s=5, alpha=0.5)
            plt.xlabel("Digistain DI (4 windows)")
            plt.ylabel("EntangleCam score (ML)")
            plt.title(f"DI vs ML score (bin {b}x{b}), r={corr:.2f}")
            plt.tight_layout()
            plt.savefig(outdir / f"DI_vs_ML_bin{b}.png", dpi=200)
            plt.close()

    pareto = pd.DataFrame(pareto_rows)
    pareto.to_csv(outdir / "pareto_resolution_vs_snr.csv", index=False)

    # Pareto plot
    plt.figure(figsize=(7,5))
    plt.plot(pareto["bin_size"], pareto["gaussian_snr_score"], marker="o", label="Gaussian SNR score")
    plt.plot(pareto["bin_size"], pareto["ml_snr_score"], marker="o", label="Learned filter SNR score")
    plt.xlabel("Spatial bin size (higher = lower resolution)")
    plt.ylabel("SNR score")
    plt.title("Resolution vs SNR tradeoff (Pareto)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pareto_resolution_vs_snr.png", dpi=200)
    plt.close()

    print("\nSaved outputs to:", outdir.resolve())
    print("Key file:", (outdir / "pareto_resolution_vs_snr.csv").name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="Long-format hyperspectral map file (X,Y,wavenumber,Absorption)")
    ap.add_argument("--out", default="outputs_entanglecam", help="Output directory")
    ap.add_argument("--windows", nargs="*", default=None, help="Optional 4 window CSV files for Digistain comparison")
    ap.add_argument("--wn_min", type=float, default=1100)
    ap.add_argument("--wn_max", type=float, default=1750)

    # baseline params
    ap.add_argument("--als_lambda", type=float, default=3e5)
    ap.add_argument("--als_p", type=float, default=0.01)
    ap.add_argument("--als_niter", type=int, default=10)

    args = ap.parse_args()

    windows = args.windows if args.windows and len(args.windows) == 4 else None
    run(
        map_path=args.map,
        outdir=args.out,
        windows=windows,
        wn_min=args.wn_min,
        wn_max=args.wn_max,
        lam=args.als_lambda,
        p=args.als_p,
        niter=args.als_niter
    )


if __name__ == "__main__":
    main()
