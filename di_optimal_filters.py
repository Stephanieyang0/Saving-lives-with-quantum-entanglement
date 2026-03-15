#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DI comparison:
- Build noise-robust matched filters for Amide and Phosphate on baseline-removed spectra
- DI_new = (wP^T A) / (wA^T A)
- DI_old = [lnT(P_peak)-lnT(P_trough)] / [lnT(A_peak)-lnT(A_trough)]
- Compare SNR + spatial stability + baseline sensitivity
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def gaussian_window(wn, mu, fwhm):
    """Gaussian weighting (not necessarily normalised)."""
    wn = np.asarray(wn, float)
    fwhm = float(fwhm)
    if fwhm <= 0:
        raise ValueError("fwhm must be > 0")
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((wn - mu) / sigma) ** 2)

def robust_mad(x, axis=0):
    """MAD scaled to sigma for approx Gaussian."""
    x = np.asarray(x)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis)
    return 1.4826 * mad

def moving_average(y, k=11):
    """Simple smoothing for noise estimation; k odd."""
    y = np.asarray(y, float)
    k = int(k)
    if k < 3:
        return y.copy()
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")

def estimate_sigma_from_residuals(A_pixels_by_wn, smooth_k=11):
    """
    Estimate sigma(ν) robustly by:
      smooth spectrum per pixel -> residual (high-freq)
      sigma(ν) = MAD(residuals across pixels) at each ν
    """
    X = np.asarray(A_pixels_by_wn, float)  # shape (Npix, Nwn)
    Xs = np.apply_along_axis(moving_average, 1, X, k=smooth_k)
    resid = X - Xs
    sigma = robust_mad(resid, axis=0)  # (Nwn,)
    # avoid divide-by-zero
    sigma = np.maximum(sigma, np.nanpercentile(sigma, 5) * 0.1 + 1e-12)
    return sigma, resid

def baseline_remove_poly(A_pixels_by_wn, wn, deg=3, anchor_ranges=None):
    """
    Baseline removal using polynomial fit on anchor regions.
    anchor_ranges: list of (low, high) wn ranges used to fit baseline.
    If None -> uses whole range (less ideal).
    """
    X = np.asarray(A_pixels_by_wn, float)
    wn = np.asarray(wn, float)

    if anchor_ranges is None:
        mask = np.isfinite(wn)
    else:
        mask = np.zeros_like(wn, dtype=bool)
        for lo, hi in anchor_ranges:
            mask |= (wn >= lo) & (wn <= hi)

    wfit = wn[mask]
    if wfit.size < deg + 2:
        # fallback: whole range
        mask = np.isfinite(wn)
        wfit = wn[mask]

    X_out = np.empty_like(X)
    for i in range(X.shape[0]):
        y = X[i, :]
        yfit = y[mask]
        # skip if too many nans
        if np.count_nonzero(np.isfinite(yfit)) < deg + 2:
            X_out[i, :] = y
            continue
        coeff = np.polyfit(wfit, yfit, deg)
        base = np.polyval(coeff, wn)
        X_out[i, :] = y - base
    return X_out

def unit_energy_normalise(w):
    """Normalise filter so ∫ w(ν)^2 dν = 1 (discrete sum)."""
    w = np.asarray(w, float)
    e = np.sqrt(np.sum(w**2))
    return w / (e + 1e-12)

def build_matched_filter(signal_shape, sigma, band_mask=None, leakage_penalty=0.0):
    """
    Matched filter (noise-whitened):
      w ∝ signal_shape / sigma^2
    with:
      - optional band mask (force to zero outside band)
      - energy normalisation
      - optional leakage penalty: discourage nonzero weights outside band
        (implemented by shrinking outside band further)
    """
    s = np.asarray(signal_shape, float)
    sig = np.asarray(sigma, float)
    w = s / (sig**2 + 1e-12)

    if band_mask is not None:
        band_mask = np.asarray(band_mask, bool)
        if leakage_penalty > 0:
            # softly suppress outside band instead of hard zero
            w_out = w.copy()
            w_out[~band_mask] *= (1.0 / (1.0 + leakage_penalty))
            w = w_out
        else:
            w = w * band_mask.astype(float)

    w = unit_energy_normalise(w)
    return w

def compute_scores(X, w):
    """Per-pixel dot product: score = w^T A."""
    return X @ w

def robust_spatial_noise_metric(score, xs, ys, k=6):
    """
    Cheap spatial stability metric without gridding:
    - take each point's k-NN median absolute deviation of score
    - return median of that (lower is more stable)
    NOTE: O(N^2) if naive. Here: approximate via coarse binning on coordinates.
    """
    df = pd.DataFrame({"x": xs, "y": ys, "s": score})
    # coarse bin to speed up:  (adjust bin_size if needed)
    bin_size = max(np.nanstd(xs), np.nanstd(ys)) / 60.0
    if not np.isfinite(bin_size) or bin_size <= 0:
        return np.nan

    df["bx"] = np.floor(df["x"] / bin_size).astype(int)
    df["by"] = np.floor(df["y"] / bin_size).astype(int)

    # within each bin, compute robust spread
    spreads = []
    for _, g in df.groupby(["bx", "by"]):
        if len(g) < 8:
            continue
        spreads.append(float(robust_mad(g["s"].values, axis=0)))
    if len(spreads) == 0:
        return np.nan
    return float(np.nanmedian(spreads))

def snr_between_two_groups(score, mask_hi, mask_lo):
    """
    SNR-like separability: |median_hi - median_lo| / pooled_MAD
    """
    a = score[mask_hi]
    b = score[mask_lo]
    if a.size < 10 or b.size < 10:
        return np.nan
    med_a = np.nanmedian(a)
    med_b = np.nanmedian(b)
    mad_a = robust_mad(a, axis=0)
    mad_b = robust_mad(b, axis=0)
    pooled = np.sqrt(mad_a**2 + mad_b**2) + 1e-12
    return float(np.abs(med_a - med_b) / pooled)

# -----------------------------
# Data loader (long format)
# -----------------------------

def load_long_map(path, wn_col="wavenumber", val_col=None, x_col="X", y_col="Y"):
    """
    Expects long table with columns: X, Y, wavenumber, value.
    value column can be ABS_clean or Absorption etc.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}
    if wn_col.lower() not in cols:
        raise ValueError(f"Cannot find wavenumber column in {df.columns}")
    wn_col = cols[wn_col.lower()]

    if x_col.lower() not in cols or y_col.lower() not in cols:
        raise ValueError(f"Need X and Y columns. Found: {df.columns}")
    x_col = cols[x_col.lower()]
    y_col = cols[y_col.lower()]

    if val_col is None:
        # try common names
        for cand in ["abs_clean", "absorption", "abs", "a", "alpha", "value"]:
            if cand in cols:
                val_col = cols[cand]
                break
    if val_col is None:
        raise ValueError("Could not infer value column. Pass --val_col explicitly.")

    wn = np.sort(df[wn_col].unique())
    # pivot to pixels x wn
    pix = df[[x_col, y_col]].drop_duplicates().reset_index(drop=True)
    pix["pid"] = np.arange(len(pix))

    df2 = df.merge(pix, on=[x_col, y_col], how="inner")
    # pivot
    M = df2.pivot_table(index="pid", columns=wn_col, values=val_col, aggfunc="mean")
    M = M.reindex(columns=wn)  # ensure ordering
    X = M.values  # (Npix, Nwn)

    xs = pix[x_col].values.astype(float)
    ys = pix[y_col].values.astype(float)

    return wn.astype(float), X.astype(float), xs, ys

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Long-format CSV/TXT map with X,Y,wavenumber,value")
    ap.add_argument("--val_col", default=None, help="Value column name (e.g. ABS_clean). If omitted, auto-tries.")
    ap.add_argument("--out", default="out_di_filters", help="Output directory")

    # Bands (adjust if your lab uses slightly different)
    ap.add_argument("--amide_min", type=float, default=1600.0)
    ap.add_argument("--amide_max", type=float, default=1700.0)
    ap.add_argument("--phos_min", type=float, default=1000.0)
    ap.add_argument("--phos_max", type=float, default=1150.0)

    # Old DI “4 windows” params (centers + FWHM)
    ap.add_argument("--A_peak", type=float, default=1650.0)
    ap.add_argument("--A_trough", type=float, default=1600.0)
    ap.add_argument("--P_peak", type=float, default=1080.0)
    ap.add_argument("--P_trough", type=float, default=1120.0)
    ap.add_argument("--fwhm_old", type=float, default=40.0)

    # Baseline removal
    ap.add_argument("--baseline_deg", type=int, default=3)
    ap.add_argument("--smooth_k", type=int, default=11)

    args = ap.parse_args()
    ensure_dir(args.out)

    # -------- Load --------
    wn, A_raw, xs, ys = load_long_map(args.input, val_col=args.val_col)
    print(f"Loaded: Npix={A_raw.shape[0]}  Nwn={A_raw.shape[1]}  wn[{wn.min():.1f},{wn.max():.1f}]")

    # Crop to common useful range (optional)
    crop_mask = (wn >= 1000) & (wn <= 1750)
    wn = wn[crop_mask]
    A_raw = A_raw[:, crop_mask]

    # -------- Baseline remove (anchor regions away from main bands) --------
    # You can tune anchors. Idea: fit slow baseline outside the informative bands.
    anchors = [
        (1000, 1050),
        (1180, 1250),
        (1450, 1500),
        (1720, 1750),
    ]
    A_bl = baseline_remove_poly(A_raw, wn, deg=args.baseline_deg, anchor_ranges=anchors)

    # -------- Robust sigma(ν) from high-frequency residuals --------
    sigma, resid = estimate_sigma_from_residuals(A_bl, smooth_k=args.smooth_k)

    # -------- Define band masks --------
    amide_mask = (wn >= args.amide_min) & (wn <= args.amide_max)
    phos_mask  = (wn >= args.phos_min) & (wn <= args.phos_max)

    # -------- Build “signal shapes” for matched filter --------
    # Matched filter needs a target shape. We use:
    #   - mean spectrum inside band minus mean outside band (peak-trough style)
    mean_spec = np.nanmedian(A_bl, axis=0)

    # Build a shape that is high in band and low outside:
    # (simple: band indicator * (mean in band - mean outside))
    mean_in_amide = np.nanmedian(mean_spec[amide_mask])
    mean_out_amide = np.nanmedian(mean_spec[~amide_mask])
    s_amide = np.zeros_like(mean_spec)
    s_amide[amide_mask] = (mean_in_amide - mean_out_amide)

    mean_in_phos = np.nanmedian(mean_spec[phos_mask])
    mean_out_phos = np.nanmedian(mean_spec[~phos_mask])
    s_phos = np.zeros_like(mean_spec)
    s_phos[phos_mask] = (mean_in_phos - mean_out_phos)

    # -------- Matched filters (noise-whitened, band-limited, unit-energy) --------
    wA = build_matched_filter(s_amide, sigma, band_mask=amide_mask, leakage_penalty=0.0)
    wP = build_matched_filter(s_phos,  sigma, band_mask=phos_mask,  leakage_penalty=0.0)

    # -------- Scores and DI_new --------
    scoreA = compute_scores(A_bl, wA)
    scoreP = compute_scores(A_bl, wP)

    # DI_new = (wP^T A)/(wA^T A)
    DI_new = scoreP / (scoreA + 1e-12)

    # -------- Old DI (supervisor equation) using Gaussian “software windows” on lnT --------
    # If your value is already A = alpha*d ~= -ln(T/(1-R)), then lnT differs by const.
    # Here we approximate lnT as -A (up to reflectivity constant). This is consistent with the “work in absorption” rule.
    lnT = -A_raw[:, :]  # if A_raw is absorbance-like; adjust if your input is actually T.

    wA_peak   = unit_energy_normalise(gaussian_window(wn, args.A_peak,   args.fwhm_old))
    wA_trough = unit_energy_normalise(gaussian_window(wn, args.A_trough, args.fwhm_old))
    wP_peak   = unit_energy_normalise(gaussian_window(wn, args.P_peak,   args.fwhm_old))
    wP_trough = unit_energy_normalise(gaussian_window(wn, args.P_trough, args.fwhm_old))

    lnT_A_peak   = lnT @ wA_peak
    lnT_A_trough = lnT @ wA_trough
    lnT_P_peak   = lnT @ wP_peak
    lnT_P_trough = lnT @ wP_trough

    num = (lnT_P_peak - lnT_P_trough)
    den = (lnT_A_peak - lnT_A_trough) + 1e-12
    DI_old = num / den

    # -------- “SNR improvement” proxy --------
    # We don’t have true labels here; make a proxy split by amide score quantiles:
    q_lo, q_hi = np.nanquantile(scoreA, [0.1, 0.9])
    mask_lo = scoreA <= q_lo
    mask_hi = scoreA >= q_hi

    snr_old = snr_between_two_groups(DI_old, mask_hi, mask_lo)
    snr_new = snr_between_two_groups(DI_new, mask_hi, mask_lo)

    # -------- Spatial stability proxy (lower is better) --------
    stab_old = robust_spatial_noise_metric(DI_old, xs, ys)
    stab_new = robust_spatial_noise_metric(DI_new, xs, ys)

    # -------- Save summary --------
    summary = {
        "Npix": int(A_raw.shape[0]),
        "Nwn": int(A_raw.shape[1]),
        "snr_proxy_old": snr_old,
        "snr_proxy_new": snr_new,
        "spatial_noise_proxy_old": stab_old,
        "spatial_noise_proxy_new": stab_new,
        "note": "snr_proxy uses top/bottom 10% amide score as proxy groups; replace with ROI labels if available."
    }
    pd.Series(summary).to_csv(os.path.join(args.out, "summary_metrics.csv"))
    print("Saved:", os.path.join(args.out, "summary_metrics.csv"))
    print(summary)

    # -------- Plots --------
    # 1) Mean spectrum + band masks
    plt.figure(figsize=(12,4))
    plt.plot(wn, mean_spec, linewidth=2)
    plt.axvspan(args.amide_min, args.amide_max, alpha=0.15, label="Amide band")
    plt.axvspan(args.phos_min, args.phos_max, alpha=0.15, label="Phosphate band")
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Mean baseline-removed A")
    plt.title("Mean spectrum (baseline-removed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "mean_spectrum_baseline_removed.png"), dpi=180)
    plt.close()

    # 2) sigma(ν)
    plt.figure(figsize=(12,4))
    plt.plot(wn, sigma, linewidth=2)
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Robust noise σ(ν) (MAD of high-freq residual)")
    plt.title("Estimated spectral noise σ(ν)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "sigma_nu.png"), dpi=180)
    plt.close()

    # 3) Matched filters wA, wP
    plt.figure(figsize=(12,4))
    plt.plot(wn, wA, label="wA (Amide matched filter)")
    plt.plot(wn, wP, label="wP (Phosphate matched filter)")
    plt.axvspan(args.amide_min, args.amide_max, alpha=0.1)
    plt.axvspan(args.phos_min, args.phos_max, alpha=0.1)
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Weight (unit-energy)")
    plt.title("Learned matched filters (noise-whitened, band-limited)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "matched_filters_wA_wP.png"), dpi=180)
    plt.close()

    # 4) Histograms DI_old vs DI_new
    plt.figure(figsize=(10,4))
    plt.hist(DI_old[np.isfinite(DI_old)], bins=80, alpha=0.5, label="DI_old")
    plt.hist(DI_new[np.isfinite(DI_new)], bins=80, alpha=0.5, label="DI_new")
    plt.xlabel("DI value")
    plt.ylabel("Count")
    plt.title("Distribution: old 4-window DI vs new optimal-filter DI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "hist_DI_old_vs_new.png"), dpi=180)
    plt.close()

    # 5) Scatter compare
    plt.figure(figsize=(5,5))
    m = np.isfinite(DI_old) & np.isfinite(DI_new)
    plt.scatter(DI_old[m], DI_new[m], s=5, alpha=0.3)
    plt.xlabel("DI_old (4-window)")
    plt.ylabel("DI_new (matched filters)")
    plt.title("Per-pixel DI comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "scatter_DI_old_vs_new.png"), dpi=180)
    plt.close()

    print(f"\nDone. Outputs in: {args.out}")

if __name__ == "__main__":
    main()
