import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress

# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "/Users/nana/Desktop/project_metadata/digistain_4method_results/DI_4methods_all_samples.csv"
EPS = 1e-12

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(CSV_FILE)

# ============================================================
# BOUNDED DI DEFINITIONS
# ============================================================
# Using:
#   P = phosphate contrast = P_peak - P_base
#   A = amide contrast     = A_peak - A_base
#
# Bounded index:
#   DI = P / (P + A)
#
# This is in [0,1] if P>=0 and A>=0.
# If either becomes negative from noise / bad baseline behaviour,
# we clip them to zero before forming the ratio.

def bounded_di(p_peak, p_base, a_peak, a_base, clip_negative=True):
    P = p_peak - p_base
    A = a_peak - a_base

    if clip_negative:
        P = np.maximum(P, 0)
        A = np.maximum(A, 0)

    denom = P + A
    out = np.where(np.abs(denom) > EPS, P / denom, np.nan)
    return out

# ============================================================
# COMPUTE BOUNDED DI FOR ORIGINAL / GAUSSIAN / TRUNCATED / BG
# ============================================================

df["DI_original_bounded"] = bounded_di(
    df["orig_P_peak"], df["orig_P_base"],
    df["orig_A_peak"], df["orig_A_base"]
)

df["DI_gaussian_bounded"] = bounded_di(
    df["gauss_P_peak"], df["gauss_P_base"],
    df["gauss_A_peak"], df["gauss_A_base"]
)

df["DI_truncated_bounded"] = bounded_di(
    df["trunc_P_peak"], df["trunc_P_base"],
    df["trunc_A_peak"], df["trunc_A_base"]
)

df["DI_bg_bounded"] = bounded_di(
    df["bg_P_peak"], df["bg_P_base"],
    df["bg_A_peak"], df["bg_A_base"]
)

# Save updated CSV
df.to_csv("DI_4methods_all_samples_with_bounded_DI.csv", index=False)

# ============================================================
# SUMMARY CHECK
# ============================================================

bounded_cols = [
    "DI_original_bounded",
    "DI_gaussian_bounded",
    "DI_truncated_bounded",
    "DI_bg_bounded"
]

summary_rows = []
for col in bounded_cols:
    s = df[col].dropna()
    summary_rows.append({
        "column": col,
        "n_valid": len(s),
        "min": s.min() if len(s) else np.nan,
        "max": s.max() if len(s) else np.nan,
        "mean": s.mean() if len(s) else np.nan,
        "median": s.median() if len(s) else np.nan,
        "std": s.std() if len(s) else np.nan,
        "n_outside_0_1": int(((s < 0) | (s > 1)).sum()) if len(s) else np.nan
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("bounded_DI_summary.csv", index=False)
print(summary_df)

# ============================================================
# PLOTTING HELPER
# ============================================================

def scatter_with_fit(df, xcol, ycol, outname, xlabel=None, ylabel=None, title=None):
    tmp = df[[xcol, ycol]].dropna().copy()
    x = tmp[xcol].values
    y = tmp[ycol].values

    rho_s, p_s = spearmanr(x, y)
    rho_p, p_p = pearsonr(x, y)
    fit = linregress(x, y)

    xx = np.linspace(np.min(x), np.max(x), 200)
    yy = fit.intercept + fit.slope * xx

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, s=14, alpha=0.45, label="Samples")
    ax.plot(xx, yy, linewidth=2, label=f"Fit: y={fit.slope:.3f}x+{fit.intercept:.3f}")

    minv = min(np.min(x), np.min(y))
    maxv = max(np.max(x), np.max(y))
    ax.plot([minv, maxv], [minv, maxv], linestyle="--", linewidth=1.5, label="Identity line")

    ax.set_xlabel(xlabel if xlabel else xcol)
    ax.set_ylabel(ylabel if ylabel else ycol)
    ax.set_title(title if title else f"{ycol} vs {xcol}")

    text = (
        f"n = {len(tmp)}\n"
        f"Spearman rho = {rho_s:.4f}\n"
        f"Pearson r = {rho_p:.4f}\n"
        f"R² = {fit.rvalue**2:.4f}"
    )
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    ax.legend()
    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.show()

# ============================================================
# MAIN PLOTS
# ============================================================

# 1. Old vs bounded original
scatter_with_fit(
    df,
    "DI_original_filter",
    "DI_original_bounded",
    "plot_old_vs_bounded_original.png",
    xlabel="Old DI original filter (unnormalised)",
    ylabel="Bounded DI original filter",
    title="Old vs bounded DI (original filter)"
)

# 2. Old vs bounded gaussian
scatter_with_fit(
    df,
    "DI_gaussian_filter",
    "DI_gaussian_bounded",
    "plot_old_vs_bounded_gaussian.png",
    xlabel="Old DI Gaussian filter (unnormalised)",
    ylabel="Bounded DI Gaussian filter",
    title="Old vs bounded DI (Gaussian filter)"
)

# 3. Bounded original vs bounded gaussian
scatter_with_fit(
    df,
    "DI_original_bounded",
    "DI_gaussian_bounded",
    "plot_bounded_original_vs_bounded_gaussian.png",
    xlabel="Bounded DI original filter (FTIR reference)",
    ylabel="Bounded DI Gaussian filter (EntangleCam proxy)",
    title="Bounded DI original vs bounded DI Gaussian"
)

# ============================================================
# OPTIONAL HISTOGRAMS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.ravel()

for ax, col in zip(axes, bounded_cols):
    s = df[col].dropna()
    ax.hist(s, bins=40)
    ax.set_title(col)
    ax.set_xlabel("DI")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig("bounded_DI_histograms.png", dpi=300)
plt.show()

print("Done. Files written:")
print("- DI_4methods_all_samples_with_bounded_DI.csv")
print("- bounded_DI_summary.csv")
print("- plot_old_vs_bounded_original.png")
print("- plot_old_vs_bounded_gaussian.png")
print("- plot_bounded_original_vs_bounded_gaussian.png")
print("- bounded_DI_histograms.png")