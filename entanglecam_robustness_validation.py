import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress

# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "/Users/nana/Desktop/viva plots/digistain_4methods_absorbance_corrected/08_DI_4methods_all_samples_absorbance_corrected.csv"
OUTPUT_DIR = "/Users/nana/Desktop/viva plots/entanglecam_boundedDI_robustness_results/entanglecam_validation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference and proxy columns
FTIR_COL = "DI_original_bounded"
ENT_COL = "DI_gaussian_bounded"

# Pretty labels
FTIR_LABEL = "FTIR reference DI"
ENT_LABEL = "EntangleCam proxy DI"

# Global style for presentation
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 22
})

# ============================================================
# HELPERS
# ============================================================

def agreement_metrics(x, y):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    x = tmp["x"].values
    y = tmp["y"].values

    if len(x) < 3:
        return None

    rho_s, p_s = spearmanr(x, y)
    rho_p, p_p = pearsonr(x, y)
    fit = linregress(x, y)

    rmse = np.sqrt(np.mean((y - x) ** 2))
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    nrmse = rmse / iqr if iqr > 0 else np.nan

    diff = y - x
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    return {
        "n": len(x),
        "spearman_rho": rho_s,
        "spearman_p": p_s,
        "pearson_r": rho_p,
        "pearson_p": p_p,
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r_squared": fit.rvalue ** 2,
        "rmse": rmse,
        "nrmse_iqr": nrmse,
        "bias": bias,
        "loa_low": loa_low,
        "loa_high": loa_high
    }

def quartile_agreement(x, y):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna().copy()
    if len(tmp) < 10:
        return None, None

    tmp["x_q"] = pd.qcut(tmp["x"], 4, labels=False, duplicates="drop")
    tmp["y_q"] = pd.qcut(tmp["y"], 4, labels=False, duplicates="drop")

    confusion = pd.crosstab(tmp["x_q"], tmp["y_q"])
    same = np.mean(tmp["x_q"] == tmp["y_q"])
    within1 = np.mean(np.abs(tmp["x_q"] - tmp["y_q"]) <= 1)

    return {
        "same_quartile_fraction": same,
        "within_one_quartile_fraction": within1,
        "n": len(tmp)
    }, confusion

def top_k_overlap(x, y, frac=0.10):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    n_top = max(1, int(np.ceil(frac * len(tmp))))
    top_x = set(tmp.nlargest(n_top, "x").index)
    top_y = set(tmp.nlargest(n_top, "y").index)
    overlap = len(top_x & top_y) / n_top
    return {
        "top_fraction": frac,
        "n_top": n_top,
        "n_total": len(tmp),
        "top_overlap_fraction": overlap
    }

# ============================================================
# PLOTS
# ============================================================

def plot_scatter(df, xcol, ycol, outpath):
    tmp = df[[xcol, ycol]].dropna()
    x = tmp[xcol].values
    y = tmp[ycol].values

    rho_s, _ = spearmanr(x, y)
    rho_p, _ = pearsonr(x, y)
    fit = linregress(x, y)

    xx = np.linspace(np.min(x), np.max(x), 200)
    yy = fit.intercept + fit.slope * xx

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))

    fig, ax = plt.subplots(figsize=(9, 7.5))
    hb = ax.hexbin(x, y, gridsize=28, mincnt=1, cmap="viridis")
    cbar = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Sample density")

    ax.plot(xx, yy, linewidth=3, label=f"Fit: y={fit.slope:.3f}x + {fit.intercept:.3f}")
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2, label="Identity line")

    ax.set_xlabel(FTIR_LABEL)
    ax.set_ylabel(ENT_LABEL)
    ax.set_title("FTIR vs EntangleCam bounded DI")

    text = (
        f"n = {len(tmp)}\n"
        f"Spearman ρ = {rho_s:.4f}\n"
        f"Pearson r = {rho_p:.4f}\n"
        f"R² = {fit.rvalue**2:.4f}"
    )
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92)
    )

    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_bland_altman(df, xcol, ycol, outpath):

    tmp = df[[xcol, ycol]].dropna().copy()

    mean_xy = 0.5 * (tmp[xcol] + tmp[ycol])
    diff = tmp[ycol] - tmp[xcol]

    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)

    low = bias - 1.96 * sd
    high = bias + 1.96 * sd

    fig, ax = plt.subplots(figsize=(9,7))

    hb = ax.hexbin(
        mean_xy,
        diff,
        gridsize=30,
        cmap="viridis",
        mincnt=1
    )

    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Sample density")

    ax.axhline(bias, color="red", linewidth=3, label=f"Bias = {bias:.3f}")
    ax.axhline(low, linestyle="--", color="black", linewidth=2,
               label=f"Lower LoA = {low:.3f}")
    ax.axhline(high, linestyle="--", color="black", linewidth=2,
               label=f"Upper LoA = {high:.3f}")

    ax.set_xlabel("Mean DI (FTIR + EntangleCam) / 2")
    ax.set_ylabel("Difference (EntangleCam − FTIR)")
    ax.set_title("Agreement between FTIR DI and EntangleCam DI")

    ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)

def plot_quartile_confusion(confusion, outpath):
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(confusion.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(confusion.shape[1]))
    ax.set_yticks(range(confusion.shape[0]))
    ax.set_xticklabels([f"Q{int(i)+1}" for i in confusion.columns])
    ax.set_yticklabels([f"Q{int(i)+1}" for i in confusion.index])

    ax.set_xlabel("EntangleCam quartile")
    ax.set_ylabel("FTIR quartile")
    ax.set_title("Quartile agreement")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(confusion.values[i, j]), ha="center", va="center")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_top10_bar(overlap, outpath):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    val = overlap["top_overlap_fraction"]
    ax.bar(["Top 10% overlap"], [val], width=0.55)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction")
    ax.set_title("High-DI preservation")
    ax.text(0, val + 0.03, f"{val:.2f}", ha="center", fontsize=15)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_distributions(df, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    axes[0].hist(df[FTIR_COL].dropna(), bins=35, edgecolor="black", alpha=0.85)
    axes[0].set_title("FTIR bounded DI")
    axes[0].set_xlabel("DI")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0, 1)

    axes[1].hist(df[ENT_COL].dropna(), bins=35, edgecolor="black", alpha=0.85)
    axes[1].set_title("EntangleCam bounded DI")
    axes[1].set_xlabel("DI")
    axes[1].set_ylabel("Count")
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_summary_dashboard(df, metrics, quart, overlap, outpath):
    """
    One-slide summary figure.
    """
    tmp = df[[FTIR_COL, ENT_COL]].dropna()
    x = tmp[FTIR_COL].values
    y = tmp[ENT_COL].values
    fit = linregress(x, y)
    xx = np.linspace(np.min(x), np.max(x), 200)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # scatter
    ax = axes[0, 0]
    hb = ax.hexbin(x, y, gridsize=28, mincnt=1, cmap="viridis")
    ax.plot(xx, fit.intercept + fit.slope * xx, linewidth=3)
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
    ax.set_title("Correlation")
    ax.set_xlabel(FTIR_LABEL)
    ax.set_ylabel(ENT_LABEL)

    # distributions
    ax = axes[0, 1]
    ax.hist(df[FTIR_COL].dropna(), bins=30, alpha=0.65, label="FTIR", edgecolor="black")
    ax.hist(df[ENT_COL].dropna(), bins=30, alpha=0.65, label="EntangleCam", edgecolor="black")
    ax.set_xlim(0, 1)
    ax.set_title("Distributions")
    ax.set_xlabel("Bounded DI")
    ax.set_ylabel("Count")
    ax.legend()

    # bland-altman
    ax = axes[1, 0]
    mean_xy = 0.5 * (tmp[FTIR_COL] + tmp[ENT_COL])
    diff = tmp[ENT_COL] - tmp[FTIR_COL]
    bias = metrics["bias"]
    low = metrics["loa_low"]
    high = metrics["loa_high"]
    ax.scatter(mean_xy, diff, s=18, alpha=0.45)
    ax.axhline(bias, linewidth=2.5)
    ax.axhline(low, linestyle="--", linewidth=2)
    ax.axhline(high, linestyle="--", linewidth=2)
    ax.set_title("Bland–Altman")
    ax.set_xlabel("Mean DI")
    ax.set_ylabel("Difference")

    # text summary
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = (
        f"n = {metrics['n']}\n\n"
        f"Spearman ρ = {metrics['spearman_rho']:.4f}\n"
        f"Pearson r = {metrics['pearson_r']:.4f}\n"
        f"R² = {metrics['r_squared']:.4f}\n\n"
        f"RMSE = {metrics['rmse']:.4f}\n"
        f"NRMSE(IQR) = {metrics['nrmse_iqr']:.4f}\n"
        f"Bias = {metrics['bias']:.4f}\n\n"
        f"Within ±1 quartile = {quart['within_one_quartile_fraction']:.3f}\n"
        f"Top 10% overlap = {overlap['top_overlap_fraction']:.3f}"
    )
    ax.text(
        0.02, 0.98, summary_text,
        va="top",
        fontsize=16,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95)
    )

    fig.suptitle("EntangleCam robustness validation using Gaussian simulated DI", y=0.99)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(CSV_FILE)

    if FTIR_COL not in df.columns or ENT_COL not in df.columns:
        raise KeyError(f"Need columns {FTIR_COL} and {ENT_COL} in {CSV_FILE}")

    metrics = agreement_metrics(df[FTIR_COL], df[ENT_COL])
    if metrics is None:
        raise RuntimeError("Not enough valid paired data for robustness analysis.")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "agreement_metrics_boundedDI.csv"), index=False)

    quart, confusion = quartile_agreement(df[FTIR_COL], df[ENT_COL])
    quart_df = pd.DataFrame([quart])
    quart_df.to_csv(os.path.join(OUTPUT_DIR, "quartile_agreement_boundedDI.csv"), index=False)
    confusion.to_csv(os.path.join(OUTPUT_DIR, "quartile_confusion_boundedDI.csv"))

    overlap = top_k_overlap(df[FTIR_COL], df[ENT_COL], frac=0.10)
    overlap_df = pd.DataFrame([overlap])
    overlap_df.to_csv(os.path.join(OUTPUT_DIR, "top10_overlap_boundedDI.csv"), index=False)

    plot_scatter(df, FTIR_COL, ENT_COL, os.path.join(OUTPUT_DIR, "01_scatter_boundedDI.png"))
    plot_bland_altman(df, FTIR_COL, ENT_COL, os.path.join(OUTPUT_DIR, "02_bland_altman_boundedDI.png"))
    plot_quartile_confusion(confusion, os.path.join(OUTPUT_DIR, "03_quartile_confusion_boundedDI.png"))
    plot_top10_bar(overlap, os.path.join(OUTPUT_DIR, "04_top10_overlap_boundedDI.png"))
    plot_distributions(df, os.path.join(OUTPUT_DIR, "05_boundedDI_distributions.png"))
    plot_summary_dashboard(df, metrics, quart, overlap, os.path.join(OUTPUT_DIR, "06_summary_dashboard_boundedDI.png"))

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("EntangleCam robustness validation using bounded DI\n\n")
        f.write(f"Reference = {FTIR_COL}\n")
        f.write(f"Proxy     = {ENT_COL}\n\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")
        f.write(quart_df.to_string(index=False))
        f.write("\n\n")
        f.write(overlap_df.to_string(index=False))
        f.write("\n\nInterpretation:\n")
        f.write("- Spearman rho tests whether EntangleCam preserves FTIR ranking.\n")
        f.write("- Pearson r and R² test numerical linear similarity.\n")
        f.write("- RMSE and NRMSE quantify absolute disagreement.\n")
        f.write("- Bland–Altman quantifies bias and limits of agreement.\n")
        f.write("- Quartile agreement checks whether samples remain in similar DI categories.\n")
        f.write("- Top-10% overlap checks whether the highest-contrast samples are preserved.\n")

    print("Done.")
    print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()