
# import os
# import re
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import spearmanr, pearsonr, linregress

# # ============================================================
# # SETTINGS
# # ============================================================

# RAW_FILE = "/Users/nana/Desktop/project_metadata/digistain_4method_results/DI_4methods_all_samples.csv"
# CLINICAL_FILE = "/Users/nana/Desktop/project_metadata/clinical_metadata(Sheet1).csv"
# MERGE_NEW = "sample"
# MERGE_CLINICAL = "Anonymised Identifier"

# OUTPUT_DIR = "correct_ratio_DI_compare_results"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # choose which clinical DI columns to compare against
# CLINICAL_DI_COLS = [
#     "DIv1 - AmidePhosphateRatios",
#     "DIv2 - DigistainIndices",
#     "DIv3 - DigistainIndicesTrimmed",
#     "DIv4 - NormalisedAmideHeights",
#     "DIv5 - NormalisedPhosphateHeights",
#     "DIv6 - UnnormalisedAmidePhosphateRatios"
# ]

# EPS = 1e-12
# SHIFT_A = 0.6798   # your scale factor

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
#     except ValueError:
#         return s

# def safe_ratio(num, den):
#     num = np.asarray(num, dtype=float)
#     den = np.asarray(den, dtype=float)
#     out = np.full_like(num, np.nan, dtype=float)
#     mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > EPS)
#     out[mask] = num[mask] / den[mask]
#     return out

# def minmax_01(x):
#     x = np.asarray(x, dtype=float)
#     out = np.full_like(x, np.nan, dtype=float)
#     mask = np.isfinite(x)
#     if mask.sum() == 0:
#         return out
#     xmin = np.nanmin(x[mask])
#     xmax = np.nanmax(x[mask])
#     if abs(xmax - xmin) < EPS:
#         out[mask] = 0.5
#     else:
#         out[mask] = (x[mask] - xmin) / (xmax - xmin)
#     return out

# def compute_corr(x, y):
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
#         "r_squared": fit.rvalue**2
#     }

# def scatter_plot(df, xcol, ycol, outpath, title):
#     tmp = df[[xcol, ycol]].dropna()
#     if len(tmp) < 3:
#         return
#     x = tmp[xcol].values
#     y = tmp[ycol].values
#     rho_s, _ = spearmanr(x, y)
#     rho_p, _ = pearsonr(x, y)
#     fit = linregress(x, y)

#     xx = np.linspace(np.min(x), np.max(x), 200)
#     yy = fit.intercept + fit.slope * xx

#     fig, ax = plt.subplots(figsize=(7.5, 6.2))
#     ax.scatter(x, y, s=16, alpha=0.45, label="Samples")
#     ax.plot(xx, yy, linewidth=2, label=f"Fit: y={fit.slope:.3f}x+{fit.intercept:.3f}")

#     minv = min(np.min(x), np.min(y))
#     maxv = max(np.max(x), np.max(y))
#     ax.plot([minv, maxv], [minv, maxv], linestyle="--", linewidth=1.4, label="Identity line")

#     ax.set_xlabel(xcol)
#     ax.set_ylabel(ycol)
#     ax.set_title(title)

#     txt = (
#         f"n = {len(tmp)}\n"
#         f"Spearman rho = {rho_s:.3f}\n"
#         f"Pearson r = {rho_p:.3f}\n"
#         f"R² = {fit.rvalue**2:.3f}"
#     )
#     ax.text(
#         0.03, 0.97, txt,
#         transform=ax.transAxes,
#         va="top",
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
#     )
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=300)
#     plt.close()

# # ============================================================
# # LOAD
# # ============================================================

# df = pd.read_csv(RAW_FILE)
# clinical = pd.read_csv(CLINICAL_FILE)

# df.columns = [str(c).strip() for c in df.columns]
# clinical.columns = [str(c).strip() for c in clinical.columns]

# df[MERGE_NEW] = df[MERGE_NEW].apply(clean_id)
# clinical[MERGE_CLINICAL] = clinical[MERGE_CLINICAL].apply(clean_id)

# # ============================================================
# # RECOMPUTE RAW RATIO FROM CHANNELS
# # DI_raw = scale * (P_peak - P_base)/(A_peak - A_base)
# # ============================================================

# def build_raw_ratio(prefix):
#     P = df[f"{prefix}_P_peak"] - df[f"{prefix}_P_base"]
#     A = df[f"{prefix}_A_peak"] - df[f"{prefix}_A_base"]
#     return SHIFT_A * safe_ratio(P, A)

# df["DI_original_rawratio"]   = build_raw_ratio("orig")
# df["DI_gaussian_rawratio"]   = build_raw_ratio("gauss")
# df["DI_truncated_rawratio"]  = build_raw_ratio("trunc")
# df["DI_bg_rawratio"]         = build_raw_ratio("bg")

# # ============================================================
# # KEEP SAME RATIO, APPLY ORIENTATION / SHIFT CORRECTIONS
# # ============================================================

# # 1. raw ratio itself
# # already built above

# # 2. reciprocal-style bounded transform
# # preserves same ratio, reverses orientation
# for col in ["DI_original_rawratio", "DI_gaussian_rawratio", "DI_truncated_rawratio", "DI_bg_rawratio"]:
#     newname = col.replace("_rawratio", "_invbounded")
#     df[newname] = 1.0 / (1.0 + np.clip(df[col], a_min=0, a_max=None))

# # 3. flipped min-max transform
# # first min-max the raw ratio, then invert
# for col in ["DI_original_rawratio", "DI_gaussian_rawratio", "DI_truncated_rawratio", "DI_bg_rawratio"]:
#     mm = minmax_01(df[col].values)
#     newname = col.replace("_rawratio", "_flipminmax")
#     df[newname] = 1.0 - mm

# # 4. linear-shifted version:
# # DI_shift = c - rawratio, with c chosen from data median
# for col in ["DI_original_rawratio", "DI_gaussian_rawratio", "DI_truncated_rawratio", "DI_bg_rawratio"]:
#     med = np.nanmedian(df[col].values)
#     newname = col.replace("_rawratio", "_linearflip")
#     df[newname] = med - (df[col] - med)

# # save all corrected versions
# df.to_csv(os.path.join(OUTPUT_DIR, "DI_with_corrected_ratio_forms.csv"), index=False)

# # ============================================================
# # MERGE WITH CLINICAL
# # ============================================================

# merged = df.merge(clinical, left_on=MERGE_NEW, right_on=MERGE_CLINICAL, how="inner")
# merged.to_csv(os.path.join(OUTPUT_DIR, "merged_correctedDI_clinical.csv"), index=False)

# # ============================================================
# # COMPARE ALL NEW FORMS TO CLINICAL DI COLUMNS
# # ============================================================

# new_cols = [
#     "DI_original_rawratio", "DI_gaussian_rawratio", "DI_truncated_rawratio", "DI_bg_rawratio",
#     "DI_original_invbounded", "DI_gaussian_invbounded", "DI_truncated_invbounded", "DI_bg_invbounded",
#     "DI_original_flipminmax", "DI_gaussian_flipminmax", "DI_truncated_flipminmax", "DI_bg_flipminmax",
#     "DI_original_linearflip", "DI_gaussian_linearflip", "DI_truncated_linearflip", "DI_bg_linearflip",
# ]

# rows = []
# for new_col in new_cols:
#     for old_col in CLINICAL_DI_COLS:
#         if old_col not in merged.columns:
#             continue
#         res = compute_corr(merged[new_col], merged[old_col])
#         if res is None:
#             rows.append({
#                 "new_DI": new_col,
#                 "clinical_DI": old_col,
#                 "n": 0,
#                 "spearman_rho": np.nan,
#                 "pearson_r": np.nan,
#                 "r_squared": np.nan
#             })
#         else:
#             rows.append({
#                 "new_DI": new_col,
#                 "clinical_DI": old_col,
#                 **res
#             })

# corr_df = pd.DataFrame(rows)
# corr_df.to_csv(os.path.join(OUTPUT_DIR, "corrected_ratio_correlations.csv"), index=False)

# # ============================================================
# # RANKING
# # ============================================================

# rank_rows = []
# for new_di in sorted(corr_df["new_DI"].unique()):
#     sub = corr_df[corr_df["new_DI"] == new_di].dropna(subset=["spearman_rho"])
#     if len(sub) == 0:
#         continue
#     best_idx = sub["spearman_rho"].abs().idxmax()
#     best = sub.loc[best_idx]
#     rank_rows.append({
#         "new_DI": new_di,
#         "mean_abs_spearman": sub["spearman_rho"].abs().mean(),
#         "best_matching_clinical_DI": best["clinical_DI"],
#         "best_match_spearman": best["spearman_rho"],
#         "mean_r_squared": sub["r_squared"].mean()
#     })

# rank_df = pd.DataFrame(rank_rows).sort_values(
#     ["mean_abs_spearman", "mean_r_squared"],
#     ascending=[False, False]
# )
# rank_df.to_csv(os.path.join(OUTPUT_DIR, "corrected_ratio_ranking.csv"), index=False)

# print(rank_df.head(20))

# # ============================================================
# # PLOT BEST MATCHES
# # ============================================================

# top_to_plot = rank_df.head(6)
# for _, row in top_to_plot.iterrows():
#     new_col = row["new_DI"]
#     old_col = row["best_matching_clinical_DI"]
#     if old_col not in merged.columns:
#         continue
#     scatter_plot(
#         merged,
#         old_col,
#         new_col,
#         os.path.join(OUTPUT_DIR, f"scatter_{new_col}_vs_{old_col}.png"),
#         title=f"{new_col} vs {old_col}"
#     )

# # ============================================================
# # SIMPLE HEATMAP OF BEST GROUP ONLY
# # ============================================================

# # focus on the most meaningful group:
# focus_cols = [
#     "DI_original_rawratio", "DI_gaussian_rawratio", "DI_truncated_rawratio",
#     "DI_original_invbounded", "DI_gaussian_invbounded", "DI_truncated_invbounded",
# ]
# focus = corr_df[corr_df["new_DI"].isin(focus_cols)].copy()
# pivot = focus.pivot(index="new_DI", columns="clinical_DI", values="spearman_rho")

# fig, ax = plt.subplots(figsize=(12, 6))
# im = ax.imshow(pivot.values, vmin=-1, vmax=1)
# ax.set_xticks(range(pivot.shape[1]))
# ax.set_yticks(range(pivot.shape[0]))
# ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
# ax.set_yticklabels(pivot.index)
# ax.set_title("Spearman correlation: corrected DI forms vs clinical metadata DI")
# for i in range(pivot.shape[0]):
#     for j in range(pivot.shape[1]):
#         val = pivot.values[i, j]
#         txt = "" if pd.isna(val) else f"{val:.2f}"
#         ax.text(j, i, txt, ha="center", va="center", fontsize=10)
# plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "focus_heatmap_correctedDI.png"), dpi=300)
# plt.close()

# print(f"Done. Results in {OUTPUT_DIR}")

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ============================================================
# SETTINGS
# ============================================================

DI_FILE = "/Users/nana/Desktop/DI_4methods_all_samples_with_bounded_DI.csv"
CLINICAL_FILE = "/Users/nana/Desktop/project_metadata/clinical_metadata(Sheet1).csv"

MERGE_COL_DI = "sample"
MERGE_COL_CLINICAL = "Anonymised Identifier"

OUTPUT_DIR = "DI_and_PA_from_integrals_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NEW_DI_COLS = [
    "DI_original_bounded",
    "DI_gaussian_bounded",
    "DI_truncated_bounded",
    "DI_bg_bounded",
]

CLINICAL_DI_COLS = [
    "DIv1 - AmidePhosphateRatios",
    "DIv2 - DigistainIndices",
    "DIv3 - DigistainIndicesTrimmed",
    "DIv4 - NormalisedAmideHeights",
    "DIv5 - NormalisedPhosphateHeights",
    "DIv6 - UnnormalisedAmidePhosphateRatios",
]

EPS = 1e-12

TITLE_MAP = {
    "DI_original_bounded": "Original DI",
    "DI_gaussian_bounded": "Gaussian DI",
    "DI_truncated_bounded": "Truncated DI",
    "DI_bg_bounded": "Background DI",
}

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
    except ValueError:
        return s

def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)

def safe_ratio(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > EPS)
    out[mask] = num[mask] / den[mask]
    return out

def compute_pa_columns(df):
    # P/A = (P_peak - P_base) / (A_peak - A_base)
    df["PA_ratio_original"] = safe_ratio(
        df["orig_P_peak"] - df["orig_P_base"],
        df["orig_A_peak"] - df["orig_A_base"]
    )
    df["PA_ratio_gaussian"] = safe_ratio(
        df["gauss_P_peak"] - df["gauss_P_base"],
        df["gauss_A_peak"] - df["gauss_A_base"]
    )
    df["PA_ratio_truncated"] = safe_ratio(
        df["trunc_P_peak"] - df["trunc_P_base"],
        df["trunc_A_peak"] - df["trunc_A_base"]
    )
    df["PA_ratio_bg"] = safe_ratio(
        df["bg_P_peak"] - df["bg_P_base"],
        df["bg_A_peak"] - df["bg_A_base"]
    )
    return df

PA_MAP = {
    "DI_original_bounded": "PA_ratio_original",
    "DI_gaussian_bounded": "PA_ratio_gaussian",
    "DI_truncated_bounded": "PA_ratio_truncated",
    "DI_bg_bounded": "PA_ratio_bg",
}

def best_clinical_match(merged, new_di_cols, clinical_di_cols):
    rows = []
    for new_col in new_di_cols:
        for old_col in clinical_di_cols:
            if old_col not in merged.columns:
                continue
            tmp = merged[[new_col, old_col]].dropna()
            if len(tmp) < 3:
                continue
            rho, _ = spearmanr(tmp[new_col], tmp[old_col])
            r, _ = pearsonr(tmp[new_col], tmp[old_col])
            fit = linregress(tmp[old_col], tmp[new_col])
            rows.append({
                "new_method": new_col,
                "clinical_DI": old_col,
                "n": len(tmp),
                "spearman_rho": rho,
                "pearson_r": r,
                "r_squared": fit.rvalue ** 2
            })
    corr_df = pd.DataFrame(rows)

    best_rows = []
    for new_col in new_di_cols:
        sub = corr_df[corr_df["new_method"] == new_col].copy()
        if len(sub) == 0:
            continue
        best = sub.iloc[sub["spearman_rho"].abs().argmax()]
        best_rows.append(best)

    return corr_df, pd.DataFrame(best_rows)

def draw_joint_panel(fig, subspec, x, y, xlab, ylab, title):
    inner = GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=subspec,
        width_ratios=[4.0, 1.2],
        height_ratios=[1.2, 4.0],
        wspace=0.05,
        hspace=0.05
    )

    ax_top = fig.add_subplot(inner[0, 0])
    ax_scatter = fig.add_subplot(inner[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_scatter)

    ax_scatter.scatter(x, y, s=16, alpha=0.35)

    rho_s, _ = spearmanr(x, y)
    rho_p, _ = pearsonr(x, y)
    fit = linregress(x, y)
    xx = np.linspace(np.min(x), np.max(x), 200)
    yy = fit.intercept + fit.slope * xx
    ax_scatter.plot(xx, yy, lw=2)

    minv = min(np.min(x), np.min(y))
    maxv = max(np.max(x), np.max(y))
    ax_scatter.plot([minv, maxv], [minv, maxv], ls="--", lw=1.3)

    ax_scatter.set_xlabel(xlab)
    ax_scatter.set_ylabel(ylab)
    ax_scatter.set_title(title, pad=8)

    txt = (
        f"n = {len(x)}\n"
        f"Spearman ρ = {rho_s:.3f}\n"
        f"Pearson r = {rho_p:.3f}\n"
        f"R² = {fit.rvalue**2:.3f}"
    )
    ax_scatter.text(
        0.03, 0.97, txt,
        transform=ax_scatter.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    ax_top.hist(x, bins=28, edgecolor="black", alpha=0.8)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("Count")

    ax_right.hist(y, bins=28, orientation="horizontal", edgecolor="black", alpha=0.8)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlabel("Count")

def draw_violin_box(ax, data, labels, title, ylabel):
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.35)
    ax.boxplot(data, labels=labels, widths=0.22, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel("Clinical DI quartile")
    ax.set_ylabel(ylabel)

# ============================================================
# LOAD + MERGE
# ============================================================

di = pd.read_csv(DI_FILE)
clinical = pd.read_csv(CLINICAL_FILE)

di.columns = [str(c).strip() for c in di.columns]
clinical.columns = [str(c).strip() for c in clinical.columns]

di[MERGE_COL_DI] = di[MERGE_COL_DI].apply(clean_id)
clinical[MERGE_COL_CLINICAL] = clinical[MERGE_COL_CLINICAL].apply(clean_id)

di = compute_pa_columns(di)

merged = di.merge(clinical, left_on=MERGE_COL_DI, right_on=MERGE_COL_CLINICAL, how="left")
merged.to_csv(os.path.join(OUTPUT_DIR, "merged_DI_clinical_with_PA.csv"), index=False)

# ============================================================
# FIND BEST CLINICAL MATCHES
# ============================================================

corr_df, best_df = best_clinical_match(merged, NEW_DI_COLS, CLINICAL_DI_COLS)
corr_df.to_csv(os.path.join(OUTPUT_DIR, "all_newDI_vs_clinicalDI_correlations.csv"), index=False)
best_df.to_csv(os.path.join(OUTPUT_DIR, "best_clinical_match_per_newDI.csv"), index=False)

# ============================================================
# FIGURE 1: 4 PANELS, SCATTER + TREND + MARGINALS
# ============================================================

fig = plt.figure(figsize=(16, 12))
outer = GridSpec(2, 2, figure=fig, wspace=0.22, hspace=0.22)

panel_order = [
    "DI_original_bounded",
    "DI_gaussian_bounded",
    "DI_truncated_bounded",
    "DI_bg_bounded",
]

for i, new_col in enumerate(panel_order):
    row = best_df[best_df["new_method"] == new_col]
    if len(row) == 0:
        continue
    clinical_col = row.iloc[0]["clinical_DI"]
    tmp = merged[[clinical_col, new_col]].dropna()

    draw_joint_panel(
        fig,
        outer[i // 2, i % 2],
        x=tmp[clinical_col].values,
        y=tmp[new_col].values,
        xlab=clinical_col,
        ylab=new_col,
        title=f"{TITLE_MAP[new_col]} vs best clinical DI"
    )

fig.suptitle("Four DI methods vs best-matching clinical metadata DI", fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.965])
plt.savefig(os.path.join(OUTPUT_DIR, "01_four_panel_joint_plots.png"), dpi=300)
plt.close()

# ============================================================
# FIGURE 2: P/A RATIO BY CLINICAL DI QUARTILES
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for ax, new_col in zip(axes, panel_order):
    row = best_df[best_df["new_method"] == new_col]
    if len(row) == 0:
        ax.set_visible(False)
        continue

    clinical_col = row.iloc[0]["clinical_DI"]
    pa_col = PA_MAP[new_col]

    tmp = merged[[clinical_col, pa_col]].dropna().copy()
    if len(tmp) < 10:
        ax.set_visible(False)
        continue

    tmp["quartile"] = pd.qcut(tmp[clinical_col], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    order = ["Q1", "Q2", "Q3", "Q4"]
    data = [tmp.loc[tmp["quartile"] == q, pa_col].values for q in order]

    draw_violin_box(
        ax,
        data=data,
        labels=order,
        title=f"{TITLE_MAP[new_col]}\nP/A ratio by {clinical_col} quartile",
        ylabel="P/A ratio"
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_PA_ratio_by_clinical_DI_quartiles.png"), dpi=300)
plt.close()

# ============================================================
# FIGURE 3: strongest single clinical DI vs method-specific P/A
# ============================================================

if len(corr_df) > 0:
    strongest = corr_df.iloc[corr_df["spearman_rho"].abs().argmax()]
    strongest_clinical = strongest["clinical_DI"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, new_col in zip(axes, panel_order):
        pa_col = PA_MAP[new_col]
        tmp = merged[[strongest_clinical, pa_col]].dropna().copy()
        if len(tmp) < 10:
            ax.set_visible(False)
            continue

        tmp["quartile"] = pd.qcut(tmp[strongest_clinical], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        order = ["Q1", "Q2", "Q3", "Q4"]
        data = [tmp.loc[tmp["quartile"] == q, pa_col].values for q in order]

        draw_violin_box(
            ax,
            data=data,
            labels=order,
            title=f"{TITLE_MAP[new_col]}\nP/A ratio by {strongest_clinical} quartile",
            ylabel="P/A ratio"
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_PA_ratio_by_strongest_clinical_DI.png"), dpi=300)
    plt.close()

# ============================================================
# SUMMARY
# ============================================================

with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
    f.write("Best clinical match per new DI method:\n\n")
    if len(best_df) > 0:
        f.write(best_df.to_string(index=False))
    f.write("\n\nP/A ratio computed directly from DI integrals:\n")
    f.write("PA_ratio_original   = (orig_P_peak - orig_P_base) / (orig_A_peak - orig_A_base)\n")
    f.write("PA_ratio_gaussian   = (gauss_P_peak - gauss_P_base) / (gauss_A_peak - gauss_A_base)\n")
    f.write("PA_ratio_truncated  = (trunc_P_peak - trunc_P_base) / (trunc_A_peak - trunc_A_base)\n")
    f.write("PA_ratio_bg         = (bg_P_peak - bg_P_base) / (bg_A_peak - bg_A_base)\n")

print("Done.")
print(f"Saved in: {OUTPUT_DIR}")