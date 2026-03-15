# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import shapiro, anderson, norm, probplot
# import seaborn as sns
# from scipy.ndimage import generic_filter
# from scipy.interpolate import griddata

# # ========== USER SETTINGS ==========
# DI_FILE = "/Users/nana/Desktop/DI_table_cropped_range.csv"   # <--- change to your DI CSV filename
# GRID_RES = 300                 # for heatmaps

# # =======================
# # LOAD DATA
# # =======================

# df = pd.read_csv(DI_FILE)
# df = df.dropna(subset=["DI"])

# di = df["DI"].values

# x = df["X"].values
# y = df["Y"].values
# di = df["DI"].values

# print("Loaded", len(df), "DI values.")
# df = df.copy()
# # Define image resolution
# grid_x, grid_y = np.mgrid[
#     x.min():x.max():500j,     # 500 × 500 grid
#     y.min():y.max():500j
# ]
# di_grid = griddata(
#     points=(x, y),
#     values=di,
#     xi=(grid_x, grid_y),
#     method='linear'
# )
# plt.figure(figsize=(7,6))
# plt.imshow(di_grid.T, origin="lower", cmap="viridis",
#            extent=[x.min(), x.max(), y.min(), y.max()])
# plt.colorbar(label="DI")
# plt.title("Interpolated DigiStain Index Map")
# plt.xlabel("X (stage coordinates)")
# plt.ylabel("Y (stage coordinates)")
# plt.show()

# from scipy.ndimage import generic_filter

# uncert = generic_filter(
#     di_grid,
#     function=lambda arr: np.nanstd(arr),
#     size=15
# )

# plt.figure(figsize=(7,6))
# plt.imshow(uncert.T, origin="lower", cmap="inferno",
#            extent=[x.min(), x.max(), y.min(), y.max()])
# plt.colorbar(label="Local DI Std Dev")
# plt.title("Spatial DI Uncertainty Map")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


# # =======================
# # BASIC STATS
# # =======================

# mean_DI = np.mean(di)
# std_DI  = np.std(di, ddof=1)
# cv_DI   = std_DI / mean_DI

# print("\n===== Basic DI Statistics =====")
# print("Mean DI:", mean_DI)
# print("Std Dev:", std_DI)
# print("Coefficient of Variation:", cv_DI)


# # =======================
# # NORMALITY TESTS
# # =======================

# print("\n===== Normality Tests =====")

# # Shapiro–Wilk
# sw_stat, sw_p = shapiro(di)
# print(f"Shapiro-Wilk: Statistic={sw_stat:.4f}, p={sw_p:.4f}")

# # Anderson–Darling
# ad_result = anderson(di)
# print("Anderson-Darling Statistic:", ad_result.statistic)
# print("Critical values:", ad_result.critical_values)
# print("Significance levels (%):", ad_result.significance_level)

# # ----------------------------
# # Convert stage coords → pixels
# # ----------------------------
# df["x_pix"] = (df["X"] - df["X"].min()) / (df["X"].max() - df["X"].min())
# df["y_pix"] = (df["Y"] - df["Y"].min()) / (df["Y"].max() - df["Y"].min())

# df["x_pix"] = (df["x_pix"] * 1000).astype(int)
# df["y_pix"] = (df["y_pix"] * 1000).astype(int)

# # =======================
# # HISTOGRAM + KDE + GAUSSIAN FIT
# # =======================
# # Clip DI range to 0–1 for cleaner statistics
# di_clipped = di[(di >= 0) & (di <= 1)]

# mean_DI = np.mean(di_clipped)
# std_DI = np.std(di_clipped, ddof=1)

# plt.figure(figsize=(8,5))
# sns.histplot(di_clipped, bins=40, kde=True, stat='density', color="skyblue", edgecolor="black")

# # Gaussian fit
# xvals = np.linspace(0, 1, 300)
# plt.plot(xvals, norm.pdf(xvals, loc=mean_DI, scale=std_DI),
#          'r-', label=f"Gaussian fit (μ={mean_DI:.2f}, σ={std_DI:.2f})")

# plt.axvline(mean_DI, color='red', linestyle='--', label="Mean")

# plt.xlim(0, 1)
# plt.title("DigiStain Index Distribution (Clipped 0–1)")
# plt.xlabel("DI Value")
# plt.ylabel("Density")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()

# # =======================
# # OPTIONAL: Moran's I for spatial autocorrelation
# # =======================

# try:
#     import pysal.lib
#     import libpysal
#     from esda.moran import Moran

#     # Build nearest-neighbor weights
#     arr = pivot.values
#     mask = ~np.isnan(arr)
#     flat = arr[mask]
#     coords = np.column_stack(np.where(mask))

#     w = libpysal.weights.KNN.from_array(coords, k=4)
#     w.transform = "R"

#     mor = Moran(flat, w)

#     print("\n===== Spatial Autocorrelation =====")
#     print("Moran's I:", mor.I)
#     print("p-value:", mor.p_sim)

# except Exception as e:
#     print("\nMoran’s I unavailable (install PySAL if needed):")
#     print("pip install pysal")
