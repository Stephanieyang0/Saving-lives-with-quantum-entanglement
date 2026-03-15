import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from di_uncertainty import simulate_DI_uncertainty, plot_DI_uncertainty
import pandas as pd

# Load your raw tissue spectrum
df = pd.read("/Users/nana/Desktop/24C15948A2_map.txt")
wn_cell = df["wavenumber"].values
abs_cell = df["Absorption"].values

# Load your filters
windows = []
for i in range(1, 5):
    wdf = pd.read_csv(f"/Users/nana/Desktop/Window{i}.csv")
    windows.append((wdf["wavenumber"].values, wdf["ABS"].values))

DI_vals = simulate_DI_uncertainty(wn_cell, abs_cell, windows, N=5000)
plot_DI_uncertainty(DI_vals)
# ==========================
# CONFIG
# ==========================

MAP_FILENAME = "/Users/nana/Desktop/24C15948A2_map.txt"
WINDOW_FILENAMES =["/Users/nana/Desktop/Window1.csv", "/Users/nana/Desktop/Window2.csv", "/Users/nana/Desktop/Window3.csv", "/Users/nana/Desktop/Window4.csv"]

SATURATION_THRESHOLD = 5.9      # absorption clipping
WN_MIN = 1100                   # physical tissue + filter region
WN_MAX = 1750
INTERP_POINTS = 2000


# ==========================
# WINDOW INTERPOLATION (PHYSICS-CORRECT)
# ==========================

def interpolate_data(cell_wn, cell_abs, win_wn, win_abs):
    """
    Interpolate cell and window spectra onto same grid.
    IMPORTANT:
      Window transmission outside measured band = 0 (NOT extrapolated)
    """

    cell_wn = np.asarray(cell_wn)
    cell_abs = np.asarray(cell_abs)
    win_wn = np.asarray(win_wn)
    win_abs = np.asarray(win_abs)

    # Sort
    order_c = np.argsort(cell_wn)
    cell_wn = cell_wn[order_c]
    cell_abs = cell_abs[order_c]

    order_w = np.argsort(win_wn)
    win_wn = win_wn[order_w]
    win_abs = win_abs[order_w]

    # Overlap range
    min_wn = max(cell_wn.min(), WN_MIN)
    max_wn = min(cell_wn.max(), WN_MAX)
    if min_wn >= max_wn:
        return None, None, None

    wn_grid = np.linspace(min_wn, max_wn, INTERP_POINTS)

    # ---- CELL INTERPOLATION ----
    f_cell = interp1d(cell_wn, cell_abs, bounds_error=False, fill_value=np.nan)
    cell_interp = f_cell(wn_grid)

    # ---- WINDOW INTERPOLATION (FILTER BEHAVIOUR) ----
    # Values outside win_wn range MUST be zero
    f_win = interp1d(win_wn, win_abs, bounds_error=False, fill_value=0.0)
    win_interp = f_win(wn_grid)

    return wn_grid, cell_interp, win_interp


# ==========================
# DIGISTAIN INDEX
# ==========================

def compute_DI(cell_wn, cell_abs, window_data):
    outputs = []

    for win in window_data:
        wn_grid, cell_interp, win_interp = interpolate_data(
            cell_wn, cell_abs,
            win["wavenumber"], win["ABS"]
        )
        if wn_grid is None:
            return np.nan

        # Convert to transmittance
        T_cell = 10 ** (-cell_interp)
        T_win = win_interp

        S = T_cell * T_win
        outputs.append(np.nansum(S))

    A, C, D, B = outputs  # check ordering: win1=amide peak, win2=phosphate peak, win3=phosphate base, win4=amide base

    if min(A, B, C, D) <= 0:
        return np.nan

    return (np.log10(A) - np.log10(B)) / (np.log10(C) - np.log10(D))


# ==========================
# MAIN PIPELINE
# ==========================

def run_digistain():
    print("Loading map...")
    df = pd.read_csv(MAP_FILENAME, sep="\t")

    df.rename(columns={"Absorption": "intensity"}, inplace=True)

    # Fix saturation
    df.loc[df["intensity"] >= SATURATION_THRESHOLD, "intensity"] = np.nan

    # Apply band-limited physical region
    mask = (df["wavenumber"] >= WN_MIN) & (df["wavenumber"] <= WN_MAX)
    df = df[mask].copy()

    # Convert stage coords → pixel grid
    Xs = np.sort(df["X"].unique())
    Ys = np.sort(df["Y"].unique())

    dx = np.median(np.diff(Xs))
    dy = np.median(np.diff(Ys))

# Compute sorted unique coordinate levels
    x_levels = np.sort(df["X"].unique())
    y_levels = np.sort(df["Y"].unique())

# Map each X/Y value to its index in the unique list
    df["x_pix"] = df["X"].map({v: i for i, v in enumerate(x_levels)})
    df["y_pix"] = df["Y"].map({v: i for i, v in enumerate(y_levels)})


    # Load windows
    window_data = []
    for f in WINDOW_FILENAMES:
        wdf = pd.read_csv(f)
        window_data.append({
            "wavenumber": wdf["wavenumber"].values,
            "ABS": wdf["ABS"].values
        })

    # Compute DI per pixel
    pixel_list = []
    for (px, py), g in df.groupby(["x_pix", "y_pix"]):

        wn = g["wavenumber"].values
        abs_ = g["intensity"].values
        mask = np.isfinite(abs_)

        if mask.sum() < 30:
            di = np.nan
        else:
            di = compute_DI(wn[mask], abs_[mask], window_data)

        pixel_list.append([px, py, di])

    result = pd.DataFrame(pixel_list, columns=["x_pix", "y_pix", "DI"])
    result.to_csv("DI_output.csv", index=False)

    # Build DI image safely
    nx = len(x_levels)
    ny = len(y_levels)

    img = np.full((ny, nx), np.nan)

    for _, row in result.iterrows():
        img[row["y_pix"], row["x_pix"]] = row["DI"]


    # Plot
    plt.figure(figsize=(8,8))
    im = plt.imshow(img, cmap="hsv", origin="lower")
    plt.colorbar(im, label="DigiStain Index")
    plt.title("DigiStain DI Map (Physics Correct)")
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    run_digistain()