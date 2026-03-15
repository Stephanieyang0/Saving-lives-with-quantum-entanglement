import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter



# ============================================================
# 0) LOAD DATA
# ============================================================

df = pd.read_csv("/Users/nana/Desktop/omnic_merge/combined_realcoords.csv")
df = df.rename(columns={"X":"x_px", "Y":"y_px"})
print("Loaded columns:", df.columns)

assert {"x_px","y_px","wavenumber","intensity"}.issubset(df.columns)


# ============================================================
# 1) GROUP SPECTRA PER PIXEL
# ============================================================

XY = df[["x_px","y_px"]].to_numpy()
uniqueXY, inv = np.unique(XY, axis=0, return_inverse=True)
nPixels = len(uniqueXY)

wavenumbers = [[] for _ in range(nPixels)]
cellData    = [[] for _ in range(nPixels)]

for i, row in df.iterrows():
    k = inv[i]
    wavenumbers[k].append(row.wavenumber)
    cellData[k].append(row.intensity)

X_pix = uniqueXY[:,0]
Y_pix = uniqueXY[:,1]


# ============================================================
# 2) CLEANING: REMOVE DUPLICATES, NON-MONOTONIC AXES, BAD PIXELS
# ============================================================

def clean_spectrum(wn, ab):
    wn = np.array(wn, float)
    ab = np.array(ab, float)

    # remove NaN
    mask = np.isfinite(wn) & np.isfinite(ab)
    wn, ab = wn[mask], ab[mask]
    if len(wn) < 5: return None, None

    # sort
    idx = np.argsort(wn)
    wn, ab = wn[idx], ab[idx]

    # unique wavenumber
    wn_u, idx_u = np.unique(wn, return_index=True)
    ab_u = ab[idx_u]

    wn, ab = wn_u, ab_u

    # still monotonic?
    if np.any(np.diff(wn) <= 0):
        return None, None

    # flat spectra → background
    if np.std(ab) < 1e-6:
        return None, None

    return wn, ab

clean_wavenumbers = []
clean_cellData = []
bad = 0

for k in range(nPixels):
    wn, ab = clean_spectrum(wavenumbers[k], cellData[k])
    clean_wavenumbers.append(wn)
    clean_cellData.append(ab)
    if wn is None:
        bad += 1

print(f"Cleaned spectra: removed {bad}/{nPixels} bad pixels.")


# ============================================================
# 3) RAW 1650 cm⁻¹ MAP (quick diagnostics)
# ============================================================

TARGET = 1650
tol = 1
raw1650 = np.full(nPixels, np.nan)

for k in range(nPixels):
    wn = clean_wavenumbers[k]
    ab = clean_cellData[k]
    if wn is None: continue
    mask = np.abs(wn - TARGET) < tol
    if mask.any(): raw1650[k] = ab[mask][0]

plt.figure(figsize=(6,5))
plt.scatter(X_pix, Y_pix, c=raw1650, s=8)
plt.gca().invert_yaxis()
plt.title("Raw Absorbance @ 1650 cm⁻¹")
plt.colorbar(); plt.show()


# ============================================================
# 4) LOAD WINDOW FILES FOR DI
# ============================================================

windowData = []
for i in range(1,5):
    T = pd.read_csv(f"Window{i}.csv")
    windowData.append({"wavenumber": T.wavenumber.values,
                       "ABS": T.ABS.values})


# ============================================================
# 5) DIGISTAIN INTERPOLATION (MATLAB logic)
# ============================================================

def interpolateData(wn_cell, ab_cell, win):
    wn_win = win["wavenumber"]
    ab_win = win["ABS"]

    minWV = max(wn_win.min(), wn_cell.min())
    maxWV = min(wn_win.max(), wn_cell.max())

    wn_new = np.linspace(minWV, maxWV, 10001)

    f_cell = interp1d(wn_cell, ab_cell, bounds_error=False, fill_value="extrapolate")
    f_win  = interp1d(wn_win,  ab_win, bounds_error=False, fill_value="extrapolate")

    return wn_new, f_cell(wn_new), f_win(wn_new)


def getDI(k, wn_list, ab_list, winData):
    if wn_list[k] is None: return np.nan

    wn = wn_list[k]
    ab = ab_list[k]

    outputs = []

    for i in range(4):
        wnew, ac, aw = interpolateData(wn, ab, winData[i])
        TRN_cell = 10**(-ac)
        TRN_win  = 10**(-aw)
        outputs.append(TRN_cell * TRN_win)

    A = np.log10(outputs[0].sum())
    C = np.log10(outputs[1].sum())
    D = np.log10(outputs[2].sum())
    B = np.log10(outputs[3].sum())

    return (A - B) / (C - D)


# ============================================================
# 6) FULL-SPECTRUM DI
# ============================================================

DI_full = np.array([getDI(k, clean_wavenumbers, clean_cellData, windowData)
                    for k in range(nPixels)])

plt.figure(figsize=(6,5))
plt.scatter(X_pix, Y_pix, c=DI_full, cmap="jet", vmin=-1, vmax=1, s=8)
plt.gca().invert_yaxis()
plt.title("Full-Spectrum DI")
plt.colorbar(); plt.show()


# ============================================================
# 7) CROPPED DI (1100–1750 cm⁻¹)
# ============================================================

LOW, HIGH = 1100, 1750

# crop spectra
wn_crop = []
ab_crop = []

for k in range(nPixels):
    wn = clean_wavenumbers[k]
    ab = clean_cellData[k]
    if wn is None:
        wn_crop.append(None); ab_crop.append(None); continue
    mask = (wn >= LOW) & (wn <= HIGH)
    wn_crop.append(wn[mask])
    ab_crop.append(ab[mask])

# crop window files
win_crop = []
for wd in windowData:
    wn = wd["wavenumber"]
    ab = wd["ABS"]
    mask = (wn >= LOW) & (wn <= HIGH)
    win_crop.append({"wavenumber": wn[mask], "ABS": ab[mask]})

DI_crop = np.array([getDI(k, wn_crop, ab_crop, win_crop)
                    for k in range(nPixels)])

plt.figure(figsize=(6,5))
plt.scatter(X_pix, Y_pix, c=DI_crop, cmap="jet", vmin=-1, vmax=1, s=8)
plt.gca().invert_yaxis()
plt.title("Cropped DI (1100–1750 cm⁻¹)")
plt.colorbar(); plt.show()


# ============================================================
# 8) PCA ON CROPPED REGION
# ============================================================

# build global axis
WN_all = np.unique(np.hstack([w for w in wn_crop if w is not None]))
WN_ref = np.sort(WN_all[(WN_all >= LOW) & (WN_all <= HIGH)])
Nw = len(WN_ref)

valid_pixels = []
spec_list = []

for k in range(nPixels):
    wn = wn_crop[k]
    ab = ab_crop[k]
    if wn is None: continue

    f = interp1d(wn, ab, bounds_error=False, fill_value="extrapolate")
    spec = f(WN_ref)

    if np.any(~np.isfinite(spec)): continue

    valid_pixels.append(k)
    spec_list.append(spec)

spec_matrix = np.vstack(spec_list)
spec_matrix = savgol_filter(spec_matrix, 11, 3)

spec_scaled = StandardScaler().fit_transform(spec_matrix)
pca = PCA(n_components=3, svd_solver="randomized")
PC = pca.fit_transform(spec_scaled)

# loadings
load = pca.components_
expl = pca.explained_variance_ratio_

plt.figure(figsize=(8,4))
for i in range(3):
    plt.plot(WN_ref, load[i], label=f"PC{i+1} ({expl[i]*100:.1f}%)")
plt.legend(); plt.title("PCA Loadings"); plt.grid(); plt.show()

# map back
PC_map = np.full((nPixels, 3), np.nan)
PC_map[valid_pixels] = PC

plt.figure(figsize=(6,5))
plt.scatter(X_pix, Y_pix, c=PC_map[:,1], cmap="coolwarm", s=10)
plt.gca().invert_yaxis()
plt.title("PC2 Map – Biochemical Contrast")
plt.colorbar(); plt.show()


# ============================================================
# 9) SPATIAL RESOLUTION (KNIFE EDGE USING DI)
# ============================================================

# choose central horizontal cut
row_mask = (np.abs(Y_pix - np.mean(Y_pix)) < 50)
x = X_pix[row_mask]
y = DI_crop[row_mask]
y_s = savgol_filter(y, 21, 3)

def erf_edge(x, x0, A, B, sigma):
    from scipy.special import erf
    return A + B*0.5*(1+erf((x-x0)/(np.sqrt(2)*sigma)))

p0 = [np.median(x), y_s.min(), y_s.max()-y_s.min(), 100]
params, _ = curve_fit(erf_edge, x, y_s, p0=p0)
x0, A, B, sigma = params
FWHM = 2.355 * sigma

plt.figure(figsize=(8,5))
plt.scatter(x, y, s=10)
plt.plot(x, erf_edge(x, *params), "r", lw=2)
plt.axvline(x0, color="k", linestyle="--")
plt.title(f"Knife-Edge Resolution = {FWHM:.1f} µm")
plt.xlabel("X (µm)"); plt.ylabel("DI")
plt.show()


# ============================================================
# 10) K-MEANS SEGMENTATION OF DI
# ============================================================

km = KMeans(n_clusters=3).fit(DI_crop.reshape(-1,1))
labels = km.labels_

plt.figure(figsize=(6,5))
plt.scatter(X_pix, Y_pix, c=labels, cmap="tab20", s=10)
plt.gca().invert_yaxis()
plt.title("K-Means Segmentation of DI")
plt.show()

print("=== PIPELINE COMPLETE ===")

# ============================================================
# 0) LOAD DATA + FIX COLUMN NAMES
# ============================================================

df = pd.read_csv("/Users/nana/Desktop/omnic_merge/combined_realcoords.csv")

# Rename columns to expected names
df = df.rename(columns={"X":"x_px", "Y":"y_px"})

assert {"x_px","y_px","wavenumber","intensity"}.issubset(df.columns)

print("Loaded:", df.shape)
print(df.columns)


# ============================================================
# 1) GROUP INTO PER-PIXEL SPECTRA
# ============================================================

XY = df[["x_px", "y_px"]].to_numpy()
uniqueXY, inv = np.unique(XY, axis=0, return_inverse=True)
nPixels = len(uniqueXY)

wavenumbers = [[] for _ in range(nPixels)]
cellData = [[] for _ in range(nPixels)]

for i, row in df.iterrows():
    k = inv[i]
    wavenumbers[k].append(row.wavenumber)
    cellData[k].append(row.intensity)

# Sort spectra
for k in range(nPixels):
    wn = np.array(wavenumbers[k])
    ab = np.array(cellData[k])
    idx = np.argsort(wn)
    wavenumbers[k] = wn[idx]
    cellData[k] = ab[idx]

X_pix = uniqueXY[:,0]
Y_pix = uniqueXY[:,1]


# ============================================================
# 2) RAW 1650 cm⁻¹ MAP
# ============================================================

TARGET = 1650
tol = 1

raw1650 = np.full(nPixels, np.nan)
for k in range(nPixels):
    mask = np.abs(wavenumbers[k] - TARGET) < tol
    if mask.any():
        raw1650[k] = cellData[k][mask][0]

plt.figure(figsize=(6,5))
plt.scatter(X_pix, Y_pix, c=raw1650, s=8)
plt.gca().invert_yaxis()
plt.title("Raw Absorbance @ 1650 cm⁻¹")
plt.colorbar()
plt.show()


# ============================================================
# 3) LOAD WINDOW FILES FOR FULL DI CALCULATION
# ============================================================

def load_window(i):
    T = pd.read_csv(f"Window{i}.csv")
    return T.wavenumber.values, T.ABS.values

windowData = []
for i in range(1,5):
    wn, ABS = load_window(i)
    windowData.append({"wavenumber": wn, "ABS": ABS})


# ============================================================
# 4) MATLAB INTERPOLATION LOGIC
# ============================================================

def interpolateData(wn_cell, ABS_cell, window_i):

    wn_win = window_i["wavenumber"]
    ABS_win = window_i["ABS"]

    minWV = max(wn_win.min(), wn_cell.min())
    maxWV = min(wn_win.max(), wn_cell.max())

    wn_new = np.linspace(minWV, maxWV, 10001)

    f_cell = interp1d(wn_cell, ABS_cell, bounds_error=False, fill_value="extrapolate")
    f_win  = interp1d(wn_win,  ABS_win, bounds_error=False, fill_value="extrapolate")

    return wn_new, f_cell(wn_new), f_win(wn_new)


def getDI(k, wn_list, ab_list, winData):

    wn = wn_list[k]
    ab = ab_list[k]

    outputs = []
    for i in range(4):
        wnew, cABS, wABS = interpolateData(wn, ab, winData[i])

        cTRN = 10**(-cABS)
        wTRN = 10**(-wABS)

        outputs.append(cTRN * wTRN)

    A = np.log10(outputs[0].sum())
    C = np.log10(outputs[1].sum())
    D = np.log10(outputs[2].sum())
    B = np.log10(outputs[3].sum())

    return (A - B) / (C - D)


# ============================================================
# 5) FULL DI MAP
# ============================================================

DI_full = np.array([getDI(k, wavenumbers, cellData, windowData)
                    for k in range(nPixels)])

plt.figure(figsize=(6,5))
plt.scatter(X_pix, Y_pix, c=DI_full, cmap="jet", vmin=-1, vmax=1, s=8)
plt.gca().invert_yaxis()
plt.title("Full-Spectrum DigiStain Index")
plt.colorbar(label="DI")
plt.show()


# ============================================================
# 6) CROPPED DI (1100–1750)
# ============================================================

LOW, HIGH = 1100, 1750
# ============================================================
# 6) CROPPED DI (1100–1750)
# ============================================================

LOW, HIGH = 1100, 1750

# Crop pixel spectra
wavenumbers_crop = []
ab_crop = []
for k in range(nPixels):
    wn = wavenumbers[k]
    ab = cellData[k]
    mask = (wn >= LOW) & (wn <= HIGH)
    wavenumbers_crop.append(wn[mask])
    ab_crop.append(ab[mask])

# Crop window spectra
win_crop = []
for wd in windowData:
    wn = wd["wavenumber"]
    ab = wd["ABS"]
    mask = (wn >= LOW) & (wn <= HIGH)
    win_crop.append({"wavenumber": wn[mask], "ABS": ab[mask]})

DI_crop = np.array([getDI(k, wavenumbers_crop, ab_crop, win_crop)
                    for k in range(nPixels)])


plt.figure(figsize=(6,5))
plt.scatter(X_pix, Y_pix, c=DI_crop, cmap="jet", vmin=-1, vmax=1, s=8)
plt.gca().invert_yaxis()
plt.title("Cropped DI (1100–1750 cm⁻¹)")
plt.colorbar(label="DI")
plt.show()

# ============================================================
# 7) PCA ON CROPPED REGION (1100–1750 cm⁻¹)
# ============================================================
# ============================================================
# 7) CLEAN PCA WITH NaN REMOVAL AND SAFE INTERPOLATION
# ============================================================

print("\n=== Building clean PCA dataset ===")

# 1. Build global wavenumber axis
WN_all = np.unique(np.hstack(wavenumbers_crop))

mask = (WN_all >= LOW) & (WN_all <= HIGH)
WN_ref = np.sort(WN_all[mask])
Nw = len(WN_ref)

print("Global PCA axis length:", Nw)

valid_pixels = []
spec_matrix_list = []

for k in range(nPixels):

    wn = wavenumbers_crop[k]
    ab = ab_crop[k]

    # Skip empty spectra
    if len(wn) < 5:
        continue

    # 2. Remove duplicates in wn
    wn_unique, idx = np.unique(wn, return_index=True)
    ab_unique = ab[idx]

    # 3. Skip constant spectra
    if np.std(ab_unique) < 1e-6:
        continue

    # 4. Safe interpolation
    try:
        f = interp1d(wn_unique, ab_unique, bounds_error=False, fill_value="extrapolate")
        spec_interp = f(WN_ref)
    except Exception as e:
        print(f"Interpolation failed for pixel {k}: {e}")
        continue

    # 5. If NaN or Inf appear, skip pixel
    if np.any(~np.isfinite(spec_interp)):
        continue

    valid_pixels.append(k)
    spec_matrix_list.append(spec_interp)

# Convert to matrix
spec_matrix = np.vstack(spec_matrix_list)
print("Final PCA matrix:", spec_matrix.shape)

# Smooth (optional)
spec_matrix_smooth = savgol_filter(spec_matrix, 11, 3)

# Standardize
scaler = StandardScaler()
spec_scaled = scaler.fit_transform(spec_matrix_smooth)

# PCA
print("Running PCA...")
pca = PCA(n_components=3, svd_solver="randomized")
PC = pca.fit_transform(spec_scaled)

loadings = pca.components_
expl = pca.explained_variance_ratio_

print("Explained variance:", expl)

# Plot loadings
plt.figure(figsize=(8,4))
for i in range(3):
    plt.plot(WN_ref, loadings[i], label=f"PC{i+1} ({expl[i]*100:.1f}%)")
plt.legend()
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Loading")
plt.title("PCA Loadings (1100–1750 cm⁻¹)")
plt.grid(True)
plt.show()

# Put PC scores back into map
PC_maps = np.full((nPixels, 3), np.nan)
PC_maps[valid_pixels] = PC

# Visualize PC2 (biochemical)
plt.figure(figsize=(7,6))
plt.scatter(X_pix, Y_pix, c=PC_maps[:,1], cmap="coolwarm", s=10)
plt.gca().invert_yaxis()
plt.title("PC2 Map — Biochemical Contrast")
plt.colorbar(label="PC2")
plt.show()



# ============================================================
# 8) SPATIAL RESOLUTION (KNIFE-EDGE ON DI)
# ============================================================

def erf_model(x, x0, A, B, sigma):
    """Error-function edge model"""
    from scipy.special import erf
    return A + B * 0.5 * (1 + erf((x - x0)/(np.sqrt(2)*sigma)))


# Pick a clean horizontal slice through DI_crop
row_mask = (np.abs(Y_pix - Y_pix.mean()) < 50)
x = X_pix[row_mask]
y = DI_crop[row_mask]

# Smooth
y_s = savgol_filter(y, 21, 3)

# Fit error function
p0 = [np.median(x), y_s.min(), y_s.max()-y_s.min(), 200]  # guess
params, _ = curve_fit(erf_model, x, y_s, p0=p0)
x0, A, B, sigma = params
FWHM = 2.355*sigma

plt.figure(figsize=(7,5))
plt.scatter(x, y, s=10, label="Data")
plt.plot(x, erf_model(x, *params), 'r', label="Fit")
plt.axvline(x0, color='k', linestyle='--', label=f"Edge @ {x0:.1f}")
plt.legend()
plt.title(f"Knife-edge Resolution = {FWHM:.1f} µm")
plt.xlabel("X (µm)")
plt.ylabel("DI")
plt.show()


# ============================================================
# 9) K-MEANS SEGMENTATION OF DI MAP
# ============================================================

km = KMeans(n_clusters=3, random_state=0)
labels = km.fit_predict(DI_crop.reshape(-1,1))

plt.figure(figsize=(6,5))
plt.scatter(X_pix, Y_pix, c=labels, cmap="tab20", s=8)
plt.gca().invert_yaxis()
plt.title("K-means Segmentation of DI Map")
plt.show()

print("DONE.")


# =====================================================================
# 1) LOAD PIXEL SPECTRAL DATA
# =====================================================================

DATAFILE = "/Users/nana/Desktop/omnic_merge/combined_realcoords.csv"   # ← your stitched map
df = pd.read_csv(DATAFILE)
import pandas as pd
df = pd.read_csv("/Users/nana/Desktop/omnic_merge/combined_realcoords.csv")
print(df.columns)

# Required columns
assert {"x_px","y_px","wavenumber","intensity"}.issubset(df.columns)


# =====================================================================
# 2) GROUP rows by (X,Y) → build per-pixel spectra
# =====================================================================

XY = df[["x_px","y_px"]].to_numpy()
uniqueXY, inv = np.unique(XY, axis=0, return_inverse=True)
nPixels = uniqueXY.shape[0]

wavenumbers = [[] for _ in range(nPixels)]
cellDataSet  = [[] for _ in range(nPixels)]

for i, row in df.iterrows():
    k = inv[i]
    wavenumbers[k].append(row.wavenumber)
    cellDataSet[k].append(row.intensity)

# Convert to sorted numpy arrays
for k in range(nPixels):
    wn = np.array(wavenumbers[k])
    ab = np.array(cellDataSet[k])
    order = np.argsort(wn)
    wavenumbers[k] = wn[order]
    cellDataSet[k] = ab[order]

X_pix = uniqueXY[:,0]
Y_pix = uniqueXY[:,1]


# =====================================================================
# 3) LOAD WINDOW DATA (Window1–Window4)
# =====================================================================

def load_window_csv(file):
    T = pd.read_csv(file)
    wn  = T["wavenumber"].values
    ABS = T["ABS"].values
    return wn, ABS

windowData = []
for i in range(1,5):
    wn, ABS = load_window_csv(f"Window{i}.csv")
    windowData.append({"wavenumber": wn, "ABS": ABS})


# =====================================================================
# 4) MATLAB-equivalent “interpolateData”
# =====================================================================

def interpolateData(wn_cell, ABS_cell, windowData_i):
    wn_win = windowData_i["wavenumber"]
    ABS_win = windowData_i["ABS"]

    # Bounds for overlap
    minWV = max(wn_win.min(), wn_cell.min())
    maxWV = min(wn_win.max(), wn_cell.max())

    wn_new = np.linspace(minWV, maxWV, 10001)

    # Interpolate with NaN fill outside range
    f_cell = interp1d(wn_cell, ABS_cell, bounds_error=False, fill_value="extrapolate")
    f_win  = interp1d(wn_win,  ABS_win,  bounds_error=False, fill_value="extrapolate")

    interp_cell = f_cell(wn_new)
    interp_win  = f_win(wn_new)

    return wn_new, interp_cell, interp_win


# =====================================================================
# 5) MATLAB-equivalent "getDigistainIndex"
# =====================================================================

def getDigistainIndex(wavenumbers, cellDataSet, windowData, k):
    cell_wn = wavenumbers[k]
    cell_abs = cellDataSet[k]

    outputs = []

    for i in range(4):
        wn_new, cell_i, win_i = interpolateData(cell_wn, cell_abs, windowData[i])
        TRN_cell = 10 ** (-cell_i)
        TRN_win  = 10 ** (-win_i)
        outputs.append(TRN_cell * TRN_win)

    A = np.log10(outputs[0].sum())
    C = np.log10(outputs[1].sum())
    D = np.log10(outputs[2].sum())
    B = np.log10(outputs[3].sum())

    DI = (A - B) / (C - D)
    return DI


# =====================================================================
# 6) (A) RAW 1650 cm⁻¹ MAP  --------------------------------------------
# =====================================================================

TARGET = 1650
tol = 1

raw1650 = np.full(nPixels, np.nan)
for k in range(nPixels):
    wn = wavenumbers[k]
    ab = cellDataSet[k]
    mask = np.abs(wn - TARGET) < tol
    if mask.any():
        raw1650[k] = ab[mask][0]

# =====================================================================
# 7) (B) FULL-SPECTRUM DigiStain Index (MATLAB-equivalent DI)
# =====================================================================

DI_full = np.full(nPixels, np.nan)
for k in range(nPixels):
    DI_full[k] = getDigistainIndex(wavenumbers, cellDataSet, windowData, k)


# =====================================================================
# 8) (C) CROPPED DIGISTAIN INDEX (1100–1750 cm⁻¹)
# =====================================================================

LOW = 1100
HIGH = 1750

# Crop window profiles
windowData_crop = []
for i in range(4):
    wn = windowData[i]["wavenumber"]
    ABS = windowData[i]["ABS"]
    mask = (wn >= LOW) & (wn <= HIGH)
    windowData_crop.append({
        "wavenumber": wn[mask],
        "ABS": ABS[mask]
    })

# Crop cell spectra
wavenumbers_crop = []
cellData_crop = []
for k in range(nPixels):
    wn = wavenumbers[k]
    ab = cellDataSet[k]
    mask = (wn >= LOW) & (wn <= HIGH)
    wavenumbers_crop.append(wn[mask])
    cellData_crop.append(ab[mask])

# Compute DI
DI_crop = np.full(nPixels, np.nan)
for k in range(nPixels):
    DI_crop[k] = getDigistainIndex(wavenumbers_crop, cellData_crop, windowData_crop, k)


# =====================================================================
# 9) PLOT ALL 3 DI MAPS
# =====================================================================

def plot_map(x, y, values, title, cmap="viridis", clim=None):
    plt.figure(figsize=(7,6))
    sc = plt.scatter(x, y, s=4, c=values, cmap=cmap)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    if clim:
        plt.clim(*clim)
    plt.colorbar(sc, label="Value")
    plt.show()


# (1) RAW 1650 cm⁻¹ absorbance
plot_map(X_pix, Y_pix, raw1650,
         "Raw intensity @ 1650 cm⁻¹",
         cmap="viridis")

# (2) Full-spectrum DI (using all wavenumbers)
plot_map(X_pix, Y_pix, DI_full,
         "Full-spectrum DI (MATLAB-equivalent)",
         cmap="jet", clim=(-1,1))

# (3) Cropped DI (1100–1750 cm⁻¹)
plot_map(X_pix, Y_pix, DI_crop,
         "Cropped DI (1100–1750 cm⁻¹)",
         cmap="jet", clim=(-1,1))
