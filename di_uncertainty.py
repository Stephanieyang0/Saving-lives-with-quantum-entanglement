import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# ============================================================
# 1. YOUR EXISTING DI COMPUTATION GOES HERE
# ============================================================

def compute_DI_fn(cell_wn, cell_abs, windows):
    """
    Replace this function body with your actual DI code.
    This stub shows the correct signature.
    """
    # === INTERPOLATION + COMPUTE DI HERE ===
    raise NotImplementedError("Insert your compute_DI() implementation here.")



# ============================================================
# 2. Perturbation models for Monte Carlo
# ============================================================

def perturb_baseline(abs_sp, sigma):
    """Random constant offset."""
    return abs_sp + np.random.normal(0, sigma)


def perturb_noise(abs_sp, sigma):
    """Pointwise spectral noise."""
    return abs_sp + np.random.normal(0, sigma, abs_sp.shape)


def perturb_window_lowfreq(win_abs, wn, sigma, cutoff=600):
    """Add uncertainty at low wavenumbers (~500 cm⁻¹)."""
    pert = win_abs.copy()
    mask = wn < cutoff
    pert[mask] += np.random.normal(0, sigma, np.sum(mask))
    return pert


def perturb_window_shift(wn, sigma):
    """Shift window central wavelength."""
    shift = np.random.normal(0, sigma)
    return wn + shift


def perturb_window_width(wn, sigma):
    """Stretch/compress filter bandwidth."""
    center = 0.5 * (wn.min() + wn.max())
    scale = 1 + np.random.normal(0, sigma)
    return center + (wn - center) * scale



# ============================================================
# 3. MONTE CARLO SIMULATION
# ============================================================

def monte_carlo_DI(
    cell_wn,
    cell_abs,
    windows,
    compute_DI_fn,          # <--- REQUIRED ARGUMENT
    n_iter=500,
    baseline_sigma=0.01,
    noise_sigma=0.005,
    window_low_sigma=0.01,
    window_shift_sigma=2.0,
    window_width_sigma=0.01,
    random_seed=None
):
    """
    Monte-Carlo DI uncertainty simulation.
    compute_DI_fn is the DI function passed in by user.
    """

    if random_seed:
        np.random.seed(random_seed)

    di_samples = []

    for _ in range(n_iter):

        # --- perturb cell spectrum ---
        pert_abs = perturb_baseline(cell_abs, baseline_sigma)
        pert_abs = perturb_noise(pert_abs, noise_sigma)

        # --- perturb windows ---
        pert_windows = []
        for win in windows:
            wn = win["wavenumber"].copy()
            absW = win["ABS"].copy()

            absW = perturb_window_lowfreq(absW, wn, window_low_sigma)
            wn = perturb_window_shift(wn, window_shift_sigma)
            wn = perturb_window_width(wn, window_width_sigma)

            pert_windows.append({"wavenumber": wn, "ABS": absW})

        # --- compute DI ---
        try:
            di_val = compute_DI_fn(cell_wn, pert_abs, pert_windows)
        except Exception:
            di_val = np.nan

        di_samples.append(di_val)

    # Convert to clean array
    di_samples = np.array(di_samples)
    di_samples = di_samples[np.isfinite(di_samples)]

    if len(di_samples) == 0:
        return di_samples, {"mean": np.nan, "std": np.nan, "cv": np.nan}

    mean_val = di_samples.mean()
    std_val  = di_samples.std(ddof=1)
    cv_val   = std_val / mean_val if mean_val != 0 else np.nan

    return di_samples, {"mean": mean_val, "std": std_val, "cv": cv_val}



# ============================================================
# 4. Plot DI uncertainty distribution
# ============================================================

def plot_monte_carlo_distribution(di_samples, title="DI Monte Carlo Distribution"):
    plt.figure(figsize=(6, 4))
    plt.hist(di_samples, bins=40, density=True, alpha=0.7, color="steelblue")
    plt.axvline(np.mean(di_samples), color="red", label="Mean DI")
    plt.xlabel("DigiStain Index")
    plt.ylabel("Probability Density")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
