import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONSTANTS
# ============================================================================
R4 = 0.002  # Nash-4 convergence radius
C_MU = -1.7549  # Muon Mandelbrot parameter
C_CORR = C_MU + R4  # Corrected parameter
ALPHA_INV = 137.035999084  # CODATA 2022
ALPHA = 1.0 / ALPHA_INV
FERMILAB_VALUE = 251e-11  # Experimental value (2021)
FERMILAB_ERROR = 59e-11   # Experimental uncertainty

# ============================================================================
# NASH-4 REGULARIZED ATTRACTOR
# ============================================================================
def z_inf_nash4(c: complex, r4: float = R4) -> complex:
    delta = abs(c + 0.75)  # Distance to period-2 window
    return -0.5 + 1j * np.sqrt(r4 * delta)

# ============================================================================
# COMPUTATION
# ============================================================================
def compute_delta_a_mu() -> dict:
    z_mu = z_inf_nash4(C_MU)
    z_corr = z_inf_nash4(C_CORR)
    delta_a = ALPHA * np.imag(z_corr - z_mu)
    return {
        'z_mu': z_mu,
        'z_corr': z_corr,
        'delta_a': delta_a,
        'delta_a_10_11': delta_a * 1e11,
        'fermilab': FERMILAB_VALUE,
        'fermilab_error': FERMILAB_ERROR,
        'discrepancy': (delta_a - FERMILAB_VALUE) / FERMILAB_ERROR,
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    results = compute_delta_a_mu()
    print("Nash-4 Regularized Attractor Values:")
    print(f"  z_inf(c_mu)   = {results['z_mu']:.6f}")
    print(f"  z_inf(c_corr) = {results['z_corr']:.6f}")
    print()
    print("Muon g-2 Correction:")
    print(f"  Delta a_mu    = {results['delta_a']:.3e}")
    print(f"  Delta a_mu    = {results['delta_a_10_11']:.1f} x 10^-11")
    print()
    print("Comparison with Fermilab (2021):")
    print(f"  Experimental  = {results['fermilab']*1e11:.0f} x 10^-11")
    print(f"  Discrepancy   = {results['discrepancy']:.2f} sigma")