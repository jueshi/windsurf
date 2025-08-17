import numpy as np
import matplotlib.pyplot as plt
import skrf
import os

from .tdr import s_params_to_tdr
from .plotting import plot_tdr

def main():
    # --- Example Usage ---

    # System characteristic impedance
    Z0 = 50.0

    # --- Test Case 1: Matched Load ---
    print("--- Test Case 1: Matched Load ---")
    freq_matched = np.linspace(0, 20e9, 201)
    s11_matched = np.zeros_like(freq_matched, dtype=complex)
    time_m, Z_m, rho_m, gamma_m = s_params_to_tdr(freq_matched, s11_matched, z0=Z0, n_fft=1024)

    # --- Test Case 2: Open Circuit ---
    print("\n--- Test Case 2: Open Circuit ---")
    freq_open = np.linspace(0, 20e9, 201)
    s11_open = np.ones_like(freq_open, dtype=complex) * (1 + 0j)
    time_o, Z_o, rho_o, gamma_o = s_params_to_tdr(freq_open, s11_open, z0=Z0, n_fft=1024)

    # --- Test Case 3: Short Circuit ---
    print("\n--- Test Case 3: Short Circuit ---")
    freq_short = np.linspace(0, 20e9, 201)
    s11_short = np.ones_like(freq_short, dtype=complex) * (-1 + 0j)
    time_s, Z_s, rho_s, gamma_s = s_params_to_tdr(freq_short, s11_short, z0=Z0, n_fft=1024)

    # --- Test Case 4: Mismatched Load (e.g., 75 Ohm load) ---
    print("\n--- Test Case 4: Mismatched Load (75 Ohm) ---")
    Zl_mismatch = 75.0
    s11_val_mismatch = (Zl_mismatch - Z0) / (Zl_mismatch + Z0)
    freq_mismatch = np.linspace(0, 20e9, 201)
    s11_mismatch = np.ones_like(freq_mismatch, dtype=complex) * s11_val_mismatch
    time_mm, Z_mm, rho_mm, gamma_mm = s_params_to_tdr(freq_mismatch, s11_mismatch, z0=Z0, n_fft=1024)

    # --- Test Case 5: Transmission Line with Mismatch at the end ---
    print("\n--- Test Case 5: TL terminated in 100 Ohm ---")
    ZL_tl = 100.0
    er_eff = 2.0
    length = 0.1
    c0 = 299792458.0
    vp = c0 / np.sqrt(er_eff)
    gamma_L_tl = (ZL_tl - Z0) / (ZL_tl + Z0)
    freq_tl_with_dc = np.linspace(0, 20e9, 401)
    s11_tl = np.zeros_like(freq_tl_with_dc, dtype=complex)
    s11_tl[0] = gamma_L_tl
    for i, f_val in enumerate(freq_tl_with_dc):
        if i == 0: continue
        beta = 2 * np.pi * f_val / vp
        s11_tl[i] = gamma_L_tl * np.exp(-2j * beta * length)
    time_tl, Z_tl, rho_tl, gamma_tl = s_params_to_tdr(freq_tl_with_dc, s11_tl, z0=Z0, n_fft=2048, window_type='hann')
    t_delay_expected = 2 * length / vp
    print(f"Expected round-trip delay for TL: {t_delay_expected*1e9:.2f} ns")

    # --- Test Case 6: Load actual S4P file ---
    print("\n--- Test Case 6: Actual S4P file data ---")
    s4p_file = "C:\\Users\\juesh\\OneDrive\\Documents\\MATLAB\\alab\\file_browsers\\sd_s_param\\sd1b_rx0_6in.s4p"

    try:
        network = skrf.Network(s4p_file)
        freq_s4p = network.f
        s11_s4p = network.s[:, 0, 0]
        ZL_s4p = 50.0
        time_s4p, Z_s4p, rho_s4p, gamma_s4p = s_params_to_tdr(freq_s4p, s11_s4p, z0=Z0, n_fft=2048, window_type='hann')
        print(f"Loaded S4P file: {s4p_file}")

        # --- Plotting Results ---
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('TDR Analysis from S-Parameters', fontsize=16)

        plot_tdr(axs[0], axs[1], time_s4p, Z_s4p, rho_s4p, gamma_s4p, f"S4P File: {os.path.basename(s4p_file)}", Z0, plot_limit_time_ns=5, Z_expected=ZL_s4p)

        Z_min = max(0, np.min(Z_s4p) * 0.9)
        Z_max = np.max(Z_s4p) * 1.1

        if Z_max - Z_min < 10:
            Z_center = (Z_min + Z_max) / 2
            Z_min = Z_center - 5
            Z_max = Z_center + 5

        axs[0].set_ylim(Z_min, Z_max)

        refl_min = min(np.min(rho_s4p), np.min(gamma_s4p))
        refl_max = max(np.max(rho_s4p), np.max(gamma_s4p))

        refl_padding = (refl_max - refl_min) * 0.1
        refl_min -= refl_padding
        refl_max += refl_padding

        if refl_max - refl_min < 0.2:
            refl_center = (refl_min + refl_max) / 2
            refl_min = refl_center - 0.1
            refl_max = refl_center + 0.1

        refl_min = max(-1.1, refl_min)
        refl_max = min(1.1, refl_max)

        axs[1].set_ylim(refl_min, refl_max)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig("tdr_analysis_s4p.png")
        print("Plot saved to tdr_analysis_s4p.png")

    except FileNotFoundError:
        print(f"S4P file not found at: {s4p_file}. Skipping this test case.")
        # Still show plots for other test cases
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('TDR Analysis from S-Parameters', fontsize=16)
        plot_tdr(axs[0,0], axs[0,1], time_m, Z_m, rho_m, gamma_m, "Matched Load", Z0, plot_limit_time_ns=5, Z_expected=Z0)
        plot_tdr(axs[1,0], axs[1,1], time_o, Z_o, rho_o, gamma_o, "Open Circuit", Z0, plot_limit_time_ns=5, Z_expected=np.inf)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig("tdr_analysis_test_cases.png")
        print("Plot saved to tdr_analysis_test_cases.png")

    except Exception as e:
        print(f"An error occurred while processing the S4P file: {e}")


if __name__ == '__main__':
    main()
