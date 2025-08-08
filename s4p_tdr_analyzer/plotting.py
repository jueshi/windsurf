import numpy as np
import matplotlib.pyplot as plt

def plot_tdr(ax_imp, ax_refl, time, Z, rho, gamma, title, Z0_val, plot_limit_time_ns=None, Z_expected=None):
    time_ns = time * 1e9

    ax_imp.plot(time_ns, Z, label=f'Z(t)')
    ax_imp.axhline(Z0_val, color='r', linestyle='--', label=f'Z0 = {Z0_val} Ohm')
    if Z_expected is not None:
        ax_imp.axhline(Z_expected, color='purple', linestyle=':', label=f'Z_expected = {Z_expected:.1f} Ohm')

    ax_imp.set_title(f'Impedance: {title}')
    ax_imp.set_xlabel('Time (ns)')
    ax_imp.set_ylabel('Impedance (Ohm)')
    ax_imp.legend()
    ax_imp.grid(True)

    min_Z_plot = 0
    max_Z_plot = Z0_val * 2.5
    if Z_expected is not None:
        min_Z_plot = min(0, Z_expected - 20, Z0_val - 20)
        max_Z_plot = max(Z0_val + 20, Z_expected + 20, Z0_val * 2.5)


    if "Open" in title: max_Z_plot = Z0_val * 5
    elif "Short" in title: min_Z_plot = -10; max_Z_plot = Z0_val+10
    elif "Matched" in title : min_Z_plot = Z0_val-10; max_Z_plot = Z0_val+10

    ax_imp.set_ylim(min_Z_plot, max_Z_plot)

    if plot_limit_time_ns:
        ax_imp.set_xlim(0, plot_limit_time_ns)

    ax_refl.plot(time_ns, rho, label='rho(t) (Impulse)')
    ax_refl.plot(time_ns, gamma, label='Gamma(t) (Step)')
    ax_refl.set_title(f'Reflection Coeffs: {title}')
    ax_refl.set_xlabel('Time (ns)')
    ax_refl.set_ylabel('Reflection Coefficient')
    ax_refl.legend()
    ax_refl.grid(True)
    if plot_limit_time_ns:
        ax_refl.set_xlim(0, plot_limit_time_ns)
    ax_refl.set_ylim(-1.1, 1.1)
