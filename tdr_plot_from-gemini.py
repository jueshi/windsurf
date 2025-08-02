import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft
from scipy.interpolate import interp1d

def s_params_to_tdr(freq, s11_complex, z0=50, n_fft=None, window_type='hann'):
    """
    Calculates TDR impedance profile from S11 parameters.

    Args:
        freq (np.ndarray): Array of frequencies (Hz). Must be sorted.
        s11_complex (np.ndarray): Array of complex S11 values corresponding to freq.
        z0 (float): Characteristic impedance of the system (Ohms).
        n_fft (int, optional): Number of points for IFFT. 
                               Defaults to the next power of 2 greater than or equal to 2*len(freq).
                               A larger N_FFT increases the time window length.
        window_type (str, optional): Type of window to apply to the frequency spectrum. 
                                     Options: 'hann', 'hamming', 'blackman', or None.

    Returns:
        tuple: (time_vector, impedance_profile, rho_t_impulse, gamma_t_step)
               time_vector (np.ndarray): Time points for the TDR plot (s).
               impedance_profile (np.ndarray): Impedance values at each time point (Ohms).
               rho_t_impulse (np.ndarray): Time-domain impulse reflection coefficient.
               gamma_t_step (np.ndarray): Time-domain step reflection coefficient.
    """

    # --- 1. Input Validation and Preparation ---
    if not isinstance(freq, np.ndarray) or not isinstance(s11_complex, np.ndarray):
        raise ValueError("freq and s11_complex must be numpy arrays.")
    if len(freq) != len(s11_complex):
        raise ValueError("freq and s11_complex must have the same length.")
    if len(freq) == 0:
        raise ValueError("Input frequency and S11 data cannot be empty.")
    if not np.all(np.diff(freq) > 0):
        # Attempt to sort if not sorted, or raise error
        if np.all(np.diff(freq) < 0): # descending
            freq = freq[::-1]
            s11_complex = s11_complex[::-1]
        else: # unsorted
            sort_indices = np.argsort(freq)
            freq = freq[sort_indices]
            s11_complex = s11_complex[sort_indices]
            print("Warning: Frequency data was unsorted and has been sorted.")
            if not np.all(np.diff(freq) > 0): # Check again after sorting
                 raise ValueError("Frequency data must be unique and sortable.")


    # --- 2. Ensure DC Point and Create Linear Frequency Vector ---
    f_max_orig = freq[-1]

    # Default N_FFT: ensure enough points for resolution and to capture original f_max
    if n_fft is None:
        n_fft = 2**(np.ceil(np.log2(2 * len(freq))).astype(int))
    
    # Target frequency vector for IFFT input (one-sided positive frequencies)
    # This vector goes from 0 up to f_max_orig.
    # n_fft // 2 points are used for the positive frequency spectrum including DC.
    if n_fft // 2 <= 1: # Need at least 2 points for linspace (0 and f_max_orig)
        target_freq_vector = np.array([0.0, f_max_orig]) if f_max_orig > 0 else np.array([0.0])
        if f_max_orig == 0 and freq[0] == 0: # Only DC point given
             target_freq_vector = np.array([0.0])
        elif n_fft//2 == 1: # Only DC point for IFFT
            target_freq_vector = np.array([0.0])
        else: # Should not happen with typical n_fft logic
            target_freq_vector = np.array([0.0, f_max_orig])

        # Ensure n_fft is at least 2 if we have non-zero f_max_orig
        if f_max_orig > 0 and n_fft < 2 : n_fft = 2
        elif f_max_orig == 0 and n_fft < 1: n_fft = 1


    else:
        target_freq_vector = np.linspace(0, f_max_orig, n_fft // 2, endpoint=True)

    # Interpolate S11 onto the new linear frequency vector
    # Handle cases where original freq might not start at DC or is sparse
    s11_interp_real = interp1d(freq, s11_complex.real, kind='linear', bounds_error=False, fill_value=(s11_complex[0].real, s11_complex[-1].real))
    s11_interp_imag = interp1d(freq, s11_complex.imag, kind='linear', bounds_error=False, fill_value=(s11_complex[0].imag, s11_complex[-1].imag))
    
    s11_positive_freq_spectrum = s11_interp_real(target_freq_vector) + 1j * s11_interp_imag(target_freq_vector)

    # --- 3. Windowing ---
    if window_type and (n_fft // 2 > 1): # Windowing makes sense if more than one freq point
        num_positive_points = len(s11_positive_freq_spectrum)
        if window_type == 'hann':
            window = np.hanning(num_positive_points)
        elif window_type == 'hamming':
            window = np.hamming(num_positive_points)
        elif window_type == 'blackman':
            window = np.blackman(num_positive_points)
        else:
            print(f"Warning: Unknown window type '{window_type}'. No window applied.")
            window = np.ones(num_positive_points)
        s11_positive_freq_spectrum_windowed = s11_positive_freq_spectrum * window
    else:
        s11_positive_freq_spectrum_windowed = s11_positive_freq_spectrum

    # --- 4. Construct Full Spectrum for IFFT (conjugate symmetric) ---
    # full_spectrum will have N_FFT points
    full_spectrum = np.zeros(n_fft, dtype=complex)
    
    # DC component
    full_spectrum[0] = s11_positive_freq_spectrum_windowed[0]

    # Positive frequencies (up to N_FFT/2 - 1)
    # s11_positive_freq_spectrum_windowed has N_FFT/2 points.
    # Indices 1 to N_FFT//2 - 1 for full_spectrum
    num_pos_freq_points_in_s11 = len(s11_positive_freq_spectrum_windowed)

    # Fill positive frequencies
    # Max index for positive frequencies in full_spectrum is n_fft//2 -1
    # Max index in s11_positive_freq_spectrum_windowed is (n_fft//2) - 1
    limit_pos = min(n_fft // 2, num_pos_freq_points_in_s11)
    full_spectrum[1:limit_pos] = s11_positive_freq_spectrum_windowed[1:limit_pos]

    # Nyquist frequency point (if N_FFT is even)
    # This is at index N_FFT/2.
    # If s11_positive_freq_spectrum_windowed includes a point for Nyquist, use it.
    # Our target_freq_vector endpoint f_max_orig corresponds to index (n_fft//2)-1 in s11_positive_freq_spectrum_windowed.
    # The IFFT's Nyquist frequency is at a slightly higher frequency if n_fft//2 points span 0 to f_max_orig.
    # So, S(f_nyquist_ifft_grid) is typically 0 if f_max_orig < f_nyquist_ifft_grid.
    if n_fft % 2 == 0:
        # If the positive spectrum data has a point that can be considered Nyquist
        # (i.e., if num_pos_freq_points_in_s11 == n_fft//2 + 1), it should be real.
        # Here, s11_positive_freq_spectrum_windowed has n_fft//2 points.
        # The point at index n_fft//2 in full_spectrum is the Nyquist point.
        # We assume the signal is bandlimited to f_max_orig, so S(f_nyquist_ifft) = 0
        # unless f_max_orig is exactly the Nyquist frequency of the IFFT grid.
        # For simplicity, if not directly provided by an extended s11_positive_freq_spectrum_windowed, set to 0.
        if num_pos_freq_points_in_s11 > n_fft // 2 : # Unlikely with current setup, but for safety
             full_spectrum[n_fft // 2] = np.real(s11_positive_freq_spectrum_windowed[n_fft // 2])
        # else: # Default, assume 0 beyond f_max_orig if f_max_orig is not the IFFT Nyquist
             # full_spectrum[n_fft // 2] = 0 # This is often implicitly handled by conjugate symmetry part if not set.
             # Let's be explicit: if our positive spectrum (length N/2) covers 0 to f_max,
             # the true Nyquist point for the IFFT (at index N/2) is considered 0 if not covered.
             pass # It will be zero from initialization if not set.


    # Negative frequencies (conjugate symmetric part: S[-f] = S*(f))
    # For S[N-k] = conj(S[k])
    # k from 1 up to n_fft//2 - 1 (if even) or (n_fft-1)//2 (if odd)
    # Example: N=8. S[7]=conj(S[1]), S[6]=conj(S[2]), S[5]=conj(S[3]). S[4] is Nyquist.
    # Example: N=7. S[6]=conj(S[1]), S[5]=conj(S[2]), S[4]=conj(S[3]).
    
    upper_conj_limit = (n_fft -1) // 2 # This is M if N=2M+1, or N/2-1 if N=2M
    for k in range(1, upper_conj_limit + 1):
        if (n_fft - k) < n_fft: # Ensure index is within bounds
            full_spectrum[n_fft - k] = np.conj(full_spectrum[k])
    
    # If N_FFT is even, the Nyquist frequency S[N_FFT/2] must be real for a real time signal.
    # If it was set from s11_positive_freq_spectrum_windowed and was complex, take real part.
    # Or, if it was from zero padding, it's already real (0).
    if n_fft > 0 and n_fft % 2 == 0 :
        full_spectrum[n_fft // 2] = np.real(full_spectrum[n_fft // 2])


    # --- 5. Perform IFFT ---
    # rho_t_impulse is the time-domain impulse reflection coefficient
    if n_fft == 0: # Handle empty case
        rho_t_impulse = np.array([])
    else:
        rho_t_impulse = ifft(full_spectrum) 
    
    # Ensure it's real (small imag parts due to numerical errors can be discarded)
    rho_t_impulse_real = np.real(rho_t_impulse)

    # --- 6. Create Time Vector ---
    # dt = 1 / (2 * f_max_orig) if f_max_orig is the highest freq in the positive spectrum.
    # Or, dt = 1 / (sampling_frequency_of_spectrum) = 1 / (N_FFT * df)
    # df = f_max_orig / ( (N_FFT/2) - 1 ) if N_FFT/2 points span 0 to f_max_orig.
    # So dt = ( (N_FFT/2) - 1 ) / (N_FFT * f_max_orig) - this is getting complicated.
    # Simpler: dt = 1 / (2 * f_max_in_IFFT_onesided_spectrum)
    # The effective one-sided bandwidth for the IFFT is determined by target_freq_vector[-1] == f_max_orig.
    if f_max_orig == 0: # DC only case
        if n_fft > 0:
            dt = 1.0 # Arbitrary, as time is not well-defined
            time_vector = np.arange(0, n_fft * dt, dt)
        else:
            dt = 0
            time_vector = np.array([])
    else:
        dt = 1.0 / (2 * f_max_orig)
        time_vector = np.arange(0, n_fft * dt, dt)
        if len(time_vector) > n_fft: # Correct length due to arange precision
            time_vector = time_vector[:n_fft]
        elif len(time_vector) < n_fft and n_fft > 0 : # If arange produced too few points
            time_vector = np.linspace(0, (n_fft-1)*dt, n_fft)


    # --- 7. Calculate Step Response ---
    # Gamma_u(t) = integral(rho_impulse(tau) dtau)
    # Approximated by cumulative sum of rho_t_impulse_real (which are samples of rho(t))
    # The output of scipy.fft.ifft is scaled by 1/N.
    # If S11 is unitless, rho_t_impulse_real from ifft is also unitless.
    # The cumulative sum of these samples gives the unitless step reflection coefficient.
    if len(rho_t_impulse_real) == 0:
        gamma_t_step = np.array([])
    else:
        gamma_t_step = np.cumsum(rho_t_impulse_real)

    # --- 8. Calculate Impedance Profile ---
    # Z(t) = Z0 * (1 + Gamma_u(t)) / (1 - Gamma_u(t))
    # Add epsilon to denominator to avoid division by zero if Gamma_u(t) is exactly 1.
    if len(gamma_t_step) == 0:
        impedance_profile = np.array([])
    else:
        denominator = 1 - gamma_t_step
        impedance_profile = z0 * (1 + gamma_t_step) / (denominator + 1e-12) # add epsilon for stability

    return time_vector, impedance_profile, rho_t_impulse_real, gamma_t_step


if __name__ == '__main__':
    # --- Example Usage ---

    # System characteristic impedance
    Z0 = 50.0

    # --- Test Case 1: Matched Load ---
    print("--- Test Case 1: Matched Load ---")
    freq_matched = np.linspace(0, 20e9, 201)  # 0 to 20 GHz
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
    # Simulate a 50 Ohm line of length 'l', terminated by 100 Ohm load.
    # S11(f) = Gamma_L * exp(-2 * j * beta * l)
    # where Gamma_L = (ZL - Z0) / (ZL + Z0)
    # beta = 2 * pi * f / vp (vp = c0 / sqrt(er_eff))
    print("\n--- Test Case 5: TL terminated in 100 Ohm ---")
    ZL_tl = 100.0
    er_eff = 2.0  # Effective dielectric constant
    length = 0.1  # meters (10 cm)
    c0 = 299792458.0  # speed of light in vacuum (m/s)
    vp = c0 / np.sqrt(er_eff)

    gamma_L_tl = (ZL_tl - Z0) / (ZL_tl + Z0)
    
    freq_tl = np.linspace(1e6, 20e9, 401) # Start from non-zero to avoid beta issues at DC for simple model
                                         # but the function handles DC extrapolation.
                                         # Let's include DC for the function to handle it.
    freq_tl_with_dc = np.linspace(0, 20e9, 401)

    s11_tl = np.zeros_like(freq_tl_with_dc, dtype=complex)
    # For f=0 (DC), S11 is just Gamma_L if the line is lossless
    s11_tl[0] = gamma_L_tl 
    
    for i, f_val in enumerate(freq_tl_with_dc):
        if i == 0: continue # Skip DC, already set
        beta = 2 * np.pi * f_val / vp
        s11_tl[i] = gamma_L_tl * np.exp(-2j * beta * length)
        
    time_tl, Z_tl, rho_tl, gamma_tl = s_params_to_tdr(freq_tl_with_dc, s11_tl, z0=Z0, n_fft=2048, window_type='hann')
    
    # Expected delay: t_delay = 2 * length / vp
    t_delay_expected = 2 * length / vp
    print(f"Expected round-trip delay for TL: {t_delay_expected*1e9:.2f} ns")


    # --- Plotting Results ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(5, 2, figsize=(14, 20))
    fig.suptitle('TDR Analysis from S-Parameters', fontsize=16)

    plot_limit_time = 2 * t_delay_expected * 1.5 if 't_delay_expected' in locals() else None # Adjust plot time range

    def plot_tdr(ax_imp, ax_refl, time, Z, rho, gamma, title, Z0_val, plot_limit_time_ns=None):
        time_ns = time * 1e9
        
        # Impedance Plot
        ax_imp.plot(time_ns, Z, label=f'Z(t)')
        ax_imp.axhline(Z0_val, color='r', linestyle='--', label=f'Z0 = {Z0_val} Ohm')
        ax_imp.set_title(f'Impedance: {title}')
        ax_imp.set_xlabel('Time (ns)')
        ax_imp.set_ylabel('Impedance (Ohm)')
        ax_imp.legend()
        ax_imp.grid(True)
        if plot_limit_time_ns:
            ax_imp.set_xlim(0, plot_limit_time_ns)
            # Auto Y limits or set reasonable ones
            if "Open" in title: ax_imp.set_ylim(Z0_val-10, Z0_val * 5) 
            elif "Short" in title: ax_imp.set_ylim(-10, Z0_val+10)
            elif "Matched Load" in title : ax_imp.set_ylim(Z0_val-10, Z0_val+10)
            elif "Mismatched" in title : ax_imp.set_ylim(Z0_val-10, Zl_mismatch + 20)
            elif "TL" in title : ax_imp.set_ylim(Z0_val-10, ZL_tl + 20)


        # Reflection Coefficients Plot
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


    plot_tdr(axs[0,0], axs[0,1], time_m, Z_m, rho_m, gamma_m, "Matched Load", Z0, plot_limit_time_ns=5)
    plot_tdr(axs[1,0], axs[1,1], time_o, Z_o, rho_o, gamma_o, "Open Circuit", Z0, plot_limit_time_ns=5)
    plot_tdr(axs[2,0], axs[2,1], time_s, Z_s, rho_s, gamma_s, "Short Circuit", Z0, plot_limit_time_ns=5)
    plot_tdr(axs[3,0], axs[3,1], time_mm, Z_mm, rho_mm, gamma_mm, f"{Zl_mismatch} Ohm Load", Z0, plot_limit_time_ns=5)
    plot_tdr(axs[4,0], axs[4,1], time_tl, Z_tl, rho_tl, gamma_tl, f"TL to {ZL_tl} Ohm (exp delay: {t_delay_expected*1e9:.2f}ns)", Z0, plot_limit_time_ns=t_delay_expected*1e9 * 2.5)
    if 't_delay_expected' in locals():
        axs[4,0].axvline(t_delay_expected*1e9, color='g', linestyle=':', label=f'Expected Delay ({t_delay_expected*1e9:.2f}ns)')
        axs[4,0].legend()
        axs[4,1].axvline(t_delay_expected*1e9, color='g', linestyle=':', label=f'Expected Delay ({t_delay_expected*1e9:.2f}ns)')
        axs[4,1].legend()


    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()
