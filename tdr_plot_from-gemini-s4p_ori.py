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
    f_min_orig = freq[0]
    f_max_orig = freq[-1]

    # Default N_FFT: ensure enough points for resolution and to capture original f_max
    if n_fft is None:
        n_fft = 2**(np.ceil(np.log2(2 * len(freq))).astype(int))
    
    # Target frequency vector for IFFT input (one-sided positive frequencies)
    # This vector goes from 0 up to f_max_orig.
    # n_fft // 2 points are used for the positive frequency spectrum including DC.
    if n_fft // 2 <= 1: 
        target_freq_vector = np.array([0.0, f_max_orig]) if f_max_orig > 0 else np.array([0.0])
        if f_max_orig == 0 and f_min_orig == 0: # Only DC point given
             target_freq_vector = np.array([0.0])
        elif n_fft//2 == 1: # Only DC point for IFFT
            target_freq_vector = np.array([0.0])
        else: 
            target_freq_vector = np.array([0.0, f_max_orig])

        if f_max_orig > 0 and n_fft < 2 : n_fft = 2
        elif f_max_orig == 0 and n_fft < 1: n_fft = 1
    else:
        target_freq_vector = np.linspace(0, f_max_orig, n_fft // 2, endpoint=True)

    # Interpolate S11 onto the new linear frequency vector
    # For fill_value, use the S11 value at the lowest provided frequency (f_min_orig)
    # for frequencies below f_min_orig (including DC if f_min_orig > 0).
    # This is a common approach for DC extrapolation.
    s11_at_f_min = s11_complex[0]
    s11_at_f_max = s11_complex[-1]

    s11_interp_real = interp1d(freq, s11_complex.real, kind='linear', bounds_error=False, 
                               fill_value=(s11_at_f_min.real, s11_at_f_max.real))
    s11_interp_imag = interp1d(freq, s11_complex.imag, kind='linear', bounds_error=False, 
                               fill_value=(s11_at_f_min.imag, s11_at_f_max.imag))
    
    s11_positive_freq_spectrum = s11_interp_real(target_freq_vector) + 1j * s11_interp_imag(target_freq_vector)

    # --- 3. Windowing ---
    if window_type and (n_fft // 2 > 1): 
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
    full_spectrum = np.zeros(n_fft, dtype=complex)
    
    full_spectrum[0] = s11_positive_freq_spectrum_windowed[0] # DC component

    num_pos_freq_points_in_s11 = len(s11_positive_freq_spectrum_windowed)
    limit_pos = min(n_fft // 2, num_pos_freq_points_in_s11)
    full_spectrum[1:limit_pos] = s11_positive_freq_spectrum_windowed[1:limit_pos]

    if n_fft % 2 == 0: # Nyquist frequency point for even N_FFT
        # If s11_positive_freq_spectrum_windowed has n_fft//2 points, its last point corresponds to f_max_orig.
        # The IFFT's Nyquist point is at index n_fft//2.
        # If f_max_orig is considered the Nyquist frequency for the *positive spectrum part*,
        # then full_spectrum[n_fft//2] could take this value (must be real).
        # However, our target_freq_vector (length n_fft//2) spans 0 to f_max_orig.
        # So, s11_positive_freq_spectrum_windowed[n_fft//2 - 1] is S11(f_max_orig).
        # The IFFT grid's Nyquist frequency might be higher than f_max_orig.
        # Typically, if data up to f_max is provided, components beyond f_max (like true Nyquist for IFFT) are zero.
        # Let's ensure it's real. If it was extrapolated from complex S11(f_max_orig), it might be complex.
        # For a real time signal, S(f_nyquist) must be real.
        # If the last point of s11_positive_freq_spectrum_windowed is meant for the Nyquist bin:
        if limit_pos == n_fft // 2 and num_pos_freq_points_in_s11 >= n_fft // 2:
             # This implies s11_positive_freq_spectrum_windowed[-1] is at f_max_orig,
             # which is also the Nyquist freq for the positive spectrum part.
             # This point should map to full_spectrum[n_fft//2] if f_max_orig is the true Nyquist.
             # This logic is tricky. For now, we assume 0 beyond f_max_orig if not explicitly given.
             # The conjugate symmetry part will handle filling.
             # Let's ensure it's real if it was somehow populated.
             full_spectrum[n_fft // 2] = np.real(full_spectrum[n_fft // 2]) # Ensure real if populated
        # Otherwise, it's zero from initialization.

    upper_conj_limit = (n_fft -1) // 2 
    for k in range(1, upper_conj_limit + 1):
        if (n_fft - k) < n_fft: 
            full_spectrum[n_fft - k] = np.conj(full_spectrum[k])
    
    if n_fft > 0 and n_fft % 2 == 0 :
        full_spectrum[n_fft // 2] = np.real(full_spectrum[n_fft // 2])


    # --- 5. Perform IFFT ---
    if n_fft == 0: 
        rho_t_impulse = np.array([])
    else:
        rho_t_impulse = ifft(full_spectrum) 
    
    rho_t_impulse_real = np.real(rho_t_impulse)

    # --- 6. Create Time Vector ---
    # The IFFT time step dt is related to the total bandwidth represented in full_spectrum.
    # The full_spectrum spans from -f_effective_max to +f_effective_max,
    # where f_effective_max corresponds to the (n_fft/2 -1)-th component if using n_fft/2 positive freqs.
    # If target_freq_vector goes up to f_max_orig with n_fft/2 points, then df = f_max_orig / (n_fft/2 - 1).
    # Total bandwidth for IFFT is N_FFT * df. Time step dt = 1 / (N_FFT * df).
    # This simplifies to dt = 1 / (2 * f_max_orig_for_IFFT_grid_spacing)
    # If our target_freq_vector's f_max_orig is the highest freq, then:
    if f_max_orig == 0: 
        if n_fft > 0:
            dt = 1.0 
            time_vector = np.arange(0, n_fft * dt, dt)
        else:
            dt = 0
            time_vector = np.array([])
    else:
        # The effective sampling frequency of the spectrum for IFFT is 2*f_max_orig
        # if target_freq_vector spans 0 to f_max_orig with n_fft/2 points.
        dt = 1.0 / (2 * target_freq_vector[-1]) if len(target_freq_vector)>1 else 1.0/(2*freq[-1]) # Use last freq of target
        if target_freq_vector[-1] == 0 and len(target_freq_vector)>1 : # if f_max_orig was 0 but target_freq_vector was [0,0]
            dt = 1.0 # Avoid division by zero, though f_max_orig == 0 case should handle this
        elif target_freq_vector[-1] == 0: # Only DC point in target
             dt = 1.0

        time_vector = np.arange(0, n_fft * dt, dt)
        if len(time_vector) > n_fft: 
            time_vector = time_vector[:n_fft]
        elif len(time_vector) < n_fft and n_fft > 0 : 
            time_vector = np.linspace(0, (n_fft-1)*dt, n_fft, endpoint=True)


    # --- 7. Calculate Step Response ---
    if len(rho_t_impulse_real) == 0:
        gamma_t_step = np.array([])
    else:
        # The IFFT output needs to be scaled. For a step response, we integrate rho(t).
        # rho(t) is an impulse response. The integral of rho(t) gives Gamma_step(t).
        # If S11 is unitless, rho(t) from IFFT is unitless (after IFFT scaling of 1/N).
        # The step response Gamma_u(t) is obtained by scaling rho_t_impulse_real
        # by the frequency step df of the *original S11 data used for IFFT*,
        # then taking cumulative sum.
        # However, a common approach for TDR from bandlimited S11 is directly
        # cumsum(ifft(S11_windowed_spectrum)).
        # The scaling of ifft result from scipy.fft.ifft is 1/N.
        # Let's verify the standard definition.
        # If S(f) is the spectrum, s(t) = IFFT(S(f)).
        # For TDR, rho(t) = IFFT(S11(f)).
        # Gamma_step(t) = integral from 0 to t of rho(tau) dtau.
        # Discrete approximation: Gamma_step[k] = sum_{i=0 to k} rho[i] * dt_time_domain
        # The dt_time_domain is already factored in by the nature of the IFFT points.
        # The values from rho_t_impulse_real are samples of rho(t).
        # A simple cumulative sum is often used directly on these samples.
        gamma_t_step = np.cumsum(rho_t_impulse_real)


    # --- 8. Calculate Impedance Profile ---
    if len(gamma_t_step) == 0:
        impedance_profile = np.array([])
    else:
        denominator = 1 - gamma_t_step
        impedance_profile = z0 * (1 + gamma_t_step) / (denominator + 1e-12) 

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
    # Load the S4P file using scikit-rf
    import skrf
    s4p_file = "C:\\Users\\juesh\\OneDrive\\Documents\\MATLAB\\alab\\file_browsers\\sd_s_param\\sd1b_rx0_6in.s4p"
    network = skrf.Network(s4p_file)
    freq_s4p = network.f
    s11_s4p = network.s[:, 0, 0]  # S11 parameter
    
    # Define a reference impedance for plotting purposes
    ZL_s4p = 50.0  # Using standard 50 Ohm as reference impedance
    
    # Process the S4P data
    time_s4p, Z_s4p, rho_s4p, gamma_s4p = s_params_to_tdr(freq_s4p, s11_s4p, z0=Z0, n_fft=2048, window_type='hann')
    print(f"Loaded S4P file: {s4p_file}")


    # --- Plotting Results ---
    plt.style.use('seaborn-v0_8-darkgrid')
    # Change from 6x2 to 1x2 subplot grid since we're only showing one case
    fig, axs = plt.subplots(1, 2, figsize=(14, 6)) # Reduced to 1 row and adjusted figure height
    fig.suptitle('TDR Analysis from S-Parameters', fontsize=16)

    plot_limit_time = 2 * t_delay_expected * 1.5 if 't_delay_expected' in locals() else None 

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

    # Comment out plots for test cases 1-5
    # plot_tdr(axs[0,0], axs[0,1], time_m, Z_m, rho_m, gamma_m, "Matched Load", Z0, plot_limit_time_ns=5, Z_expected=Z0)
    # plot_tdr(axs[1,0], axs[1,1], time_o, Z_o, rho_o, gamma_o, "Open Circuit", Z0, plot_limit_time_ns=5, Z_expected=np.inf)
    # plot_tdr(axs[2,0], axs[2,1], time_s, Z_s, rho_s, gamma_s, "Short Circuit", Z0, plot_limit_time_ns=5, Z_expected=0)
    # plot_tdr(axs[3,0], axs[3,1], time_mm, Z_mm, rho_mm, gamma_mm, f"{Zl_mismatch} Ohm Load", Z0, plot_limit_time_ns=5, Z_expected=Zl_mismatch)
    
    # plot_tl_time_limit = t_delay_expected*1e9 * 2.5
    # plot_tdr(axs[4,0], axs[4,1], time_tl, Z_tl, rho_tl, gamma_tl, f"TL to {ZL_tl} Ohm (exp delay: {t_delay_expected*1e9:.2f}ns)", Z0, plot_limit_time_ns=plot_tl_time_limit, Z_expected=ZL_tl)
    # if 't_delay_expected' in locals():
    #     axs[4,0].axvline(t_delay_expected*1e9, color='g', linestyle=':', label=f'Expected Delay ({t_delay_expected*1e9:.2f}ns)')
    #     axs[4,0].legend() # Re-call legend to include axvline
    #     axs[4,1].axvline(t_delay_expected*1e9, color='g', linestyle=':', label=f'Expected Delay ({t_delay_expected*1e9:.2f}ns)')
    #     axs[4,1].legend() # Re-call legend

    # Only keep the plot for Test Case 6 (Simulated S4P)
    # Use axs[0] and axs[1] instead of axs[5,0] and axs[5,1] since we now have a 1x2 grid
    plot_tdr(axs[0], axs[1], time_s4p, Z_s4p, rho_s4p, gamma_s4p, f"S4P File: sd1b_rx0_6in.s4p", Z0, plot_limit_time_ns=5, Z_expected=ZL_s4p)
    
    # Automatically adjust y-axis range based on actual data values
    # Calculate the min and max values of the impedance data with some padding
    Z_min = max(0, np.min(Z_s4p) * 0.9)  # 10% below minimum, but not negative
    Z_max = np.max(Z_s4p) * 1.1  # 10% above maximum
    
    # Set reasonable limits if the data is very flat
    if Z_max - Z_min < 10:
        Z_center = (Z_min + Z_max) / 2
        Z_min = Z_center - 5
        Z_max = Z_center + 5
    
    # Apply the calculated range
    axs[0].set_ylim(Z_min, Z_max)
    
    # Automatically adjust y-axis range for reflection coefficient plot
    # Find min and max values from both rho and gamma data
    refl_min = min(np.min(rho_s4p), np.min(gamma_s4p))
    refl_max = max(np.max(rho_s4p), np.max(gamma_s4p))
    
    # Add padding (10%)
    refl_padding = (refl_max - refl_min) * 0.1
    refl_min = refl_min - refl_padding
    refl_max = refl_max + refl_padding
    
    # Ensure reasonable range if data is very flat
    if refl_max - refl_min < 0.2:  # Minimum range of 0.2 for reflection coefficient
        refl_center = (refl_min + refl_max) / 2
        refl_min = refl_center - 0.1
        refl_max = refl_center + 0.1
    
    # Ensure range doesn't exceed theoretical limits of reflection coefficient (-1 to 1)
    refl_min = max(-1.1, refl_min)  # Allow slight extension beyond -1 for visibility
    refl_max = min(1.1, refl_max)   # Allow slight extension beyond 1 for visibility
    
    # Apply the calculated range
    axs[1].set_ylim(refl_min, refl_max)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout for suptitle
    plt.show()
