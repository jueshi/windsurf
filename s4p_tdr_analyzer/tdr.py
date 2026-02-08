import numpy as np
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
        if limit_pos == n_fft // 2 and num_pos_freq_points_in_s11 >= n_fft // 2:
             full_spectrum[n_fft // 2] = np.real(full_spectrum[n_fft // 2])

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
    if f_max_orig == 0:
        if n_fft > 0:
            dt = 1.0
            time_vector = np.arange(0, n_fft * dt, dt)
        else:
            dt = 0
            time_vector = np.array([])
    else:
        dt = 1.0 / (2 * target_freq_vector[-1]) if len(target_freq_vector)>1 else 1.0/(2*freq[-1])
        if target_freq_vector[-1] == 0 and len(target_freq_vector)>1 :
            dt = 1.0
        elif target_freq_vector[-1] == 0:
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
        gamma_t_step = np.cumsum(rho_t_impulse_real)


    # --- 8. Calculate Impedance Profile ---
    if len(gamma_t_step) == 0:
        impedance_profile = np.array([])
    else:
        denominator = 1 - gamma_t_step
        impedance_profile = z0 * (1 + gamma_t_step) / (denominator + 1e-12)

    return time_vector, impedance_profile, rho_t_impulse_real, gamma_t_step
