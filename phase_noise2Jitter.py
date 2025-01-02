import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def convert_phase_noise_to_jitter(freq, psd):
    """
    Convert phase noise PSD to RMS jitter
    
    Parameters:
    -----------
    freq : array_like
        Frequency points in Hz
    psd : array_like
        Phase noise power spectral density in dBc/Hz
    
    Returns:
    --------
    jitter_rms : float
        Root Mean Square (RMS) jitter in seconds
    integrated_jitter : array
        Cumulative integrated jitter at each frequency point
    """
    # Convert PSD from dBc/Hz to linear scale
    psd_linear = 10**(psd/10)
    
    # Calculate integrated phase noise (jitter)
    # Integrate from lowest to highest frequency
    # Use trapezoid rule for numerical integration
    jitter_rms = np.sqrt(integrate.trapezoid(psd_linear, freq))
    
    # Calculate cumulative integrated jitter from low to high frequencies
    integrated_jitter = np.sqrt(np.cumsum(psd_linear) * np.diff(np.concatenate((freq, [freq[-1]*10]))))
    
    return jitter_rms, integrated_jitter

# Example usage
def main():
    # Generate example phase noise data
    freq = np.logspace(1, 6, 100)  # 10 Hz to 1 MHz
    psd = -50 - 10*np.log10(freq) + 10*np.random.normal(size=len(freq))
    
    # Convert phase noise to jitter
    jitter_rms, integrated_jitter = convert_phase_noise_to_jitter(freq, psd)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Phase Noise PSD
    plt.subplot(2, 1, 1)
    plt.semilogx(freq, psd)
    plt.title('Phase Noise Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase Noise (dBc/Hz)')
    plt.grid(True)
    
    # Integrated Jitter
    plt.subplot(2, 1, 2)
    plt.loglog(freq, integrated_jitter)
    plt.title('Cumulative Integrated Jitter (Low to High Frequencies)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('RMS Jitter (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print RMS Jitter
    print(f"RMS Jitter: {jitter_rms:.2e} seconds")

if __name__ == "__main__":
    '''Key points about converting phase noise to jitter:

    Conversion Process:
    Convert phase noise from dBc/Hz to linear scale
    Integrate the power spectral density across frequency
    Take the square root to get RMS jitter

    Jitter Calculation:
    Uses scipy.integrate.trapezoid for numerical integration
    Calculates total RMS jitter by integrating across entire frequency range
    Provides cumulative integrated jitter at each frequency point
    
    Visualization:
    First plot shows original phase noise PSD
    Second plot shows cumulative integrated jitter
    
    Interpretation:
    Lower frequency components contribute more to jitter
    RMS jitter represents total timing uncertainty of the signal
    Practical considerations:

    Actual phase noise data typically comes from spectrum analyzers
    Integration limits depend on specific application and measurement setup
    Different frequency ranges (e.g., 1 Hz to 10 kHz) might be more relevant for specific systems'''
    
    main()
