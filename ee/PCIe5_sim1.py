import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# PCIe Gen 5 Specifications
BIT_RATE = 32e9  # 32 Gbps
UI = 1 / BIT_RATE  # Unit Interval
SAMPLE_RATE = 10 * BIT_RATE  # Oversampling for detailed analysis

def generate_pcie_signal(n_ui=10, jitter_rms=0.2, noise_amplitude=0.3, 
                 deterministic_jitter=0.1, inter_symbol_interference=0.2):
    """
    Generate a PCIe Gen 5 signal with enhanced jitter and noise modeling
    
    Parameters:
    - n_ui: Number of Unit Intervals to simulate
    - jitter_rms: RMS jitter as a fraction of UI (increased)
    - noise_amplitude: Amplitude of added noise (increased)
    - deterministic_jitter: Peak-to-peak deterministic jitter
    - inter_symbol_interference: ISI effect strength
    
    Returns:
    - time array
    - signal array
    """
    # Time array
    t = np.linspace(0, n_ui * UI, int(n_ui * SAMPLE_RATE / BIT_RATE))
    
    # Base signal (NRZ)
    bits = np.random.randint(2, size=n_ui)
    signal_base = 2 * bits - 1
    signal_base = np.repeat(signal_base, int(SAMPLE_RATE / BIT_RATE))
    
    # Random Jitter (Gaussian)
    random_jitter = np.random.normal(0, jitter_rms * UI, len(t))
    
    # Deterministic Jitter (Sinusoidal)
    det_jitter_freq = 1 / (2 * UI)  # Half UI frequency
    deterministic_jitter_wave = deterministic_jitter * UI * np.sin(2 * np.pi * det_jitter_freq * t)
    
    # Inter-Symbol Interference (ISI) - Simple channel response
    isi_kernel = signal.gaussian(int(SAMPLE_RATE / BIT_RATE), std=inter_symbol_interference * int(SAMPLE_RATE / BIT_RATE))
    isi_kernel /= np.sum(isi_kernel)
    isi_effect = signal.convolve(signal_base, isi_kernel, mode='same')
    
    # Combine Jitter and ISI
    t_jittered = t + random_jitter + deterministic_jitter_wave
    
    # Interpolate to get jittered signal
    signal_jittered = np.interp(t, t_jittered, signal_base + isi_effect)
    
    # Add Noise
    noise = np.random.normal(0, noise_amplitude, len(t))
    signal_noisy = signal_jittered + noise
    
    return t, signal_noisy

def plot_eye_diagram(t, signal_data, title='PCIe Gen 5 Eye Diagram'):
    """
    Plot eye diagram from signal data
    
    Parameters:
    - t: Time array
    - signal_data: Signal array
    - title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Eye Diagram
    plt.subplot(1, 2, 1)
    ui_samples = int(SAMPLE_RATE / BIT_RATE)
    n_periods = len(t) // ui_samples
    
    for i in range(n_periods - 1):
        start = i * ui_samples
        end = (i + 1) * ui_samples
        plt.plot(np.linspace(-UI/2, UI/2, ui_samples), 
                 signal_data[start:end], 
                 color='blue', alpha=0.1)
    
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Bathtub Curve (Simplified)
    plt.subplot(1, 2, 2)
    threshold_range = np.linspace(-0.5, 0.5, 100)
    ber_approx = []
    
    for threshold in threshold_range:
        # Approximate Bit Error Rate
        errors = np.sum(signal_data > threshold)
        ber_approx.append(errors / len(signal_data))
    
    plt.semilogy(threshold_range, ber_approx)
    plt.title('Bathtub Curve')
    plt.xlabel('Threshold (V)')
    plt.ylabel('Bit Error Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Simulation
def main():
    # Generate PCIe Gen 5 Signal
    t, signal_data = generate_pcie_signal(
        n_ui=20,  # Simulate 20 Unit Intervals
        jitter_rms=0.03,  # 3% RMS Jitter
        noise_amplitude=0.15,  # Signal noise
        deterministic_jitter=0.1,  # 10% Peak-to-peak deterministic jitter
        inter_symbol_interference=0.2  # ISI effect strength
    )
    
    # Plot Eye Diagram and Bathtub Curve
    plot_eye_diagram(t, signal_data)
    
    # Calculate Jitter Metrics
    ui_samples = int(SAMPLE_RATE / BIT_RATE)
    total_jitter = np.std(t) * UI * 100  # Total Jitter in % of UI
    rms_jitter = np.std(t) * UI * 100  # RMS Jitter in % of UI
    
    print(f"Total Jitter: {total_jitter:.2f}% of UI")
    print(f"RMS Jitter: {rms_jitter:.2f}% of UI")

if __name__ == "__main__":
    main()
