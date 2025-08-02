import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# PCIe Gen 5 Specifications
BIT_RATE = 32e9  # 32 Gbps
UI = 1 / BIT_RATE  # Unit Interval
SAMPLE_RATE = 10 * BIT_RATE  # Oversampling for detailed analysis

def generate_pcie_signal(n_ui=10, jitter_rms=0.02, noise_amplitude=0.1):
    """
    Generate a PCIe Gen 5 signal with jitter and noise
    
    Parameters:
    - n_ui: Number of Unit Intervals to simulate
    - jitter_rms: RMS jitter as a fraction of UI
    - noise_amplitude: Amplitude of added noise
    
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
    
    # Add Jitter (Gaussian)
    jitter = np.random.normal(0, jitter_rms * UI, len(t))
    t_jittered = t + jitter
    
    # Interpolate to get jittered signal
    signal_jittered = np.interp(t, t_jittered, signal_base)
    
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
        noise_amplitude=0.15  # Signal noise
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
