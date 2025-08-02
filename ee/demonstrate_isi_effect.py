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
    kernel_length = int(SAMPLE_RATE / BIT_RATE)
    x = np.linspace(-3, 3, kernel_length)
    isi_kernel = np.exp(-x**2 / (2 * inter_symbol_interference**2))
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
    plt.savefig('eye_diagram.png')
    plt.close()

def create_isi_kernel(kernel_length, isi_strength=0.2):
    """
    Create an Inter-Symbol Interference (ISI) kernel
    
    Parameters:
    - kernel_length: Length of the kernel
    - isi_strength: Controls the spread of interference
    
    Returns:
    - ISI kernel
    """
    x = np.linspace(-3, 3, kernel_length)
    kernel = np.exp(-x**2 / (2 * isi_strength**2))
    kernel /= np.sum(kernel)  # Normalize
    return kernel

def visualize_kernel_and_effect():
    """
    Visualize ISI kernel and its effect on a digital signal
    """
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Different ISI strengths to compare
    isi_strengths = [0.05, 0.2, 0.5, 1.0]
    
    for i, strength in enumerate(isi_strengths):
        # Create kernel
        kernel_length = 100
        isi_kernel = create_isi_kernel(kernel_length, strength)
        
        # Plot kernel
        row = i // 2
        col = i % 2
        
        axs[row, col].plot(isi_kernel, label=f'ISI Strength = {strength}')
        axs[row, col].set_title(f'ISI Kernel (Strength = {strength})')
        axs[row, col].set_xlabel('Kernel Position')
        axs[row, col].set_ylabel('Kernel Value')
        axs[row, col].legend()
    
    plt.tight_layout()
    plt.savefig('isi_kernel_and_effect.png')
    plt.close()

def demonstrate_isi_effect():
    """
    Show how ISI affects a digital signal with fine-grained detail
    """
    # Create a precise square wave digital signal to highlight ISI
    signal_length = 1000
    original_signal = np.zeros(signal_length)
    
    # Create a clean square wave pattern
    original_signal[:signal_length//2] = 1    # First half at +1
    original_signal[signal_length//2:] = -1   # Second half at -1
    
    # Create figure with multiple subplots
    plt.figure(figsize=(16, 10))
    
    # Fine-grained ISI strengths
    isi_strengths = [0.01, 0.25, 0.5, 1]
    
    for i, strength in enumerate(isi_strengths, 1):
        # Create kernel
        kernel_length = 100  # Adjusted kernel length
        isi_kernel = create_isi_kernel(kernel_length, strength)
        
        # Apply convolution (simulates signal spreading)
        isi_signal = signal.convolve(original_signal, isi_kernel, mode='same')
        
        # Plot
        plt.subplot(2, 2, i)
        plt.plot(original_signal, label='Original Signal', color='black', linewidth=2, alpha=0.5)
        plt.plot(isi_signal, label=f'ISI (Strength = {strength})', color='red', linewidth=2)
        
        plt.title(f'Signal with ISI Strength = {strength}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Add grid for better visibility
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Zoom in on a specific region to show detail
        plt.xlim(0, 1000)  # Full signal view
        plt.ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('isi_signal_effect.png')
    plt.close()

    # Kernel visualization
    plt.figure(figsize=(16, 5))
    for i, strength in enumerate(isi_strengths):
        kernel_length = 100
        isi_kernel = create_isi_kernel(kernel_length, strength)
        
        plt.subplot(1, 4, i+1)
        plt.plot(isi_kernel)
        plt.title(f'Kernel (Strength = {strength})')
        plt.xlabel('Kernel Position')
        plt.ylabel('Kernel Value')
    
    plt.tight_layout()
    plt.savefig('isi_kernels.png')
    plt.close()

def main():
    # Generate PCIe Gen 5 Signal
    t, signal_data = generate_pcie_signal(
        n_ui=20,  # Simulate 20 Unit Intervals
        jitter_rms=0.2,  # 20% RMS Jitter (increased)
        noise_amplitude=0.3,  # Increased noise
        deterministic_jitter=0.1,  # 10% Peak-to-peak deterministic jitter
        inter_symbol_interference=0.2  # ISI effect strength
    )
    
    # Plot Eye Diagram and Bathtub Curve
    plot_eye_diagram(t, signal_data)
    
    # Visualize kernels
    visualize_kernel_and_effect()
    
    # Demonstrate ISI effect on signal
    demonstrate_isi_effect()
    
    # Calculate Jitter Metrics
    ui_samples = int(SAMPLE_RATE / BIT_RATE)
    
    # Total Jitter Calculation
    total_jitter_ui = np.max(t) - np.min(t)
    total_jitter_percent = (total_jitter_ui / UI) * 100
    
    # RMS Jitter Calculation
    rms_jitter_ui = np.std(t)
    rms_jitter_percent = (rms_jitter_ui / UI) * 100
    
    # Peak-to-Peak Jitter
    pk_to_pk_jitter_ui = np.ptp(t)
    pk_to_pk_jitter_percent = (pk_to_pk_jitter_ui / UI) * 100
    
    print("Jitter Metrics:")
    print(f"Total Jitter: {total_jitter_percent:.2f}% of UI")
    print(f"RMS Jitter: {rms_jitter_percent:.2f}% of UI")
    print(f"Peak-to-Peak Jitter: {pk_to_pk_jitter_percent:.2f}% of UI")
    
    # Signal Quality Metrics
    signal_mean = np.mean(signal_data)
    signal_std = np.std(signal_data)
    
    print("\nSignal Quality:")
    print(f"Signal Mean: {signal_mean:.4f}")
    print(f"Signal Standard Deviation: {signal_std:.4f}")

if __name__ == "__main__":
    main()
