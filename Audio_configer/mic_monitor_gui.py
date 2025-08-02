import tkinter as tk
from tkinter import ttk
import subprocess
import sys

class MicMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Microphone Monitor")
        self.root.geometry("800x600")  # Increased window size
        
        # Initialize audio stream as None
        self.stream = None
        
        # Add audio callback function
        def audio_callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            outdata[:] = indata
        
        self.audio_callback = audio_callback
        
        # Main frame for better organization
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Device selection frame
        device_frame = ttk.LabelFrame(main_frame, text="Device Selection", padding="10")
        device_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Input device
        self.input_label = ttk.Label(device_frame, text="Input Device:")
        self.input_label.pack(pady=5)
        
        self.input_var = tk.StringVar()
        self.input_combobox = ttk.Combobox(device_frame, textvariable=self.input_var, width=60)
        self.input_combobox.pack(pady=5)
        
        # Output device
        self.output_label = ttk.Label(device_frame, text="Output Device:")
        self.output_label.pack(pady=5)
        
        self.output_var = tk.StringVar()
        self.output_combobox = ttk.Combobox(device_frame, textvariable=self.output_var, width=60)
        self.output_combobox.pack(pady=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        # Buttons
        self.start_btn = ttk.Button(button_frame, text="Start Monitoring", command=self.start_monitoring)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Monitoring", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.close_btn = ttk.Button(button_frame, text="Close", command=self.on_closing)
        self.close_btn.pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(pady=10)
        
        # Add a proper close method
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.load_devices()
        
    def load_devices(self):
        # Get available audio devices using sounddevice
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            
            input_devices = [f"{d['name']} (ID: {d['index']})" 
                           for d in devices if d['max_input_channels'] > 0]
            output_devices = [f"{d['name']} (ID: {d['index']})" 
                            for d in devices if d['max_output_channels'] > 0]
            
            self.input_combobox['values'] = input_devices
            self.output_combobox['values'] = output_devices
            
            if input_devices:
                self.input_var.set(input_devices[0])
            if output_devices:
                self.output_var.set(output_devices[0])
                
        except Exception as e:
            self.status_var.set(f"Error loading devices: {str(e)}")
    
    def start_monitoring(self):
        input_device = self.input_var.get()
        output_device = self.output_var.get()
        
        if not input_device or not output_device:
            self.status_var.set("Please select both input and output devices")
            return
            
        try:
            import sounddevice as sd
            
            # Get device IDs from the selected strings
            input_id = int(input_device.split('ID: ')[1].rstrip(')'))
            output_id = int(output_device.split('ID: ')[1].rstrip(')'))
            
            # Set default devices
            sd.default.device = (input_id, output_id)
            
            # Test if devices are working
            sd.check_input_settings()
            sd.check_output_settings()
            
            # Start the audio stream
            self.stream = sd.Stream(
                device=(input_id, output_id),
                samplerate=44100,
                blocksize=1024,
                channels=1,
                callback=self.audio_callback
            )
            self.stream.start()
            
            self.status_var.set("Monitoring started")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.status_var.set(f"Error starting monitoring: {str(e)}")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def stop_monitoring(self):
        try:
            # Stop and close the audio stream
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            self.status_var.set("Monitoring stopped")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
        except Exception as e:
            self.status_var.set(f"Error stopping monitoring: {str(e)}")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def on_closing(self):
        try:
            # Cleanup method to properly close the application
            if self.stream is not None:
                # Close any open streams or resources
                self.stream.close()
                self.stream = None
            
            # Clean up Tkinter variables
            if hasattr(self, 'input_var'):
                self.input_var.set('')
                del self.input_var
            if hasattr(self, 'output_var'):
                self.output_var.set('')
                del self.output_var
            if hasattr(self, 'status_var'):
                self.status_var.set('')
                del self.status_var
            
            # Destroy all widgets
            for widget in self.root.winfo_children():
                widget.destroy()
            
            # Quit the Tkinter main loop and destroy the window
            self.root.quit()
            self.root.destroy()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MicMonitorApp(root)
    root.mainloop()
