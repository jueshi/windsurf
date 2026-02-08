import speech_recognition as sr
import pyaudio
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import queue

def get_stereo_mix_device_index():
    """Finds the device index for 'Stereo Mix'."""
    p = pyaudio.PyAudio()
    device_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0 and 'stereo mix' in info['name'].lower():
            device_index = i
            break
    p.terminate()
    return device_index

def get_audio_devices():
    """Return a dict of {name: index} using SpeechRecognition's device listing.
    This guarantees indices align with sr.Microphone(device_index=...).
    """
    devices = {}
    try:
        names = sr.Microphone.list_microphone_names()
        for i, name in enumerate(names):
            # Filter to input-capable devices is implicit; SR will raise on invalid indices.
            devices[name] = i
    except Exception:
        # Fallback to PyAudio if SR enumeration fails
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                devices[info['name']] = i
        p.terminate()
    return devices

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Audio Transcriber")
        self.root.geometry("600x400")

        # Audio source selection
        self.source_frame = tk.Frame(root)
        self.source_frame.pack(pady=5)
        tk.Label(self.source_frame, text="Audio Source:").pack(side=tk.LEFT, padx=5)
        self.audio_devices = get_audio_devices()
        self.source_var = tk.StringVar()
        self.source_menu = ttk.Combobox(self.source_frame, textvariable=self.source_var,
                                        values=list(self.audio_devices.keys()), state="readonly", width=40)
        if self.audio_devices:
            default_device = next((name for name in self.audio_devices if 'stereo mix' in name.lower()),
                                  list(self.audio_devices.keys())[0])
            self.source_menu.set(default_device)
        self.source_menu.pack(side=tk.LEFT, padx=5)

        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20)
        self.text_area.pack(pady=10, padx=10)

        self.start_button = tk.Button(root, text="Start Transcription", command=self.start_transcription)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(root, text="Stop Transcription", command=self.stop_transcription, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=10)

        self.status_label = tk.Label(root, text="Status: Not Started", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.transcription_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.transcription_thread = None

    def start_transcription(self):
        # Use selected device from dropdown
        selected = self.source_var.get()
        if not selected:
            self.update_status("No audio source selected.")
            self.text_area.insert(tk.END, "No audio source selected. Please choose one and try again.\n")
            return
        self.device_index = self.audio_devices[selected]

        self.stop_event.clear()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.source_menu.config(state=tk.DISABLED)
        self.update_status("Starting...")

        self.transcription_thread = threading.Thread(target=self.transcribe_audio, daemon=True)
        self.transcription_thread.start()
        self.root.after(100, self.process_queue)

    def stop_transcription(self):
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.stop_event.set()
            self.transcription_thread.join(timeout=1) # Wait for the thread to finish
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.source_menu.config(state=tk.NORMAL)
        self.update_status("Stopped.")

    def transcribe_audio(self):
        r = sr.Recognizer()
        mic = None
        source = None
        try:
            # Resolve device name for status
            names = sr.Microphone.list_microphone_names()
            device_name = names[self.device_index] if 0 <= self.device_index < len(names) else f"index {self.device_index}"

            mic = sr.Microphone(device_index=self.device_index)
            # Manually manage context to avoid NoneType close errors if __enter__ fails
            source = mic.__enter__()

            self.update_status("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=2)
            self.update_status(f"Listening on '{device_name}' (Device {self.device_index})...")

            while not self.stop_event.is_set():
                try:
                    audio = r.listen(source, timeout=1, phrase_time_limit=10)
                    # Add a language hint for better accuracy (change if needed)
                    text = r.recognize_google(audio, language="en-US")
                    self.transcription_queue.put(text)
                except sr.WaitTimeoutError:
                    continue  # No speech detected
                except sr.UnknownValueError:
                    self.transcription_queue.put("(...)")
                except sr.RequestError as e:
                    error_msg = f"API Error: {e}"
                    self.transcription_queue.put(error_msg)
                    self.update_status(error_msg)
                    break
        except (AssertionError, OSError, ValueError) as e:
            msg = f"Microphone error on device {self.device_index}: {e}"
            self.transcription_queue.put(msg)
            self.update_status("Error opening device. Try a different source.")
        finally:
            if mic is not None:
                try:
                    mic.__exit__(None, None, None)
                except Exception:
                    pass

    def process_queue(self):
        try:
            while not self.transcription_queue.empty():
                message = self.transcription_queue.get_nowait()
                self.text_area.insert(tk.END, message + "\n")
                self.text_area.see(tk.END)
        finally:
            if not self.stop_event.is_set() and self.transcription_thread.is_alive():
                self.root.after(100, self.process_queue)

    def update_status(self, text):
        self.status_label.config(text=f"Status: {text}")

    def on_closing(self):
        self.stop_transcription()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
