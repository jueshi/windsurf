import speech_recognition as sr
import pyaudio
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import queue
import whisper
import torch
import wave
import os

def get_audio_devices():
    p = pyaudio.PyAudio()
    devices = {}
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            devices[info['name']] = i
    p.terminate()
    return devices

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Audio Transcriber")
        self.root.geometry("600x500")

        # Audio source selection
        self.source_frame = tk.Frame(root)
        self.source_frame.pack(pady=5)
        tk.Label(self.source_frame, text="Audio Source:").pack(side=tk.LEFT, padx=5)
        self.audio_devices = get_audio_devices()
        self.source_var = tk.StringVar()
        self.source_menu = ttk.Combobox(self.source_frame, textvariable=self.source_var, values=list(self.audio_devices.keys()), state="readonly", width=40)
        if self.audio_devices:
            # Try to set 'Stereo Mix' as default
            default_device = next((name for name in self.audio_devices if 'stereo mix' in name.lower()), list(self.audio_devices.keys())[0])
            self.source_menu.set(default_device)
        self.source_menu.pack(side=tk.LEFT, padx=5)

        # Engine selection
        self.engine_frame = tk.Frame(root)
        self.engine_frame.pack(pady=5)
        tk.Label(self.engine_frame, text="Engine:").pack(side=tk.LEFT, padx=5)
        self.engine_var = tk.StringVar(value="Google")
        self.engine_menu = ttk.Combobox(self.engine_frame, textvariable=self.engine_var, values=["Google", "Whisper"], state="readonly")
        self.engine_menu.pack(side=tk.LEFT, padx=5)
        self.engine_menu.bind("<<ComboboxSelected>>", self.on_engine_change)

        # Whisper model selection (initially hidden)
        self.model_frame = tk.Frame(root)
        tk.Label(self.model_frame, text="Whisper Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="base")
        self.model_menu = ttk.Combobox(self.model_frame, textvariable=self.model_var, values=["tiny", "base", "small", "medium"], state="readonly")
        self.model_menu.pack(side=tk.LEFT, padx=5)

        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20)
        self.text_area.pack(pady=10, padx=10)

        self.start_button = tk.Button(root, text="Start Transcription", command=self.start_transcription)
        self.start_button.pack(side=tk.LEFT, padx=10, expand=True)

        self.stop_button = tk.Button(root, text="Stop Transcription", command=self.stop_transcription, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=10, expand=True)

        self.status_label = tk.Label(root, text="Status: Not Started", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.transcription_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.transcription_thread = None
        self.whisper_model = None

        self.on_engine_change(None) # Set initial visibility of model selection

    def on_engine_change(self, event):
        if self.engine_var.get() == "Whisper":
            self.model_frame.pack(pady=5)
        else:
            self.model_frame.pack_forget()

    def start_transcription(self):
        selected_device_name = self.source_var.get()
        if not selected_device_name:
            self.update_status("No audio source selected.")
            return
        self.device_index = self.audio_devices[selected_device_name]

        self.stop_event.clear()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.source_menu.config(state=tk.DISABLED)
        self.engine_menu.config(state=tk.DISABLED)
        self.model_menu.config(state=tk.DISABLED)
        self.update_status("Starting...")

        target_func = self.transcribe_audio_google if self.engine_var.get() == "Google" else self.transcribe_audio_whisper
        self.transcription_thread = threading.Thread(target=target_func, daemon=True)
        self.transcription_thread.start()
        self.root.after(100, self.process_queue)

    def stop_transcription(self):
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.stop_event.set()
            self.transcription_thread.join(timeout=2)
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.source_menu.config(state=tk.NORMAL)
        self.engine_menu.config(state=tk.NORMAL)
        self.model_menu.config(state=tk.NORMAL)
        self.update_status("Stopped.")

    def transcribe_audio_google(self):
        r = sr.Recognizer()
        # Tune thresholds a bit for Stereo Mix sources
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.8
        r.non_speaking_duration = 0.3
        with sr.Microphone(device_index=self.device_index) as source:
            self.update_status("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=2)
            self.update_status("Listening with Google...")
            while not self.stop_event.is_set():
                try:
                    # Force periodic segmentation so continuous audio doesn't stall
                    audio = r.record(source, duration=5)
                    text = r.recognize_google(audio)
                    self.transcription_queue.put(text)
                except sr.UnknownValueError:
                    self.transcription_queue.put("(...)")
                    continue
                except sr.RequestError as e:
                    self.transcription_queue.put(f"API Error: {e}")
                    break

    def transcribe_audio_whisper(self):
        r = sr.Recognizer()
        model_name = self.model_var.get()
        self.update_status(f"Preparing Whisper model '{model_name}'...")

        with sr.Microphone(device_index=self.device_index) as source:
            self.update_status("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=2)
            self.update_status(f"Listening with Whisper ({model_name})...")

            while not self.stop_event.is_set():
                try:
                    # Periodic fixed-size capture for reliability with continuous sources
                    audio = r.record(source, duration=8)
                    text = r.recognize_whisper(audio, model=model_name)
                    self.transcription_queue.put(text)
                except sr.UnknownValueError:
                    self.transcription_queue.put("(...)")
                    continue
                except sr.RequestError as e:
                    self.transcription_queue.put(f"Whisper API error: {e}")
                    break
                except Exception as e:
                    self.transcription_queue.put(f"An unexpected error occurred: {e}")
                    break

    def process_queue(self):
        try:
            while not self.transcription_queue.empty():
                message = self.transcription_queue.get_nowait()
                # Add a prefix to make it clear this is a transcription result
                if message.strip(): # Avoid adding prefix to empty lines
                    display_message = f"Transcription: {message}\n"
                    self.text_area.insert(tk.END, display_message)
                    self.text_area.see(tk.END)
        finally:
            if not self.stop_event.is_set() and (self.transcription_thread and self.transcription_thread.is_alive()):
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
