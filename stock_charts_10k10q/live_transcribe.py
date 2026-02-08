import speech_recognition as sr
import pyaudio
import tkinter as tk
from tkinter import scrolledtext
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

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Audio Transcriber")
        self.root.geometry("600x400")

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
        self.device_index = get_stereo_mix_device_index()
        if self.device_index is None:
            self.update_status("'Stereo Mix' not found. Please enable it.")
            self.text_area.insert(tk.END, "'Stereo Mix' device not found. Please make sure it's enabled in your sound settings.\n")
            return

        self.stop_event.clear()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
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
        self.update_status("Stopped.")

    def transcribe_audio(self):
        r = sr.Recognizer()
        with sr.Microphone(device_index=self.device_index) as source:
            self.update_status("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=2)
            self.update_status(f"Listening on 'Stereo Mix' (Device {self.device_index})...")

            while not self.stop_event.is_set():
                try:
                    audio = r.listen(source, timeout=1, phrase_time_limit=10)
                    text = r.recognize_google(audio)
                    self.transcription_queue.put(text)
                except sr.WaitTimeoutError:
                    continue # No speech detected
                except sr.UnknownValueError:
                    self.transcription_queue.put("(...)")
                except sr.RequestError as e:
                    error_msg = f"API Error: {e}"
                    self.transcription_queue.put(error_msg)
                    self.update_status(error_msg)
                    break

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
