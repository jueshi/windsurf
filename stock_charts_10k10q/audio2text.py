import whisper
import os
from tqdm import tqdm

# Load a model – choose from tiny / base / small / medium / large
model = whisper.load_model("small")     # memory: ~1 GB (CPU) / 6 GB (GPU)

import tkinter as tk
from tkinter import filedialog

# Progress callback function
def progress_callback(segment, total_segments):
    if not hasattr(progress_callback, 'pbar'):
        progress_callback.pbar = tqdm(total=total_segments, desc="Transcribing", unit="segment")
    progress_callback.pbar.update(1)
    if segment == total_segments - 1:  # If last segment
        progress_callback.pbar.close()

root = tk.Tk()
root.withdraw()

audio_file = filedialog.askopenfilename()
folder = os.path.dirname(audio_file)
file_name = os.path.basename(audio_file)
# Transcribe with progress
print("Starting transcription...")
result = model.transcribe(audio_file, progress_callback=progress_callback)   # result is a dict: {'text': '...', 'segments': [...]
print(result["text"])
with open(rf"{folder}\{file_name}.txt", "w") as f:
    f.write(result["text"])


# choco install ffmpeg
# ffmpeg -version
# pip install torch
# python -c "import torch; print(torch.cuda.is_available())"
# pip install -U openai-whisper
# whisper --help
# python "C:\Users\JueShi\OneDrive - Astera Labs, Inc\Documents\windsurf\audio2text.py"