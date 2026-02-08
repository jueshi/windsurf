# mic_monitor.py
'''Here's an explanation of the key components of the mic_monitor.py code:

Imports:
import sounddevice as sd
import numpy as np
import sys

sounddevice: Main audio library for input/output
numpy: Used for audio data processing
sys: For error handling and system operations

Audio Callback Function:
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    outdata[:] = indata

This function is called for each audio block
Copies input audio data directly to output
Handles any status/error messages

Main Monitoring Function:
def start_mic_monitoring(input_device=None, output_device=None):

Takes optional input/output device parameters
If not specified, uses system defaults

Device Listing:

devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"{i}: {device['name']}...")

Lists all available audio devices
Shows device capabilities (input/output channels)

Stream Configuration:
samplerate = 44100  # CD quality
blocksize = 1024    # Audio buffer size
channels = 1        # Mono audio

Audio Stream Setup:
with sd.Stream(
    device=(input_device, output_device),
    samplerate=samplerate,
    blocksize=blocksize,
    channels=channels,
    callback=audio_callback
):

Creates audio stream with specified parameters
Uses callback function for real-time processing

Main Loop:
while True:
    sd.sleep(1000)

    Keeps the program running
    Processes audio in background via callback

Error Handling:
    except KeyboardInterrupt:
        print("\nStopping microphone monitoring...")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

Gracefully handles Ctrl+C interruption
Catches and reports other errors

The program:

Lists available audio devices
Sets up audio stream with default devices
Processes audio in real-time using callback
Runs until interrupted
Handles errors gracefully

You can modify the input/output devices by passing device IDs to start_mic_monitoring() or edit the audio settings (samplerate, blocksize, channels) as needed.
'''

import sounddevice as sd
import numpy as np
import sys

def audio_callback(indata, outdata, frames, time, status):
    """Audio callback function that copies input to output"""
    if status:
        print(status, file=sys.stderr)
    outdata[:] = indata

def start_mic_monitoring(input_device=None, output_device=None):
    """Start real-time microphone monitoring"""
    try:
        # Get available devices
        devices = sd.query_devices()
        print("Available audio devices:")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")

        # Set default devices if not specified
        if input_device is None:
            input_device = sd.default.device[0]
        if output_device is None:
            output_device = sd.default.device[1]

        # Print selected devices
        print(f"\nUsing input device: {devices[input_device]['name']}")
        print(f"Using output device: {devices[output_device]['name']}")

        # Audio settings
        samplerate = 44100
        blocksize = 1024
        channels = 1

        # Start audio stream
        with sd.Stream(
            device=(input_device, output_device),
            samplerate=samplerate,
            blocksize=blocksize,
            channels=channels,
            callback=audio_callback
        ):
            print("\nMicrophone monitoring started. Press Ctrl+C to stop.")
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\nStopping microphone monitoring...")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Start monitoring with default devices
    start_mic_monitoring()
