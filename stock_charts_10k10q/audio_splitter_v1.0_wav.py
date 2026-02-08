#!/usr/bin/env python3
"""
WAV Audio Splitter

This script splits a WAV file into two equal-sized parts based on duration.
Requires the pydub library for audio manipulation.

Usage:
    python audio_splitter_v1.0_wav.py input_file.wav
    
Or use the split_wav_file() function directly in your code.

Installation:
    pip install pydub

Note: Most uncompressed PCM WAV files work without ffmpeg. For compressed WAV
codecs, you may need ffmpeg:
    - Windows: Download from https://ffmpeg.org/download.html
    - macOS: brew install ffmpeg
    - Linux: sudo apt-get install ffmpeg
"""

import os
import sys
from pathlib import Path

try:
    from pydub import AudioSegment
except ImportError:
    print("Error: pydub library is required. Install it with: pip install pydub")
    sys.exit(1)


def split_wav_file(input_file, output_dir=None):
    """
    Split a WAV file into two equal-sized parts.
    
    Args:
        input_file (str): Path to the input WAV file
        output_dir (str, optional): Directory to save output files. 
                                  If None, uses the same directory as input file.
    
    Returns:
        tuple: Paths to the two output files (part1_path, part2_path)
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        Exception: If there's an error processing the audio file
    """
    # Validate input file
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if input_path.suffix.lower() != '.wav':
        raise ValueError("Input file must be a WAV file (.wav)")
    
    # Set output directory
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file names
    base_name = input_path.stem
    part1_path = output_dir / f"{base_name}_part1.wav"
    part2_path = output_dir / f"{base_name}_part2.wav"
    
    try:
        print(f"Loading WAV file: {input_file}")
        # Load the WAV file
        audio = AudioSegment.from_wav(input_file)
        
        # Get the duration and calculate midpoint
        duration_ms = len(audio)
        midpoint_ms = duration_ms // 2
        
        print(f"Total duration: {duration_ms / 1000:.2f} seconds")
        print(f"Split point: {midpoint_ms / 1000:.2f} seconds")
        
        # Split the audio into two parts
        first_half = audio[:midpoint_ms]
        second_half = audio[midpoint_ms:]
        
        # Export the two parts
        print(f"Exporting first half to: {part1_path}")
        first_half.export(part1_path, format="wav")
        
        print(f"Exporting second half to: {part2_path}")
        second_half.export(part2_path, format="wav")
        
        # Display file information
        print(f"\nSplit completed successfully!")
        print(f"Original file: {input_file} ({duration_ms / 1000:.2f}s)")
        print(f"Part 1: {part1_path} ({len(first_half) / 1000:.2f}s)")
        print(f"Part 2: {part2_path} ({len(second_half) / 1000:.2f}s)")
        
        return str(part1_path), str(part2_path)
        
    except Exception as e:
        raise Exception(f"Error processing audio file: {str(e)}")


def main():
    """
    Command-line interface for the WAV splitter.
    """
    input_file = None
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    else:
        # Open a file dialog to browse for a WAV file
        try:
            import tkinter as _tk
            from tkinter import filedialog as _fd
            _root = _tk.Tk(); _root.withdraw()
            input_file = _fd.askopenfilename(
                title="Select a WAV file to split",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )
            try:
                _root.update(); _root.destroy()
            except Exception:
                pass
            if not input_file:
                print("No file selected. Exiting.")
                sys.exit(1)
        except Exception:
            print("Usage: python audio_splitter_v1.0_wav.py <input_wav_file>")
            print("Example: python audio_splitter_v1.0_wav.py lecture.wav")
            sys.exit(1)
    
    try:
        part1, part2 = split_wav_file(input_file)
        print(f"\n✅ Successfully split WAV file!")
        print(f"📁 Output files:")
        print(f"   • {part1}")
        print(f"   • {part2}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()