#!/usr/bin/env python3
"""
MP3 Audio Splitter

This script splits an MP3 file into two equal-sized parts based on duration.
Requires the pydub library for audio manipulation.

Usage:
    python audio_splitter.py input_file.mp3
    
Or use the split_mp3_file() function directly in your code.

Installation:
    pip install pydub

Note: For MP3 support, you may also need to install ffmpeg:
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


def split_mp3_file(input_file, output_dir=None):
    """
    Split an MP3 file into two equal-sized parts.
    
    Args:
        input_file (str): Path to the input MP3 file
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
    
    if not input_path.suffix.lower() == '.mp3':
        raise ValueError("Input file must be an MP3 file")
    
    # Set output directory
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file names
    base_name = input_path.stem
    part1_path = output_dir / f"{base_name}_part1.mp3"
    part2_path = output_dir / f"{base_name}_part2.mp3"
    
    try:
        print(f"Loading MP3 file: {input_file}")
        # Load the MP3 file
        audio = AudioSegment.from_mp3(input_file)
        
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
        first_half.export(part1_path, format="mp3")
        
        print(f"Exporting second half to: {part2_path}")
        second_half.export(part2_path, format="mp3")
        
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
    Command-line interface for the MP3 splitter.
    """
    if len(sys.argv) != 2:
        print("Usage: python audio_splitter.py <input_mp3_file>")
        print("Example: python audio_splitter.py my_song.mp3")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        part1, part2 = split_mp3_file(input_file)
        print(f"\n✅ Successfully split MP3 file!")
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