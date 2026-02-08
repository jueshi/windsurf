#!/usr/bin/env python3
"""
Media (Audio/Video) Splitter

Split any supported media file (video or audio) into two equal parts by duration.
This uses ffmpeg/ffprobe, performs stream copy (no re-encode) for speed/quality,
and supports many formats out of the box as long as ffmpeg is installed.

Usage:
    python audio_splitter_v1.1_mp4.py <input_media>

Or call split_media_file() directly.

Requirements:
  - ffmpeg and ffprobe must be installed and available on PATH
    • Windows: https://ffmpeg.org/download.html (add bin folder to PATH)
    • macOS:   brew install ffmpeg
    • Linux:   sudo apt-get install ffmpeg
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional


SUPPORTED_EXTS = {
    # Video
    ".mp4", ".mov", ".mkv", ".avi", ".wmv", ".webm", ".flv", ".m4v", ".ts", ".m2ts",
    # Audio
    ".mp3", ".wav", ".aac", ".m4a", ".ogg", ".flac", ".wma", ".aiff", ".aif", ".alac"
}


def _require_ffmpeg():
    """Ensure ffmpeg and ffprobe are available on PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        raise EnvironmentError(
            "ffmpeg/ffprobe not found. Install ffmpeg and ensure both 'ffmpeg' and 'ffprobe' are on PATH."
        )
    return ffmpeg_path, ffprobe_path


def _ffprobe_duration_seconds(ffprobe: str, input_file: str) -> float:
    """Use ffprobe to get the media duration in seconds (float)."""
    try:
        cmd = [
            ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", input_file
        ]
        # Capture bytes and decode explicitly as UTF-8 to avoid locale issues on Windows (e.g., GBK)
        res = subprocess.run(cmd, capture_output=True, text=False, check=True)
        stdout = res.stdout.decode("utf-8", errors="replace")
        info = json.loads(stdout)
        # Prefer container format duration
        fmt = info.get("format", {})
        if "duration" in fmt:
            return float(fmt["duration"]) if fmt["duration"] is not None else 0.0
        # Fallback to first stream with duration
        for s in info.get("streams", []):
            if "duration" in s and s["duration"] is not None:
                return float(s["duration"])  # type: ignore
        raise ValueError("Unable to determine media duration")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}") from e


def split_media_file(input_file: str, output_dir: Optional[str] = None):
    """
    Split a media file (video or audio) into two equal-duration parts using ffmpeg copy.

    Args:
        input_file: Path to input media.
        output_dir: Optional output directory. Defaults to the input's directory.

    Returns:
        (part1_path, part2_path)

    Notes:
        - Uses stream copy (-c copy). Splits at nearest keyframe boundaries.
        - Works for many formats supported by ffmpeg.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    ext = input_path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        # Allow any file; warn but continue — ffmpeg might still support it
        print(f"Warning: Extension '{ext}' not in predefined supported list. Attempting with ffmpeg anyway...")

    if output_dir is None:
        output_dir_path = input_path.parent
    else:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    ffmpeg, ffprobe = _require_ffmpeg()

    # Determine midpoint
    duration = _ffprobe_duration_seconds(ffprobe, str(input_path))
    if duration <= 0:
        raise ValueError("Could not detect media duration or duration is zero.")
    mid = duration / 2.0

    # Output paths preserving extension
    base = input_path.stem
    part1_path = output_dir_path / f"{base}_part1{ext}"
    part2_path = output_dir_path / f"{base}_part2{ext}"

    print(f"Total duration: {duration:.2f}s; splitting at {mid:.2f}s")

    # Build commands: part1 uses -t, part2 uses -ss
    cmd1 = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path), "-t", f"{mid}", "-c", "copy", str(part1_path)]
    cmd2 = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{mid}", "-i", str(input_path), "-c", "copy", str(part2_path)]

    try:
        print(f"Exporting first half to: {part1_path}")
        subprocess.run(cmd1, check=True)
        print(f"Exporting second half to: {part2_path}")
        subprocess.run(cmd2, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed while splitting: {e}") from e

    print("\nSplit completed successfully!")
    return str(part1_path), str(part2_path)


def main():
    """CLI for the media splitter."""
    input_file = None
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    else:
        # GUI file picker for convenience
        try:
            import tkinter as _tk
            from tkinter import filedialog as _fd
            _root = _tk.Tk(); _root.withdraw()
            patterns = [
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.wmv *.webm *.flv *.m4v *.ts *.m2ts"),
                ("Audio files", "*.mp3 *.wav *.aac *.m4a *.ogg *.flac *.wma *.aiff *.aif *.alac"),
                ("All files", "*.*"),
            ]
            input_file = _fd.askopenfilename(title="Select a media file to split", filetypes=patterns)
            try:
                _root.update(); _root.destroy()
            except Exception:
                pass
            if not input_file:
                print("No file selected. Exiting.")
                sys.exit(1)
        except Exception:
            print("Usage: python audio_splitter_v1.1_mp4.py <input_media>")
            sys.exit(1)

    try:
        part1, part2 = split_media_file(input_file)
        print("\n✅ Successfully split media file!")
        print("📁 Output files:")
        print(f"   • {part1}")
        print(f"   • {part2}")
    except (FileNotFoundError, EnvironmentError, ValueError, RuntimeError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()