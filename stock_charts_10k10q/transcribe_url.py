#!/usr/bin/env python3
"""
Standalone URL -> transcript script.

- Downloads bestaudio/best from a URL using yt-dlp to a temp folder
- Optionally extracts a cached 16k mono WAV for large video files via ffmpeg
- Transcribes with OpenAI Whisper and saves <title>.txt to the output folder

Requirements:
  pip install yt-dlp openai-whisper ffmpeg-python  # ffmpeg executable must be installed on PATH

Usage:
  python transcribe_url.py --url "https://www.youtube.com/watch?v=..." --out "./transcripts" --model small
"""

import os
import sys
import shutil
import tempfile
import argparse
import subprocess

def ensure_ffmpeg_in_path() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("Error: FFmpeg executable not found on PATH.")
        print("Install FFmpeg and ensure its 'bin' folder is on PATH.")
        print("Download: https://ffmpeg.org/download.html or Windows builds: https://www.gyan.dev/ffmpeg/builds/")
        sys.exit(1)

def ensure_yt_dlp_installed():
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception:
        print("yt-dlp not installed. Please run: pip install yt-dlp")
        return False

def is_video_file(path: str) -> bool:
    _, ext = os.path.splitext(path or "")
    return (ext or "").lower() in {
        ".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v", ".webm", ".mts", ".m2ts", ".ts",
        ".3gp", ".flv", ".mpeg",
    }

def _file_size_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return 0

def _cache_wav_path_for(src_path: str) -> str:
    import hashlib
    home = os.path.expanduser("~")
    cache_dir = os.path.join(home, ".audio2text_cache")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        stat = os.stat(src_path)
        key = f"{os.path.abspath(src_path)}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8", errors="ignore")
    except Exception:
        key = os.path.abspath(src_path).encode("utf-8", errors="ignore")
    digest = hashlib.sha1(key).hexdigest()
    return os.path.join(cache_dir, f"{digest}.wav")

def get_or_make_cached_wav_for_large_video(input_path: str, threshold_mb: int = 200) -> str | None:
    try:
        if not is_video_file(input_path):
            return None
        size_mb = _file_size_bytes(input_path) / (1024 * 1024.0)
        if size_mb < float(threshold_mb):
            return None
        out_wav = _cache_wav_path_for(input_path)
        if os.path.isfile(out_wav):
            return out_wav
        # Extract mono 16 kHz WAV using ffmpeg
        cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_wav]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.isfile(out_wav):
            return out_wav
        return None
    except Exception:
        return None

def download_media_to_temp(url: str) -> tuple[str | None, str | None]:
    if not ensure_yt_dlp_installed():
        return None, None
    import yt_dlp

    tmpdir = tempfile.mkdtemp(prefix="url2txt_")
    out_tmpl = os.path.join(tmpdir, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "quiet": False,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Try to resolve the filepath similar to yt-dlp logic
            title = info.get("title") or "download"
            ext = info.get("ext") or "m4a"
            candidate = os.path.join(tmpdir, f"{title}.{ext}")
            if os.path.isfile(candidate):
                return candidate, tmpdir
            # Fallback: pick newest file in tmpdir
            files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if os.path.isfile(os.path.join(tmpdir, f))]
            if not files:
                return None, tmpdir
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return files[0], tmpdir
    except Exception as e:
        print(f"yt-dlp download failed: {e}")
        return None, tmpdir

def main():
    parser = argparse.ArgumentParser(description="Transcribe a URL to text using Whisper.")
    parser.add_argument("--url", required=True, help="Media URL (YouTube, etc.)")
    parser.add_argument("--out", default=".", help="Output folder for transcript .txt")
    parser.add_argument("--model", default="small", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--threshold-mb", type=int, default=200, help="Extract cached 16k WAV for videos larger than this (MB)")
    args = parser.parse_args()

    # Validate env/tools
    ensure_ffmpeg_in_path()

    # Import whisper lazily to allow nice message if missing
    try:
        import whisper
    except Exception:
        print("openai-whisper not installed. Please run: pip install openai-whisper")
        sys.exit(1)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print("Downloading media...")
    media_path, tmpdir = download_media_to_temp(args.url)
    if not media_path:
        print("Failed to download media.")
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(1)

    try:
        base_name = os.path.splitext(os.path.basename(media_path))[0]
        print(f"Downloaded: {media_path}")

        # For large videos, extract cached wav
        cached = get_or_make_cached_wav_for_large_video(media_path, threshold_mb=args.threshold_mb)
        input_for_whisper = cached if cached else media_path

        print(f"Loading Whisper model '{args.model}'...")
        model = whisper.load_model(args.model)

        print("Transcribing...")
        result = model.transcribe(input_for_whisper, fp16=False, verbose=True)
        text = (result.get("text") or "").strip()

        out_txt = os.path.join(out_dir, f"{base_name}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved transcript: {out_txt}")
    finally:
        try:
            if tmpdir and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()

# Install prerequisites (if not already):
# pip install yt-dlp openai-whisper ffmpeg-python
# Ensure the ffmpeg executable is on your PATH.
# Run:
# python transcribe_url.py --url "https://www.youtube.com/watch?v=VIDEOID" --out "./transcripts" --model small


# python stock_charts_10k10q/transcribe_url.py --url "https://www.youtube.com/watch?v=VIDEOID" --out "C:/Users/juesh/Videos" --model small


