import os
import argparse
from pydub import AudioSegment
from tkinter import filedialog, Tk

# On Windows, you may need to explicitly set the path to the ffmpeg executable.
# AudioSegment.converter = "C:\\path\\to\\ffmpeg.exe"

def convert_wav_directory_to_mp3(source_dir, dest_dir=None):
    """
    Converts all .wav files in the source directory to .mp3 format and saves them
    in the destination directory.

    Args:
        source_dir (str): The directory containing .wav files.
        dest_dir (str, optional): The directory to save .mp3 files. 
                                  If None, saves them in the source directory. 
                                  Defaults to None.
    """
    if dest_dir is None:
        dest_dir = source_dir

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Find all .wav files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.wav'):
            wav_path = os.path.join(source_dir, filename)
            mp3_filename = os.path.splitext(filename)[0] + '.mp3'
            mp3_path = os.path.join(dest_dir, mp3_filename)

            try:
                # Load the .wav file
                print(f'Loading {wav_path}...')
                audio = AudioSegment.from_wav(wav_path)

                # Export as .mp3
                print(f'Converting to {mp3_path}...')
                audio.export(mp3_path, format='mp3')
                print(f'Successfully converted {filename} to {mp3_filename}')

            except Exception as e:
                print(f'Error converting {filename}: {e}')

def convert_single_wav_to_mp3(wav_path, mp3_path=None):
    """
    Converts a single .wav file to .mp3 format.

    Args:
        wav_path (str): The path to the input .wav file.
        mp3_path (str, optional): The path to save the output .mp3 file. 
                                  If None, saves it in the same directory. 
                                  Defaults to None.
    """
    if mp3_path is None:
        mp3_path = os.path.splitext(wav_path)[0] + '.mp3'

    try:
        # Load the .wav file
        print(f'Loading {wav_path}...')
        audio = AudioSegment.from_wav(wav_path)

        # Export as .mp3
        print(f'Converting to {mp3_path}...')
        audio.export(mp3_path, format='mp3')
        print(f'Successfully converted {os.path.basename(wav_path)} to {os.path.basename(mp3_path)}')

    except Exception as e:
        print(f'Error converting {os.path.basename(wav_path)}: {e}')

def browse_for_file():
    """
    Opens a file dialog to select a .wav file and returns its path.
    """
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    filepath = filedialog.askopenfilename(
        title="Select a .wav file",
        filetypes=[("WAV files", "*.wav")]
    )
    return filepath

def main():
    """
    Main function to handle command-line arguments for the conversion script.
    """
    parser = argparse.ArgumentParser(
        description='Convert .wav files to .mp3 format.',
        epilog='''
        ------------------------------------------------------------------------
        Installation Requirements:
        ------------------------------------------------------------------------
        This script requires the `pydub` library and `ffmpeg`.

        1. Install pydub:
           pip install pydub

        2. Install ffmpeg:
           - Windows: Download the ffmpeg build from https://ffmpeg.org/download.html, 
             unzip it, and add the 'bin' directory to your system's PATH.
           - macOS (using Homebrew): brew install ffmpeg
           - Linux (using apt): sudo apt-get install ffmpeg
        ------------------------------------------------------------------------
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-s', '--source',
                        help='The source directory containing .wav files.')
    parser.add_argument('-i', '--input',
                        help='A single .wav file to convert.')
    parser.add_argument('-d', '--destination',
                        help='(Optional) The destination directory for the .mp3 files. ' \
                             'If not provided, .mp3 files are saved in the source directory.',
                        default=None)

    args = parser.parse_args()

    # If no arguments are provided, open the file browser by default
    if not args.input and not args.source:
        filepath = browse_for_file()
        if filepath:
            convert_single_wav_to_mp3(filepath, args.destination)
        else:
            print("No file selected.")
        return

    if args.input:
        if not os.path.isfile(args.input):
            print(f'Error: Input file not found at {args.input}')
            return
        convert_single_wav_to_mp3(args.input, args.destination)
    elif args.source:
        if not os.path.isdir(args.source):
            print(f'Error: Source directory not found at {args.source}')
            return
        convert_wav_directory_to_mp3(args.source, args.destination)

if __name__ == '__main__':
    main()
