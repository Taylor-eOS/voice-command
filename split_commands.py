import os, sys
import math, statistics
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import settings

def split_audio_on_silence(input_path, output_dir, min_silence_len=settings.MIN_SILENCE, silence_thresh=None, keep_silence=100):
    audio = AudioSegment.from_file(input_path)
    if silence_thresh is None:
        silence_thresh = audio.dBFS - 16
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    for i, (start, end) in enumerate(nonsilent_ranges, 1):
        s = max(0, start - keep_silence)
        e = min(len(audio), end + keep_silence)
        chunk = audio[s:e]
        name = f"{i}.wav"
        chunk.export(os.path.join(output_dir, name), format="wav")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} input.wav")
        sys.exit(1)
    input_path = sys.argv[1]
    base, _ = os.path.splitext(os.path.basename(input_path))
    output_dir = f"{base}_segments"
    split_audio_on_silence(input_path, output_dir)
    print(f"segments saved to {output_dir}")

if __name__ == "__main__":
    main()

