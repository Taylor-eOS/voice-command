import os
import time
import tkinter as tk
import subprocess
import threading
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

class RecorderGUI:
    def __init__(self, master):
        self.master = master
        master.title("Segmented Recorder")
        self.recording = False
        self.process = None
        self.marks = []
        self.start_time = None
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        self.temp_file = os.path.join(self.output_dir, "temp_recording.wav")
        self.label = tk.Label(master, text="Record words with pauses, mark between words")
        self.label.pack(pady=10)
        self.record_button = tk.Button(master, text="Record", command=self.start_recording)
        self.record_button.pack(side=tk.LEFT, padx=10)
        self.mark_button = tk.Button(master, text="Mark", command=self.mark_segment, state=tk.DISABLED)
        self.mark_button.pack(side=tk.LEFT, padx=10)
        self.stop_button = tk.Button(master, text="Stop", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        self.status = tk.Label(master, text="Idle")
        self.status.pack(pady=10)

    def start_recording(self):
        if self.recording:
            return
        self.marks = []
        self.status.config(text="Recording...")
        self.record_button.config(state=tk.DISABLED)
        self.mark_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        self.recording = True
        self.start_time = time.time()
        def record():
            self.process = subprocess.Popen([
                "arecord", "-f", "cd", "-t", "wav", "-q", "-r", "16000", "-c", "1", self.temp_file])
            self.process.wait()
        threading.Thread(target=record, daemon=True).start()

    def mark_segment(self):
        if not self.recording:
            return
        elapsed = time.time() - self.start_time
        ms = int(elapsed * 1000)
        self.marks.append(ms)
        self.status.config(text=f"Segments: {len(self.marks) + 1}")

    def stop_recording(self):
        if not self.recording:
            return
        self.status.config(text="Stopping...")
        if self.process:
            self.process.terminate()
            self.process.wait()
        self.recording = False
        self.record_button.config(state=tk.NORMAL)
        self.mark_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.status.config(text="Processing...")
        self.master.after(100, self.split_segments)

    def split_segments(self):
        audio = AudioSegment.from_wav(self.temp_file)
        boundaries = [0] + self.marks + [len(audio)]
        silence_thresh = audio.dBFS - 20
        for i in range(len(boundaries) - 1):
            start_ms = boundaries[i]
            end_ms = boundaries[i + 1]
            segment = audio[start_ms:end_ms]
            self.status.config(text=f"Processing segment {i+1} of {len(boundaries) - 1}")
            self.master.update()
            nonsilent_ranges = detect_nonsilent(segment, min_silence_len=100, silence_thresh=silence_thresh)
            if nonsilent_ranges:
                first_sound = nonsilent_ranges[0][0]
                last_sound = nonsilent_ranges[-1][1]
                trimmed = segment[first_sound:last_sound]
                trimmed.export(os.path.join(self.output_dir, f"word_{i+1}.wav"), format="wav")
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        self.status.config(text=f"Saved {len(boundaries) - 1} segments")

def main():
    root = tk.Tk()
    app = RecorderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
