from pathlib import Path
import librosa

audio_files = list(Path("data/MEMD_audio").glob("*.mp3"))
print(f"Found {len(audio_files)} audio files")

test_file = audio_files[0]
y, sr = librosa.load(test_file, sr=22050)

print("Loaded:", test_file)
print("Sample rate:", sr)
print("Duration (seconds):", len(y) / sr)