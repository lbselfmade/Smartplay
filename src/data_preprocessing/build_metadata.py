from pathlib import Path
import pandas as pd

file1 = Path("data/annotations/annotations per each rater/song_level/static_annotations_songs_1_2000.csv")
file2 = Path("data/annotations/annotations per each rater/song_level/static_annotations_songs_2000_2058.csv")

# ---- File 1: per-rater annotations, so we average ourselves ----
df1 = pd.read_csv(file1)
df1.columns = df1.columns.str.strip()

df1 = df1[["SongId", "Valence", "Arousal"]]

song_level_1 = (
    df1.groupby("SongId", as_index=False)
       .agg(
           valence_mean=("Valence", "mean"),
           arousal_mean=("Arousal", "mean")
       )
)

# ---- File 2: already contains average values ----
df2 = pd.read_csv(file2)
df2.columns = df2.columns.str.strip()

df2 = df2[["SongId", "Valence_Average", "Arousal_Average"]].rename(
    columns={
        "Valence_Average": "valence_mean",
        "Arousal_Average": "arousal_mean"
    }
)

# ---- Combine both song-level tables ----
song_level = pd.concat([song_level_1, df2], ignore_index=True)

# Remove any accidental duplicates by SongId, keeping the first occurrence
song_level = song_level.drop_duplicates(subset="SongId")

# Build audio path
audio_dir = Path("data/MEMD_audio")
song_level["audio_path"] = song_level["SongId"].apply(lambda x: str(audio_dir / f"{x}.mp3"))

# Check if audio exists
song_level["audio_exists"] = song_level["audio_path"].apply(lambda p: Path(p).exists())

matched = song_level[song_level["audio_exists"]].copy()

print("First 5 rows:")
print(matched.head())

print(f"\nTotal songs in combined annotations: {len(song_level)}")
print(f"Songs with matching audio: {len(matched)}")

output_path = Path("data/processed_metadata.csv")
matched.to_csv(output_path, index=False)

print(f"\nSaved processed metadata to: {output_path}")