import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ---- Load data ----
audio_embeddings = np.load("data/embeddings/audio_embeddings.npy")
audio_song_ids = np.load("data/embeddings/audio_song_ids.npy")

processed = pd.read_csv("data/processed_metadata.csv")

# ---- Load metadata for titles/artists ----
meta_2013 = pd.read_csv("data/metadata_2013.csv", header=0,
                         names=["song_id", "filename", "artist", "title", "start", "end", "genre"],
                         on_bad_lines='skip', engine='python')
meta_2014 = pd.read_csv("data/metadata_2014.csv", usecols=[0, 1, 3],
                         names=["song_id", "artist", "title"], header=0,
                         on_bad_lines='skip', engine='python')
meta_2015 = pd.read_csv("data/metadata_2015.csv", header=0,
                         names=["song_id", "filename", "artist", "title", "start", "end", "genre"],
                         on_bad_lines='skip', engine='python')

metadata = pd.concat([meta_2013, meta_2014, meta_2015], ignore_index=True)
metadata["song_id"] = pd.to_numeric(metadata["song_id"], errors='coerce')
metadata = metadata.dropna(subset=["song_id"])
metadata["song_id"] = metadata["song_id"].astype(int)
metadata["artist"] = metadata["artist"].str.strip()
metadata["title"] = metadata["title"].str.strip()

# ---- Emotion labels ----
threshold = 5
def emotion_class(valence, arousal):
    if valence >= threshold and arousal >= threshold:
        return "happy"
    elif valence >= threshold and arousal < threshold:
        return "calm"
    elif valence < threshold and arousal >= threshold:
        return "angry"
    else:
        return "sad"

emotion_map = {"sad": 0, "happy": 1, "angry": 2, "calm": 3}
reverse_map = {0: "sad", 1: "happy", 2: "angry", 3: "calm"}

processed["emotion"] = processed.apply(
    lambda row: emotion_class(row["valence_mean"], row["arousal_mean"]), axis=1
)
processed["label"] = processed["emotion"].map(emotion_map)

# ---- Merge with metadata ----
song_df = processed.merge(metadata[["song_id", "artist", "title"]],
                           left_on="SongId", right_on="song_id", how="left")

# ---- Add embeddings ----
id_to_embedding = {sid: audio_embeddings[i] for i, sid in enumerate(audio_song_ids)}
song_df["embedding"] = song_df["SongId"].map(id_to_embedding)
song_df = song_df.dropna(subset=["embedding"])

print(f"Total songs in recommender: {len(song_df)}")
print(f"Class distribution:")
print(song_df["emotion"].value_counts())

# ============================================================
# APPROACH 1 — Emotion Filtering
# ============================================================
def recommend_by_filter(mood, song_df, n=10):
    """Filter songs by predicted emotion label."""
    filtered = song_df[song_df["emotion"] == mood].copy()
    recommendations = filtered.sample(min(n, len(filtered)), random_state=42)
    return recommendations[["SongId", "artist", "title", "emotion"]]

# ============================================================
# APPROACH 2 — Cosine Similarity
# ============================================================
def recommend_by_cosine(mood, song_df, n=10):
    """Find songs most similar to the centroid of the target emotion."""
    target_songs = song_df[song_df["emotion"] == mood]
    centroid = np.stack(target_songs["embedding"].values).mean(axis=0)
    
    all_embeddings = np.stack(song_df["embedding"].values)
    similarities = cosine_similarity([centroid], all_embeddings)[0]
    
    song_df = song_df.copy()
    song_df["similarity"] = similarities
    recommendations = song_df.nlargest(n, "similarity")
    return recommendations[["SongId", "artist", "title", "emotion", "similarity"]]

# ============================================================
# APPROACH 3 — Centroid-Based
# ============================================================
def recommend_by_centroid(mood, song_df, n=10):
    """Compute centroid per emotion class, rank all songs by distance to target centroid."""
    centroids = {}
    for emotion in ["sad", "happy", "angry", "calm"]:
        songs = song_df[song_df["emotion"] == emotion]
        centroids[emotion] = np.stack(songs["embedding"].values).mean(axis=0)
    
    target_centroid = centroids[mood]
    all_embeddings = np.stack(song_df["embedding"].values)
    similarities = cosine_similarity([target_centroid], all_embeddings)[0]
    
    song_df = song_df.copy()
    song_df["centroid_similarity"] = similarities
    # Only return songs from target emotion class
    target_songs = song_df[song_df["emotion"] == mood]
    recommendations = target_songs.nlargest(n, "centroid_similarity")
    return recommendations[["SongId", "artist", "title", "emotion", "centroid_similarity"]]

# ============================================================
# EVALUATION — Precision@N
# ============================================================
def precision_at_n(recommendations, target_mood, n=10):
    """What % of top N recommendations match the target emotion."""
    top_n = recommendations.head(n)
    correct = (top_n["emotion"] == target_mood).sum()
    return correct / n

# ---- Test all three approaches ----
print("\n" + "="*60)
print("RECOMMENDER EVALUATION — Precision@10")
print("="*60)

moods = ["sad", "happy", "angry", "calm"]

for mood in moods:
    print(f"\nMood: {mood.upper()}")
    
    # Approach 1
    rec1 = recommend_by_filter(mood, song_df, n=10)
    p1 = precision_at_n(rec1, mood)
    print(f"  Approach 1 (Filter)     Precision@10: {p1:.2f}")
    
    # Approach 2
    rec2 = recommend_by_cosine(mood, song_df, n=10)
    p2 = precision_at_n(rec2, mood)
    print(f"  Approach 2 (Cosine)     Precision@10: {p2:.2f}")
    
    # Approach 3
    rec3 = recommend_by_centroid(mood, song_df, n=10)
    p3 = precision_at_n(rec3, mood)
    print(f"  Approach 3 (Centroid)   Precision@10: {p3:.2f}")

# ---- Sample output for happy ----
print("\n" + "="*60)
print("SAMPLE RECOMMENDATIONS FOR MOOD: HAPPY")
print("="*60)

print("\nApproach 1 (Filter):")
print(recommend_by_filter("happy", song_df, n=5).to_string(index=False))

print("\nApproach 2 (Cosine):")
print(recommend_by_cosine("happy", song_df, n=5).to_string(index=False))

print("\nApproach 3 (Centroid):")
print(recommend_by_centroid("happy", song_df, n=5).to_string(index=False))

# ---- Save results ----
Path("outputs").mkdir(exist_ok=True)
results = []
for mood in moods:
    rec1 = recommend_by_filter(mood, song_df, n=10)
    rec2 = recommend_by_cosine(mood, song_df, n=10)
    rec3 = recommend_by_centroid(mood, song_df, n=10)
    results.append({
        "mood": mood,
        "filter_precision": precision_at_n(rec1, mood),
        "cosine_precision": precision_at_n(rec2, mood),
        "centroid_precision": precision_at_n(rec3, mood)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/recommender_evaluation.csv", index=False)
print("\nSaved evaluation to outputs/recommender_evaluation.csv")
print(results_df)