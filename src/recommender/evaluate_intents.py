import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ---- Page config ----
st.set_page_config(
    page_title="Emotion-Aware Music Recommender",
    page_icon="🎵",
    layout="centered"
)

# ---- Load data ----
@st.cache_data
def load_data():
    audio_embeddings = np.load("data/embeddings/audio_embeddings.npy")
    audio_song_ids = np.load("data/embeddings/audio_song_ids.npy")
    processed = pd.read_csv("data/processed_metadata.csv")

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

    processed["emotion"] = processed.apply(
        lambda row: emotion_class(row["valence_mean"], row["arousal_mean"]), axis=1
    )

    song_df = processed.merge(metadata[["song_id", "artist", "title"]],
                               left_on="SongId", right_on="song_id", how="left")

    id_to_embedding = {sid: audio_embeddings[i] for i, sid in enumerate(audio_song_ids)}
    song_df["embedding"] = song_df["SongId"].map(id_to_embedding)
    song_df = song_df.dropna(subset=["embedding"])
    song_df = song_df[song_df["artist"].notna() & song_df["title"].notna()]
    song_df = song_df[song_df["artist"] != "NaN"]

    return song_df

# ---- Mood mappings ----
INTENT_MAPS = {
    "Mood Congruence — match my mood":           {"sad": "sad",   "angry": "angry", "calm": "calm",  "happy": "happy"},
    "Mood Enhancement — improve my mood":         {"sad": "happy", "angry": "calm",  "calm": "happy", "happy": "happy"},
    "Arousal Regulation — change my energy":      {"sad": "calm",  "angry": "calm",  "calm": "happy", "happy": "calm"},
    "Gradual Transition — ease into a new mood":  {"sad": "calm",  "angry": "sad",   "calm": "happy", "happy": "happy"},
    "Contrast — opposite of my mood":             {"sad": "angry", "angry": "happy", "calm": "angry", "happy": "sad"},
}

# ---- Centroid recommendation ----
def recommend_by_centroid(target_mood, song_df, n=10):
    centroids = {}
    for emotion in ["sad", "happy", "angry", "calm"]:
        songs = song_df[song_df["emotion"] == emotion]
        centroids[emotion] = np.stack(songs["embedding"].values).mean(axis=0)
    target_centroid = centroids[target_mood]
    all_embeddings = np.stack(song_df["embedding"].values)
    similarities = cosine_similarity([target_centroid], all_embeddings)[0]
    song_df = song_df.copy()
    song_df["similarity"] = similarities
    target_songs = song_df[song_df["emotion"] == target_mood]
    return target_songs.nlargest(n, "similarity")[["SongId", "artist", "title", "emotion", "similarity"]]

# ---- UI ----
st.title("🎵 Emotion-Aware Music Recommender")
st.markdown("Tell us how you feel and what you want from your music.")

song_df = load_data()

mood_emoji = {"happy": "😊 Happy", "sad": "😢 Sad", "angry": "😠 Angry", "calm": "😌 Calm"}
emotion_emoji = {"happy": "😊", "sad": "😢", "angry": "😠", "calm": "😌"}

mood = st.selectbox(
    "How are you feeling right now?",
    options=list(mood_emoji.keys()),
    format_func=lambda x: mood_emoji[x]
)

intent = st.radio(
    "What do you want from your music?",
    options=list(INTENT_MAPS.keys())
)

n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if st.button("🎶 Generate Playlist"):
    target_mood = INTENT_MAPS[intent][mood]

    if target_mood != mood:
        st.info(f"You're feeling {mood_emoji[mood]} — recommending {mood_emoji[target_mood]} music based on your intent.")

    results = recommend_by_centroid(target_mood, song_df, n)

    st.markdown(f"### Your {mood_emoji[target_mood]} Playlist")

    for i, row in results.iterrows():
        st.markdown(f"**{row['artist']}** — {row['title']} {emotion_emoji[row['emotion']]}")

st.markdown("---")
st.markdown("*Powered by Audio CNN + DEAM Dataset*")
st.markdown("💬 Please rate your experience using the feedback form shared with you.")