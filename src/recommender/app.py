import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

# ---- Mood mapping logic ----
ENHANCEMENT_MAP = {
    "sad": "happy",
    "angry": "calm",
    "calm": "happy",
    "happy": "happy"
}

AROUSAL_REGULATION_MAP = {
    "sad": "calm",
    "angry": "calm",
    "calm": "happy",
    "happy": "calm"
}

GRADUAL_MAP = {
    "sad": "calm",
    "angry": "sad",
    "calm": "happy",
    "happy": "happy"
}

CONTRAST_MAP = {
    "sad": "angry",
    "angry": "happy",
    "calm": "angry",
    "happy": "sad"
}

def get_target_mood(current_mood, intent):
    if intent == "Mood Congruence — match my mood":
        return current_mood
    elif intent == "Mood Enhancement — improve my mood":
        return ENHANCEMENT_MAP[current_mood]
    elif intent == "Arousal Regulation — change my energy":
        return AROUSAL_REGULATION_MAP[current_mood]
    elif intent == "Gradual Transition — ease into a new mood":
        return GRADUAL_MAP[current_mood]
    elif intent == "Contrast — opposite of my mood":
        return CONTRAST_MAP[current_mood]
    return current_mood

# ---- Centroid-based recommendation ----
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

# ---- Mood selector ----
mood_emoji = {"happy": "😊 Happy", "sad": "😢 Sad", "angry": "😠 Angry", "calm": "😌 Calm"}
emotion_emoji = {"happy": "😊", "sad": "😢", "angry": "😠", "calm": "😌"}

mood = st.selectbox(
    "How are you feeling right now?",
    options=list(mood_emoji.keys()),
    format_func=lambda x: mood_emoji[x]
)

# ---- Intent selector ----
intent = st.radio(
    "What do you want from your music?",
    options=[
        "Mood Congruence — match my mood",
        "Mood Enhancement — improve my mood",
        "Arousal Regulation — change my energy",
        "Gradual Transition — ease into a new mood",
        "Contrast — opposite of my mood"
    ]
)

n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

# ---- Generate ----
if st.button("🎶 Generate Playlist"):
    target_mood = get_target_mood(mood, intent)

    if target_mood != mood:
        st.info(f"You're feeling {mood_emoji[mood]} — recommending {mood_emoji[target_mood]} music based on your intent.")

    results = recommend_by_centroid(target_mood, song_df, n)

    st.markdown(f"### Your {mood_emoji[target_mood]} Playlist")
    st.markdown(f"*{len(results)} songs recommended*")

    for i, row in results.iterrows():
        artist = row["artist"]
        title = row["title"]
        emotion = row["emotion"]
        st.markdown(f"**{artist}** — {title} {emotion_emoji[emotion]}")

    st.markdown("---")
    st.markdown("*Powered by Audio CNN + DEAM Dataset*")