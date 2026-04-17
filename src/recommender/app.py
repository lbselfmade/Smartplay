import streamlit as st
import numpy as np
import pandas as pd
import os
from urllib.parse import quote
from sklearn.metrics.pairwise import cosine_similarity

# ---- Page config ----
st.set_page_config(
    page_title="SmartPlay - Emotion-Aware Music",
    page_icon="",
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

# ---- Mood mapping ----
ENHANCEMENT_MAP  = {"sad": "happy", "angry": "calm",  "calm": "happy", "happy": "happy"}
AROUSAL_MAP      = {"sad": "calm",  "angry": "calm",  "calm": "happy", "happy": "calm"}
GRADUAL_MAP      = {"sad": "calm",  "angry": "sad",   "calm": "happy", "happy": "happy"}
CONTRAST_MAP     = {"sad": "angry", "angry": "happy", "calm": "angry", "happy": "sad"}

STRATEGY_DESCRIPTIONS = {
    "Mood Congruence — match my mood":
        "Recommends music that reflects how you are already feeling.",
    "Mood Enhancement — improve my mood":
        "Recommends music to lift your mood toward something more positive.",
    "Arousal Regulation — change my energy":
        "Recommends music to adjust your energy level up or down.",
    "Gradual Transition — ease into a new mood":
        "Recommends music that gently shifts your emotional state.",
    "Contrast — opposite of my mood":
        "Recommends music from the opposite emotional quadrant."
}

def get_target_mood(current_mood, intent):
    if intent == "Mood Congruence — match my mood":
        return current_mood
    elif intent == "Mood Enhancement — improve my mood":
        return ENHANCEMENT_MAP[current_mood]
    elif intent == "Arousal Regulation — change my energy":
        return AROUSAL_MAP[current_mood]
    elif intent == "Gradual Transition — ease into a new mood":
        return GRADUAL_MAP[current_mood]
    elif intent == "Contrast — opposite of my mood":
        return CONTRAST_MAP[current_mood]
    return current_mood

# ---- Recommendation ----
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

# ---- Audio helpers ----
def get_local_audio_path(song_id):
    path = f"data/MEMD_audio/{int(song_id)}.mp3"
    return path if os.path.exists(path) else None

def get_youtube_url(artist, title):
    query = quote(f"{artist} {title}")
    return f"https://www.youtube.com/results?search_query={query}"

# ---- UI ----
st.title("SmartPlay")
st.markdown("#### Emotion-Aware Music Recommendation")
st.markdown("Select your current mood and choose how you would like music to influence how you feel.")
st.markdown("---")

song_df = load_data()

mood_labels = {"happy": "Happy", "sad": "Sad", "angry": "Angry", "calm": "Calm"}

col1, col2 = st.columns(2)

with col1:
    mood = st.selectbox(
        "How are you feeling right now?",
        options=list(mood_labels.keys()),
        format_func=lambda x: mood_labels[x]
    )

with col2:
    n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

st.markdown("##### What do you want from your music?")
intent = st.radio(
    "",
    options=list(STRATEGY_DESCRIPTIONS.keys()),
    label_visibility="collapsed"
)

st.caption(STRATEGY_DESCRIPTIONS[intent])
st.markdown("---")

if st.button("Generate Playlist", use_container_width=True):
    target_mood = get_target_mood(mood, intent)

    if target_mood != mood:
        st.info(f"You are feeling {mood_labels[mood]}. Recommending {mood_labels[target_mood]} music based on your selected strategy.")
    else:
        st.success(f"Recommending {mood_labels[target_mood]} music to match your current mood.")

    results = recommend_by_centroid(target_mood, song_df, n)

    st.markdown(f"### Your {mood_labels[target_mood]} Playlist")
    st.markdown(f"*{len(results)} songs ranked by audio similarity to the {target_mood} mood centroid*")
    st.markdown("---")

    has_local_audio = get_local_audio_path(results.iloc[0]["SongId"]) is not None

    for idx, (i, row) in enumerate(results.iterrows()):
        artist = row["artist"]
        title = row["title"]
        emotion = row["emotion"]
        similarity = row["similarity"]
        song_id = row["SongId"]

        st.markdown(f"**{idx+1}. {artist}** — {title}")
        st.caption(f"Mood: {emotion.capitalize()}  |  Similarity score: {similarity:.3f}")

        if has_local_audio:
            audio_path = get_local_audio_path(song_id)
            if audio_path:
                st.audio(audio_path, format="audio/mp3")
        else:
            yt_url = get_youtube_url(artist, title)
            st.markdown(
                f'<a href="{yt_url}" target="_blank" style="'
                f'display:inline-block;padding:6px 16px;background:#FF0000;'
                f'color:white;border-radius:4px;text-decoration:none;font-size:14px;">'
                f'Search on YouTube</a>',
                unsafe_allow_html=True
            )

        st.markdown("---")

    st.markdown("*Powered by audio CNN embeddings trained on the DEAM dataset*")