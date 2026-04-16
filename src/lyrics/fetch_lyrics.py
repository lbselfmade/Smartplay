import pandas as pd
import requests
import re
import time
from pathlib import Path

# ---- Genius API token ----
GENIUS_TOKEN = "t4ZB66Sj9e1zXDoRNIsGjmF_58kzzp-oQZAWwt5ry8ypLzv7GWbAkA_oE3a0zMab"

# ---- Load processed metadata ----
processed = pd.read_csv("data/processed_metadata.csv")

# ---- Load and combine DEAM metadata files ----
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

print(f"Total metadata entries: {len(metadata)}")
print(metadata.head())

# ---- Merge with processed metadata ----
merged = processed.merge(metadata[["song_id", "artist", "title"]],
                          left_on="SongId", right_on="song_id", how="left")

print(f"\nMerged shape: {merged.shape}")
print(f"Songs with title/artist: {merged['title'].notna().sum()}")
print(merged[["SongId", "artist", "title"]].head())

# ---- Genius API search ----
def search_genius(artist, title, token):
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": f"{artist} {title}"}
    try:
        response = requests.get("https://api.genius.com/search", headers=headers, params=params, timeout=10)
        data = response.json()
        hits = data["response"]["hits"]
        if hits:
            return hits[0]["result"]["url"]
        return None
    except Exception as e:
        print(f"Search error: {e}")
        return None

def scrape_lyrics(url):
    try:
        response = requests.get(url, timeout=10)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
        if not containers:
            return None
        lyrics = "\n".join([c.get_text(separator="\n") for c in containers])
        lyrics = re.sub(r"\[.*?\]", "", lyrics)
        lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)
        return lyrics.strip()
    except Exception as e:
        print(f"Scrape error: {e}")
        return None

# ---- Collect lyrics ----
results = []

for idx, row in merged.iterrows():
    song_id = row["SongId"]
    artist = row.get("artist", "")
    title = row.get("title", "")

    if pd.isna(artist) or pd.isna(title):
        results.append({"SongId": song_id, "artist": artist, "title": title, "lyrics": None})
        continue

    if idx % 50 == 0:
        print(f"Processing {idx}/{len(merged)}...")

    url = search_genius(artist, title, GENIUS_TOKEN)
    if url:
        lyrics = scrape_lyrics(url)
    else:
        lyrics = None

    results.append({"SongId": song_id, "artist": artist, "title": title, "lyrics": lyrics})
    time.sleep(0.5)

# ---- Save ----
lyrics_df = pd.DataFrame(results)
lyrics_df.to_csv("data/lyrics.csv", index=False)

print(f"\nDone! Total songs: {len(lyrics_df)}")
print(f"Songs with lyrics: {lyrics_df['lyrics'].notna().sum()}")
print(f"Songs without lyrics: {lyrics_df['lyrics'].isna().sum()}")