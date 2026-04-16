import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import re

# ---- Emotion map ----
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

# ---- Load lyrics ----
lyrics_df = pd.read_csv("data/lyrics.csv")
processed = pd.read_csv("data/processed_metadata.csv")

# ---- Filter valid lyrics ----
lyrics_df = lyrics_df[lyrics_df["lyrics"].notna()]
lyrics_df = lyrics_df[lyrics_df["lyrics"].str.len().between(100, 5000)]

# ---- Merge with emotion labels ----
merged = lyrics_df.merge(processed[["SongId", "valence_mean", "arousal_mean"]],
                          on="SongId", how="left")
merged = merged.dropna(subset=["valence_mean", "arousal_mean"])
merged["emotion_class"] = merged.apply(
    lambda row: emotion_class(row["valence_mean"], row["arousal_mean"]), axis=1
)
merged["label"] = merged["emotion_class"].map(emotion_map)

print(f"Songs with valid lyrics: {len(merged)}")

# ---- Clean lyrics ----
def clean_lyrics(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

merged["lyrics"] = merged["lyrics"].apply(clean_lyrics)

# ---- Dataset ----
class LyricsDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }

# ---- Load tokenizer and model ----
MODEL_NAME = "distilroberta-base"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---- Classifier model ----
class LyricsClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(LyricsClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output  # Return embedding not classification

# ---- Device ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = LyricsClassifier(num_classes=4).to(device)
model.load_state_dict(torch.load("models/lyrics_model.pth", map_location=device), strict=False)
model.eval()

# ---- DataLoader ----
dataset = LyricsDataset(merged["lyrics"].tolist(), tokenizer)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# ---- Extract embeddings ----
print("Extracting lyric embeddings...")
all_embeddings = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        embeddings = model(input_ids, attention_mask)
        all_embeddings.append(embeddings.cpu().numpy())

lyric_embeddings = np.concatenate(all_embeddings, axis=0)
print(f"Lyric embeddings shape: {lyric_embeddings.shape}")

# ---- Save ----
Path("data/embeddings").mkdir(exist_ok=True)
np.save("data/embeddings/lyric_embeddings.npy", lyric_embeddings)
np.save("data/embeddings/lyric_song_ids.npy", merged["SongId"].values)
np.save("data/embeddings/lyric_labels.npy", merged["label"].values)

print("Saved lyric embeddings to data/embeddings/lyric_embeddings.npy")
print("Saved song IDs to data/embeddings/lyric_song_ids.npy")
print("Saved labels to data/embeddings/lyric_labels.npy")