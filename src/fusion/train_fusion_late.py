import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import pandas as pd
import re

# ---- CNN Model ----
class AudioCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AudioCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 250, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_out(x)
        return self.classifier(x)

# ---- Lyrics Model ----
class LyricsClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(LyricsClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained("distilroberta-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# ---- Lyrics Dataset ----
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

# ---- Load data ----
processed = pd.read_csv("data/processed_metadata.csv")
lyrics_df = pd.read_csv("data/lyrics.csv")
X_audio = np.load("data/X_features.npy")[:1802]
y_audio = np.load("data/y_labels.npy")[:1802]
audio_song_ids = processed["SongId"].values[:1802]

# ---- Normalise audio ----
X_audio = (X_audio - X_audio.mean()) / X_audio.std()

# ---- Filter valid lyrics ----
lyrics_df = lyrics_df[lyrics_df["lyrics"].notna()]
lyrics_df = lyrics_df[lyrics_df["lyrics"].str.len().between(100, 5000)]

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

merged = lyrics_df.merge(processed[["SongId", "valence_mean", "arousal_mean"]],
                          on="SongId", how="left")
merged = merged.dropna(subset=["valence_mean", "arousal_mean"])
merged["label"] = merged.apply(
    lambda row: emotion_map[emotion_class(row["valence_mean"], row["arousal_mean"])], axis=1
)

def clean_lyrics(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

merged["lyrics"] = merged["lyrics"].apply(clean_lyrics)

# ---- Match songs with both audio and lyrics ----
audio_id_to_idx = {sid: idx for idx, sid in enumerate(audio_song_ids)}

matched_audio_X = []
matched_lyrics_text = []
matched_labels = []

for _, row in merged.iterrows():
    sid = row["SongId"]
    if sid in audio_id_to_idx:
        matched_audio_X.append(X_audio[audio_id_to_idx[sid]])
        matched_lyrics_text.append(row["lyrics"])
        matched_labels.append(row["label"])

matched_audio_X = np.array(matched_audio_X)
matched_labels = np.array(matched_labels)

print(f"Matched songs: {len(matched_labels)}")

# ---- Train/test split ----
indices = np.arange(len(matched_labels))
idx_train, idx_test, y_train, y_test = train_test_split(
    indices, matched_labels, test_size=0.2, random_state=42, stratify=matched_labels
)

# ---- Device ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load CNN ----
audio_model = AudioCNN(num_classes=4).to(device)
audio_model.load_state_dict(torch.load("models/audio_cnn.pth", map_location=device), strict=False)
audio_model.eval()

# ---- Get audio predictions ----
print("Getting audio predictions...")
X_test_audio = matched_audio_X[idx_test][:, np.newaxis, :, :]
X_test_audio_t = torch.tensor(X_test_audio, dtype=torch.float32)
audio_loader = DataLoader(TensorDataset(X_test_audio_t), batch_size=16, shuffle=False)

audio_probs = []
with torch.no_grad():
    for batch in audio_loader:
        out = audio_model(batch[0].to(device))
        probs = torch.softmax(out, dim=1)
        audio_probs.append(probs.cpu().numpy())

audio_probs = np.concatenate(audio_probs, axis=0)

# ---- Load Lyrics model ----
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
lyrics_model = LyricsClassifier(num_classes=4).to(device)
lyrics_model.load_state_dict(torch.load("models/lyrics_model.pth", map_location=device), strict=False)
lyrics_model.eval()

# ---- Get lyrics predictions ----
print("Getting lyrics predictions...")
test_lyrics = [matched_lyrics_text[i] for i in idx_test]
lyrics_dataset = LyricsDataset(test_lyrics, tokenizer)
lyrics_loader = DataLoader(lyrics_dataset, batch_size=16, shuffle=False)

lyrics_probs = []
with torch.no_grad():
    for batch in lyrics_loader:
        out = lyrics_model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probs = torch.softmax(out, dim=1)
        lyrics_probs.append(probs.cpu().numpy())

lyrics_probs = np.concatenate(lyrics_probs, axis=0)

# ---- Late fusion: average probabilities ----
print("Fusing predictions...")
fused_probs = (audio_probs + lyrics_probs) / 2
final_preds = np.argmax(fused_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, final_preds, target_names=["sad", "happy", "angry", "calm"], zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, final_preds))