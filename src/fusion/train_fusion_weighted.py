import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# ---- Weighted Fusion Model ----
class WeightedFusion(nn.Module):
    def __init__(self, num_classes=4):
        super(WeightedFusion, self).__init__()
        # Learnable weights for each modality
        self.audio_weight = nn.Parameter(torch.tensor(0.7))
        self.lyric_weight = nn.Parameter(torch.tensor(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, audio_logits, lyric_logits):
        # Normalise weights
        total = self.audio_weight.abs() + self.lyric_weight.abs()
        w_audio = self.audio_weight.abs() / total
        w_lyric = self.lyric_weight.abs() / total
        # Weighted combination
        weighted = w_audio * audio_logits + w_lyric * lyric_logits
        # Also concatenate for richer representation
        combined = torch.cat([audio_logits, lyric_logits], dim=1)
        return self.classifier(combined), w_audio.item(), w_lyric.item()

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

# ---- Load Lyrics model ----
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
lyrics_model = LyricsClassifier(num_classes=4).to(device)
lyrics_model.load_state_dict(torch.load("models/lyrics_model.pth", map_location=device), strict=False)
lyrics_model.eval()

# ---- Extract logits for all matched songs ----
print("Extracting audio logits...")
all_audio_X = matched_audio_X[:, np.newaxis, :, :]
all_audio_t = torch.tensor(all_audio_X, dtype=torch.float32)
audio_loader = DataLoader(TensorDataset(all_audio_t), batch_size=16, shuffle=False)

audio_logits_all = []
with torch.no_grad():
    for batch in audio_loader:
        out = audio_model(batch[0].to(device))
        audio_logits_all.append(out.cpu().numpy())
audio_logits_all = np.concatenate(audio_logits_all, axis=0)

print("Extracting lyric logits...")
all_lyrics_dataset = LyricsDataset(matched_lyrics_text, tokenizer)
lyrics_loader = DataLoader(all_lyrics_dataset, batch_size=16, shuffle=False)

lyric_logits_all = []
with torch.no_grad():
    for batch in lyrics_loader:
        out = lyrics_model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        lyric_logits_all.append(out.cpu().numpy())
lyric_logits_all = np.concatenate(lyric_logits_all, axis=0)

# ---- Split logits ----
audio_logits_train = torch.tensor(audio_logits_all[idx_train], dtype=torch.float32)
audio_logits_test = torch.tensor(audio_logits_all[idx_test], dtype=torch.float32)
lyric_logits_train = torch.tensor(lyric_logits_all[idx_train], dtype=torch.float32)
lyric_logits_test = torch.tensor(lyric_logits_all[idx_test], dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# ---- DataLoader for fusion ----
train_dataset = TensorDataset(audio_logits_train, lyric_logits_train, y_train_t)
test_dataset = TensorDataset(audio_logits_test, lyric_logits_test, y_test_t)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---- Train weighted fusion ----
fusion_model = WeightedFusion(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 2.0, 2.0]).to(device))
optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

EPOCHS = 30

print("\nTraining weighted fusion model...")
for epoch in range(EPOCHS):
    fusion_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for audio_batch, lyric_batch, label_batch in train_loader:
        audio_batch = audio_batch.to(device)
        lyric_batch = lyric_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()
        outputs, w_audio, w_lyric = fusion_model(audio_batch, lyric_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == label_batch).sum().item()
        total += label_batch.size(0)

    train_acc = correct / total
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Audio weight: {w_audio:.3f} | Lyric weight: {w_lyric:.3f}")

# ---- Evaluation ----
fusion_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for audio_batch, lyric_batch, label_batch in test_loader:
        outputs, w_audio, w_lyric = fusion_model(audio_batch.to(device), lyric_batch.to(device))
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(label_batch.numpy())

print(f"\nFinal learned weights — Audio: {w_audio:.3f}, Lyrics: {w_lyric:.3f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["sad", "happy", "angry", "calm"], zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ---- Save ----
Path("models").mkdir(exist_ok=True)
torch.save(fusion_model.state_dict(), "models/fusion_weighted.pth")
print("Model saved to models/fusion_weighted.pth")