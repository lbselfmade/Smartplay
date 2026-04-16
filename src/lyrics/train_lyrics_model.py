import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import re

# ---- Emotion map ----
emotion_map = {
    "sad": 0,
    "happy": 1,
    "angry": 2,
    "calm": 3,
}

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

# ---- Load lyrics ----
lyrics_df = pd.read_csv("data/lyrics.csv")
processed = pd.read_csv("data/processed_metadata.csv")

# ---- Filter out bad lyrics ----
lyrics_df = lyrics_df[lyrics_df["lyrics"].notna()]
lyrics_df = lyrics_df[lyrics_df["lyrics"].str.len().between(100, 5000)]

print(f"Songs with valid lyrics: {len(lyrics_df)}")

# ---- Merge with emotion labels ----
merged = lyrics_df.merge(processed[["SongId", "valence_mean", "arousal_mean"]],
                          on="SongId", how="left")
merged = merged.dropna(subset=["valence_mean", "arousal_mean"])
merged["emotion_class"] = merged.apply(
    lambda row: emotion_class(row["valence_mean"], row["arousal_mean"]), axis=1
)
merged["label"] = merged["emotion_class"].map(emotion_map)

print(f"Class distribution:")
print(merged["emotion_class"].value_counts())

# ---- Clean lyrics ----
def clean_lyrics(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

merged["lyrics"] = merged["lyrics"].apply(clean_lyrics)

# ---- Dataset ----
class LyricsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
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
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    merged["lyrics"].tolist(),
    merged["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=merged["label"]
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---- Load tokenizer and model ----
MODEL_NAME = "distilroberta-base"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---- DataLoaders ----
train_dataset = LyricsDataset(X_train, y_train, tokenizer)
test_dataset = LyricsDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---- Classifier on top of DistilRoBERTa ----
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
        return self.classifier(cls_output)

# ---- Device ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = LyricsClassifier(num_classes=4).to(device)

# ---- Loss and optimiser ----
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ---- Training loop ----
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

# ---- Evaluation ----
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["sad", "happy", "angry", "calm"], zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ---- Save model ----
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/lyrics_model.pth")
print("Model saved to models/lyrics_model.pth")