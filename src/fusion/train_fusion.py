import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# ---- Load embeddings ----
audio_embeddings = np.load("data/embeddings/audio_embeddings.npy")
audio_song_ids = np.load("data/embeddings/audio_song_ids.npy")

lyric_embeddings = np.load("data/embeddings/lyric_embeddings.npy")
lyric_song_ids = np.load("data/embeddings/lyric_song_ids.npy")
lyric_labels = np.load("data/embeddings/lyric_labels.npy")

print(f"Audio embeddings: {audio_embeddings.shape}")
print(f"Lyric embeddings: {lyric_embeddings.shape}")

# ---- Match songs that have both audio and lyrics ----
audio_id_to_idx = {sid: idx for idx, sid in enumerate(audio_song_ids)}

matched_audio = []
matched_lyrics = []
matched_labels = []
matched_ids = []

for i, sid in enumerate(lyric_song_ids):
    if sid in audio_id_to_idx:
        audio_idx = audio_id_to_idx[sid]
        matched_audio.append(audio_embeddings[audio_idx])
        matched_lyrics.append(lyric_embeddings[i])
        matched_labels.append(lyric_labels[i])
        matched_ids.append(sid)

matched_audio = np.array(matched_audio)
matched_lyrics = np.array(matched_lyrics)
matched_labels = np.array(matched_labels)

print(f"\nMatched songs (have both audio and lyrics): {len(matched_ids)}")
print(f"Audio embedding shape: {matched_audio.shape}")
print(f"Lyric embedding shape: {matched_lyrics.shape}")

import collections
print("Class distribution:", collections.Counter(matched_labels.tolist()))

# ---- Concatenate embeddings (early fusion) ----
fused = np.concatenate([matched_audio, matched_lyrics], axis=1)
print(f"Fused embedding shape: {fused.shape}")

# ---- Normalise ----
fused = (fused - fused.mean()) / fused.std()

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    fused, matched_labels, test_size=0.2, random_state=42, stratify=matched_labels
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---- Convert to tensors ----
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# ---- DataLoaders ----
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---- Fusion Classifier ----
class FusionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(FusionClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ---- Device ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nUsing device: {device}")

input_dim = fused.shape[1]  # 256 + 768 = 1024
model = FusionClassifier(input_dim=input_dim, num_classes=4).to(device)

# ---- Class weights ----
class_weights = torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ---- Training loop ----
EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

# ---- Evaluation ----
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["sad", "happy", "angry", "calm"], zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ---- Save model ----
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/fusion_model.pth")
print("Model saved to models/fusion_model.pth")