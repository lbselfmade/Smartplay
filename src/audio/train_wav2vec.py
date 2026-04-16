import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import pandas as pd

# ---- Dataset ----
import librosa

class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, sample_rate=16000, max_length=160000):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_length = max_length

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        waveform, sr = librosa.load(self.audio_paths[idx], sr=self.sample_rate, mono=True)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if waveform.shape[0] > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_length - waveform.shape[0]))
        return waveform, self.labels[idx]

# ---- Classifier on top of WAV2VEC2 ----
class Wav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(Wav2Vec2Classifier, self).__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2 = bundle.get_model()
        # Freeze wav2vec2 weights
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features, _ = self.wav2vec2.extract_features(x)
        # Use last layer features, average over time
        last_features = features[-1]
        pooled = last_features.mean(dim=1)
        return self.classifier(pooled)

# ---- Load metadata ----
processed = pd.read_csv("data/processed_metadata.csv")

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
processed["label"] = processed.apply(
    lambda row: emotion_map[emotion_class(row["valence_mean"], row["arousal_mean"])], axis=1
)

print(f"Total songs: {len(processed)}")
print("Class distribution:")
print(processed["label"].value_counts())

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    processed["audio_path"].tolist(),
    processed["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=processed["label"]
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ---- DataLoaders ----
train_dataset = AudioDataset(X_train, y_train)
test_dataset = AudioDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# ---- Device ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Model ----
model = Wav2Vec2Classifier(num_classes=4).to(device)

# ---- Loss and optimiser ----
class_weights = torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ---- Training loop ----
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for waveforms, labels in train_loader:
        waveforms = waveforms.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(waveforms)
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
    for waveforms, labels in test_loader:
        waveforms = waveforms.to(device)
        outputs = model(waveforms)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["sad", "happy", "angry", "calm"], zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ---- Save ----
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/wav2vec2_classifier.pth")
print("Model saved to models/wav2vec2_classifier.pth")