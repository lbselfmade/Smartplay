import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import pandas as pd

# ---- Load embeddings ----
X = np.load("data/embeddings/wav2vec2_embeddings.npy")
song_ids = np.load("data/embeddings/wav2vec2_song_ids.npy")

print(f"Embeddings shape: {X.shape}")

# ---- Load labels ----
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

y = processed["label"].values

print("Class distribution:")
import collections
print(collections.Counter(y.tolist()))

# ---- Normalise ----
X = (X - X.mean()) / X.std()

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ---- Convert to tensors ----
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# ---- DataLoaders ----
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---- Classifier ----
class Wav2Vec2Classifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=4):
        super(Wav2Vec2Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ---- Device ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = Wav2Vec2Classifier().to(device)

# ---- Loss and optimiser ----
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

# ---- Save ----
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/wav2vec2_classifier.pth")
print("Model saved to models/wav2vec2_classifier.pth")