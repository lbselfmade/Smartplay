import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

# ---- CNN Model (same architecture as training) ----
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
        return x

    def get_embedding(self, x):
        return self.forward(x)

# ---- Load features ----
X = np.load("data/X_features.npy")
y = np.load("data/y_labels.npy")
X = X[:1802]
y = y[:1802]

# ---- Load processed metadata to get SongIds ----
processed = pd.read_csv("data/processed_metadata.csv")
song_ids = processed["SongId"].values[:1802]

# ---- Normalise ----
X = (X - X.mean()) / X.std()

# ---- Add channel dimension ----
X = X[:, np.newaxis, :, :]

# ---- Convert to tensor ----
X_tensor = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# ---- Load model ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = AudioCNN(num_classes=4).to(device)
model.load_state_dict(torch.load("models/audio_cnn.pth", map_location=device), strict=False)
model.eval()

# ---- Extract embeddings ----
print("Extracting audio embeddings...")
all_embeddings = []

with torch.no_grad():
    for batch in loader:
        x_batch = batch[0].to(device)
        embeddings = model.get_embedding(x_batch)
        all_embeddings.append(embeddings.cpu().numpy())

audio_embeddings = np.concatenate(all_embeddings, axis=0)
print(f"Audio embeddings shape: {audio_embeddings.shape}")

# ---- Save embeddings and song IDs ----
Path("data/embeddings").mkdir(exist_ok=True)
np.save("data/embeddings/audio_embeddings.npy", audio_embeddings)
np.save("data/embeddings/audio_song_ids.npy", song_ids)

print("Saved audio embeddings to data/embeddings/audio_embeddings.npy")
print("Saved song IDs to data/embeddings/audio_song_ids.npy")