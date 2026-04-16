import numpy as np
import torch
import torchaudio
import librosa
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd

# ---- Dataset ----
class AudioDataset(Dataset):
    def __init__(self, audio_paths, sample_rate=16000, max_length=160000):
        self.audio_paths = audio_paths
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
        return waveform

# ---- Load metadata ----
processed = pd.read_csv("data/processed_metadata.csv")
audio_paths = processed["audio_path"].tolist()
song_ids = processed["SongId"].values

print(f"Total songs: {len(audio_paths)}")

# ---- Device ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load WAV2VEC2 ----
print("Loading WAV2VEC2...")
bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec2 = bundle.get_model().to(device)
wav2vec2.eval()

# ---- DataLoader ----
dataset = AudioDataset(audio_paths)
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

# ---- Extract embeddings ----
print("Extracting WAV2VEC2 embeddings...")
all_embeddings = []

with torch.no_grad():
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        features, _ = wav2vec2.extract_features(batch)
        last_features = features[-1]
        pooled = last_features.mean(dim=1)
        all_embeddings.append(pooled.cpu().numpy())
        if i % 20 == 0:
            print(f"Processing batch {i}/{len(loader)}...")

embeddings = np.concatenate(all_embeddings, axis=0)
print(f"Embeddings shape: {embeddings.shape}")

# ---- Save ----
Path("data/embeddings").mkdir(exist_ok=True)
np.save("data/embeddings/wav2vec2_embeddings.npy", embeddings)
np.save("data/embeddings/wav2vec2_song_ids.npy", song_ids)
print("Saved wav2vec2 embeddings to data/embeddings/wav2vec2_embeddings.npy")