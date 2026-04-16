import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
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

for i, sid in enumerate(lyric_song_ids):
    if sid in audio_id_to_idx:
        audio_idx = audio_id_to_idx[sid]
        matched_audio.append(audio_embeddings[audio_idx])
        matched_lyrics.append(lyric_embeddings[i])
        matched_labels.append(lyric_labels[i])

matched_audio = np.array(matched_audio)
matched_lyrics = np.array(matched_lyrics)
matched_labels = np.array(matched_labels)

print(f"\nMatched songs: {len(matched_labels)}")

import collections
print("Class distribution:", collections.Counter(matched_labels.tolist()))

# ---- Concatenate embeddings ----
fused = np.concatenate([matched_audio, matched_lyrics], axis=1)
print(f"Fused embedding shape: {fused.shape}")

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    fused, matched_labels, test_size=0.2, random_state=42, stratify=matched_labels
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---- Scale features ----
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- SVM with GridSearchCV ----
print("\nRunning GridSearchCV to find best SVM params...")
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto"]
}

svm = SVC(class_weight="balanced", random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.4f}")

# ---- Evaluate best model ----
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["sad", "happy", "angry", "calm"], zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---- Save model and scaler ----
Path("models").mkdir(exist_ok=True)
joblib.dump(best_model, "models/fusion_svm.pkl")
joblib.dump(scaler, "models/fusion_scaler.pkl")
print("\nSVM model saved to models/fusion_svm.pkl")
print("Scaler saved to models/fusion_scaler.pkl")