import pandas as pd 
from pathlib import Path
import librosa 
import numpy as np
from sklearn.model_selection import train_test_split


processed_metadata = pd.read_csv(Path("data/processed_metadata.csv"))

features = []
labels = []

print(len(processed_metadata))

threshold = 5

def emotion_class(valence, arousal):
    if valence >= threshold and arousal >= threshold:
        return "happy"
    elif valence >= threshold and arousal < threshold:
        return "calm"
    elif valence < threshold and arousal >= threshold:
        return "angry"
    elif valence < threshold and arousal < threshold:
        return "sad"
    
processed_metadata["emotion_class"] = processed_metadata.apply(lambda row: emotion_class(row["valence_mean"], row["arousal_mean"]), axis = 1) 

max_length = 2000

for index, row in processed_metadata.iterrows():
    
    if index % 100 == 0:
        print(f"Processing {index}/{len(processed_metadata)}...")

    
    audio_path = row["audio_path"]
    emotion_label = row["emotion_class"]
    waveform, sample_rate = librosa.load(audio_path)
    db_feature = librosa.power_to_db(librosa.feature.melspectrogram(y = waveform))

    current_length = db_feature.shape[1]
    if current_length > max_length:
        deficit = current_length - max_length
        db_feature = db_feature[:, :max_length]
    elif current_length < max_length:
        deficit = max_length - current_length
        db_feature = np.pad(db_feature, ((0,0), (0, deficit)), mode="constant")

    features.append(db_feature)
    labels.append(emotion_label)

    import traceback

for index, row in processed_metadata.iterrows():
    try:
        audio_path = row["audio_path"]
        emotion_label = row["emotion_class"]
        waveform, sample_rate = librosa.load(audio_path)
        db_feature = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform))

        current_length = db_feature.shape[1]
        if current_length > max_length:
            db_feature = db_feature[:, :max_length]
        elif current_length < max_length:
            deficit = max_length - current_length
            db_feature = np.pad(db_feature, ((0,0), (0, deficit)), mode="constant")

        features.append(db_feature)
        labels.append(emotion_label)
    except Exception as e:
        print(f"Error at index {index}, file {row['audio_path']}: {e}")
        traceback.print_exc()
        break

emotion_map = {
    "sad" : 0,
    "happy" : 1,
    "angry" : 2,
    "calm" : 3,
}


encoded_labels = []
for label in labels:
    encoded_labels.append(emotion_map[label])
x = np.array(features)
y = np.array(encoded_labels)

np.save("data/X_features.npy", x)
np.save("data/y_labels.npy", y)

print(len(x))
print(len(y))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42, stratify= y)