import pandas as pd
from pathlib import Path
import librosa 
import numpy as np

processed_metadata = pd.read_csv(Path("data/processed_metadata.csv"))
print (len(processed_metadata))
print(processed_metadata.shape)
print(processed_metadata.columns)
print(processed_metadata.head())

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

print(processed_metadata["emotion_class"].value_counts())
print(processed_metadata[["valence_mean","arousal_mean","emotion_class"]].head())