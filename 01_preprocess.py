
import librosa
import numpy as np
import pandas as pd
import os

# Function to extract features from a single audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, mono=True, duration=30)
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # Average the features across time to get a single vector
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Path to the GTZAN dataset
dataset_path = 'genres_original'
features_list = []
labels = []

for genre in os.listdir(dataset_path):
    genre_path = os.path.join(dataset_path, genre)
    if os.path.isdir(genre_path):
        for filename in os.listdir(genre_path):
            file_path = os.path.join(genre_path, filename)
            # Ensure it's a valid audio file
            if file_path.endswith('.wav'):
                print(f"Processing: {file_path}")
                features = extract_features(file_path)
                if features is not None:
                    features_list.append(features)
                    labels.append(genre)

# Create a DataFrame and save to CSV
df = pd.DataFrame(features_list)
df['genre'] = labels
df.to_csv('features.csv', index=False)
print("Features saved to features.csv")
