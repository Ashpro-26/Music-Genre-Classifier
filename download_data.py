
import deeplake
import os
import numpy as np
import soundfile as sf

# Load the dataset
ds = deeplake.load('hub://activeloop/gtzan-genre')

# Create a directory to store the genres
dataset_path = 'genres'
os.makedirs(dataset_path, exist_ok=True)

# Get the genre names from the labels
genre_names = ds.info.class_names

# Create subdirectories for each genre
for genre_name in genre_names:
    genre_path = os.path.join(dataset_path, genre_name)
    os.makedirs(genre_path, exist_ok=True)

# Iterate through the dataset and save the audio files
for i, sample in enumerate(ds):
    # Get the audio data and label
    audio_data = sample['audio'].numpy()
    label_index = sample['genres'].numpy()[0]
    genre = genre_names[label_index]

    # Generate a filename
    filename = f"{genre}.{i:05d}.wav"
    file_path = os.path.join(dataset_path, genre, filename)

    # Save the audio data as a .wav file
    # The audio data is mono, and the sample rate is 22050
    sf.write(file_path, audio_data, 22050)

    print(f"Saved: {file_path}")

print("Dataset downloaded and organized successfully.")
