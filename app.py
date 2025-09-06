
import streamlit as st
import joblib
import librosa
import numpy as np
import pandas as pd
import os
import subprocess

# Load the trained model
# This assumes 'model.pkl' is in the same directory
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please make sure the model file is uploaded.")
    st.stop()

# Feature extraction function
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, mono=True, duration=30)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# Function to download audio from YouTube
def download_audio_from_youtube(url):
    try:
        st.info("Downloading the first 30 seconds of audio from YouTube...")
        # Use yt-dlp to download only the first 30 seconds of audio
        # --download-sections "*0-30": download from the beginning to 30 seconds
        command = f'yt-dlp -x --audio-format wav --download-sections "*0-30" -o "temp_youtube_audio.%(ext)s" --force-overwrite "{url}"'
        
        # Run the command with a timeout
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=120)
        
        st.info("Audio downloaded successfully. Analyzing...")
        
        # Find the downloaded file
        downloaded_file = None
        for file in os.listdir('.'):
            if file.startswith('temp_youtube_audio'):
                downloaded_file = file
                break
        
        if downloaded_file:
            return downloaded_file
        else:
            st.error("Failed to find the downloaded audio file from YouTube.")
            return None

    except subprocess.TimeoutExpired:
        st.error("The download process took too long and was timed out. Please try a different video.")
        return None
    except subprocess.CalledProcessError as e:
        st.error(f"Error downloading or processing YouTube video. Please check the URL.")
        st.error(f"yt-dlp error: {e.stderr}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

st.title("Music Genre Classifier")
st.write("Upload a song (.wav, .mp3) or provide a YouTube link to get its predicted genre.")

# Option 1: File Uploader
st.header("Option 1: Upload an Audio File")
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio_file.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio("temp_audio_file.wav")

    with st.spinner('Analyzing...'):
        features = extract_features("temp_audio_file.wav")
        if features is not None:
            prediction = model.predict([features])
            prediction_proba = model.predict_proba([features])

            st.success("Analysis Complete!")
            st.subheader("Predicted Genre:")
            st.write(f"### {prediction[0].capitalize()}")

            st.subheader("Prediction Probabilities:")
            df_proba = pd.DataFrame(prediction_proba, columns=model.classes_).T
            df_proba.columns = ['probability']
            st.bar_chart(df_proba)

# Option 2: YouTube Link
st.header("Option 2: Provide a YouTube Link")
youtube_url = st.text_input("Enter a YouTube video URL:")

if st.button("Classify from YouTube"):
    if youtube_url:
        # No spinner here, as the function now provides its own status updates
        audio_file = download_audio_from_youtube(youtube_url)
        if audio_file:
            st.audio(audio_file)
            features = extract_features(audio_file)
            if features is not None:
                prediction = model.predict([features])
                prediction_proba = model.predict_proba([features])

                st.success("Analysis Complete!")
                st.subheader("Predicted Genre:")
                st.write(f"### {prediction[0].capitalize()}")

                st.subheader("Prediction Probabilities:")
                df_proba = pd.DataFrame(prediction_proba, columns=model.classes_).T
                df_proba.columns = ['probability']
                st.bar_chart(df_proba)
    else:
        st.warning("Please enter a YouTube URL.")
