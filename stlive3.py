import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import pickle
import io
import time

# Load the trained model
with open("emotion_detection_model_new_all.49_emo.pkl", "rb") as file:
    model = pickle.load(file)

# Function to extract features from audio
def extract_features(audio_data, sample_rate, max_length=500):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40, n_fft=1024)
    
    # Pad or truncate the audio features to the maximum length
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    
    # Flatten the MFCCs to obtain a feature vector
    audio_features = mfccs.flatten()
    return audio_features

# Function for real-time emotion detection using microphone
def predict_emotion_live(audio_data, sample_rate):
    # Extract features from the live audio
    audio_features = extract_features(audio_data, sample_rate)
    
    # Standardize the features using the same scaler used during training
    scaler = StandardScaler()
    audio_features_scaled = scaler.fit_transform([audio_features])  # Scale the features
    
    # Predict the emotion using the trained model
    predicted_emotion = model.predict(audio_features_scaled)
    return predicted_emotion[0]

# Function to display emoji based on emotion
def display_emoji(emotion):
    emojis = {
        "neutral": "😐",
        "calm": "😌",
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "fearful": "😨",
        "disgust": "😖",
        "surprised": "😮"
    }
    return emojis.get(emotion, "Unknown")

def main():
    st.title("Real-time Emotion Detection")
    st.write("You can either start recording or upload an audio file.")

    # Button to start recording
    if st.button("Start Recording"):
        duration = 3  # Duration for capturing audio in seconds
        sample_rate = 44100  # Standard audio CD sample rate

        st.write("Recording...")
        with st.spinner("Recording..."):
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
            sd.wait()  # Wait for the audio recording to complete

            # Predict the emotion for the captured audio
            predicted_emotion = predict_emotion_live(np.squeeze(audio_data), sample_rate)
            st.write("Predicted Emotion:", predicted_emotion)
            st.write("Emoji:", display_emoji(predicted_emotion))

    # File uploader to upload an audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        audio_data = uploaded_file.read()

        # Convert audio bytes to numpy array
        audio, sample_rate = librosa.load(io.BytesIO(audio_data), sr=None)

        # Predict the emotion for the uploaded audio
        predicted_emotion = predict_emotion_live(audio, sample_rate)
        st.write("Predicted Emotion:", predicted_emotion)
        st.write("Emoji:", display_emoji(predicted_emotion))

if __name__ == "__main__":
    main()
