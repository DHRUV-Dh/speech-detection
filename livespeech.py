import sounddevice as sd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
with open("/Volumes/Time Machine Backups/Desktop/speech/emotion_detection_model_new_4_emo.pkl", "rb") as file:
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

# Perform real-time emotion detection using microphone
def real_time_emotion_detection():
    duration = 3  # Duration for capturing audio in seconds
    sample_rate = 44100  # Standard audio CD sample rate

    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()  # Wait for the audio recording to complete

    # Predict the emotion for the captured audio
    predicted_emotion = predict_emotion_live(np.squeeze(audio_data), sample_rate)
    print("Predicted Emotion:", predicted_emotion)

# Call the function for real-time emotion detection using microphone
real_time_emotion_detection()
