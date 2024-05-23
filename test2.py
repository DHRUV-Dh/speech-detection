import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
# Define the path to the directory containing the audio files
audio_dir = "/Volumes/Time Machine Backups/Desktop/speech/Audio_Speech_Actors_01-24/all"

# Define the maximum length for padding/truncating the audio features
max_length = 500

# Create empty lists to store the features and labels
features = []
labels = []

# Emotions in the dataset
emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

observed_emotions = ["calm", "happy", "fearful", "disgust"]

# Loop through each audio file in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        # Load the audio file
        file_path = os.path.join(audio_dir, filename)
        audio, sr = librosa.load(file_path, sr=None)
        
        # Extract audio features (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=1024)
        
        # Pad or truncate the audio features to the maximum length
        if mfccs.shape[1] < max_length:
            pad_width = max_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]
        
        # Flatten the MFCCs to obtain a feature vector
        audio_features = mfccs.flatten()
        
        # Extract the emotion label from the file name
        parts = filename.split("-")
        if len(parts) < 3:
            print(f"Skipping file with invalid name: {filename}")
            continue
        emotion = emotions.get(parts[2], "unknown")
        
        if emotion not in observed_emotions:
            continue
        
        # Append the features and labels to the respective lists
        features.append(audio_features)
        labels.append(emotion)

# Convert the lists to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=9)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # Initialize and train the machine learning model (SVM)
# model = SVC(kernel='linear')
# model.fit(X_train_scaled, y_train)
with open("/Volumes/Time Machine Backups/Desktop/speech/emotion_detection_model_new_4_emo.pkl", "rb") as file:
    model = pickle.load(file)
# Evaluate the model
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print(X_test)