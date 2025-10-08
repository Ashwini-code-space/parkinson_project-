import os
import librosa
import numpy as np
import pandas as pd
import parselmouth  # for jitter/shimmer/HNR
from parselmouth.praat import call

# -------------------------------
# Function to extract features
# -------------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Duration
        duration = librosa.get_duration(y=y, sr=sr)

        # Pitch (F0)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

        # Energy
        energy = np.mean(librosa.feature.rms(y=y))

        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # Spectral Features
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = [np.mean(mfcc) for mfcc in mfccs]

        # Jitter, Shimmer, HNR using Praat via Parselmouth
        snd = parselmouth.Sound(file_path)
        pointProcess = call(snd, "To PointProcess (periodic, cc)", 75, 500)

        jitter_local = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        # Put all features in one row
        features = {
            "duration": duration,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "energy": energy,
            "zcr": zcr,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff": rolloff,
            "spectral_flatness": flatness,
            "jitter": jitter_local,
            "shimmer": shimmer_local,
            "hnr": hnr
        }

        # Add MFCCs
        for i, mfcc in enumerate(mfccs_mean):
            features[f"mfcc_{i+1}"] = mfcc

        return features

    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None

# -------------------------------
# Process Healthy & PD folders
# -------------------------------
def process_dataset(healthy_folder, pd_folder, demo_file, output_csv):
    all_data = []

    # Process Healthy
    for file in os.listdir(healthy_folder):
        if file.endswith(".wav"):
            fpath = os.path.join(healthy_folder, file)
            feats = extract_features(fpath)
            if feats:
                feats["filename"] = os.path.splitext(file)[0]  # remove .wav
                feats["source"] = "HealthyFolder"
                all_data.append(feats)

    # Process PD
    for file in os.listdir(pd_folder):
        if file.endswith(".wav"):
            fpath = os.path.join(pd_folder, file)
            feats = extract_features(fpath)
            if feats:
                feats["filename"] = os.path.splitext(file)[0]  # remove .wav
                feats["source"] = "PDFolder"
                all_data.append(feats)

    df_features = pd.DataFrame(all_data)

    # Load demographics file
    if demo_file.endswith(".csv"):
        df_demo = pd.read_csv(demo_file)
    else:
        df_demo = pd.read_excel(demo_file)

    # Rename "Sample ID" -> filename
    df_demo.rename(columns={"Sample ID": "filename"}, inplace=True)

    # Strip .wav if present
    df_demo["filename"] = df_demo["filename"].astype(str).str.replace(".wav", "", regex=False)

    # Merge on filename
    df = pd.merge(df_features, df_demo, on="filename", how="left")

    # Warn if missing demographics
    missing = df[df.isna().any(axis=1)]["filename"].tolist()
    if missing:
        print(f"⚠️ Warning: {len(missing)} files missing demographic info. Example: {missing[:5]}")

    # Save dataset
    df.to_csv(output_csv, index=False)
    print(f"✅ Final dataset saved to {output_csv} with shape {df.shape}")

# -------------------------------
# Run pipeline
# -------------------------------
healthy_folder = "Healthy"
pd_folder = "PD"
demo_file = r"C:\Users\Vishal Raj\OneDrive\Desktop\park\Demographics_age_sex.xlsx"   # update path if needed
output_csv = "Final_Parkinson_Dataset.csv"

process_dataset(healthy_folder, pd_folder, demo_file, output_csv)
