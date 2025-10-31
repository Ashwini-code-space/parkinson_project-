import os
import pandas as pd
import librosa
import numpy as np
from scipy.fft import fft, fftfreq
from pathlib import Path

def analyze_voice_features(audio_file, duration=7.0):
    """
    Analyzes voice features for tremors, stumbling, monotone, breathiness, hoarseness, and loudness.
    
    Args:
        audio_file (str): Path to the audio file (WAV format assumed).
        duration (float): Duration to analyze (seconds).
    
    Returns:
        dict: Dictionary of features with metrics. Returns None if error.
    """
    try:
        # Load audio
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file not found - {audio_file}")
            return None
        
        y, sr = librosa.load(audio_file, sr=None, duration=duration)
        
        # Preprocess: Normalize amplitude
        y = y / np.max(np.abs(y))
        
        # Initialize results dictionary
        results = {}
        
        # Common parameters
        frame_length = 2048  # ~50 ms at 44.1 kHz
        hop_length = 512
        
        # 1. Tremors (Amplitude and Pitch Fluctuations)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)  # Amplitude variability
        rms_peaks = len([i for i in range(1, len(rms)-1) if rms[i] > rms[i-1] and rms[i] > rms[i+1] and rms[i] > rms_mean + rms_std])
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = [pitches[np.argmax(magnitudes[:, i]), i] for i in range(pitches.shape[1]) if np.max(magnitudes[:, i]) > 0]
        pitch_values = [p for p in pitch_values if p > 0]  # Filter out invalid pitches
        pitch_mean = np.mean(pitch_values) if pitch_values else 0
        pitch_std = np.std(pitch_values) if pitch_values else 0
        pitch_peaks = len([i for i in range(1, len(pitch_values)-1) if pitch_values[i] > pitch_values[i-1] and pitch_values[i] > pitch_values[i+1] and pitch_values[i] > pitch_mean + pitch_std]) if pitch_values else 0
        
        # Tremor frequency (FFT on RMS)
        fft_freqs = fftfreq(len(rms), hop_length/sr)
        rms_spectrum = np.abs(fft(rms - rms_mean))
        valid_freqs = (fft_freqs > 0) & (fft_freqs <= 10)  # Focus on 0–10 Hz for tremors
        if np.any(valid_freqs):
            tremor_freq = fft_freqs[valid_freqs][np.argmax(rms_spectrum[valid_freqs])]
        else:
            tremor_freq = 0
        
        results['tremor_amplitude_peaks'] = rms_peaks
        results['tremor_pitch_peaks'] = pitch_peaks
        results['tremor_amplitude_std'] = rms_std
        results['tremor_pitch_std'] = pitch_std
        results['tremor_frequency_hz'] = tremor_freq
        
        # 2. Stumbling (Irregular amplitude dips)
        threshold = rms_mean * 0.9  # 10% below mean
        silence_threshold = 0.01  # Avoid complete silence
        stumble_frames = [i for i in range(len(rms)) if silence_threshold < rms[i] < threshold]
        num_stumbles = len(stumble_frames) // 2  # Approximate distinct events
        stumble_duration = len(stumble_frames) * hop_length / sr
        stumble_frequency = num_stumbles / duration if duration > 0 else 0
        
        results['stumble_num_events'] = num_stumbles
        results['stumble_total_duration_sec'] = stumble_duration
        results['stumble_frequency_per_sec'] = stumble_frequency
        
        # 3. Monotone (Pitch range and variability)
        pitch_range = np.max(pitch_values) - np.min(pitch_values) if pitch_values else 0
        
        results['monotone_pitch_range_hz'] = pitch_range
        results['monotone_pitch_std_hz'] = pitch_std
        
        # 4. Breathiness (Approximate using spectral flatness)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        avg_flatness = np.mean(spectral_flatness)  # Higher flatness indicates more noise (breathiness)
        
        results['breathiness_spectral_flatness'] = avg_flatness
        
        # 5. Hoarseness (Approximate using jitter-like pitch variation and shimmer-like amplitude variation)
        # Jitter: Cycle-to-cycle pitch variation (%)
        if len(pitch_values) > 1:
            jitter_diffs = [abs(pitch_values[i] - pitch_values[i-1]) / pitch_values[i-1] for i in range(1, len(pitch_values))]
            jitter = np.mean(jitter_diffs) * 100
        else:
            jitter = 0
        
        # Shimmer: Cycle-to-cycle amplitude variation (%)
        if len(rms) > 1:
            shimmer_diffs = [abs(rms[i] - rms[i-1]) / rms[i-1] for i in range(1, len(rms)) if rms[i-1] > 0]
            shimmer = np.mean(shimmer_diffs) * 100
        else:
            shimmer = 0
        
        results['hoarseness_jitter_percent'] = jitter
        results['hoarseness_shimmer_percent'] = shimmer
        
        # 6. Loudness
        mean_rms = rms_mean
        loudness_variability = rms_std
        
        results['loudness_mean_rms'] = mean_rms
        results['loudness_variability'] = loudness_variability
        
        return results
        
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

def process_csv_with_audio_features(csv_file, audio_dir, output_csv, audio_column='filename', duration=7.0):
    """
    Processes an Excel/CSV file with audio file names, locates files in a single audio folder,
    preserves existing columns, and adds voice feature columns.
    
    Args:
        csv_file (str): Path to input Excel/CSV file with audio file names.
        audio_dir (str): Directory containing all audio files.
        output_csv (str): Path to save output CSV with feature columns.
        audio_column (str): Column name with audio file names (default: 'filename').
        duration (float): Audio duration to analyze (seconds).
    
    Returns:
        pd.DataFrame: Updated DataFrame with original and new feature columns.
    """
    # Load file (CSV or Excel)
    if csv_file.lower().endswith('.xlsx'):
        df = pd.read_excel(csv_file, engine='openpyxl')
    else:
        df = pd.read_csv(csv_file)
    
    # Ensure audio_column exists
    if audio_column not in df.columns:
        raise ValueError(f"Column '{audio_column}' not found in file. Available columns: {list(df.columns)}")
    
    # Feature columns to add
    feature_columns = [
        'tremor_amplitude_peaks', 'tremor_pitch_peaks', 'tremor_amplitude_std', 'tremor_pitch_std', 'tremor_frequency_hz',
        'stumble_num_events', 'stumble_total_duration_sec', 'stumble_frequency_per_sec',
        'monotone_pitch_range_hz', 'monotone_pitch_std_hz',
        'breathiness_spectral_flatness',
        'hoarseness_jitter_percent', 'hoarseness_shimmer_percent',
        'loudness_mean_rms', 'loudness_variability'
    ]
    
    # Initialize new columns with NaN if not already present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # Process each audio file
    for idx, row in df.iterrows():
        audio_filename = str(row[audio_column])  # Convert to string
        
        # Append .wav if extension is missing
        if not audio_filename.lower().endswith('.wav'):
            audio_filename = audio_filename + '.wav'
        
        audio_path = Path(audio_dir) / audio_filename
        
        print(f"Processing {idx+1}/{len(df)}: {audio_filename}")
        
        features = analyze_voice_features(str(audio_path), duration)
        if features:
            for feat, value in features.items():
                df.at[idx, feat] = value
    
    # Save to output CSV
    df.to_csv(output_csv, index=False)
    print(f"\nOutput saved to: {output_csv}")
    
    return df

# Example usage
if __name__ == "__main__":
    # Update these paths
    csv_file = r"C:/Users/Vishal Raj/OneDrive/Desktop/park/adding more/all features train/updated_training_dataset_healthy.xlsx"  # Your Excel file
    audio_dir = r"C:\Users\Vishal Raj\OneDrive\Desktop\park\adding more\all features train\audio"  # Directory with all audio files
    output_csv = r"C:\Users\Vishal Raj\OneDrive\Desktop\park\adding more\all features train\output_training_dataset_for_healthy.csv"  # Output CSV
    
    df_updated = process_csv_with_audio_features(csv_file, audio_dir, output_csv, audio_column='filename')
    
    # Display first few rows with all columns
    print("\nUpdated DataFrame (first 5 rows, showing all columns):")
    print(df_updated.head())
