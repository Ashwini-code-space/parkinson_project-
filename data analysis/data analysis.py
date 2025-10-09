import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==== CONFIGURATION ====
data_path = r"C:\Users\Vishal Raj\OneDrive\Desktop\park\Final_Parkinson_Dataset.csv"
output_folder = r"C:\Users\Vishal Raj\OneDrive\Desktop\park\Full_EDA"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "Plots"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "Plots", "Acoustic_Features"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "Plots", "Spectral_Features"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "Plots", "MFCC_Features"), exist_ok=True)

# ==== LOAD DATA ====
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()  # remove extra spaces

# ==== DATA CLEANING ====
df = df.drop_duplicates()

# Fill missing numeric values with mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill missing categorical values
df['Sex'] = df['Sex'].fillna(df['Sex'].mode()[0])

# Encode label: PD=1, Healthy=0
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])  # PD->1, Healthy->0
df['Age'] = df['Age'].astype(float)

# ==== BASIC DATA ANALYSIS ====
summary = df.describe()
summary.to_csv(os.path.join(output_folder, "data_summary.csv"))

class_counts = df['Label'].value_counts()
class_counts.to_csv(os.path.join(output_folder, "class_distribution.csv"))

sex_counts = df['Sex'].value_counts()
sex_counts.to_csv(os.path.join(output_folder, "sex_distribution.csv"))

# Correlation only on numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlation = df[numeric_cols].corr()
correlation.to_csv(os.path.join(output_folder, "correlation_matrix.csv"))

# ==== FEATURE SCALING ====
features = df.drop(['Label', 'Sex', 'filename', 'source'], axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['Label'] = df['Label']
scaled_df['Sex'] = df['Sex']
scaled_df.to_csv(os.path.join(output_folder, "cleaned_scaled_data.csv"), index=False)

# ==== FEATURE GROUPS ====
acoustic_features = ['pitch_mean', 'pitch_std', 'jitter', 'shimmer', 'hnr', 'duration', 'energy', 'zcr']
spectral_features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness']
mfcc_features = [f'mfcc_{i}' for i in range(1,14)]

# ==== HISTOGRAMS ====
def plot_histograms(feature_list, folder):
    for col in feature_list:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(folder, f"{col}_histogram.png"))
        plt.close()

plot_histograms(acoustic_features, os.path.join(output_folder, "Plots", "Acoustic_Features"))
plot_histograms(spectral_features, os.path.join(output_folder, "Plots", "Spectral_Features"))
plot_histograms(mfcc_features, os.path.join(output_folder, "Plots", "MFCC_Features"))

# ==== BOXPLOTS (PD vs Healthy) ====
def plot_boxplots(feature_list, folder):
    for col in feature_list:
        plt.figure(figsize=(6,4))
        sns.boxplot(x='Label', y=col, data=df)
        plt.title(f'{col} vs Label (PD=1, Healthy=0)')
        plt.savefig(os.path.join(folder, f"{col}_boxplot.png"))
        plt.close()

plot_boxplots(acoustic_features, os.path.join(output_folder, "Plots", "Acoustic_Features"))
plot_boxplots(spectral_features, os.path.join(output_folder, "Plots", "Spectral_Features"))
plot_boxplots(mfcc_features, os.path.join(output_folder, "Plots", "MFCC_Features"))

# ==== CORRELATION HEATMAPS ====
def plot_heatmap(feature_list, filename):
    numeric_subset = [f for f in feature_list + ['Label'] if f in numeric_cols]
    plt.figure(figsize=(10,8))
    sns.heatmap(df[numeric_subset].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f"Correlation Heatmap")
    plt.savefig(filename)
    plt.close()

plot_heatmap(acoustic_features, os.path.join(output_folder, "Plots", "acoustic_correlation.png"))
plot_heatmap(spectral_features, os.path.join(output_folder, "Plots", "spectral_correlation.png"))
plot_heatmap(mfcc_features, os.path.join(output_folder, "Plots", "mfcc_correlation.png"))

# ==== PD vs Healthy Feature Comparison (NUMERIC ONLY) ====
numeric_cols_features = [col for col in numeric_cols if col not in ['Label']]
pd_features = df[df['Label']==1][numeric_cols_features].mean()
healthy_features = df[df['Label']==0][numeric_cols_features].mean()

comparison = pd.concat([pd_features, healthy_features], axis=1)
comparison.columns = ['PD_mean', 'Healthy_mean']
comparison.to_csv(os.path.join(output_folder, "PD_vs_Healthy_feature_comparison.csv"))

# ==== SAVE CLEANED DATA ====
df.to_csv(os.path.join(output_folder, "cleaned_original_data.csv"), index=False)
scaled_df.to_csv(os.path.join(output_folder, "cleaned_scaled_data.csv"), index=False)

print("EDA and data cleaning completed. All outputs saved in:", output_folder)
