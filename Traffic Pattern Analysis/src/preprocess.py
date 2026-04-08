"""
preprocess.py
-------------
Loads and preprocesses the CICIDS2017 dataset for HTTPS vs SSH
traffic classification.

Steps:
  1. Load and combine raw CSV files from CICFlowMeter output
  2. Filter rows for HTTPS (port 443) and SSH (port 22)
  3. Encode labels: HTTPS -> 0, SSH -> 1
  4. Drop rows with missing values in selected features
  5. Scale features to [0, 1] using MinMaxScaler
  6. Perform stratified train/test split
  7. Apply SMOTE oversampling on the training set

Usage:
    python preprocess.py --data_dir <path_to_csv_folder> --output_dir <path_to_save>
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Features selected via Mutual Information scoring (see feature_selection.py)
SELECTED_FEATURES = [
    'Init_Win_bytes_backward',
    'Flow IAT Max',
    'Fwd IAT Max',
    'Max Packet Length',
    'Bwd Packets/s'
]

SAMPLE_SIZE_PER_CLASS = 500  # Reduce to 100 for small-dataset experiments
TEST_SIZE = 0.3
RANDOM_STATE = 42


def load_and_combine(data_dir):
    """Load and combine multiple CICIDS2017 CSV files from a directory."""
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    frames = []
    for fname in csv_files:
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        frames.append(df)
        print(f"  Loaded {fname}: {df.shape[0]} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined dataset: {combined.shape[0]} rows, {combined.shape[1]} columns")
    return combined


def filter_and_label(df):
    """Keep only HTTPS (443) and SSH (22) traffic and assign binary labels."""
    df = df[df['Destination Port'].isin([443, 22])].copy()
    df['Protocol_Type'] = df['Destination Port'].map({443: 'HTTPS', 22: 'SSH'})
    df['Target'] = df['Protocol_Type'].map({'HTTPS': 0, 'SSH': 1})
    print(f"\nProtocol counts:\n{df['Protocol_Type'].value_counts()}")
    return df


def sample_and_scale(df, features, sample_size, test_size, random_state):
    """Drop NaNs, sample balanced classes, scale features, and split."""
    df = df.dropna(subset=features)

    # Balance classes by sampling
    df_0 = df[df['Target'] == 0].sample(n=sample_size, random_state=random_state)
    df_1 = df[df['Target'] == 1].sample(n=sample_size, random_state=random_state)
    df_balanced = pd.concat([df_0, df_1])

    X = df_balanced[features].values
    y = df_balanced['Target'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE oversampling to the training set."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE — class counts: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


def main():
    parser = argparse.ArgumentParser(description="Preprocess CICIDS2017 for QSVM traffic classification")
    parser.add_argument('--data_dir', required=True, help="Directory containing CICIDS2017 CSV files")
    parser.add_argument('--output_dir', default='data/processed', help="Directory to save processed arrays")
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE_PER_CLASS,
                        help="Number of samples per class (default: 500)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Loading data ===")
    df = load_and_combine(args.data_dir)

    print("\n=== Filtering and labelling ===")
    df = filter_and_label(df)

    print("\n=== Sampling, scaling, and splitting ===")
    X_train, X_test, y_train, y_test, scaler = sample_and_scale(
        df, SELECTED_FEATURES, args.sample_size, TEST_SIZE, RANDOM_STATE
    )

    print("\n=== Applying SMOTE ===")
    X_train, y_train = apply_smote(X_train, y_train, RANDOM_STATE)

    # Save processed arrays
    np.save(os.path.join(args.output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(args.output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(args.output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(args.output_dir, 'y_test.npy'), y_test)
    print(f"\nProcessed data saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
