"""
preprocess.py
-------------
Loads and preprocesses the SWaT (Secure Water Treatment) dataset for
quantum/classical CPS anomaly detection.

Supports two dataset configurations:
  A) SWaT 2015 only  — Normal + Attack from the same year
  B) Hybrid (default) — SWaT Feb 2026 normal + SWaT 2015 attack data

The SWaT dataset is available upon request from the iTrust Centre for
Research in Cyber Security at SUTD:
  https://itrust.sutd.edu.sg/itrust-labs_datasets/

Attack rows must first be extracted from the attack Excel file using
trim_attack_data.py before running this script.

Steps:
  1. Load and combine raw sensor data
  2. Map 2026 column format to 2015 base format (hybrid mode)
  3. Handle 'Bad Input' strings, alarm encoding, variance filtering
  4. StandardScaler normalization
  5. Balanced sampling + stratified train/test split
  6. Save processed arrays for classical and quantum pipelines

Usage:
    python preprocess.py --mode hybrid
                         --normal_day1 data/raw/SWaT_19-Feb-2026.csv
                         --normal_day2 data/raw/SWaT_20-Feb-2026.csv
                         --attack_csv  data/raw/swat_attacks_only.csv
                         --output_dir  data/processed
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Column mapping: 2026 suffix format → 2015 base format ─────────────────────
COL_MAPPING = {
    'FIT101.Pv': 'FIT101', 'LIT101.Pv': 'LIT101', 'MV101.Status': 'MV101',
    'P101.Status': 'P101',
    'FIT201.Pv': 'FIT201', 'AIT201.Pv': 'AIT201', 'AIT202.Pv': 'AIT202',
    'AIT203.Pv': 'AIT203', 'MV201.Status': 'MV201', 'P201.Status': 'P201',
    'P203.Status': 'P203', 'P205.Status': 'P205',
    'DPIT301.Pv': 'DPIT301', 'FIT301.Pv': 'FIT301', 'LIT301.Pv': 'LIT301',
    'MV302.Status': 'MV302', 'MV304.Status': 'MV304', 'P301.Status': 'P301',
    'AIT402.Pv': 'AIT402', 'FIT401.Pv': 'FIT401', 'LIT401.Pv': 'LIT401',
    'P401.Status': 'P401', 'P403.Status': 'P403', 'UV401.Status': 'UV401',
    'AIT501.Pv': 'AIT501', 'AIT502.Pv': 'AIT502', 'AIT503.Pv': 'AIT503',
    'AIT504.Pv': 'AIT504',
    'FIT501.Pv': 'FIT501', 'FIT502.Pv': 'FIT502', 'FIT503.Pv': 'FIT503',
    'P501.Status': 'P501',
    'PIT501.Pv': 'PIT501', 'PIT502.Pv': 'PIT502', 'PIT503.Pv': 'PIT503',
    'P601.Status': 'P601',
}

N_NORMAL_SAMPLE = 54584   # Match attack count for balanced dataset
N_ATTACK_SAMPLE = 54584   # All available SWaT 2015 attack rows
TEST_SIZE       = 0.30
RANDOM_STATE    = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_col_names(df):
    df.columns = df.columns.str.strip()
    return df


def safe_numeric(series):
    """Convert a column to float, replacing 'Bad Input' with NaN, fill with median."""
    s = series.astype(str).str.strip().replace('Bad Input', np.nan)
    s = pd.to_numeric(s, errors='coerce')
    return s.fillna(s.median())


def encode_alarms(df):
    """Encode Active/Inactive alarm columns to 1/0; drop fully-bad columns."""
    alarm_cols = [c for c in df.columns if 'Alarm' in c or 'alarm' in c]
    for col in alarm_cols:
        vals = df[col].astype(str).str.strip().str.lower()
        if set(vals.unique()) <= {'bad input', 'nan'}:
            df.drop(columns=[col], inplace=True)
        else:
            df[col] = vals.map({'active': 1, 'inactive': 0}).fillna(0).astype(int)
    return df


# ── Mode A: SWaT 2015 only ────────────────────────────────────────────────────

def load_2015_only(normal_xlsx, attack_xlsx):
    print("Loading SWaT 2015 Normal...")
    df_norm = clean_col_names(pd.read_excel(normal_xlsx, skiprows=1, engine='openpyxl'))
    print(f"  {len(df_norm):,} rows x {df_norm.shape[1]} cols")

    print("Loading SWaT 2015 Attack...")
    df_atk = clean_col_names(pd.read_excel(attack_xlsx, skiprows=1, engine='openpyxl'))
    print(f"  {len(df_atk):,} rows x {df_atk.shape[1]} cols")

    # Find label column
    label_col = None
    for col in df_atk.columns:
        uvals = df_atk[col].astype(str).str.lower().str.strip().unique()
        if any(v in ['attack', 'normal', 'a ttack'] for v in uvals):
            label_col = col
            break
    print(f"  Label column: '{label_col}'")

    drop_kw = ['timestamp', 'time', label_col, 'normal/attack']
    feat_cols = [c for c in df_atk.columns
                 if not any(k.lower() in c.lower() for k in drop_kw)]

    for df_tmp in [df_norm, df_atk]:
        for col in feat_cols:
            if col in df_tmp.columns:
                df_tmp[col] = pd.to_numeric(df_tmp[col], errors='coerce')
                df_tmp[col].fillna(df_tmp[col].median(), inplace=True)

    df_atk['label'] = df_atk[label_col].astype(str).str.lower().str.strip()\
                          .apply(lambda x: 1 if 'attack' in x else 0)
    df_norm['label'] = 0

    common = [c for c in feat_cols if c in df_norm.columns and c in df_atk.columns]

    # Remove low-variance features
    sel = VarianceThreshold(threshold=0.01)
    sel.fit(df_norm[common].fillna(0))
    common = [c for c, m in zip(common, sel.get_support()) if m]
    print(f"  Features after variance filter: {len(common)}")

    df_normal_s = df_norm[common + ['label']].sample(
        n=min(N_NORMAL_SAMPLE, len(df_norm)), random_state=RANDOM_STATE)
    df_attack_s = df_atk[df_atk['label'] == 1][common + ['label']]
    return df_normal_s, df_attack_s, common


# ── Mode B: Hybrid 2026 normal + 2015 attacks ─────────────────────────────────

def load_hybrid(day1_csv, day2_csv, attack_csv):
    print("Loading SWaT 2026 Normal (Day 1 + Day 2)...")
    df1 = pd.read_csv(day1_csv, low_memory=False)
    df2 = pd.read_csv(day2_csv, low_memory=False)
    df_norm = clean_col_names(pd.concat([df1, df2], ignore_index=True))
    print(f"  {len(df_norm):,} rows x {df_norm.shape[1]} cols")

    print("Loading SWaT 2015 Attack data...")
    df_atk = clean_col_names(pd.read_csv(attack_csv, low_memory=False))
    df_atk['label'] = 1
    print(f"  {len(df_atk):,} rows x {df_atk.shape[1]} cols")

    feature_cols = list(COL_MAPPING.keys())

    # Build normal matrix from 2026 data
    X_norm = pd.DataFrame()
    for c26 in feature_cols:
        if c26 in df_norm.columns:
            X_norm[c26] = safe_numeric(df_norm[c26]).values
        else:
            X_norm[c26] = 0.0
    X_norm['label'] = 0

    # Build attack matrix from 2015 data
    X_atk = pd.DataFrame()
    for c26, c15 in COL_MAPPING.items():
        if c15 in df_atk.columns:
            X_atk[c26] = safe_numeric(df_atk[c15]).values
        else:
            X_atk[c26] = 0.0
    X_atk['label'] = 1

    return X_norm, X_atk, feature_cols


# ── Combined + scale ───────────────────────────────────────────────────────────

def combine_and_scale(df_normal_s, df_attack_s, feature_cols, output_dir):
    n_atk = min(N_ATTACK_SAMPLE, len(df_attack_s))
    df_attack_s = df_attack_s.sample(n=n_atk, random_state=RANDOM_STATE)

    df_combined = pd.concat([df_normal_s, df_attack_s], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X_raw = df_combined[feature_cols].values.astype(float)
    y     = df_combined['label'].values.astype(int)

    # Fill any residual NaNs with column means
    col_means = np.nanmean(X_raw, axis=0)
    nans = np.where(np.isnan(X_raw))
    X_raw[nans] = np.take(col_means, nans[1])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"\nFinal dataset: {len(df_combined):,} samples | {len(feature_cols)} features")
    print(f"  Normal: {(y == 0).sum():,} | Attack: {(y == 1).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'),  X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'),  y_test)

    df_out = pd.DataFrame(X_scaled, columns=feature_cols)
    df_out['label'] = y
    df_out.to_csv(os.path.join(output_dir, 'swat_final_labeled.csv'), index=False)

    with open(os.path.join(output_dir, 'feature_cols.txt'), 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f"\nSaved to: {output_dir}")
    print("  X_train.npy / X_test.npy / y_train.npy / y_test.npy")
    print("  swat_final_labeled.csv")
    print("  feature_cols.txt")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Preprocess SWaT dataset for CPS anomaly detection')
    parser.add_argument('--mode', choices=['2015', 'hybrid'], default='hybrid',
                        help='Dataset configuration: "2015" for SWaT 2015 only, '
                             '"hybrid" for SWaT 2026 normal + 2015 attacks (default)')
    parser.add_argument('--normal_day1', default='data/raw/SWaT_19-Feb-2026.csv')
    parser.add_argument('--normal_day2', default='data/raw/SWaT_20-Feb-2026.csv')
    parser.add_argument('--normal_xlsx', default='data/raw/SWaT_Dataset_Normal_v1.xlsx')
    parser.add_argument('--attack_csv',  default='data/raw/swat_attacks_only.csv')
    parser.add_argument('--attack_xlsx', default='data/raw/SWaT_Dataset_Attack_v0.xlsx')
    parser.add_argument('--output_dir',  default='data/processed')
    args = parser.parse_args()

    if args.mode == '2015':
        df_normal, df_attack, feat_cols = load_2015_only(
            args.normal_xlsx, args.attack_xlsx)
    else:
        df_normal, df_attack, feat_cols = load_hybrid(
            args.normal_day1, args.normal_day2, args.attack_csv)

    combine_and_scale(df_normal, df_attack, feat_cols, args.output_dir)


if __name__ == '__main__':
    main()
