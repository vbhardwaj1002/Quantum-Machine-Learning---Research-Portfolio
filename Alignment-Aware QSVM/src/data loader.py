"""
data_loader.py
--------------
Loads and standardises the NCBI MicroBIGG-E genomic feature matrices
for Ampicillin (AMP), Ciprofloxacin (CIP), and Cefotaxime (CTX).

Raw feature matrices contain acquired resistance genes and chromosomal
mutations as binary presence/absence columns, plus a 'Label' column
(0 = susceptible, 1 = resistant) and an 'Antibiotic' column.

This script:
  1. Standardises each antibiotic's feature matrix to a shared column schema
  2. Applies SMOTE+ENN rebalancing to correct extreme class imbalance
  3. Saves per-antibiotic balanced CSVs ready for downstream use

Usage:
    python data_loader.py --input_dir data/raw --output_dir data/processed
"""

import argparse
import os

import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split

# Shared column schema across all three antibiotics
# (union of all resistance genes observed in AMP, CIP, CTX datasets)
TARGET_COLUMNS = [
    'ampC', 'ampC_C-11T', 'ampC_C-42T', 'ampC_T-32A',
    'blaCARB-2', 'blaCMY', 'blaCMY-131', 'blaCMY-141', 'blaCMY-145',
    'blaCMY-146', 'blaCMY-148', 'blaCMY-16', 'blaCMY-196', 'blaCMY-2',
    'blaCMY-219', 'blaCMY-4', 'blaCMY-42', 'blaCMY-6',
    'blaCTX-M', 'blaCTX-M-1', 'blaCTX-M-104', 'blaCTX-M-130',
    'blaCTX-M-132', 'blaCTX-M-14', 'blaCTX-M-15', 'blaCTX-M-231',
    'blaCTX-M-24', 'blaCTX-M-27', 'blaCTX-M-3', 'blaCTX-M-32',
    'blaCTX-M-55', 'blaCTX-M-64', 'blaCTX-M-65', 'blaCTX-M-9',
    'blaDHA', 'blaDHA-1', 'blaEC', 'blaEC-5', 'blaFOX-5',
    'blaHER-3', 'blaLAP-1', 'blaLAP-2',
    'blaOXA', 'blaOXA-1', 'blaOXA-10', 'blaOXA-1205', 'blaOXA-1207',
    'blaOXA-193', 'blaOXA-453', 'blaOXA-460', 'blaOXA-489',
    'blaOXA-61_G-57T', 'blaOXA-9',
    'blaSFO-1', 'blaSHV-1', 'blaSHV-11', 'blaSHV-30', 'blaSHV-7',
    'blaTEM', 'blaTEM-1', 'blaTEM-103', 'blaTEM-12', 'blaTEM-135',
    'blaTEM-166', 'blaTEM-169', 'blaTEM-176', 'blaTEM-19', 'blaTEM-190',
    'blaTEM-215', 'blaTEM-30', 'blaTEM-32', 'blaTEM-33', 'blaTEM-34',
    'blaTEM-35', 'blaTEM-40', 'blaVEB-1', 'blaVEB-9', 'cdiA',
    'gyrA_A84P', 'gyrA_D87N', 'gyrA_D87V', 'gyrA_D87Y',
    'gyrA_S83A', 'gyrA_S83L', 'gyrA_S83V', 'gyrA_T86I',
    'parC_A108T', 'parC_A108V', 'parC_A56T', 'parC_A85T',
    'parC_E84G', 'parC_E84K', 'parC_E84V', 'parC_S57T',
    'parC_S80I', 'parC_S80R',
    'parE_D475E', 'parE_E460D', 'parE_E460K', 'parE_I355T',
    'parE_I464F', 'parE_I529L', 'parE_L416F', 'parE_L445H',
    'parE_P439S', 'parE_S458A', 'parE_S458T',
    'qepA', 'qepA1', 'qepA4', 'qepA8', 'qepA9',
    'qnrA', 'qnrA1', 'qnrB', 'qnrB1', 'qnrB19', 'qnrB2',
    'qnrB4', 'qnrB6', 'qnrB7', 'qnrD1',
    'qnrS', 'qnrS1', 'qnrS13', 'qnrS2', 'qnrS4', 'qnrVC1',
    'Label', 'Antibiotic',
]

ANTIBIOTICS = ['AMP', 'CIP', 'CTX']
DROP_COLS = ['BioSample', 'Isolate', 'Antibiotic', 'Label']
RANDOM_STATE = 42
TEST_SIZE = 0.2


def standardise(df, antibiotic):
    """Align a raw feature matrix to the shared TARGET_COLUMNS schema."""
    df = df.replace({True: 1, False: 0, 'True': 1, 'False': 0})
    out = pd.DataFrame(0, index=df.index, columns=TARGET_COLUMNS)
    for col in df.columns:
        if col in out.columns:
            out[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    if 'Label' in df.columns:
        out['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)
    out['Antibiotic'] = antibiotic
    return out


def apply_smoteenn(df, antibiotic, random_state=RANDOM_STATE):
    """
    Apply SMOTE+ENN rebalancing on the training split of one antibiotic's data.

    SMOTE+ENN combines synthetic oversampling of the minority class (SMOTE)
    with Edited Nearest Neighbours (ENN) undersampling to clean noisy boundary
    samples, which is well-suited for the extreme imbalance in AMR datasets
    (e.g., AMP: 0.39% susceptible).
    """
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].astype(int)
    y = df['Label'].astype(int)

    if len(y.unique()) < 2:
        print(f"  Skipping {antibiotic}: only one class present — {y.unique()}")
        return None

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=random_state
    )

    smote_enn = SMOTEENN(random_state=random_state)
    X_res, y_res = smote_enn.fit_resample(X_train, y_train)

    balanced = pd.concat(
        [pd.DataFrame(X_res, columns=feature_cols),
         pd.Series(y_res, name='Label')],
        axis=1
    )
    balanced['Antibiotic'] = antibiotic
    print(f"  {antibiotic}: {y_train.value_counts().to_dict()} -> {y_res.value_counts().to_dict()}")
    return balanced


def load_and_process(input_dir, output_dir):
    """Main pipeline: standardise -> SMOTE+ENN -> save per-antibiotic CSV."""
    os.makedirs(output_dir, exist_ok=True)

    for abx in ANTIBIOTICS:
        raw_path = os.path.join(input_dir, f'{abx}_FeatureMatrix.csv')
        if not os.path.exists(raw_path):
            print(f"File not found: {raw_path} — skipping {abx}")
            continue

        print(f"\nProcessing {abx}...")
        df_raw = pd.read_csv(raw_path)
        df_std = standardise(df_raw, abx)

        std_path = os.path.join(output_dir, f'{abx}_FeatureMatrix_Standardized.csv')
        df_std.to_csv(std_path, index=False)
        print(f"  Standardised matrix saved: {std_path}  shape={df_std.shape}")

        df_balanced = apply_smoteenn(df_std, abx)
        if df_balanced is not None:
            bal_path = os.path.join(output_dir, f'Balanced_{abx}_SMOTEENN.csv')
            df_balanced.to_csv(bal_path, index=False)
            print(f"  Balanced matrix saved:     {bal_path}  shape={df_balanced.shape}")

    # Merge all balanced datasets
    balanced_frames = []
    for abx in ANTIBIOTICS:
        bal_path = os.path.join(output_dir, f'Balanced_{abx}_SMOTEENN.csv')
        if os.path.exists(bal_path):
            balanced_frames.append(pd.read_csv(bal_path))

    if balanced_frames:
        merged = pd.concat(balanced_frames, ignore_index=True)
        merged_path = os.path.join(output_dir, 'Balanced_AMR_AllAntibiotics.csv')
        merged.to_csv(merged_path, index=False)
        print(f"\nMerged balanced dataset saved: {merged_path}  shape={merged.shape}")
        print(merged['Antibiotic'].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(description='Load and preprocess AMR feature matrices')
    parser.add_argument('--input_dir',  default='data/raw',       help='Folder containing raw *_FeatureMatrix.csv files')
    parser.add_argument('--output_dir', default='data/processed', help='Folder to save processed/balanced CSVs')
    args = parser.parse_args()
    load_and_process(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
