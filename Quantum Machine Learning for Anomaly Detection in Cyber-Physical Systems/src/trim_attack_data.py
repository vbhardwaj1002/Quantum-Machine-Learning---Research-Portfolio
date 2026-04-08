"""
trim_attack_data.py
-------------------
Extracts only the attack rows from the SWaT 2015 attack Excel file
and saves them as a CSV for use in preprocess.py.

The SWaT_Dataset_Attack_v0.xlsx file contains both 'Normal' and 'Attack'
labelled rows. This script filters to attack rows only and saves a
lean CSV to avoid loading the full Excel file in downstream scripts.

Usage:
    python trim_attack_data.py --input  data/raw/SWaT_Dataset_Attack_v0.xlsx
                               --output data/raw/swat_attacks_only.csv
"""

import argparse

import pandas as pd


def extract_attack_rows(input_path, output_path):
    print(f"Loading: {input_path}")
    df = pd.read_excel(input_path, skiprows=1, engine='openpyxl')
    print(f"  Shape: {df.shape}")

    # The label is in the last column
    label_col = df.columns[-1]
    print(f"  Label column: '{label_col}'")
    print(f"  Unique values: {df[label_col].astype(str).str.lower().str.strip().unique()}")

    df_attacks = df[df[label_col].astype(str).str.lower().str.strip() == 'attack']
    df_attacks.to_csv(output_path, index=False)
    print(f"\nSaved {len(df_attacks):,} attack rows to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract attack rows from SWaT attack Excel file')
    parser.add_argument('--input',  default='data/raw/SWaT_Dataset_Attack_v0.xlsx',
                        help='Path to SWaT_Dataset_Attack_v0.xlsx')
    parser.add_argument('--output', default='data/raw/swat_attacks_only.csv',
                        help='Output path for the filtered attack CSV')
    args = parser.parse_args()
    extract_attack_rows(args.input, args.output)


if __name__ == '__main__':
    main()
