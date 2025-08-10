# scripts/run_feature_pipeline.py

import pandas as pd
from src.preprocess.clean_features import ElectricityDataProcessor

def main():
    # Load your input data
    input_path = "data/raw/hourly_fuel_fpl.csv"  # update if different
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # Convert 'period' column to datetime
    if 'period' in df.columns:
        df['period'] = pd.to_datetime(df['period'])
    else:
        raise ValueError("Column 'period' not found in DataFrame.")

    if 'value' not in df.columns:
        raise ValueError("Column 'value' not found in DataFrame.")

    # Initialize processor
    processor = ElectricityDataProcessor()

    # Run feature engineering pipeline
    print("Running feature engineering...")
    processed_path = processor.create_and_save_all_features(
        df,
        target_col='value',
        filename="daily_features.csv",
        scale_features=True
    )

    print(f"✅ Feature engineering complete. File saved at: {processed_path}")

if __name__ == "__main__":
    main()
