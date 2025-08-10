import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import logging
from typing import Optional
from src.utils.aws_utils import S3Handler
from datetime import datetime

class DataCleaner:
    """Handles cleaning and filtering of electricity data"""

    def __init__(self, log_level: int = logging.INFO):
        self.logger = self._setup_logger(log_level)
        self.s3_handler = S3Handler()

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def create_time_features(self, df: pd.DataFrame, period_col: str = 'period') -> pd.DataFrame:
        """
        Create time-based features from datetime column

        Args:
            df: DataFrame with datetime column
            period_col: Name of datetime column

        Returns:
            DataFrame with additional time features
        """
        df = df.copy()

        # Ensure datetime column is datetime type
        df[period_col] = pd.to_datetime(df[period_col])

        # Extract time components
        df['hour'] = df[period_col].dt.hour
        df['day_of_week'] = df[period_col].dt.dayofweek
        df['day_of_month'] = df[period_col].dt.day
        df['month'] = df[period_col].dt.month
        df['quarter'] = df[period_col].dt.quarter
        df['year'] = df[period_col].dt.year

        # Cyclical features (to capture cyclical nature of time)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 4).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 17)).astype(int)
        df['is_peak_hour'] = ((df['hour'] >= 16) & (df['hour'] <= 20)).astype(int)

        # Season features
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })

        self.logger.info(f"Created time features: {len(df.columns) - len(df.columns)} new features")
        return df

    def clean_data(self, input_key: str, output_key: Optional[str] = None) -> pd.DataFrame:
        try:
            self.logger.info(f"Reading data from S3: {input_key}")
            df = self.s3_handler.read_df_from_s3(input_key)

            initial_rows = len(df)
            initial_cols = len(df.columns)

            # Select required columns
            required_cols = ['period', 'type-name', 'value']
            df = df[required_cols]

            # Remove rows with NaN values
            df = df.dropna()

            # Create time features
            df = self.create_time_features(df, period_col='period')

            # Sort by period
            df = df.sort_values('period')

            self.logger.info(
                f"Cleaned data: "
                f"{initial_rows:,} → {len(df):,} rows, "
                f"{initial_cols} → {len(df.columns)} columns"
            )

            if output_key:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if not output_key.endswith('.csv'):
                    output_key = f"{output_key}_{timestamp}.csv"

                s3_path = self.s3_handler.upload_df_to_s3(df, output_key)
                self.logger.info(f"Saved cleaned data to {s3_path}")

            return df

        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            raise


def main():
    """Main function to demonstrate usage"""
    cleaner = DataCleaner()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Clean the data
        cleaned_df = cleaner.clean_data(
            input_key="raw/hourly_fuel_fpl.csv",
            output_key=f"processed/cleaned_electricity_data_{timestamp}.csv"
        )

        # Display sample of cleaned data
        print("\nFirst few rows of cleaned data:")
        print(cleaned_df.head())
        # Display basic statistics
        print("\nBasic statistics:")
        print(cleaned_df.describe())

    except Exception as e:
        logging.error(f"Failed to process data: {str(e)}")
        raise


if __name__ == "__main__":
    main()