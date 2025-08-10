import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class ElectricityDataProcessor:
    """
    Feature engineering and data preprocessing for electricity demand forecasting
    """

    def __init__(self, processed_data_dir: str = "data/processed"):
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def remove_initial_nan_values(self, df: pd.DataFrame, target_col: str = 'value') -> pd.DataFrame:
        """
        Remove NaN values at the beginning of the dataset

        Args:
            df: DataFrame with potential NaN values at start
            target_col: Target column to check for NaN values

        Returns:
            DataFrame with initial NaN values removed
        """
        df = df.copy()
        df = df.sort_values('period')

        # Find the first non-NaN value in the target column
        if target_col in df.columns:
            first_valid_idx = df[target_col].first_valid_index()
            if first_valid_idx is not None:
                # Keep only rows from the first valid value onwards
                df = df.loc[first_valid_idx:].reset_index(drop=True)
                logger.info(f"Removed initial NaN values. Dataset now starts from index {first_valid_idx}")
            else:
                logger.warning(f"No valid values found in {target_col} column")

        return df

    def create_time_features(self, df: pd.DataFrame, period: str = 'period') -> pd.DataFrame:
        """
        Create time-based features from datetime column

        Args:
            df: DataFrame with datetime column
            period: Name of datetime column

        Returns:
            DataFrame with additional time features
        """
        df = df.copy()

        # Ensure datetime column is datetime type
        df[period] = pd.to_datetime(df[period])

        # Extract time components
        df['hour'] = df[period].dt.hour
        df['day_of_week'] = df[period].dt.dayofweek
        df['day_of_month'] = df[period].dt.day
        df['month'] = df[period].dt.month
        df['quarter'] = df[period].dt.quarter
        df['year'] = df[period].dt.year

        # Cyclical features (to capture cyclical nature of time)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 17)).astype(int)
        df['is_peak_hour'] = ((df['hour'] >= 16) & (df['hour'] <= 20)).astype(int)

        # Season features
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })

        logger.info(f"Created time features: {df.shape[1] - df.shape[1] + 15} new features")
        return df

    def create_lag_features(self, df: pd.DataFrame,
                            value_col: str = 'value',
                            lags: List[int] = [1, 2, 3, 6, 12, 24, 48, 168]) -> pd.DataFrame:
        """
        Create lag features for time series forecasting

        Args:
            df: DataFrame sorted by time
            value_col: Column to create lags for
            lags: List of lag periods (in hours)

        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        df = df.sort_values('period')

        for lag in lags:
            df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)

        logger.info(f"Created {len(lags)} lag features")
        return df

    def create_rolling_features(self, df: pd.DataFrame,
                                value_col: str = 'value',
                                windows: List[int] = [3, 6, 12, 24, 48, 168]) -> pd.DataFrame:
        """
        Create rolling window features

        Args:
            df: DataFrame sorted by time
            value_col: Column to create rolling features for
            windows: List of window sizes (in hours)

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        df = df.sort_values('period')

        for window in windows:
            df[f'{value_col}_rolling_mean_{window}'] = df[value_col].rolling(window).mean()
            df[f'{value_col}_rolling_std_{window}'] = df[value_col].rolling(window).std()
            df[f'{value_col}_rolling_min_{window}'] = df[value_col].rolling(window).min()
            df[f'{value_col}_rolling_max_{window}'] = df[value_col].rolling(window).max()

        logger.info(f"Created rolling features for {len(windows)} windows")
        return df

    def create_fuel_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to fuel types

        Args:
            df: DataFrame with fuel type data

        Returns:
            DataFrame with fuel type features
        """

        df = df.copy()
        df_nonan = df.dropna(subset=['value'])

        if not df_nonan.empty:
            dominant_fuel = df_nonan.loc[df_nonan.groupby('period')['value'].idxmax()]
            dominant_fuel = dominant_fuel[['period', 'type-name']].rename(columns={'type-name': 'dominant_fuel'})
            df = df.merge(dominant_fuel, on='period', how='left')
        else:
            logger.warning("No valid rows found to compute dominant fuel type.")

        # One-hot encode fuel types
        fuel_dummies = pd.get_dummies(df['type-name'], prefix='fuel')
        df = pd.concat([df, fuel_dummies], axis=1)

        # Create fuel mix features (if multiple fuel types per period)
        if df.groupby('period').size().max() > 1:
            # Calculate total generation per period
            total_gen = df.groupby('period')['value'].sum().reset_index()
            total_gen.columns = ['period', 'total_generation']
            df = df.merge(total_gen, on='period', how='left')

            # Calculate fuel type percentage
            df['fuel_percentage'] = df['value'] / df['total_generation'] * 100

            # Create dominant fuel type feature
            dominant_fuel = df.loc[df.groupby('period')['value'].idxmax()]
            dominant_fuel = dominant_fuel[['period', 'type-name']].rename(columns={'type-name': 'dominant_fuel'})
            df = df.merge(dominant_fuel, on='period', how='left')

        logger.info("Created fuel type features")
        return df

    def create_statistical_features(self, df: pd.DataFrame,
                                    value_col: str = 'value') -> pd.DataFrame:
        """
        Create statistical features

        Args:
            df: DataFrame with value column
            value_col: Column to create features for

        Returns:
            DataFrame with statistical features
        """
        df = df.copy()

        # Z-score (standardized value)
        df[f'{value_col}_zscore'] = (df[value_col] - df[value_col].mean()) / df[value_col].std()

        # Percentile rank
        df[f'{value_col}_percentile'] = df[value_col].rank(pct=True)

        # Difference features
        df[f'{value_col}_diff_1'] = df[value_col].diff(1)
        df[f'{value_col}_diff_24'] = df[value_col].diff(24)  # Day-over-day
        df[f'{value_col}_diff_168'] = df[value_col].diff(168)  # Week-over-week

        # Rate of change
        df[f'{value_col}_pct_change_1'] = df[value_col].pct_change(1)
        df[f'{value_col}_pct_change_24'] = df[value_col].pct_change(24)

        logger.info("Created statistical features")
        return df

    def detect_outliers(self, df: pd.DataFrame,
                        value_col: str = 'value',
                        method: str = 'iqr') -> pd.DataFrame:
        """
        Detect and flag outliers

        Args:
            df: DataFrame with value column
            value_col: Column to check for outliers
            method: Method for outlier detection ('iqr', 'zscore')

        Returns:
            DataFrame with outlier flags
        """
        df = df.copy()

        if method == 'iqr':
            Q1 = df[value_col].quantile(0.25)
            Q3 = df[value_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df['is_outlier'] = (df[value_col] < lower_bound) | (df[value_col] > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs((df[value_col] - df[value_col].mean()) / df[value_col].std())
            df['is_outlier'] = z_scores > 3

        outlier_count = df['is_outlier'].sum()
        logger.info(f"Detected {outlier_count} outliers using {method} method")

        return df

    def handle_missing_values(self, df: pd.DataFrame,
                              strategy: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            df: DataFrame with missing values
            strategy: Strategy for handling missing values

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        missing_before = df.isnull().sum().sum()

        if strategy == 'interpolate':
            # Time-based interpolation for time series
            df = df.sort_values('period')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='time')

        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill')

        elif strategy == 'backward_fill':
            df = df.fillna(method='bfill')

        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        missing_after = df.isnull().sum().sum()
        logger.info(f"Handled missing values: {missing_before} -> {missing_after}")

        return df

    def scale_features(self, df: pd.DataFrame,
                       feature_cols: List[str] = None,
                       method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features

        Args:
            df: DataFrame with features to scale
            feature_cols: List of columns to scale (if None, scale all numeric)
            method: Scaling method ('standard', 'minmax')

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude certain columns from scaling
            exclude_cols = ['period', 'year', 'month', 'day_of_month', 'hour', 'day_of_week']
            feature_cols = [col for col in feature_cols if col not in exclude_cols]

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()

        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        logger.info(f"Scaled {len(feature_cols)} features using {method} scaling")

        return df

    def create_all_features(self, df: pd.DataFrame,
                            target_col: str = 'value',
                            scale_features: bool = True) -> pd.DataFrame:
        """
        Create all features in one pipeline

        Args:
            df: Raw DataFrame
            target_col: Target column name
            scale_features: Whether to scale features

        Returns:
            DataFrame with all features
        """
        logger.info("Starting feature engineering pipeline")

        # Remove initial NaN values first
        df = self.remove_initial_nan_values(df, target_col)

        # Create time features
        df = self.create_time_features(df)

        # Create lag features
        df = self.create_lag_features(df, target_col)

        # Create rolling features
        df = self.create_rolling_features(df, target_col)
        print("Hi, this is create fuel type features")
        # Create fuel type features
        if 'type-name' in df.columns:
            df = self.create_fuel_type_features(df)

        # Create statistical features
        df = self.create_statistical_features(df, target_col)

        # Detect outliers
        df = self.detect_outliers(df, target_col)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Scale features
        if scale_features:
            df = self.scale_features(df)

        # Store feature names
        self.feature_names = df.columns.tolist()

        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df

    def prepare_sequences(self, df: pd.DataFrame,
                          target_col: str = 'value',
                          sequence_length: int = 24,
                          forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM/RNN models

        Args:
            df: DataFrame with features
            target_col: Target column name
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast

        Returns:
            Tuple of (X, y) arrays for training
        """
        df = df.sort_values('period')

        # Remove any remaining NaN values before sequence creation
        df = df.dropna()

        # Select feature columns (exclude non-numeric and target)
        feature_cols = [col for col in df.columns if col not in ['period', target_col]]
        feature_cols = [col for col in feature_cols if df[col].dtype in [np.float64, np.int64]]

        X, y = [], []

        for i in range(len(df) - sequence_length - forecast_horizon + 1):
            # Input sequence
            seq_x = df[feature_cols].iloc[i:i + sequence_length].values

            # Target sequence
            seq_y = df[target_col].iloc[i + sequence_length:i + sequence_length + forecast_horizon].values

            X.append(seq_x)
            y.append(seq_y)

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Prepared sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y

    def save_processed_data(self, df: pd.DataFrame,
                            filename: str = None,
                            save_scaler: bool = True,
                            save_metadata: bool = True) -> str:
        """
        Save processed data to data/processed directory

        Args:
            df: Processed DataFrame to save
            filename: Custom filename (if None, uses timestamp)
            save_scaler: Whether to save the fitted scaler
            save_metadata: Whether to save processing metadata

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_electricity_data_{timestamp}.csv"

        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'

        filepath = self.processed_data_dir / filename

        # Save the DataFrame
        df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")

        # Save the scaler if it exists
        if save_scaler and self.scaler is not None:
            scaler_path = self.processed_data_dir / f"scaler_{filename.replace('.csv', '.pkl')}"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Scaler saved to {scaler_path}")

        # Save metadata
        if save_metadata:
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'feature_names': self.feature_names,
                'data_types': df.dtypes.to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'scaler_type': type(self.scaler).__name__ if self.scaler else None
            }

            metadata_path = self.processed_data_dir / f"metadata_{filename.replace('.csv', '.json')}"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to {metadata_path}")

        return str(filepath)

    def load_processed_data(self, filename: str,
                            load_scaler: bool = True) -> pd.DataFrame:
        """
        Load processed data from data/processed directory

        Args:
            filename: Name of the file to load
            load_scaler: Whether to load the associated scaler

        Returns:
            Loaded DataFrame
        """
        filepath = self.processed_data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")

        df = pd.read_csv(filepath)

        # Convert period back to datetime if it exists
        if 'period' in df.columns:
            df['period'] = pd.to_datetime(df['period'])

        # Load scaler if requested
        if load_scaler:
            scaler_path = self.processed_data_dir / f"scaler_{filename.replace('.csv', '.pkl')}"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Scaler loaded from {scaler_path}")

        logger.info(f"Processed data loaded from {filepath}")
        return df

    def save_sequences(self, X: np.ndarray, y: np.ndarray,
                       filename_prefix: str = None) -> Dict[str, str]:
        """
        Save prepared sequences for model training

        Args:
            X: Input sequences
            y: Target sequences
            filename_prefix: Prefix for filenames

        Returns:
            Dictionary with paths to saved files
        """
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"sequences_{timestamp}"

        X_path = self.processed_data_dir / f"{filename_prefix}_X.npy"
        y_path = self.processed_data_dir / f"{filename_prefix}_y.npy"

        np.save(X_path, X)
        np.save(y_path, y)

        # Save sequence metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'X_shape': X.shape,
            'y_shape': y.shape,
            'X_dtype': str(X.dtype),
            'y_dtype': str(y.dtype)
        }

        metadata_path = self.processed_data_dir / f"{filename_prefix}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Sequences saved: X to {X_path}, y to {y_path}")

        return {
            'X_path': str(X_path),
            'y_path': str(y_path),
            'metadata_path': str(metadata_path)
        }

    def load_sequences(self, filename_prefix: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load prepared sequences

        Args:
            filename_prefix: Prefix used when saving sequences

        Returns:
            Tuple of (X, y) arrays
        """
        X_path = self.processed_data_dir / f"{filename_prefix}_X.npy"
        y_path = self.processed_data_dir / f"{filename_prefix}_y.npy"

        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Sequence files not found with prefix {filename_prefix}")

        X = np.load(X_path)
        y = np.load(y_path)

        logger.info(f"Sequences loaded: X shape {X.shape}, y shape {y.shape}")
        return X, y

    def list_processed_files(self) -> Dict[str, List[str]]:
        """
        List all processed files in the data/processed directory

        Returns:
            Dictionary categorizing different file types
        """
        files = {
            'data_files': [],
            'scaler_files': [],
            'metadata_files': [],
            'sequence_files': []
        }

        for file in self.processed_data_dir.glob('*'):
            if file.suffix == '.csv':
                files['data_files'].append(file.name)
            elif file.suffix == '.pkl':
                files['scaler_files'].append(file.name)
            elif file.suffix == '.json':
                files['metadata_files'].append(file.name)
            elif file.suffix == '.npy':
                files['sequence_files'].append(file.name)

        return files

    def create_and_save_all_features(self, df: pd.DataFrame,
                                     target_col: str = 'value',
                                     filename: str = None,
                                     scale_features: bool = True) -> str:
        """
        Create all features and save to processed directory

        Args:
            df: Raw DataFrame
            target_col: Target column name
            filename: Custom filename for saving
            scale_features: Whether to scale features

        Returns:
            Path to saved processed file
        """
        # Create all features
        processed_df = self.create_all_features(df, target_col, scale_features)

        # Save processed data
        saved_path = self.save_processed_data(
            processed_df,
            filename=filename,
            save_scaler=True,
            save_metadata=True
        )

        return saved_path

    def main(self):
        """
        Main function to demonstrate ElectricityDataProcessor functionality
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Initialize the processor
        processor = ElectricityDataProcessor(processed_data_dir="data/processed")

        try:
            # Load raw data (adjust path as needed)
            raw_data = pd.read_csv("data/raw/electricity_demand.csv")

            # Convert period to datetime if needed
            if 'period' in raw_data.columns:
                raw_data['period'] = pd.to_datetime(raw_data['period'])

            # Process the data and save features
            saved_path = processor.create_and_save_all_features(
                df=raw_data,
                target_col='value',
                filename='electricity_features.csv',
                scale_features=True
            )
            logger.info(f"Complete pipeline executed. Processed data saved to {saved_path}")

            # Optional: Create and save sequences for time series models
            processed_data = processor.load_processed_data('electricity_features.csv')
            X, y = processor.prepare_sequences(
                df=processed_data,
                target_col='value',
                sequence_length=24,
                forecast_horizon=1
            )

            sequence_paths = processor.save_sequences(X, y, filename_prefix='electricity_sequences')
            logger.info(f"Sequences saved to: {sequence_paths}")

        except Exception as e:
            logger.error(f"Error in processing pipeline: {str(e)}")
            raise

    if __name__ == "__main__":
        main()
