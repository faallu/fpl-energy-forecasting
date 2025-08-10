import logging
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.ingest.fetch_data import fetch_eia_fuel_data, API_KEY
from src.preprocess.clean import DataCleaner
from src.utils.aws_utils import S3Handler


class ETLPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.s3_handler = S3Handler()
        self.cleaner = DataCleaner()

    def run_pipeline(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # 1. Extract
            raw_df = fetch_eia_fuel_data(API_KEY)
            raw_key = f"raw/hourly_fuel_fpl_{timestamp}.csv"
            self.s3_handler.upload_df_to_s3(raw_df, raw_key)

            # 2. Transform
            cleaned_df = self.cleaner.clean_data(
                input_key=raw_key,
                output_key=f"processed/cleaned_data_{timestamp}.csv"
            )

            # 3. Load (prepare for modeling)
            model_ready_key = f"model_ready/fuel_data_{timestamp}.csv"
            self.s3_handler.upload_df_to_s3(cleaned_df, model_ready_key)

            return True

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    pipeline = ETLPipeline()
    pipeline.run_pipeline()