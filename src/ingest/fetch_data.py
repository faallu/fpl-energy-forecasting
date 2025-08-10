import os
import requests
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.aws_utils import S3Handler
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("EIA_API_KEY")  # Make sure this is defined in your .env

BASE_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
OUTPUT_PATH = "data/raw/hourly_fuel_fpl.csv"

def fetch_eia_fuel_data(api_key, respondent="FPL", frequency="hourly", max_pages=10):
    """Fetch hourly fuel type data from EIA v2 API"""
    params = {
        "api_key": api_key,
        "frequency": frequency,
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "offset": 0,
        "length": 5000,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc"
    }

    all_rows = []
    page = 0

    while True:
        print(f"Fetching page {page + 1}")
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        page_data = response.json()["response"]["data"]
        all_rows.extend(page_data)

        if len(page_data) < params["length"] or page >= max_pages - 1:
            break
        params["offset"] += params["length"]
        page += 1

    df = pd.DataFrame(all_rows)
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

def save_to_s3(df, key):
    s3_handler = S3Handler()
    s3_path = s3_handler.upload_df_to_s3(df, key)
    print(f"Saved to {s3_path}")

if __name__ == "__main__":
    print("Fetching EIA fuel type data (FPL hourly)...")
    df = fetch_eia_fuel_data(API_KEY)
    save_to_s3(df, "raw/hourly_fuel_fpl.csv")