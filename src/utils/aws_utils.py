import boto3
import pandas as pd
from io import StringIO
import os
from dotenv import load_dotenv

class S3Handler:
    def __init__(self):
        load_dotenv()
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket = os.getenv('S3_BUCKET_NAME')

    def upload_df_to_s3(self, df: pd.DataFrame, key: str) -> str:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=csv_buffer.getvalue()
        )
        return f"s3://{self.bucket}/{key}"

    def read_df_from_s3(self, key: str) -> pd.DataFrame:
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return pd.read_csv(obj['Body'])