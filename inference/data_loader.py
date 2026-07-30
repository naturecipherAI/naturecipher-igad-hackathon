"""
Data loader — reads processed history and model artifacts from private S3 bucket.

Required environment variables (set in .env, never committed):
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION        (default: us-east-1)
    S3_BUCKET         (default: naturecipher-forecast)

Request credentials from: kelvin@naturecipherai.com
"""
import io
import os
import tempfile
import logging

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

REGIONS = ['asal_north', 'asal_northeast', 'asal_eastern']


def _s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
    )


def _bucket():
    return os.getenv('S3_BUCKET', 'naturecipher-forecast')


def load_historical_parquet(region: str) -> pd.DataFrame:
    """
    Load processed historical time series for a region directly from S3.
    Concatenates all parquet splits (period1_train, period2_train, etc.)
    from the processed_v3/{region}/ prefix. Never writes to local disk.

    Args:
        region: One of asal_north, asal_northeast, asal_eastern

    Returns:
        DataFrame with ERA5/CHIRPS/NDVI/LST features, monthly rows
    """
    if region not in REGIONS:
        raise ValueError(f"Unknown region: {region}. Must be one of {REGIONS}")

    prefix = f'processed_v3/{region}/'
    logger.info(f"Loading parquets from s3://{_bucket()}/{prefix}...")

    s3 = _s3_client()
    resp = s3.list_objects_v2(Bucket=_bucket(), Prefix=prefix)
    parquet_keys = [
        obj['Key'] for obj in resp.get('Contents', [])
        if obj['Key'].endswith('.parquet')
    ]

    if not parquet_keys:
        raise FileNotFoundError(f"No parquet files found at s3://{_bucket()}/{prefix}")

    frames = []
    for key in sorted(parquet_keys):
        response = s3.get_object(Bucket=_bucket(), Key=key)
        frames.append(pd.read_parquet(io.BytesIO(response['Body'].read())))
        logger.info(f"  Loaded {key}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=['year', 'month']).sort_values(['year', 'month']).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} rows for {region}")
    return df


def load_drought_model(region: str) -> str:
    """
    Download XGBoost block_3 drought classifier from S3 to a temp file.

    Args:
        region: One of asal_north, asal_northeast, asal_eastern

    Returns:
        Path to temp JSON model file (caller responsible for cleanup)
    """
    key = f'models/{region}/block_3_model.json'
    logger.info(f"Downloading drought model {key} from S3...")

    s3 = _s3_client()
    tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False, prefix=f'{region}_')
    s3.download_fileobj(_bucket(), key, tmp)
    tmp.close()

    logger.info(f"Model saved to {tmp.name}")
    return tmp.name


def load_bridge_model(region: str, bridge_name: str) -> str:
    """
    Download a cascade bridge model from S3 to a temp file.

    Args:
        region: One of asal_north, asal_northeast, asal_eastern
        bridge_name: e.g. bridge1_chirps, bridge2_ndvi, bridge3a_lst_day, bridge3b_lst_night

    Returns:
        Path to temp JSON model file
    """
    key = f'models/cascade_bridges/{region}/{bridge_name}.json'
    logger.info(f"Downloading bridge model {key} from S3...")

    s3 = _s3_client()
    tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False, prefix=f'{region}_{bridge_name}_')
    s3.download_fileobj(_bucket(), key, tmp)
    tmp.close()

    return tmp.name


def load_bridge_metadata(region: str) -> dict:
    """Load bridge feature metadata from S3."""
    import json

    key = f'models/cascade_bridges/{region}/bridge_metadata.json'
    s3 = _s3_client()
    response = s3.get_object(Bucket=_bucket(), Key=key)
    return json.loads(response['Body'].read())


def load_validation_reference() -> dict:
    """Load the Jan-Mar 2026 retrospective validation output from S3."""
    import json

    key = 'validation/forecast_2026.json'
    s3 = _s3_client()
    response = s3.get_object(Bucket=_bucket(), Key=key)
    return json.loads(response['Body'].read())


def check_credentials() -> bool:
    """Verify S3 credentials are configured and bucket is accessible."""
    required = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    missing = [k for k in required if not os.getenv(k)]

    if missing:
        print(f"\n❌ Missing credentials: {missing}")
        print("Request credentials from: kelvin@naturecipherai.com")
        print("Then: cp credentials.env .env\n")
        return False

    try:
        s3 = _s3_client()
        s3.head_bucket(Bucket=_bucket())
        print(f"✅ S3 connection verified: s3://{_bucket()}")
        return True
    except Exception as e:
        print(f"❌ S3 connection failed: {e}")
        return False


if __name__ == "__main__":
    check_credentials()
