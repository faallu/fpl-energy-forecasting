import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.aws_utils import S3Handler


def load_data():
    """Load the latest processed data from S3"""
    s3 = S3Handler()
    try:
        # Load the most recent cleaned data file
        df = s3.read_df_from_s3("model_ready/fuel_data_20250729_232047.csv")
        df['period'] = pd.to_datetime(df['period'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def main():
    st.title("FPL Electricity Generation Dashboard")

    df = load_data()
    if df is None:
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['period'].min().date(), df['period'].max().date()),
        min_value=df['period'].min().date(),
        max_value=df['period'].max().date()
    )

    # Filter data based on date range
    mask = (df['period'].dt.date >= date_range[0]) & (df['period'].dt.date <= date_range[1])
    filtered_df = df[mask]

    # Main charts
    st.header("Generation Mix Over Time")

    fig_mix = px.area(
        filtered_df,
        x='period',
        y='value',
        color='type-name',
        title="Generation Mix by Fuel Type",
        labels={'value': 'Generation (MW)', 'type-name': 'Fuel Type', 'period': 'Time'}
    )
    st.plotly_chart(fig_mix)

    # Daily patterns
    st.header("Daily Generation Patterns")

    # Calculate daily averages by fuel type and hour
    hourly_avg = filtered_df.groupby(['hour', 'type-name'])['value'].mean().reset_index()

    fig_daily = px.line(
        hourly_avg,
        x='hour',
        y='value',
        color='type-name',
        title="Average Hourly Generation by Fuel Type",
        labels={'hour': 'Hour of Day', 'value': 'Average Generation (MW)', 'type-name': 'Fuel Type'}
    )
    st.plotly_chart(fig_daily)

if __name__ == "__main__":
    main()