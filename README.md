# FPL Energy Forecasting

## Overview
This project forecasts energy demand and generation for Florida Power & Light (FPL) using machine learning and time series models. It includes data ingestion, preprocessing, modeling, and deployment components.

## Directory Structure

- `api/` — API-related code
- `data/`
  - `raw/` — Raw data files
  - `processed/` — Cleaned and processed datasets
- `deployment/` — Deployment scripts and configs
- `docs/` — Documentation
- `mlruns/` — MLflow experiment tracking
- `models/` — Model definitions and artifacts
- `notebooks/` — Jupyter notebooks for exploration
- `outputs/` — Generated outputs and results
- `scripts/` — Utility and pipeline scripts
- `src/`
  - `dashboard/` — Dashboard app for visualization
  - `ingest/` — Data fetching scripts
  - `models/` — Model implementations
  - `notebooks/` — Source notebooks
  - `pipeline/` — ETL pipeline code
  - `predict/` — Prediction scripts
  - `preprocess/` — Data cleaning and feature engineering
  - `utils/` — Utility functions
- `tests/` — Unit and integration tests

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/faallu/fpl-energy-forecasting.git
   cd fpl-energy-forecasting
