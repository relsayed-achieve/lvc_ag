# AHL Weekly Performance Dashboard

This is a comprehensive Dash-based BI tool that connects to Google BigQuery and allows you to track and visualize LVC vintage performance, Channel Mix, and Mortgage Advisor performance metrics.

## Setup

1.  **Clone/Open the project**
2.  **Install dependencies**:
    ```bash
    pip install -r requirements_dash.txt
    ```

## Authentication

### Local / Browser Auth (Recommended for Local Dev)
Run the following command in your terminal and log in with your Google account:
```bash
gcloud auth application-default login
```
This sets up "Application Default Credentials" (ADC) which the app detects automatically to query BigQuery.

## Running the App

```bash
python lvc_day7_vintage_dashboard.py
```
After running, navigate to `http://127.0.0.1:8051/` in your web browser.
