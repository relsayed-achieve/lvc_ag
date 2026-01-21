# GBQ + Streamlit BI Tool

This is a simple BI tool that connects to Google BigQuery and allows you to run SQL queries and visualize the results.

## Setup

1.  **Clone/Open the project**
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Authentication

### Option 1: Local / Browser Auth (Recommended for Local Dev)
Run the following command in your terminal and log in with your Google account:
```bash
gcloud auth application-default login
```
This sets up "Application Default Credentials" (ADC) which the app detects automatically.

### Option 2: Service Account Key
To connect using a Service Account Key (JSON):
1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  Navigate to **IAM & Admin** > **Service Accounts**.
3.  Create a Service Account (or use an existing one) with **BigQuery Data Viewer** and **BigQuery Job User** roles.
4.  Go to the **Keys** tab and create a new key (JSON type).
5.  Save the JSON file and upload it in the app sidebar.

## Running the App

```bash
streamlit run app.py
```
