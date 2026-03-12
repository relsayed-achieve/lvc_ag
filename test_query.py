from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()
query = """
SELECT sub_group, CreditRange, avg_secured_competitor_apr, avg_achieve_loan_apr 
FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
WHERE DATE(lead_created_date) >= '2026-02-01'
AND sub_group = 'LT PL'
LIMIT 5
"""
try:
    df = client.query(query).to_dataframe()
    print(df)
except Exception as e:
    print(e)
