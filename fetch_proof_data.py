import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

query = """
SELECT 
    lendage_guid,
    initial_lead_score_datetime,
    first_dial_phx,
    first_call_attempt_datetime as ma_first_call,
    CASE WHEN sf__contacted_guid IS NOT NULL THEN 'Yes' ELSE 'No' END as ma_contacted
FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
WHERE lendage_guid IN (
    'd1b33012-41f2-4c70-b775-16561c93819c',
    'f311f0eb-00b7-467f-bf15-0fe098502358',
    '64a78abe-d5e9-4e13-9833-0803cbd8fcea',
    '09a714c9-c26c-4d65-8a04-c761bd89b4bf'
)
"""

df = client.query(query).to_dataframe()
print(df.to_markdown(index=False))
