"""
One-off script to list columns for the outreach tab join.
Run: python fetch_outreach_schema.py
Requires: gcloud auth application-default login (or GOOGLE_APPLICATION_CREDENTIALS)
"""
from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

# Table A: vintages
q_a = """
SELECT column_name, data_type
FROM `ffn-dw-bigquery-prd.Ramzi.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'lendage_lead_vintages_table'
ORDER BY ordinal_position
"""
print("=== ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table ===\n")
for row in client.query(q_a).result():
    print(f"  {row.column_name}\t{row.data_type}")
print()

# Table B: comms (cross-project)
q_b = """
SELECT column_name, data_type
FROM `lendage-data-platform.standardized_data.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'lendage_consumer_comms_lead_data'
ORDER BY ordinal_position
"""
print("=== lendage-data-platform.standardized_data.lendage_consumer_comms_lead_data ===\n")
try:
    for row in client.query(q_b).result():
        print(f"  {row.column_name}\t{row.data_type}")
except Exception as e:
    print(f"  Error (check access to lendage-data-platform): {e}")
