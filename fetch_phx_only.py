from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

query = """
SELECT 
    lendage_guid, 
    DATE(sent_to_sales_date) as sts_date,
    first_dial_phx,
    first_call_attempt_datetime,
    initial_lead_value_cohort
FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
WHERE DATE(sent_to_sales_date) >= '2026-02-23'
  AND first_call_attempt_datetime IS NULL 
  AND first_dial_phx IS NOT NULL
ORDER BY sent_to_sales_date DESC
"""

df = client.query(query).to_dataframe()

if df.empty:
    print("No leads found matching the criteria.")
else:
    print(f"Total leads found: {len(df)}")
    df.to_csv('phx_only_leads_last_2_weeks.csv', index=False)
    print("Saved all results to 'phx_only_leads_last_2_weeks.csv'.")
    print("\nHere is a sample of the first 50 leads:")
    print(df[['lendage_guid', 'sts_date', 'initial_lead_value_cohort', 'first_dial_phx']].head(50).to_markdown(index=False))
