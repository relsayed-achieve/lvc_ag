import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

query = """
SELECT 
    EXTRACT(HOUR FROM initial_lead_score_datetime) as score_hour,
    COUNT(DISTINCT lendage_guid) as lead_count,
    SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN 1 ELSE 0 END) as fas_count,
    SUM(CASE WHEN sf__contacted_guid IS NOT NULL THEN 1 ELSE 0 END) as contact_count
FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
WHERE DATE(sent_to_sales_date) >= '2025-12-19'
  AND adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
  AND initial_lead_score_datetime IS NOT NULL
GROUP BY score_hour
ORDER BY score_hour
"""

df = client.query(query).to_dataframe()

df['fas_rate'] = df['fas_count'] / df['lead_count']
df['contact_rate'] = df['contact_count'] / df['lead_count']

print("Lead Distribution by Score Hour (LVC 1-8, since 12/19):")
print(df.to_markdown(index=False))

# Grouping by business vs non-business hours roughly
# Assuming standard MT/PT business hours are roughly 7 AM - 5 PM (adjust as needed based on actual timezone of the data)
df['time_block'] = pd.cut(df['score_hour'], 
                          bins=[-1, 6, 17, 24], 
                          labels=['Night/Early Morning (0-6)', 'Business Hours (7-17)', 'Evening/Night (18-23)'])

block_summary = df.groupby('time_block').agg({
    'lead_count': 'sum',
    'fas_count': 'sum',
    'contact_count': 'sum'
}).reset_index()

block_summary['pct_of_total_leads'] = block_summary['lead_count'] / block_summary['lead_count'].sum()
block_summary['fas_rate'] = block_summary['fas_count'] / block_summary['lead_count']
block_summary['contact_rate'] = block_summary['contact_count'] / block_summary['lead_count']

print("\nLead Distribution by Time Block:")
print(block_summary.to_markdown(index=False))
