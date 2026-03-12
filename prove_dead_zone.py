import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

query = """
SELECT 
    lendage_guid,
    initial_lead_score_datetime,
    EXTRACT(HOUR FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") as score_hour_pt,
    first_dial_phx,
    TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, HOUR) as hours_to_phx_dial,
    TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, MINUTE) / 60.0 as exact_hours_to_phx_dial
FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
WHERE DATE(sent_to_sales_date) >= '2026-01-27'
  AND adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
  AND first_dial_phx IS NOT NULL
  AND initial_lead_score_datetime IS NOT NULL
  -- Filtering for evening/night (using UTC or adjusting to PT? The exact hour doesn't matter as much as the result that they take 24+ hrs, but let's filter for 5PM to 7AM PT roughly)
  AND (EXTRACT(HOUR FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") >= 17 
       OR EXTRACT(HOUR FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") <= 7)
ORDER BY exact_hours_to_phx_dial DESC
LIMIT 20
"""

df = client.query(query).to_dataframe()

print("Examples of Evening/Night Leads (Post 1/27) and PHX Time-to-Dial:")
print(df.to_markdown(index=False))

# Let's also get the distribution of hours to dial for JUST evening/night leads
dist_query = """
WITH night_leads AS (
    SELECT 
        lendage_guid,
        TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, MINUTE) / 60.0 as exact_hours_to_phx_dial
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(sent_to_sales_date) >= '2026-01-27'
      AND adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
      AND first_dial_phx IS NOT NULL
      AND initial_lead_score_datetime IS NOT NULL
      AND (EXTRACT(HOUR FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") >= 17 
           OR EXTRACT(HOUR FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") <= 7)
)
SELECT 
    COUNT(lendage_guid) as total_night_leads_dialed_by_phx,
    COUNTIF(exact_hours_to_phx_dial < 24) as dialed_under_24h,
    COUNTIF(exact_hours_to_phx_dial BETWEEN 24 AND 36) as dialed_24_to_36h,
    COUNTIF(exact_hours_to_phx_dial > 36) as dialed_over_36h,
    AVG(exact_hours_to_phx_dial) as avg_hours_to_dial
FROM night_leads
"""

dist_df = client.query(dist_query).to_dataframe()
print("\nDistribution for Evening/Night Leads (Post 1/27):")
print(dist_df.to_markdown(index=False))
