import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

# 1. Overall Pre vs Post, Weekdays Only
query_overall = """
WITH base AS (
    SELECT 
        lendage_guid,
        CASE WHEN DATE(sent_to_sales_date) BETWEEN '2025-12-19' AND '2026-01-26' THEN 'Pre (12/19-1/26)'
             WHEN DATE(sent_to_sales_date) BETWEEN '2026-01-27' AND '2026-03-06' THEN 'Post (1/27-3/6)' END AS period,
        sf__contacted_guid IS NOT NULL as had_contact,
        full_app_submit_datetime IS NOT NULL as had_fas,
        e_loan_amount,
        phoenix_fronter_contact_datetime
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
      AND DATE(sent_to_sales_date) BETWEEN '2025-12-19' AND '2026-03-06'
      AND EXTRACT(DAYOFWEEK FROM sent_to_sales_date) NOT IN (1, 7)
)
SELECT 
    period,
    COUNT(DISTINCT lendage_guid) as lead_count,
    SUM(CASE WHEN had_contact THEN 1 ELSE 0 END) / COUNT(DISTINCT lendage_guid) as contact_rate,
    SUM(CASE WHEN had_fas THEN 1 ELSE 0 END) / COUNT(DISTINCT lendage_guid) as fas_rate,
    SUM(CASE WHEN had_fas THEN e_loan_amount ELSE 0 END) as fas_dollars,
    SUM(CASE WHEN phoenix_fronter_contact_datetime IS NOT NULL THEN 1 ELSE 0 END) / COUNT(DISTINCT lendage_guid) as phx_contact_rate
FROM base
WHERE period IS NOT NULL
GROUP BY period
ORDER BY period DESC
"""

# 2. PHX Performance by Channel in Pre-Period (where they had no 24hr delay)
query_channel = """
WITH phx_channels AS (
    SELECT 
        COALESCE(lead_source, 'Unknown') as lead_channel,
        COUNT(DISTINCT lendage_guid) as lead_count,
        SUM(CASE WHEN first_dial_phx IS NOT NULL THEN 1 ELSE 0 END) as phx_dialed_count,
        SUM(CASE WHEN phoenix_fronter_contact_datetime IS NOT NULL THEN 1 ELSE 0 END) as phx_contacted_count,
        SUM(CASE WHEN CAST(phoenix_transfer_flag AS STRING) IN ('1', '1.0', 'True', 'true', 'Yes', 'yes') THEN 1 ELSE 0 END) as phx_transfer_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN 1 ELSE 0 END) as fas_count
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(sent_to_sales_date) BETWEEN '2025-12-19' AND '2026-01-26' -- PRE period
      AND adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
      AND EXTRACT(DAYOFWEEK FROM sent_to_sales_date) NOT IN (1, 7)
    GROUP BY lead_channel
)
SELECT 
    lead_channel,
    lead_count,
    phx_dialed_count,
    phx_contacted_count,
    phx_transfer_count,
    phx_contacted_count / NULLIF(phx_dialed_count, 0) as phx_contact_rate_when_dialed,
    phx_transfer_count / NULLIF(phx_contacted_count, 0) as phx_transfer_rate_when_contacted,
    fas_count / NULLIF(lead_count, 0) as overall_fas_rate
FROM phx_channels
ORDER BY lead_count DESC
"""

print("--- 1. Weekday-Only Overall Pre vs Post ---")
df_overall = client.query(query_overall).to_dataframe()
print(df_overall.to_markdown(index=False))

print("\n--- 2. Channel Performance for PHX (Pre Period, Weekday-Only) ---")
df_channel = client.query(query_channel).to_dataframe()
print(df_channel.to_markdown(index=False))
