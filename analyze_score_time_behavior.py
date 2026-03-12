import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

query = """
WITH lead_timing AS (
    SELECT 
        lendage_guid,
        initial_lead_score_datetime,
        EXTRACT(HOUR FROM initial_lead_score_datetime) as score_hour,
        EXTRACT(DAYOFWEEK FROM initial_lead_score_datetime) as score_dow,
        first_call_attempt_datetime,
        first_dial_phx,
        sf__contacted_guid IS NOT NULL as had_contact,
        full_app_submit_datetime IS NOT NULL as had_fas,
        DATE(sent_to_sales_date) as sts_date
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(sent_to_sales_date) >= '2026-01-27' -- Post 24hr delay period
      AND adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
      AND initial_lead_score_datetime IS NOT NULL
)
SELECT 
    CASE 
        WHEN score_hour BETWEEN 17 AND 23 THEN '1. Evening (5pm-Midnight)'
        WHEN score_hour BETWEEN 0 AND 6 THEN '2. Night/Early Morning (Midnight-7am)'
        WHEN score_hour BETWEEN 7 AND 11 THEN '3. Morning (7am-Noon)'
        WHEN score_hour BETWEEN 12 AND 16 THEN '4. Afternoon (Noon-5pm)'
    END as score_time_block,
    COUNT(DISTINCT lendage_guid) as lead_count,
    
    -- Speed to first MA call (in hours, to see if they call within the 4 hour window)
    COUNT(DISTINCT CASE WHEN TIMESTAMP_DIFF(first_call_attempt_datetime, initial_lead_score_datetime, HOUR) <= 4 THEN lendage_guid END) as ma_called_within_4hr,
    COUNT(DISTINCT CASE WHEN TIMESTAMP_DIFF(first_call_attempt_datetime, initial_lead_score_datetime, HOUR) <= 24 THEN lendage_guid END) as ma_called_within_24hr,
    
    -- Contact & Conversion
    SUM(CASE WHEN had_contact THEN 1 ELSE 0 END) as contact_count,
    SUM(CASE WHEN had_fas THEN 1 ELSE 0 END) as fas_count
FROM lead_timing
GROUP BY 1
ORDER BY 1
"""

df = client.query(query).to_dataframe()

df['pct_of_leads'] = df['lead_count'] / df['lead_count'].sum()
df['ma_call_rate_4hr'] = df['ma_called_within_4hr'] / df['lead_count']
df['ma_call_rate_24hr'] = df['ma_called_within_24hr'] / df['lead_count']
df['contact_rate'] = df['contact_count'] / df['lead_count']
df['fas_rate'] = df['fas_count'] / df['lead_count']

print("MA Call Behavior by Score Time Block (Post 1/27):")
print(df[['score_time_block', 'lead_count', 'pct_of_leads', 'ma_call_rate_4hr', 'ma_call_rate_24hr', 'contact_rate', 'fas_rate']].to_markdown(index=False))
