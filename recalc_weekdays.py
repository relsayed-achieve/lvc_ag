import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

print("--- 1. Lead Distribution by Score Time Block (Weekday Only) ---")
query1 = """
WITH lead_timing AS (
    SELECT 
        lendage_guid,
        initial_lead_score_datetime,
        EXTRACT(HOUR FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") as score_hour,
        first_call_attempt_datetime,
        sf__contacted_guid IS NOT NULL as had_contact,
        full_app_submit_datetime IS NOT NULL as had_fas
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(sent_to_sales_date) >= '2026-01-27' 
      AND adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
      AND initial_lead_score_datetime IS NOT NULL
      -- Exclude weekends (1=Sunday, 7=Saturday)
      AND EXTRACT(DAYOFWEEK FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") NOT IN (1, 7)
)
SELECT 
    CASE 
        WHEN score_hour BETWEEN 7 AND 11 THEN 'Morning (7am-Noon)'
        WHEN score_hour BETWEEN 12 AND 16 THEN 'Afternoon (Noon-5pm)'
        WHEN score_hour BETWEEN 17 AND 23 THEN 'Evening (5pm-Midnight)'
        WHEN score_hour BETWEEN 0 AND 6 THEN 'Night (Midnight-7am)'
    END as Time_Scored,
    COUNT(DISTINCT lendage_guid) as lead_count,
    COUNT(DISTINCT CASE WHEN TIMESTAMP_DIFF(first_call_attempt_datetime, initial_lead_score_datetime, HOUR) <= 4 THEN lendage_guid END) as ma_called_within_4hr,
    SUM(CASE WHEN had_contact THEN 1 ELSE 0 END) as contact_count
FROM lead_timing
GROUP BY 1
ORDER BY 
    CASE Time_Scored 
        WHEN 'Morning (7am-Noon)' THEN 1
        WHEN 'Afternoon (Noon-5pm)' THEN 2
        WHEN 'Evening (5pm-Midnight)' THEN 3
        WHEN 'Night (Midnight-7am)' THEN 4
    END
"""
df1 = client.query(query1).to_dataframe()
df1['pct_of_leads'] = df1['lead_count'] / df1['lead_count'].sum()
df1['ma_call_rate_4hr'] = df1['ma_called_within_4hr'] / df1['lead_count']
df1['contact_rate'] = df1['contact_count'] / df1['lead_count']
print(df1[['Time_Scored', 'pct_of_leads', 'ma_call_rate_4hr', 'contact_rate']].to_markdown(index=False))


print("\n--- 2. PHX Time to First Dial (Weekday Only) ---")
query2 = """
WITH phx_timing AS (
    SELECT 
        lendage_guid,
        CASE WHEN DATE(sent_to_sales_date) BETWEEN '2025-12-19' AND '2026-01-26' THEN 'Pre (12/19-1/26)'
             WHEN DATE(sent_to_sales_date) BETWEEN '2026-01-27' AND '2026-03-06' THEN 'Post (1/27-3/6)' END AS period,
        TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, MINUTE) / 60.0 as hours_to_phx_dial
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(sent_to_sales_date) BETWEEN '2025-12-19' AND '2026-03-06'
      AND adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
      AND first_dial_phx IS NOT NULL
      AND initial_lead_score_datetime IS NOT NULL
      -- Exclude weekends
      AND EXTRACT(DAYOFWEEK FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") NOT IN (1, 7)
)
SELECT 
    period,
    COUNT(lendage_guid) as leads_dialed_by_phx,
    APPROX_QUANTILES(hours_to_phx_dial, 100)[OFFSET(50)] as median_hours_to_phx_dial,
    COUNT(CASE WHEN hours_to_phx_dial < 4 THEN 1 END) / COUNT(lendage_guid) as pct_phx_dial_under_4h,
    COUNT(CASE WHEN hours_to_phx_dial < 24 THEN 1 END) / COUNT(lendage_guid) as pct_phx_dial_under_24h,
    COUNT(CASE WHEN hours_to_phx_dial BETWEEN 24 AND 48 THEN 1 END) / COUNT(lendage_guid) as pct_phx_dial_24_to_48h,
    COUNT(CASE WHEN hours_to_phx_dial > 48 THEN 1 END) / COUNT(lendage_guid) as pct_phx_dial_over_48h
FROM phx_timing
WHERE period IS NOT NULL
GROUP BY period
ORDER BY period DESC
"""
df2 = client.query(query2).to_dataframe()
print(df2.to_markdown(index=False))


print("\n--- 3. Examples of Evening/Night Leads Not Dialed by PHX Until 24+ Hours (Weekday Only) ---")
query3 = """
SELECT 
    lendage_guid,
    initial_lead_score_datetime,
    first_call_attempt_datetime as ma_first_call,
    CASE WHEN sf__contacted_guid IS NOT NULL THEN 'Yes' ELSE 'No' END as ma_contacted,
    first_dial_phx,
    TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, MINUTE) / 60.0 as exact_hours_to_phx_dial
FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
WHERE DATE(sent_to_sales_date) >= '2026-01-27'
  AND adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
  AND first_dial_phx IS NOT NULL
  AND initial_lead_score_datetime IS NOT NULL
  -- Exclude weekends (Score date)
  AND EXTRACT(DAYOFWEEK FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") NOT IN (1, 7)
  -- We want evening/night leads (e.g. Mon-Thu evening so it doesn't just hit the weekend)
  AND EXTRACT(DAYOFWEEK FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") IN (2, 3, 4, 5) -- Mon, Tue, Wed, Thu
  AND (EXTRACT(HOUR FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") >= 17 
       OR EXTRACT(HOUR FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") <= 7)
  -- Delayed > 30 hours
  AND TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, HOUR) > 30
ORDER BY initial_lead_score_datetime DESC
LIMIT 4
"""
df3 = client.query(query3).to_dataframe()
print(df3.to_markdown(index=False))
