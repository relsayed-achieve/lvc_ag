import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

query = """
WITH phx_timing AS (
    SELECT 
        lendage_guid,
        initial_lead_score_datetime,
        first_dial_phx,
        DATE(sent_to_sales_date) as sts_date,
        CASE WHEN DATE(sent_to_sales_date) BETWEEN '2025-12-19' AND '2026-01-26' THEN 'Pre (12/19-1/26)'
             WHEN DATE(sent_to_sales_date) BETWEEN '2026-01-27' AND '2026-03-06' THEN 'Post (1/27-3/6)' END AS period,
        TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, MINUTE) as minutes_to_phx_dial,
        TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, HOUR) as hours_to_phx_dial
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(sent_to_sales_date) BETWEEN '2025-12-19' AND '2026-03-06'
      AND adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
      AND first_dial_phx IS NOT NULL
      AND initial_lead_score_datetime IS NOT NULL
)
SELECT 
    period,
    COUNT(lendage_guid) as leads_dialed_by_phx,
    AVG(hours_to_phx_dial) as avg_hours_to_phx_dial,
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

df = client.query(query).to_dataframe()

print("PHX Time to First Dial (from Score Time):")
print(df.to_markdown(index=False))
