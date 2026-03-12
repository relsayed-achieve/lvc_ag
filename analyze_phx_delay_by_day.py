import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

query = """
WITH phx_timing AS (
    SELECT 
        lendage_guid,
        CASE 
            WHEN initial_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
            WHEN initial_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
            ELSE 'Other'
        END as lvc_group,
        EXTRACT(DAYOFWEEK FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") as dow_num,
        FORMAT_DATETIME('%A', DATETIME(initial_lead_score_datetime, "America/Los_Angeles")) as weekday,
        TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, MINUTE) / 60.0 as hours_to_phx_dial
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(sent_to_sales_date) BETWEEN '2026-01-27' AND '2026-03-06'
      AND initial_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
      AND first_dial_phx IS NOT NULL
      AND initial_lead_score_datetime IS NOT NULL
)
SELECT 
    weekday,
    lvc_group,
    COUNT(lendage_guid) as total_leads_dialed_by_phx,
    COUNTIF(hours_to_phx_dial < 24) as under_24h,
    COUNTIF(hours_to_phx_dial >= 24 AND hours_to_phx_dial < 28) as hours_24_to_28,
    COUNTIF(hours_to_phx_dial >= 28 AND hours_to_phx_dial < 32) as hours_28_to_32,
    COUNTIF(hours_to_phx_dial >= 32 AND hours_to_phx_dial < 36) as hours_32_to_36,
    COUNTIF(hours_to_phx_dial >= 36) as over_36h,
    dow_num
FROM phx_timing
GROUP BY weekday, lvc_group, dow_num
ORDER BY 
    CASE dow_num
        WHEN 2 THEN 1 -- Monday
        WHEN 3 THEN 2 -- Tuesday
        WHEN 4 THEN 3 -- Wednesday
        WHEN 5 THEN 4 -- Thursday
        WHEN 6 THEN 5 -- Friday
        WHEN 7 THEN 6 -- Saturday
        WHEN 1 THEN 7 -- Sunday
    END,
    lvc_group
"""

df = client.query(query).to_dataframe()

# Drop the sorting column for display
df_display = df.drop(columns=['dow_num'])

print("PHX Time to First Dial (Post 1/27 - 3/6) by Weekday and LVC Group:")
print(df_display.to_markdown(index=False))

# Let's also do a percentage version to make it easier to read
for col in ['under_24h', 'hours_24_to_28', 'hours_28_to_32', 'hours_32_to_36', 'over_36h']:
    df_display[col + '_pct'] = (df_display[col] / df_display['total_leads_dialed_by_phx'] * 100).map('{:.1f}%'.format)

print("\nAs Percentages:")
pct_cols = ['weekday', 'lvc_group', 'total_leads_dialed_by_phx', 'under_24h_pct', 'hours_24_to_28_pct', 'hours_28_to_32_pct', 'hours_32_to_36_pct', 'over_36h_pct']
print(df_display[pct_cols].to_markdown(index=False))
