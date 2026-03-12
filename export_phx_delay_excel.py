import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

# --- Query 1: The Bucket Counts by Weekday and LVC ---
query_buckets = """
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

df_buckets = client.query(query_buckets).to_dataframe()
df_buckets = df_buckets.drop(columns=['dow_num'])

# Create the percentage version
df_pct = df_buckets.copy()
for col in ['under_24h', 'hours_24_to_28', 'hours_28_to_32', 'hours_32_to_36', 'over_36h']:
    df_pct[col + '_pct'] = (df_pct[col] / df_pct['total_leads_dialed_by_phx']).round(4)

df_pct = df_pct[['weekday', 'lvc_group', 'total_leads_dialed_by_phx', 'under_24h_pct', 'hours_24_to_28_pct', 'hours_28_to_32_pct', 'hours_32_to_36_pct', 'over_36h_pct']]


# --- Query 2: Detail for leads taking 32+ hours ---
query_details = """
SELECT 
    lendage_guid,
    DATE(lead_created_date) as lead_created_date,
    initial_lead_score_datetime as score_datetime,
    FORMAT_DATETIME('%A', DATETIME(initial_lead_score_datetime, "America/Los_Angeles")) as score_weekday,
    first_call_attempt_datetime as ma_first_dial,
    first_dial_phx as phx_first_dial,
    initial_lead_value_cohort,
    CASE WHEN sf__contacted_guid IS NOT NULL THEN 'Yes' ELSE 'No' END as ma_contacted,
    TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, MINUTE) / 60.0 as hours_to_phx_dial
FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
WHERE DATE(sent_to_sales_date) BETWEEN '2026-01-27' AND '2026-03-06'
  AND initial_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
  AND first_dial_phx IS NOT NULL
  AND initial_lead_score_datetime IS NOT NULL
  AND TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, HOUR) >= 32
ORDER BY hours_to_phx_dial DESC
"""

df_details = client.query(query_details).to_dataframe()

# Remove timezone info so Excel can save it properly
for col in ['score_datetime', 'ma_first_dial', 'phx_first_dial']:
    if pd.api.types.is_datetime64_any_dtype(df_details[col]):
        df_details[col] = df_details[col].dt.tz_localize(None)


# Write to Excel using openpyxl instead of xlsxwriter since it might be missing
with pd.ExcelWriter('/Users/relsayed/Documents/cursor/lvc_ag/PHX_Dial_Delay_Analysis.xlsx', engine='openpyxl') as writer:
    df_buckets.to_excel(writer, sheet_name='Counts by Weekday', index=False)
    df_pct.to_excel(writer, sheet_name='Percentages by Weekday', index=False)
    df_details.to_excel(writer, sheet_name='32+ Hours Detail', index=False)

print("Excel file created: PHX_Dial_Delay_Analysis.xlsx")
