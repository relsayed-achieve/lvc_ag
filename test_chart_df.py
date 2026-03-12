import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

dz_sql = """
WITH phx_timing AS (
    SELECT 
        lendage_guid,
        CASE 
            WHEN initial_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
            WHEN initial_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
            ELSE 'Other'
        END as lvc_group,
        initial_lead_value_cohort,
        EXTRACT(DAYOFWEEK FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") as dow_num,
        FORMAT_DATETIME('%A', DATETIME(initial_lead_score_datetime, "America/Los_Angeles")) as weekday,
        TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, MINUTE) / 60.0 as hours_to_phx_dial,
        DATE(lead_created_date) as lead_created_date,
        initial_lead_score_datetime as score_datetime,
        first_call_attempt_datetime as ma_first_dial,
        first_dial_phx as phx_first_dial,
        CASE WHEN sf__contacted_guid IS NOT NULL THEN 'Yes' ELSE 'No' END as ma_contacted
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(sent_to_sales_date) BETWEEN '2026-01-27' AND '2026-03-06'
      AND initial_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
      AND first_dial_phx IS NOT NULL
      AND initial_lead_score_datetime IS NOT NULL
)
SELECT * FROM phx_timing
"""

df_dz = client.query(dz_sql).to_dataframe()

df_dz_weekday_chart = df_dz[~df_dz['dow_num'].isin([1, 7])].copy()
df_dz_weekday_chart['bucket'] = pd.cut(df_dz_weekday_chart['hours_to_phx_dial'], 
                                        bins=[-1, 24, 28, 32, 36, 999999],
                                        labels=['< 24 hrs', '24 - 28 hrs', '28 - 32 hrs', '32 - 36 hrs', '36+ hrs'])

df_dz_weekday_chart['score_week'] = pd.to_datetime(df_dz_weekday_chart['score_datetime']).dt.to_period('W-SUN').dt.start_time
df_dz_weekday_chart['score_week_label'] = df_dz_weekday_chart['score_week'].dt.strftime('%m/%d')

weekly_dz = df_dz_weekday_chart.groupby(['score_week_label', 'bucket'], observed=False).agg(
    leads=('lendage_guid', 'count')
).reset_index()

weekly_totals = weekly_dz.groupby('score_week_label')['leads'].sum().reset_index().rename(columns={'leads': 'total_leads'})
weekly_dz = weekly_dz.merge(weekly_totals, on='score_week_label')
weekly_dz['pct'] = weekly_dz['leads'] / weekly_dz['total_leads']

print(weekly_dz.head(20).to_string())
