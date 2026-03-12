import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="ffn-dw-bigquery-prd")

call_start_date = '2025-11-01'
call_end_date = '2026-03-06'
comparison_date_str = '2026-01-27'

query_ma = f"""
WITH call_data AS (
    SELECT 
        lendage_guid,
        DATE(sent_to_sales_date) as sts_date,
        mortgage_advisor,
        call_attempts,
        first_call_attempt_datetime,
        sf__contacted_guid,
        full_app_submit_datetime,
        initial_sales_assigned_datetime,
        CASE 
            WHEN DATE(sent_to_sales_date) < '{comparison_date_str}' THEN 'Pre'
            ELSE 'Post'
        END as period,
        TIMESTAMP_DIFF(first_call_attempt_datetime, initial_sales_assigned_datetime, MINUTE) as minutes_to_first_call,
        TIMESTAMP_DIFF(first_call_attempt_datetime, initial_sales_assigned_datetime, HOUR) as hours_to_first_call,
        adjusted_lead_value_cohort
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(sent_to_sales_date) BETWEEN '{call_start_date}' AND '{call_end_date}'
    AND sent_to_sales_date IS NOT NULL
    AND TIME(sent_to_sales_datetime) BETWEEN '07:30:00' AND '16:30:00'
    AND adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
)
SELECT 
    period,
    COUNT(DISTINCT lendage_guid) as total_leads,
    SUM(COALESCE(call_attempts, 0)) as total_call_attempts,
    COUNT(DISTINCT CASE WHEN first_call_attempt_datetime IS NOT NULL THEN lendage_guid END) as leads_with_call,
    COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted,
    COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
    AVG(CASE WHEN minutes_to_first_call > 0 AND minutes_to_first_call < 10080 THEN minutes_to_first_call END) as avg_minutes_to_first_call,
    COUNT(DISTINCT CASE WHEN minutes_to_first_call <= 5 THEN lendage_guid END) as called_within_5min,
    COUNT(DISTINCT CASE WHEN minutes_to_first_call <= 15 THEN lendage_guid END) as called_within_15min,
    COUNT(DISTINCT CASE WHEN minutes_to_first_call <= 60 THEN lendage_guid END) as called_within_1hr
FROM call_data
GROUP BY period
ORDER BY period DESC
"""

df_ma = client.query(query_ma).to_dataframe()
print("MA Call Metrics:")
print(df_ma.to_markdown())

query_overlap = f"""
WITH phx_data AS (
    SELECT 
        lendage_guid,
        DATE(sent_to_sales_date) as sts_date,
        first_call_attempt_datetime,
        first_dial_phx,
        phoenix_transfer_flag,
        CASE 
            WHEN DATE(sent_to_sales_date) < '{comparison_date_str}' THEN 'Pre'
            ELSE 'Post'
        END as period,
        CASE 
            WHEN first_call_attempt_datetime IS NOT NULL AND first_dial_phx IS NULL THEN 'MA Only'
            WHEN first_call_attempt_datetime IS NULL AND first_dial_phx IS NOT NULL THEN 'PHX Only'
            WHEN first_call_attempt_datetime IS NOT NULL AND first_dial_phx IS NOT NULL THEN 'Both MA & PHX'
            ELSE 'No Calls'
        END as call_type
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(sent_to_sales_date) BETWEEN '{call_start_date}' AND '{call_end_date}'
    AND sent_to_sales_date IS NOT NULL
    AND adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
)
SELECT 
    period,
    call_type,
    COUNT(DISTINCT lendage_guid) as lead_count
FROM phx_data
GROUP BY period, call_type
ORDER BY period DESC, call_type
"""

df_overlap = client.query(query_overlap).to_dataframe()
print("\nCall Type Distribution:")
print(df_overlap.to_markdown())

query_outreach = f"""
        WITH base AS (
            SELECT
                a.lendage_guid,
                DATE(a.sent_to_sales_date) AS sts_date,
                CASE WHEN DATE(a.sent_to_sales_date) BETWEEN '2025-12-19' AND '2026-01-26' THEN 'Pre'
                     WHEN DATE(a.sent_to_sales_date) BETWEEN '2026-01-27' AND '2026-03-06' THEN 'Post' END AS period,
                a.initial_lead_value_cohort,
                a.first_call_attempt_datetime,
                a.first_dial_phx,
                a.sf__contacted_guid IS NOT NULL AS had_contact,
                a.full_app_submit_datetime IS NOT NULL AS had_fas,
                COALESCE(a.e_loan_amount, 0) AS e_loan_amount
            FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table` a
            WHERE a.initial_lead_value_cohort IN ('1','2','3','4','5','6','7','8')
            AND DATE(a.sent_to_sales_date) BETWEEN '2025-12-19' AND '2026-03-06'
            AND a.sent_to_sales_date IS NOT NULL
        )
        SELECT 
            period,
            COUNT(DISTINCT lendage_guid) as lead_count,
            SUM(CASE WHEN had_contact THEN 1 ELSE 0 END) as contact_count,
            SUM(CASE WHEN had_fas THEN 1 ELSE 0 END) as fas_count,
            SUM(CASE WHEN had_fas THEN e_loan_amount ELSE 0 END) as fas_dollars
        FROM base
        WHERE period IS NOT NULL
        GROUP BY period
"""

df_outreach = client.query(query_outreach).to_dataframe()
print("\nOutreach / FAS Metrics:")
print(df_outreach.to_markdown())
