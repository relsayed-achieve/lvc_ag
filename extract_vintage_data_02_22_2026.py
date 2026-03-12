"""
Extract Vintage Analysis Data for Week of 02/22/2026
This script queries BigQuery directly to get all the vintage analysis data
"""

import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
import json

# Initialize BigQuery client
project_id = "ffn-dw-bigquery-prd"
client = bigquery.Client(project=project_id)

# Target week
target_week = '2026-02-22'
target_date = pd.to_datetime(target_week)
target_end = target_date + timedelta(days=6)

print(f"Extracting Vintage Analysis Data for Week: {target_week}")
print(f"Week Range: {target_date.strftime('%Y-%m-%d')} to {target_end.strftime('%Y-%m-%d')}")
print("="*80)

# Query 1: Get Executive Summary (Vintage) Data
query_vintage = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
        COUNT(DISTINCT lendage_guid) as total_leads,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sts_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars,
        AVG(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount END) as avg_loan,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{target_week}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1
"""

print("\n1. Querying Executive Summary (Vintage) Data...")
df_vintage_summary = client.query(query_vintage).to_dataframe()

if not df_vintage_summary.empty:
    row = df_vintage_summary.iloc[0]
    print(f"\n   Total Leads: {row['total_leads']:,.0f}")
    print(f"   STS Volume: {row['sts_count']:,.0f}")
    print(f"   FAS Day 7 Qty: {row['fas_count']:,.0f}")
    print(f"   FAS Day 7 $: ${row['fas_dollars']:,.2f}")
    print(f"   Avg LP2C: {row['avg_lp2c']:.2f}%")
    print(f"   Avg Loan: ${row['avg_loan']:,.2f}")
else:
    print("   No data found!")

# Query 2: Get LVC Breakdown (Vintage)
query_lvc = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
        CASE
            WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
            WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
            WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
            WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
            ELSE 'Other'
        END as lvc_group,
        COUNT(DISTINCT lendage_guid) as total_leads,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sts_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{target_week}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2
    ORDER BY 2
"""

print("\n2. Querying LVC Breakdown (Vintage) Data...")
df_lvc = client.query(query_lvc).to_dataframe()

if not df_lvc.empty:
    print(f"\n   LVC Group Breakdown:")
    print("   " + "-"*100)
    print(f"   {'LVC Group':<15} {'Total Leads':>12} {'STS Count':>12} {'FAS Day 7 Qty':>15} {'FAS Day 7 $':>18} {'Avg LP2C':>10}")
    print("   " + "-"*100)
    
    for _, row in df_lvc.iterrows():
        print(f"   {row['lvc_group']:<15} {row['total_leads']:>12,.0f} {row['sts_count']:>12,.0f} {row['fas_count']:>15,.0f} ${row['fas_dollars']:>17,.2f} {row['avg_lp2c']:>9.2f}%")
    
    print("   " + "-"*100)
    print(f"   {'TOTAL':<15} {df_lvc['total_leads'].sum():>12,.0f} {df_lvc['sts_count'].sum():>12,.0f} {df_lvc['fas_count'].sum():>15,.0f} ${df_lvc['fas_dollars'].sum():>17,.2f}")
else:
    print("   No LVC data found!")

# Query 3: Get Channel Breakdown (Vintage)
query_channel = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
        COALESCE(sub_group, 'Unknown') as channel,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sts_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{target_week}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2
    ORDER BY fas_count DESC
"""

print("\n3. Querying Channel Breakdown (Vintage) Data...")
df_channel = client.query(query_channel).to_dataframe()

if not df_channel.empty:
    print(f"\n   Channel Breakdown (Top 20):")
    print("   " + "-"*100)
    print(f"   {'Channel':<30} {'STS Count':>12} {'FAS Day 7 Qty':>15} {'FAS Day 7 $':>18} {'Avg LP2C':>10}")
    print("   " + "-"*100)
    
    for _, row in df_channel.head(20).iterrows():
        print(f"   {row['channel']:<30} {row['sts_count']:>12,.0f} {row['fas_count']:>15,.0f} ${row['fas_dollars']:>17,.2f} {row['avg_lp2c']:>9.2f}%")
    
    print("   " + "-"*100)
    print(f"   {'TOTAL (All Channels)':<30} {df_channel['sts_count'].sum():>12,.0f} {df_channel['fas_count'].sum():>15,.0f} ${df_channel['fas_dollars'].sum():>17,.2f}")
else:
    print("   No channel data found!")

# Query 4: Get detailed LVC + Channel breakdown
query_lvc_channel = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
        CASE
            WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
            WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
            WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
            WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
            ELSE 'Other'
        END as lvc_group,
        COALESCE(sub_group, 'Unknown') as channel,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sts_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{target_week}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2, 3
    ORDER BY 2, fas_count DESC
"""

print("\n4. Querying LVC + Channel Breakdown (Vintage) Data...")
df_lvc_channel = client.query(query_lvc_channel).to_dataframe()

if not df_lvc_channel.empty:
    print(f"\n   LVC + Channel Breakdown (showing top channels per LVC):")
    
    for lvc in ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']:
        lvc_data = df_lvc_channel[df_lvc_channel['lvc_group'] == lvc]
        if not lvc_data.empty:
            print(f"\n   {lvc}:")
            print("   " + "-"*100)
            print(f"   {'Channel':<30} {'STS Count':>12} {'FAS Day 7 Qty':>15} {'FAS Day 7 $':>18} {'Avg LP2C':>10}")
            print("   " + "-"*100)
            
            for _, row in lvc_data.head(10).iterrows():
                print(f"   {row['channel']:<30} {row['sts_count']:>12,.0f} {row['fas_count']:>15,.0f} ${row['fas_dollars']:>17,.2f} {row['avg_lp2c']:>9.2f}%")
            
            print("   " + "-"*100)
            print(f"   {lvc + ' TOTAL':<30} {lvc_data['sts_count'].sum():>12,.0f} {lvc_data['fas_count'].sum():>15,.0f} ${lvc_data['fas_dollars'].sum():>17,.2f}")
else:
    print("   No LVC + Channel data found!")

# Save to CSV files for reference
print("\n" + "="*80)
print("Saving data to CSV files...")

df_vintage_summary.to_csv('vintage_summary_02_22_2026.csv', index=False)
print("   - vintage_summary_02_22_2026.csv")

df_lvc.to_csv('vintage_lvc_breakdown_02_22_2026.csv', index=False)
print("   - vintage_lvc_breakdown_02_22_2026.csv")

df_channel.to_csv('vintage_channel_breakdown_02_22_2026.csv', index=False)
print("   - vintage_channel_breakdown_02_22_2026.csv")

df_lvc_channel.to_csv('vintage_lvc_channel_breakdown_02_22_2026.csv', index=False)
print("   - vintage_lvc_channel_breakdown_02_22_2026.csv")

print("\nDone!")
