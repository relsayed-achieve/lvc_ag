"""
Extract all dashboard data for the week of 2/22/2026
This script queries all the same data that the dashboard displays
"""

import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery
import json

def get_bq_client():
    """Get BigQuery client using default credentials"""
    try:
        project_id = "ffn-dw-bigquery-prd"
        client = bigquery.Client(project=project_id)
        return client
    except Exception as e:
        print(f"BigQuery connection error: {e}")
        return None

def run_query(query):
    """Execute BigQuery query and return DataFrame"""
    client = get_bq_client()
    if client:
        try:
            return client.query(query).to_dataframe()
        except Exception as e:
            print(f"Query error: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def get_executive_summary_data(target_week):
    """Get executive summary metrics for target week vs prev 12 weeks - IN-PERIOD based"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=12)
    prev_end = target_date - timedelta(days=1)
    
    query = f"""
    WITH fas_data AS (
        SELECT
            DATE_TRUNC(DATE(full_app_submit_datetime), WEEK(SUNDAY)) as fas_week,
            COUNT(DISTINCT lendage_guid) as fas_count,
            SUM(e_loan_amount) as fas_dollars,
            AVG(e_loan_amount) as avg_loan,
            AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE full_app_submit_datetime IS NOT NULL
        AND DATE(full_app_submit_datetime) >= '{prev_start.strftime('%Y-%m-%d')}'
        AND DATE(full_app_submit_datetime) <= '{target_end.strftime('%Y-%m-%d')}'
        GROUP BY 1
    ),
    sts_data AS (
        SELECT
            DATE_TRUNC(DATE(sent_to_sales_date), WEEK(SUNDAY)) as sts_week,
            COUNT(DISTINCT lendage_guid) as sts_count
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE sent_to_sales_date IS NOT NULL
        AND DATE(sent_to_sales_date) >= '{prev_start.strftime('%Y-%m-%d')}'
        AND DATE(sent_to_sales_date) <= '{target_end.strftime('%Y-%m-%d')}'
        GROUP BY 1
    )
    SELECT 
        COALESCE(f.fas_week, s.sts_week) as week_start,
        COALESCE(f.fas_count, 0) as fas_count,
        COALESCE(f.fas_dollars, 0) as fas_dollars,
        COALESCE(f.avg_loan, 0) as avg_loan,
        COALESCE(s.sts_count, 0) as sts_count,
        COALESCE(f.avg_lp2c, 0) as avg_lp2c
    FROM fas_data f
    FULL OUTER JOIN sts_data s ON f.fas_week = s.sts_week
    WHERE COALESCE(f.fas_week, s.sts_week) IS NOT NULL
    ORDER BY 1 DESC
    """
    return run_query(query)

def get_vintage_summary_data(target_week):
    """Get vintage-based summary metrics for target week vs prev 4 weeks - VINTAGE based"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    prev_end = target_date - timedelta(days=1)
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
        COUNT(DISTINCT lendage_guid) as total_leads,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sts_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars,
        AVG(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount END) as avg_loan,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1
    ORDER BY 1 DESC
    """
    return run_query(query)

def get_lvc_breakdown_inperiod(target_week):
    """Get LVC group breakdown for target week vs prev 4 weeks - IN-PERIOD based"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    prev_end = target_date - timedelta(days=1)
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(full_app_submit_datetime), WEEK(SUNDAY)) as fas_week,
        CASE
            WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
            WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
            WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
            WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
            ELSE 'Other'
        END as lvc_group,
        COUNT(DISTINCT lendage_guid) as fas_count,
        SUM(e_loan_amount) as fas_dollars,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE full_app_submit_datetime IS NOT NULL
    AND DATE(full_app_submit_datetime) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(full_app_submit_datetime) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2
    ORDER BY 1 DESC, 3 DESC
    """
    return run_query(query)

def get_channel_breakdown(target_week):
    """Get channel breakdown for target week vs prev 4 weeks"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    prev_end = target_date - timedelta(days=1)
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(full_app_submit_datetime), WEEK(SUNDAY)) as fas_week,
        COALESCE(sub_group, 'Unknown') as channel,
        COUNT(DISTINCT lendage_guid) as fas_count,
        SUM(e_loan_amount) as fas_dollars,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE full_app_submit_datetime IS NOT NULL
    AND DATE(full_app_submit_datetime) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(full_app_submit_datetime) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2
    ORDER BY 1 DESC, 3 DESC
    """
    return run_query(query)

def get_ma_performance(target_week):
    """Get MA rep performance for target week vs prev 4 weeks"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    prev_end = target_date - timedelta(days=1)
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(full_app_submit_datetime), WEEK(SUNDAY)) as fas_week,
        COALESCE(mortgage_advisor, 'Unassigned') as ma_rep,
        COUNT(DISTINCT lendage_guid) as fas_count,
        SUM(e_loan_amount) as fas_dollars,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE full_app_submit_datetime IS NOT NULL
    AND DATE(full_app_submit_datetime) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(full_app_submit_datetime) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2
    ORDER BY 1 DESC, 3 DESC
    """
    return run_query(query)

def get_daily_comparison(target_week):
    """Get daily vintage data for target week and same days in prev 4 weeks"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    
    query = f"""
    SELECT
        DATE(lead_created_date) as lead_date,
        FORMAT_DATE('%A', DATE(lead_created_date)) as day_name,
        EXTRACT(DAYOFWEEK FROM DATE(lead_created_date)) as day_of_week,
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as week_start,
        COUNT(DISTINCT lendage_guid) as total_leads,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sts_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2, 3, 4
    ORDER BY 1 DESC
    """
    return run_query(query)

def get_persona_breakdown(target_week):
    """Get persona breakdown for target week vs prev 4 weeks"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    prev_end = target_date - timedelta(days=1)
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(full_app_submit_datetime), WEEK(SUNDAY)) as fas_week,
        COALESCE(persona, 'Unknown') as persona,
        COUNT(DISTINCT lendage_guid) as fas_count,
        SUM(e_loan_amount) as fas_dollars,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE full_app_submit_datetime IS NOT NULL
    AND DATE(full_app_submit_datetime) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(full_app_submit_datetime) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2
    ORDER BY 1 DESC, 3 DESC
    """
    return run_query(query)

def main():
    target_week = '2026-02-22'  # Week of February 22, 2026
    
    print("=" * 80)
    print(f"EXTRACTING DATA FOR WEEK OF {target_week}")
    print("=" * 80)
    
    # 1. Executive Summary (In-Period)
    print("\n\n1. EXECUTIVE SUMMARY - IN-PERIOD (Based on FAS Submit Date)")
    print("-" * 80)
    exec_summary = get_executive_summary_data(target_week)
    if not exec_summary.empty:
        target_week_data = exec_summary[exec_summary['week_start'] == target_week]
        if not target_week_data.empty:
            row = target_week_data.iloc[0]
            print(f"\nTarget Week: {target_week}")
            print(f"  FAS Count: {row['fas_count']:,.0f}")
            print(f"  FAS Dollars: ${row['fas_dollars']:,.0f}")
            print(f"  Avg Loan: ${row['avg_loan']:,.0f}")
            print(f"  STS Count: {row['sts_count']:,.0f}")
            print(f"  Avg LP2C: {row['avg_lp2c']:.2f}%")
        
        print("\n\nPrevious 12 Weeks Comparison:")
        print(exec_summary.to_string(index=False))
    else:
        print("No data available")
    
    # 2. Vintage Summary
    print("\n\n2. VINTAGE SUMMARY (Based on Lead Created Date)")
    print("-" * 80)
    vintage_summary = get_vintage_summary_data(target_week)
    if not vintage_summary.empty:
        target_week_data = vintage_summary[vintage_summary['vintage_week'] == target_week]
        if not target_week_data.empty:
            row = target_week_data.iloc[0]
            print(f"\nTarget Week Vintage: {target_week}")
            print(f"  Total Leads: {row['total_leads']:,.0f}")
            print(f"  STS Count: {row['sts_count']:,.0f}")
            print(f"  FAS Count: {row['fas_count']:,.0f}")
            print(f"  FAS Dollars: ${row['fas_dollars']:,.0f}")
            print(f"  Avg Loan: ${row['avg_loan']:,.0f}")
            print(f"  Avg LP2C: {row['avg_lp2c']:.2f}%")
        
        print("\n\nPrevious 4 Weeks Comparison:")
        print(vintage_summary.to_string(index=False))
    else:
        print("No data available")
    
    # 3. LVC Breakdown (In-Period)
    print("\n\n3. LVC GROUP BREAKDOWN - IN-PERIOD")
    print("-" * 80)
    lvc_breakdown = get_lvc_breakdown_inperiod(target_week)
    if not lvc_breakdown.empty:
        target_week_lvc = lvc_breakdown[lvc_breakdown['fas_week'] == target_week]
        if not target_week_lvc.empty:
            print(f"\nTarget Week LVC Breakdown: {target_week}")
            for _, row in target_week_lvc.iterrows():
                print(f"\n  {row['lvc_group']}:")
                print(f"    FAS Count: {row['fas_count']:,.0f}")
                print(f"    FAS Dollars: ${row['fas_dollars']:,.0f}")
                print(f"    Avg LP2C: {row['avg_lp2c']:.2f}%")
        
        print("\n\nFull LVC Breakdown (All Weeks):")
        print(lvc_breakdown.to_string(index=False))
    else:
        print("No data available")
    
    # 4. Channel Breakdown
    print("\n\n4. CHANNEL BREAKDOWN")
    print("-" * 80)
    channel_breakdown = get_channel_breakdown(target_week)
    if not channel_breakdown.empty:
        target_week_channel = channel_breakdown[channel_breakdown['fas_week'] == target_week]
        if not target_week_channel.empty:
            print(f"\nTarget Week Channel Breakdown: {target_week}")
            for _, row in target_week_channel.iterrows():
                print(f"\n  {row['channel']}:")
                print(f"    FAS Count: {row['fas_count']:,.0f}")
                print(f"    FAS Dollars: ${row['fas_dollars']:,.0f}")
                print(f"    Avg LP2C: {row['avg_lp2c']:.2f}%")
        
        print("\n\nFull Channel Breakdown (All Weeks):")
        print(channel_breakdown.to_string(index=False))
    else:
        print("No data available")
    
    # 5. MA Performance
    print("\n\n5. MA PERFORMANCE")
    print("-" * 80)
    ma_performance = get_ma_performance(target_week)
    if not ma_performance.empty:
        target_week_ma = ma_performance[ma_performance['fas_week'] == target_week]
        if not target_week_ma.empty:
            print(f"\nTarget Week MA Performance: {target_week}")
            for _, row in target_week_ma.iterrows():
                print(f"\n  {row['ma_rep']}:")
                print(f"    FAS Count: {row['fas_count']:,.0f}")
                print(f"    FAS Dollars: ${row['fas_dollars']:,.0f}")
                print(f"    Avg LP2C: {row['avg_lp2c']:.2f}%")
        
        print("\n\nFull MA Performance (All Weeks):")
        print(ma_performance.to_string(index=False))
    else:
        print("No data available")
    
    # 6. Daily Comparison
    print("\n\n6. DAILY COMPARISON (Vintage-Based)")
    print("-" * 80)
    daily_comparison = get_daily_comparison(target_week)
    if not daily_comparison.empty:
        target_date = pd.to_datetime(target_week)
        target_end = target_date + timedelta(days=6)
        target_week_daily = daily_comparison[
            (daily_comparison['lead_date'] >= target_date) &
            (daily_comparison['lead_date'] <= target_end)
        ]
        if not target_week_daily.empty:
            print(f"\nTarget Week Daily Breakdown: {target_week}")
            for _, row in target_week_daily.iterrows():
                print(f"\n  {row['lead_date']} - {row['day_name']} (Day {row['day_of_week']}):")
                print(f"    Total Leads: {row['total_leads']:,.0f}")
                print(f"    STS Count: {row['sts_count']:,.0f}")
                print(f"    FAS Count: {row['fas_count']:,.0f}")
                print(f"    FAS Dollars: ${row['fas_dollars']:,.0f}")
        
        print("\n\nFull Daily Comparison (5 Weeks):")
        print(daily_comparison.to_string(index=False))
    else:
        print("No data available")
    
    # 7. Persona Breakdown
    print("\n\n7. PERSONA BREAKDOWN")
    print("-" * 80)
    persona_breakdown = get_persona_breakdown(target_week)
    if not persona_breakdown.empty:
        target_week_persona = persona_breakdown[persona_breakdown['fas_week'] == target_week]
        if not target_week_persona.empty:
            print(f"\nTarget Week Persona Breakdown: {target_week}")
            for _, row in target_week_persona.iterrows():
                print(f"\n  {row['persona']}:")
                print(f"    FAS Count: {row['fas_count']:,.0f}")
                print(f"    FAS Dollars: ${row['fas_dollars']:,.0f}")
                print(f"    Avg LP2C: {row['avg_lp2c']:.2f}%")
        
        print("\n\nFull Persona Breakdown (All Weeks):")
        print(persona_breakdown.to_string(index=False))
    else:
        print("No data available")
    
    print("\n\n" + "=" * 80)
    print("DATA EXTRACTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
