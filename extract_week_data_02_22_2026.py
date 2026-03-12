"""
Extract dashboard data for week of 02/22/2026
This script calls the same data functions used by the dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
import json

# === BigQuery Setup ===
def get_bigquery_client():
    """Initialize BigQuery client using default credentials"""
    try:
        project_id = "ffn-dw-bigquery-prd"
        client = bigquery.Client(project=project_id)
        return client
    except Exception as e:
        print(f"Error initializing BigQuery client: {e}")
        return None

def run_query(query):
    """Execute BigQuery query and return DataFrame"""
    client = get_bigquery_client()
    if client is None:
        return pd.DataFrame()
    
    try:
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        print(f"Query error: {e}")
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
    ORDER BY 1
    """
    return run_query(query)

def get_breakdown_data(target_week):
    """Get breakdown by LVC, Persona, MA, Channel for target week vs prev 4 weeks"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    
    query = f"""
    WITH fas_data AS (
        SELECT
            DATE_TRUNC(DATE(full_app_submit_datetime), WEEK(SUNDAY)) as fas_week,
            CASE
                WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
                WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
                WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
                WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
                ELSE 'Other'
            END as lvc_group,
            persona,
            mortgage_advisor,
            COALESCE(sub_group, 'Unknown') as channel,
            lendage_guid,
            e_loan_amount
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE full_app_submit_datetime IS NOT NULL
        AND DATE(full_app_submit_datetime) >= '{prev_start.strftime('%Y-%m-%d')}'
        AND DATE(full_app_submit_datetime) <= '{target_end.strftime('%Y-%m-%d')}'
    )
    SELECT
        fas_week,
        lvc_group,
        persona,
        mortgage_advisor,
        channel,
        COUNT(DISTINCT lendage_guid) as fas_count,
        SUM(e_loan_amount) as fas_dollars
    FROM fas_data
    GROUP BY 1, 2, 3, 4, 5
    """
    return run_query(query)

def calculate_z_score(current_val, prev_mean, prev_std):
    """Calculate z-score for statistical significance"""
    if prev_std == 0 or pd.isna(prev_std):
        return 0
    return (current_val - prev_mean) / prev_std

def format_change(change, z_score):
    """Format change with statistical significance indicator"""
    if abs(z_score) >= 2:
        sig = " ***" if abs(z_score) >= 3 else " **"
    else:
        sig = ""
    return f"{change:+.1f}%{sig}"

def main():
    target_week = "2026-02-22"
    print(f"\n{'='*80}")
    print(f"DASHBOARD DATA EXTRACTION FOR WEEK OF {target_week}")
    print(f"{'='*80}\n")
    
    # Get executive summary data
    print("Fetching executive summary data...")
    df_exec = get_executive_summary_data(target_week)
    
    if df_exec.empty:
        print("ERROR: No data returned from query")
        return
    
    # Convert week_start to datetime
    df_exec['week_start'] = pd.to_datetime(df_exec['week_start'])
    
    # Separate target week and previous weeks
    target_date = pd.to_datetime(target_week)
    df_target = df_exec[df_exec['week_start'] == target_date]
    df_prev = df_exec[df_exec['week_start'] < target_date].tail(4)  # Last 4 weeks
    
    if df_target.empty:
        print(f"ERROR: No data found for target week {target_week}")
        print("\nAvailable weeks in data:")
        print(df_exec['week_start'].dt.strftime('%Y-%m-%d').tolist())
        return
    
    target_row = df_target.iloc[0]
    
    # Calculate overall metrics
    target_fas_count = target_row['fas_count']
    target_fas_dollars = target_row['fas_dollars']
    target_avg_loan = target_row['avg_loan']
    target_sts_count = target_row['sts_count']
    target_lp2c = target_row['avg_lp2c']
    
    # Calculate FAS rate (FAS/STS)
    target_fas_rate = (target_fas_count / target_sts_count * 100) if target_sts_count > 0 else 0
    
    # Previous 4 weeks stats
    prev_fas_rates = (df_prev['fas_count'] / df_prev['sts_count'] * 100).values
    prev_fas_rates = prev_fas_rates[~np.isnan(prev_fas_rates)]
    prev_mean = np.mean(prev_fas_rates) if len(prev_fas_rates) > 0 else 0
    prev_std = np.std(prev_fas_rates) if len(prev_fas_rates) > 0 else 0
    fas_rate_change = target_fas_rate - prev_mean
    fas_rate_z = calculate_z_score(target_fas_rate, prev_mean, prev_std)
    
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE METRICS")
    print("="*80)
    print(f"FAS Rate (FAS/STS): {target_fas_rate:.1f}% (Change: {format_change(fas_rate_change, fas_rate_z)}, Z-score: {fas_rate_z:.2f})")
    print(f"Total FAS $:        ${target_fas_dollars:,.0f}")
    print(f"Total FAS Count:    {target_fas_count:,.0f}")
    print(f"Total STS Count:    {target_sts_count:,.0f}")
    print(f"Avg Loan Size:      ${target_avg_loan:,.0f}")
    print(f"Avg LP2C Score:     {target_lp2c:.1f}%")
    
    # Get breakdown data
    print("\nFetching breakdown data (Channel, LVC, MA)...")
    df_breakdown = get_breakdown_data(target_week)
    
    if df_breakdown.empty:
        print("ERROR: No breakdown data returned")
        return
    
    df_breakdown['fas_week'] = pd.to_datetime(df_breakdown['fas_week'])
    df_bd_target = df_breakdown[df_breakdown['fas_week'] == target_date]
    df_bd_prev = df_breakdown[df_breakdown['fas_week'] < target_date]
    
    # Channel Analysis
    print("\n" + "="*80)
    print("CHANNEL ANALYSIS")
    print("="*80)
    
    channel_target = df_bd_target.groupby('channel').agg({
        'fas_count': 'sum',
        'fas_dollars': 'sum'
    }).reset_index()
    
    channel_prev = df_bd_prev.groupby('channel').agg({
        'fas_count': 'sum',
        'fas_dollars': 'sum'
    }).reset_index()
    
    # Average over 4 weeks
    channel_prev['fas_count'] = channel_prev['fas_count'] / 4
    channel_prev['fas_dollars'] = channel_prev['fas_dollars'] / 4
    
    # Sort by FAS count
    channel_target = channel_target.sort_values('fas_count', ascending=False)
    
    for _, ch_row in channel_target.head(10).iterrows():
        channel = ch_row['channel']
        target_count = ch_row['fas_count']
        target_dollars = ch_row['fas_dollars']
        
        prev_row = channel_prev[channel_prev['channel'] == channel]
        if not prev_row.empty:
            prev_count = prev_row.iloc[0]['fas_count']
            change = ((target_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
            print(f"{channel:20s}: {target_count:4.0f} FAS (${target_dollars/1000:6.0f}K) | Change: {change:+6.1f}%")
        else:
            print(f"{channel:20s}: {target_count:4.0f} FAS (${target_dollars/1000:6.0f}K) | New")
    
    # LVC Cohort Analysis
    print("\n" + "="*80)
    print("LVC COHORT ANALYSIS")
    print("="*80)
    
    lvc_target = df_bd_target.groupby('lvc_group').agg({
        'fas_count': 'sum',
        'fas_dollars': 'sum'
    }).reset_index()
    
    lvc_prev = df_bd_prev.groupby('lvc_group').agg({
        'fas_count': 'sum',
        'fas_dollars': 'sum'
    }).reset_index()
    
    # Average over 4 weeks
    lvc_prev['fas_count'] = lvc_prev['fas_count'] / 4
    lvc_prev['fas_dollars'] = lvc_prev['fas_dollars'] / 4
    
    lvc_order = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']
    for lvc in lvc_order:
        lvc_row = lvc_target[lvc_target['lvc_group'] == lvc]
        if not lvc_row.empty:
            target_count = lvc_row.iloc[0]['fas_count']
            target_dollars = lvc_row.iloc[0]['fas_dollars']
            
            prev_row = lvc_prev[lvc_prev['lvc_group'] == lvc]
            if not prev_row.empty:
                prev_count = prev_row.iloc[0]['fas_count']
                change = ((target_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
                print(f"{lvc:15s}: {target_count:4.0f} FAS (${target_dollars/1000:6.0f}K) | Change: {change:+6.1f}%")
            else:
                print(f"{lvc:15s}: {target_count:4.0f} FAS (${target_dollars/1000:6.0f}K) | New")
    
    # MA Performance Analysis
    print("\n" + "="*80)
    print("MA PERFORMANCE ANALYSIS")
    print("="*80)
    
    ma_target = df_bd_target.groupby('mortgage_advisor').agg({
        'fas_count': 'sum',
        'fas_dollars': 'sum'
    }).reset_index()
    
    ma_prev = df_bd_prev.groupby('mortgage_advisor').agg({
        'fas_count': 'sum',
        'fas_dollars': 'sum'
    }).reset_index()
    
    # Average over 4 weeks
    ma_prev['fas_count'] = ma_prev['fas_count'] / 4
    ma_prev['fas_dollars'] = ma_prev['fas_dollars'] / 4
    
    # Calculate changes
    ma_changes = []
    for _, ma_row in ma_target.iterrows():
        ma = ma_row['mortgage_advisor']
        target_count = ma_row['fas_count']
        target_dollars = ma_row['fas_dollars']
        
        prev_row = ma_prev[ma_prev['mortgage_advisor'] == ma]
        if not prev_row.empty:
            prev_count = prev_row.iloc[0]['fas_count']
            change = target_count - prev_count
            pct_change = (change / prev_count * 100) if prev_count > 0 else 0
            
            ma_changes.append({
                'ma': ma,
                'fas_count': target_count,
                'fas_dollars': target_dollars,
                'change': change,
                'pct_change': pct_change
            })
    
    if ma_changes:
        df_ma_changes = pd.DataFrame(ma_changes)
        
        # Top performers by count
        print("\nTop 10 Performers (by FAS count):")
        top_performers = df_ma_changes.nlargest(10, 'fas_count')
        for _, row in top_performers.iterrows():
            print(f"  {row['ma']:25s}: {row['fas_count']:4.0f} FAS (${row['fas_dollars']/1000:6.0f}K) | Change: {row['pct_change']:+6.1f}%")
        
        # Biggest gainers
        print("\nBiggest Gainers (by absolute change):")
        gainers = df_ma_changes.nlargest(10, 'change')
        for _, row in gainers.iterrows():
            print(f"  {row['ma']:25s}: {row['fas_count']:4.0f} FAS | Change: {row['change']:+5.0f} ({row['pct_change']:+6.1f}%)")
        
        # Biggest decliners
        print("\nBiggest Decliners (by absolute change):")
        decliners = df_ma_changes.nsmallest(10, 'change')
        for _, row in decliners.iterrows():
            print(f"  {row['ma']:25s}: {row['fas_count']:4.0f} FAS | Change: {row['change']:+5.0f} ({row['pct_change']:+6.1f}%)")
    else:
        print("No MA comparison data available")
    
    print("\n" + "="*80)
    print("LEGEND:")
    print("  ** = Statistically significant at 2 standard deviations (p < 0.05)")
    print("  *** = Highly significant at 3 standard deviations (p < 0.01)")
    print("  Z-score: Measures how many standard deviations away from the mean")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
