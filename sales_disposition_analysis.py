"""
Sales Disposition Analysis
Analyzes fallout reasons by vintage for contacted leads
Source: lendage-data-platform-ops.report_views.v_sales_disposition
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go

# === BigQuery Connection ===
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

# === Step 1: Explore the schema ===
def get_schema():
    """Get the schema of the sales disposition view"""
    query = """
    SELECT column_name, data_type, is_nullable
    FROM `lendage-data-platform-ops`.report_views.INFORMATION_SCHEMA.COLUMNS
    WHERE table_name = 'v_sales_disposition'
    ORDER BY ordinal_position
    """
    return run_query(query)

# === Step 2: Sample data to understand field values ===
def get_sample_data(limit=100):
    """Get sample data to understand field values"""
    query = f"""
    SELECT *
    FROM `lendage-data-platform-ops.report_views.v_sales_disposition`
    WHERE contacted_flag = TRUE
    LIMIT {limit}
    """
    return run_query(query)

# === Step 3: Main Analysis Query ===
def get_fallout_analysis(start_date='2025-10-01', end_date=None):
    """
    Analyze fallout reasons by vintage for contacted leads
    
    Fallout reason logic (from Tableau calc):
    - IF NOT ISNULL([Ineligible Reason]) THEN "Ineligible - " + [Ineligible Reason]
    - ELSEIF NOT ISNULL([Inactive Reason]) THEN "Inactive - " + [Inactive Reason]
    - ELSE NULL END
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    query = f"""
    WITH base_data AS (
        SELECT
            -- Vintage (lead created date) aggregation
            DATE(lead_created_date) as vintage_date,
            DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
            
            -- Core fields
            call_activity,
            lead_call_result,
            ineligible_reason,
            inactive_reason,
            
            -- Derived fallout reason (Tableau calc logic)
            CASE
                WHEN ineligible_reason IS NOT NULL THEN CONCAT('Ineligible - ', ineligible_reason)
                WHEN inactive_reason IS NOT NULL THEN CONCAT('Inactive - ', inactive_reason)
                ELSE NULL
            END as fallout_reason,
            
            -- Lead identifier
            lendage_guid
            
        FROM `lendage-data-platform-ops.report_views.v_sales_disposition`
        WHERE contacted_flag = TRUE
        AND DATE(lead_created_date) >= '{start_date}'
        AND DATE(lead_created_date) <= '{end_date}'
    )
    SELECT
        vintage_week,
        call_activity,
        lead_call_result,
        fallout_reason,
        COUNT(DISTINCT lendage_guid) as lead_count
    FROM base_data
    GROUP BY 1, 2, 3, 4
    ORDER BY vintage_week DESC, lead_count DESC
    """
    return run_query(query)

def get_weekly_trends(start_date='2025-10-01', end_date=None):
    """Get weekly trends for fallout reasons"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    query = f"""
    WITH base_data AS (
        SELECT
            DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
            
            -- Derived fallout reason
            CASE
                WHEN ineligible_reason IS NOT NULL THEN CONCAT('Ineligible - ', ineligible_reason)
                WHEN inactive_reason IS NOT NULL THEN CONCAT('Inactive - ', inactive_reason)
                ELSE 'No Fallout'
            END as fallout_reason,
            
            lendage_guid
            
        FROM `lendage-data-platform-ops.report_views.v_sales_disposition`
        WHERE contacted_flag = TRUE
        AND DATE(lead_created_date) >= '{start_date}'
        AND DATE(lead_created_date) <= '{end_date}'
    )
    SELECT
        vintage_week,
        fallout_reason,
        COUNT(DISTINCT lendage_guid) as lead_count
    FROM base_data
    GROUP BY 1, 2
    ORDER BY vintage_week DESC, lead_count DESC
    """
    return run_query(query)

def get_call_result_trends(start_date='2025-10-01', end_date=None):
    """Get weekly trends for lead_call_result"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
        lead_call_result,
        COUNT(DISTINCT lendage_guid) as lead_count
    FROM `lendage-data-platform-ops.report_views.v_sales_disposition`
    WHERE contacted_flag = TRUE
    AND DATE(lead_created_date) >= '{start_date}'
    AND DATE(lead_created_date) <= '{end_date}'
    GROUP BY 1, 2
    ORDER BY vintage_week DESC, lead_count DESC
    """
    return run_query(query)

def get_call_activity_trends(start_date='2025-10-01', end_date=None):
    """Get weekly trends for call_activity"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
        call_activity,
        COUNT(DISTINCT lendage_guid) as lead_count
    FROM `lendage-data-platform-ops.report_views.v_sales_disposition`
    WHERE contacted_flag = TRUE
    AND DATE(lead_created_date) >= '{start_date}'
    AND DATE(lead_created_date) <= '{end_date}'
    GROUP BY 1, 2
    ORDER BY vintage_week DESC, lead_count DESC
    """
    return run_query(query)

def get_last_two_weeks_comparison():
    """Compare last 2 weeks vs previous weeks for fallout trends"""
    today = datetime.now()
    two_weeks_ago = today - timedelta(weeks=2)
    four_weeks_ago = today - timedelta(weeks=4)
    
    query = f"""
    WITH base_data AS (
        SELECT
            DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
            
            CASE
                WHEN DATE(lead_created_date) >= '{two_weeks_ago.strftime('%Y-%m-%d')}' THEN 'Last 2 Weeks'
                ELSE 'Previous Weeks'
            END as period,
            
            call_activity,
            lead_call_result,
            
            CASE
                WHEN ineligible_reason IS NOT NULL THEN CONCAT('Ineligible - ', ineligible_reason)
                WHEN inactive_reason IS NOT NULL THEN CONCAT('Inactive - ', inactive_reason)
                ELSE 'No Fallout'
            END as fallout_reason,
            
            lendage_guid
            
        FROM `lendage-data-platform-ops.report_views.v_sales_disposition`
        WHERE contacted_flag = TRUE
        AND DATE(lead_created_date) >= '2025-10-01'
        AND DATE(lead_created_date) <= '{today.strftime('%Y-%m-%d')}'
    )
    SELECT
        period,
        vintage_week,
        call_activity,
        lead_call_result,
        fallout_reason,
        COUNT(DISTINCT lendage_guid) as lead_count
    FROM base_data
    GROUP BY 1, 2, 3, 4, 5
    ORDER BY period, vintage_week DESC, lead_count DESC
    """
    return run_query(query)


def get_detailed_weekly_analysis():
    """Comprehensive weekly analysis with call_activity, lead_call_result, and fallout"""
    today = datetime.now()
    
    query = f"""
    WITH base_data AS (
        SELECT
            DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
            call_activity,
            lead_call_result,
            ineligible_reason,
            inactive_reason,
            
            CASE
                WHEN ineligible_reason IS NOT NULL THEN CONCAT('Ineligible - ', ineligible_reason)
                WHEN inactive_reason IS NOT NULL THEN CONCAT('Inactive - ', inactive_reason)
                ELSE NULL
            END as fallout_reason,
            
            CASE
                WHEN ineligible_reason IS NOT NULL THEN 'Ineligible'
                WHEN inactive_reason IS NOT NULL THEN 'Inactive'
                ELSE 'No Fallout'
            END as fallout_category,
            
            lendage_guid
            
        FROM `lendage-data-platform-ops.report_views.v_sales_disposition`
        WHERE contacted_flag = TRUE
        AND DATE(lead_created_date) >= '2025-10-01'
        AND DATE(lead_created_date) <= '{today.strftime('%Y-%m-%d')}'
    )
    SELECT
        vintage_week,
        call_activity,
        lead_call_result,
        fallout_category,
        fallout_reason,
        COUNT(DISTINCT lendage_guid) as lead_count
    FROM base_data
    GROUP BY 1, 2, 3, 4, 5
    ORDER BY vintage_week DESC, lead_count DESC
    """
    return run_query(query)


def analyze_trends():
    """Generate a comprehensive trend analysis report"""
    print("=" * 80)
    print("SALES DISPOSITION TREND ANALYSIS")
    print("Vintage: Lead Created Date | Filter: contacted_flag = TRUE")
    print("Period: October 2025 - Today")
    print("=" * 80)
    
    df = get_detailed_weekly_analysis()
    if df.empty:
        print("No data returned!")
        return
    
    df['vintage_week'] = pd.to_datetime(df['vintage_week'])
    
    # Total contacted leads by week
    print("\n" + "=" * 80)
    print("1. WEEKLY CONTACTED LEADS (Vintage)")
    print("=" * 80)
    weekly_total = df.groupby('vintage_week')['lead_count'].sum().reset_index()
    weekly_total = weekly_total.sort_values('vintage_week')
    
    # Calculate WoW change
    weekly_total['prev_week'] = weekly_total['lead_count'].shift(1)
    weekly_total['wow_change'] = weekly_total['lead_count'] - weekly_total['prev_week']
    weekly_total['wow_pct'] = (weekly_total['wow_change'] / weekly_total['prev_week'] * 100).round(1)
    
    print(weekly_total[['vintage_week', 'lead_count', 'wow_change', 'wow_pct']].tail(12).to_string(index=False))
    
    # Last 2 weeks highlight
    last_2_weeks = weekly_total.tail(2)
    prev_4_weeks = weekly_total.tail(6).head(4)
    
    last_2_avg = last_2_weeks['lead_count'].mean()
    prev_4_avg = prev_4_weeks['lead_count'].mean()
    change_pct = ((last_2_avg - prev_4_avg) / prev_4_avg * 100)
    
    print(f"\n📊 Last 2 Weeks Avg: {last_2_avg:,.0f} | Previous 4 Weeks Avg: {prev_4_avg:,.0f} | Change: {change_pct:+.1f}%")
    
    # Call Activity breakdown
    print("\n" + "=" * 80)
    print("2. CALL ACTIVITY BREAKDOWN BY WEEK")
    print("=" * 80)
    call_activity_weekly = df.groupby(['vintage_week', 'call_activity'])['lead_count'].sum().reset_index()
    call_activity_pivot = call_activity_weekly.pivot(index='vintage_week', columns='call_activity', values='lead_count').fillna(0)
    call_activity_pivot = call_activity_pivot.sort_index()
    
    print("\nCall Activity Distribution (Last 8 Weeks):")
    print(call_activity_pivot.tail(8).to_string())
    
    # Call Activity % distribution for last 2 weeks vs previous
    print("\n📊 Call Activity % - Last 2 Weeks vs Previous 4 Weeks:")
    last_2_weeks_dates = weekly_total.tail(2)['vintage_week'].tolist()
    prev_4_weeks_dates = weekly_total.tail(6).head(4)['vintage_week'].tolist()
    
    last_2_activity = df[df['vintage_week'].isin(last_2_weeks_dates)].groupby('call_activity')['lead_count'].sum()
    prev_4_activity = df[df['vintage_week'].isin(prev_4_weeks_dates)].groupby('call_activity')['lead_count'].sum()
    
    activity_comparison = pd.DataFrame({
        'Last 2 Weeks': last_2_activity,
        'Previous 4 Weeks': prev_4_activity
    }).fillna(0)
    activity_comparison['Last 2 Weeks %'] = (activity_comparison['Last 2 Weeks'] / activity_comparison['Last 2 Weeks'].sum() * 100).round(1)
    activity_comparison['Prev 4 Weeks %'] = (activity_comparison['Previous 4 Weeks'] / activity_comparison['Previous 4 Weeks'].sum() * 100).round(1)
    activity_comparison['Change (pp)'] = (activity_comparison['Last 2 Weeks %'] - activity_comparison['Prev 4 Weeks %']).round(1)
    print(activity_comparison.to_string())
    
    # Lead Call Result breakdown
    print("\n" + "=" * 80)
    print("3. LEAD CALL RESULT BREAKDOWN BY WEEK")
    print("=" * 80)
    result_weekly = df.groupby(['vintage_week', 'lead_call_result'])['lead_count'].sum().reset_index()
    result_pivot = result_weekly.pivot(index='vintage_week', columns='lead_call_result', values='lead_count').fillna(0)
    result_pivot = result_pivot.sort_index()
    
    print("\nLead Call Result Distribution (Last 8 Weeks):")
    print(result_pivot.tail(8).to_string())
    
    # Lead Call Result % comparison
    print("\n📊 Lead Call Result % - Last 2 Weeks vs Previous 4 Weeks:")
    last_2_result = df[df['vintage_week'].isin(last_2_weeks_dates)].groupby('lead_call_result')['lead_count'].sum()
    prev_4_result = df[df['vintage_week'].isin(prev_4_weeks_dates)].groupby('lead_call_result')['lead_count'].sum()
    
    result_comparison = pd.DataFrame({
        'Last 2 Weeks': last_2_result,
        'Previous 4 Weeks': prev_4_result
    }).fillna(0)
    result_comparison['Last 2 Weeks %'] = (result_comparison['Last 2 Weeks'] / result_comparison['Last 2 Weeks'].sum() * 100).round(1)
    result_comparison['Prev 4 Weeks %'] = (result_comparison['Previous 4 Weeks'] / result_comparison['Previous 4 Weeks'].sum() * 100).round(1)
    result_comparison['Change (pp)'] = (result_comparison['Last 2 Weeks %'] - result_comparison['Prev 4 Weeks %']).round(1)
    result_comparison = result_comparison.sort_values('Change (pp)', ascending=False)
    print(result_comparison.to_string())
    
    # Fallout Category breakdown
    print("\n" + "=" * 80)
    print("4. FALLOUT CATEGORY BREAKDOWN BY WEEK")
    print("=" * 80)
    fallout_cat_weekly = df.groupby(['vintage_week', 'fallout_category'])['lead_count'].sum().reset_index()
    fallout_cat_pivot = fallout_cat_weekly.pivot(index='vintage_week', columns='fallout_category', values='lead_count').fillna(0)
    fallout_cat_pivot = fallout_cat_pivot.sort_index()
    
    # Add percentages
    fallout_cat_pivot['Total'] = fallout_cat_pivot.sum(axis=1)
    for col in ['Ineligible', 'Inactive', 'No Fallout']:
        if col in fallout_cat_pivot.columns:
            fallout_cat_pivot[f'{col} %'] = (fallout_cat_pivot[col] / fallout_cat_pivot['Total'] * 100).round(1)
    
    print("\nFallout Category (Last 8 Weeks):")
    cols_to_show = [c for c in fallout_cat_pivot.columns if '%' in c or c == 'Total']
    print(fallout_cat_pivot[cols_to_show].tail(8).to_string())
    
    # Detailed Fallout Reasons
    print("\n" + "=" * 80)
    print("5. TOP FALLOUT REASONS - Last 2 Weeks vs Previous 4 Weeks")
    print("=" * 80)
    
    # Filter out 'No Fallout' and get detailed reasons
    fallout_df = df[df['fallout_reason'].notna()]
    
    last_2_fallout = fallout_df[fallout_df['vintage_week'].isin(last_2_weeks_dates)].groupby('fallout_reason')['lead_count'].sum()
    prev_4_fallout = fallout_df[fallout_df['vintage_week'].isin(prev_4_weeks_dates)].groupby('fallout_reason')['lead_count'].sum()
    
    fallout_comparison = pd.DataFrame({
        'Last 2 Weeks': last_2_fallout,
        'Previous 4 Weeks': prev_4_fallout
    }).fillna(0)
    fallout_comparison['Last 2 Weeks %'] = (fallout_comparison['Last 2 Weeks'] / fallout_comparison['Last 2 Weeks'].sum() * 100).round(1)
    fallout_comparison['Prev 4 Weeks %'] = (fallout_comparison['Previous 4 Weeks'] / fallout_comparison['Previous 4 Weeks'].sum() * 100).round(1)
    fallout_comparison['Change (pp)'] = (fallout_comparison['Last 2 Weeks %'] - fallout_comparison['Prev 4 Weeks %']).round(1)
    fallout_comparison = fallout_comparison.sort_values('Last 2 Weeks', ascending=False)
    
    print("\nTop 20 Fallout Reasons:")
    print(fallout_comparison.head(20).to_string())
    
    # Biggest movers
    print("\n📈 Biggest INCREASES in Fallout Reasons (Last 2 Weeks vs Previous 4 Weeks):")
    increases = fallout_comparison[fallout_comparison['Change (pp)'] > 0].sort_values('Change (pp)', ascending=False)
    print(increases.head(10).to_string())
    
    print("\n📉 Biggest DECREASES in Fallout Reasons:")
    decreases = fallout_comparison[fallout_comparison['Change (pp)'] < 0].sort_values('Change (pp)')
    print(decreases.head(10).to_string())
    
    # Summary stats
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_last_2 = df[df['vintage_week'].isin(last_2_weeks_dates)]['lead_count'].sum()
    total_prev_4 = df[df['vintage_week'].isin(prev_4_weeks_dates)]['lead_count'].sum()
    
    fallout_last_2 = fallout_df[fallout_df['vintage_week'].isin(last_2_weeks_dates)]['lead_count'].sum()
    fallout_prev_4 = fallout_df[fallout_df['vintage_week'].isin(prev_4_weeks_dates)]['lead_count'].sum()
    
    print(f"\nTotal Contacted Leads:")
    print(f"  Last 2 Weeks: {total_last_2:,}")
    print(f"  Previous 4 Weeks: {total_prev_4:,}")
    
    print(f"\nFallout Rate:")
    print(f"  Last 2 Weeks: {fallout_last_2/total_last_2*100:.1f}% ({fallout_last_2:,} / {total_last_2:,})")
    print(f"  Previous 4 Weeks: {fallout_prev_4/total_prev_4*100:.1f}% ({fallout_prev_4:,} / {total_prev_4:,})")
    
    return df


if __name__ == "__main__":
    analyze_trends()
