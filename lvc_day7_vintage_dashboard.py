"""
LVC Day 7 Vintage Analysis Dashboard
Dash application for analyzing FAS Day 7 performance by lead vintage
Two tabs: Overview & Deeper Dive (selectable week)
Modular layout with collapsible sections

FAS Day 7 Logic:
- Numerator: FAS within 7 days of lead creation (full_app_submit_datetime between lead_created_date and lead_created_date + 7 days)
- Denominator: Sent to Sales leads where lead_created_date <= today - 7 days
- Only includes leads with complete 7-day window
"""

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State, ALL, MATCH, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from google.cloud import bigquery
from google.oauth2 import service_account
import json

# === DARK THEME COLORS (Cyber/Tech Style) ===
COLORS = {
    'background': '#0a1628',
    'card_bg': '#0d1e36',
    'text': '#e0e6ed',
    'text_muted': '#6b7c93',
    'border': '#1a3a5c',
    'primary': '#00d4aa',
    'success': '#00d4aa',
    'warning': '#f0b429',
    'danger': '#ff6b6b',
    'accent1': '#00d4aa',
    'accent2': '#00b894',
    'accent3': '#74b9ff',
    'chart_cyan': '#00d4aa',
    'chart_green': '#2ecc71',
    'chart_yellow': '#f39c12',
    'chart_purple': '#9b59b6',
    'chart_gray': '#7f8c8d',
}

# === MODULAR LAYOUT CONFIGURATION ===
OVERVIEW_LAYOUT_CONFIG = [
    {'id': 'key-metrics', 'title': 'Key Performance Metrics', 'subtitle': 'Vintage: Leads with complete 7-day window', 'visible': True, 'collapsed': False},
    {'id': 'what-changed', 'title': 'What Changed vs Previous 4 Weeks', 'subtitle': 'Weekly Performance Analysis', 'visible': True, 'collapsed': False},
    {'id': 'weekly-trends', 'title': 'Weekly Metrics', 'subtitle': 'Last 12 Weeks (Sun-Sat) | Vintage Analysis', 'visible': True, 'collapsed': False},
    {'id': 'vintage-analysis', 'title': 'Vintage Analysis', 'subtitle': 'Vintage: Based on Lead Created Date - Tracking Lead Cohort Performance', 'visible': True, 'collapsed': False},
    {'id': 'daily-comparison', 'title': 'Current Week by Day vs Previous 4 Weeks', 'subtitle': 'Vintage: Comparing by Lead Created Date', 'visible': True, 'collapsed': False},
]

# Vintage Deeper Dive - based on lead_created_date with FAS Day 7 logic
VINTAGE_DEEPDIVE_CONFIG = [
    {'id': 'vintage-kpi', 'title': 'Week Deep Dive', 'subtitle': 'What Drove the Performance?', 'visible': True, 'collapsed': False},
    {'id': 'vintage-lvc', 'title': 'LVC Group Analysis', 'subtitle': 'Which Lead Segments Drove Growth?', 'visible': True, 'collapsed': False},
    {'id': 'vintage-channel-v2', 'title': 'Channel Mix Analysis', 'subtitle': 'Top 10 Channels + Other | Based on Lead Created Date (Vintage)', 'visible': True, 'collapsed': False},
    {'id': 'vintage-ma', 'title': 'MA Performance Analysis', 'subtitle': 'Based on Lead Created Date (Vintage)', 'visible': True, 'collapsed': False},
]

# In-Period Deeper Dive - metrics based on event occurrence dates
INPERIOD_DEEPDIVE_CONFIG = [
    {'id': 'inperiod-kpi', 'title': 'Week Deep Dive - In Period', 'subtitle': 'Metrics by Event Date (STS, Contact, FAS, Funded)', 'visible': True, 'collapsed': False},
    {'id': 'inperiod-lvc', 'title': 'LVC Group Analysis', 'subtitle': 'Lead Quality Segments | In-Period Logic', 'visible': True, 'collapsed': False},
    {'id': 'inperiod-channel', 'title': 'Channel Mix Analysis', 'subtitle': 'Top Channels | In-Period Logic', 'visible': True, 'collapsed': False},
    {'id': 'inperiod-ma', 'title': 'MA Performance Analysis', 'subtitle': 'By Starring Group & Individual MA | In-Period Logic', 'visible': True, 'collapsed': False},
]

# === STYLES ===
CARD_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'borderRadius': '12px',
    'border': f"1px solid {COLORS['border']}",
    'padding': '20px',
    'marginBottom': '16px',
    'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)',
}

METRIC_CARD_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'borderRadius': '12px',
    'border': f"1px solid {COLORS['border']}",
    'padding': '24px 16px',
    'textAlign': 'center',
    'minHeight': '130px',
    'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)',
    'position': 'relative',
}

TABLE_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'color': COLORS['text'],
}

TAB_STYLE = {
    'backgroundColor': 'transparent',
    'color': COLORS['text_muted'],
    'border': 'none',
    'borderBottom': f"2px solid transparent",
    'padding': '10px 24px',
    'fontSize': '13px',
    'fontWeight': '500',
    'letterSpacing': '0.5px',
    'textTransform': 'uppercase',
    'cursor': 'pointer',
}

TAB_SELECTED_STYLE = {
    'backgroundColor': 'transparent',
    'color': COLORS['primary'],
    'border': 'none',
    'borderBottom': f"2px solid {COLORS['primary']}",
    'padding': '10px 24px',
    'fontSize': '13px',
    'fontWeight': '600',
    'letterSpacing': '0.5px',
    'textTransform': 'uppercase',
}

# === Initialize Dash App ===
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="LVC Day 7 Vintage Analysis"
)

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

# === Data Query Functions with FAS Day 7 Logic ===
def get_available_weeks():
    """Get list of available week vintages for dropdown - only weeks with complete 7-day window"""
    query = """
    SELECT DISTINCT 
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as week_start
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '2025-10-01'
    AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
    ORDER BY week_start DESC
    LIMIT 20
    """
    df = run_query(query)
    if not df.empty:
        df['week_start'] = pd.to_datetime(df['week_start'])
        return df['week_start'].dt.strftime('%Y-%m-%d').tolist()
    return []

def get_day7_summary_data(target_week):
    """Get FAS Day 7 summary metrics for target week vs prev 12 weeks - VINTAGE based with Day 7 logic"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=12)
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            THEN lendage_guid 
        END) as sts_eligible,
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as day7_eligible_sts,
        COUNT(DISTINCT CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as fas_day7_count,
        SUM(CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN e_loan_amount 
            ELSE 0 
        END) as fas_day7_dollars,
        AVG(CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN e_loan_amount 
        END) as avg_loan,
        COUNT(DISTINCT lendage_guid) as total_leads,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1
    ORDER BY 1
    """
    return run_query(query)

def get_vintage_summary_data(target_week):
    """Get vintage-based summary metrics with FAS Day 7 logic"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
        COUNT(DISTINCT lendage_guid) as total_leads,
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            THEN lendage_guid 
        END) as sts_eligible,
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as day7_eligible_sts,
        COUNT(DISTINCT CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as fas_day7_count,
        SUM(CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN e_loan_amount 
            ELSE 0 
        END) as fas_day7_dollars,
        AVG(CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN e_loan_amount 
        END) as avg_loan,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1
    ORDER BY 1
    """
    return run_query(query)

def get_daily_comparison_data(target_week):
    """Get daily vintage data with FAS Day 7 logic"""
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
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            THEN lendage_guid 
        END) as sts_eligible,
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as day7_eligible_sts,
        COUNT(DISTINCT CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as fas_day7_count,
        SUM(CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN e_loan_amount 
            ELSE 0 
        END) as fas_day7_dollars
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2, 3, 4
    ORDER BY 1
    """
    return run_query(query)

def get_lvc_day7_breakdown_data(target_week):
    """Get LVC group breakdown with FAS Day 7 logic - VINTAGE based"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    
    query = f"""
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
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            THEN lendage_guid 
        END) as sts_eligible,
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as day7_eligible_sts,
        COUNT(DISTINCT CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as fas_day7_count,
        SUM(CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN e_loan_amount 
            ELSE 0 
        END) as fas_day7_dollars,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2
    """
    return run_query(query)

def get_what_changed_data(target_week):
    """Get breakdown by LVC, Persona, MA, Channel with FAS Day 7 logic"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
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
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as day7_eligible_sts,
        COUNT(DISTINCT CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as fas_day7_count,
        SUM(CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN e_loan_amount 
            ELSE 0 
        END) as fas_day7_dollars
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2, 3, 4, 5
    """
    return run_query(query)

def get_vintage_deep_dive_data(target_week):
    """Get comprehensive data for vintage-based deep dive analysis with FAS Day 7 logic"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=8)
    
    query = f"""
    SELECT
        DATE(lead_created_date) as vintage_date,
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
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
        COUNT(DISTINCT lendage_guid) as lead_count,
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            THEN lendage_guid 
        END) as sts_eligible,
        COUNT(DISTINCT CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as fas_day7_count,
        SUM(CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN e_loan_amount 
            ELSE 0 
        END) as fas_day7_dollars,
        AVG(CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN e_loan_amount 
        END) as avg_loan,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c,
        -- SF Contact Day 7: contacted within 7 days of lead creation (must be sent to sales)
        COUNT(DISTINCT CASE 
            WHEN sf__contacted_guid IS NOT NULL 
            AND DATE(sf_contacted_date) >= DATE(lead_created_date)
            AND DATE(sf_contacted_date) <= DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            AND sent_to_sales_date IS NOT NULL
            THEN lendage_guid 
        END) as sf_contact_day7_count,
        -- Day 7 eligible denominator (for SF Contact % Day 7)
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as day7_eligible_sts,
        AVG(SAFE_CAST(fico AS FLOAT64)) as avg_fico,
        AVG(SAFE_CAST(lead_value_cohort AS FLOAT64)) as avg_lvc,
        COUNT(DISTINCT CASE WHEN funded_date IS NOT NULL THEN lendage_guid END) as funded_count,
        SUM(CASE WHEN funded_date IS NOT NULL THEN e_loan_amount ELSE 0 END) as funded_dollars
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2, 3, 4, 5, 6
    """
    return run_query(query)

def get_ma_performance_data(target_week):
    """Get MA performance data with starring aggregation - fetches 6 weeks for trend analysis"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=6)
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
        COALESCE(CAST(starring AS STRING), 'Unknown') as starring_group,
        mortgage_advisor,
        COUNT(DISTINCT lendage_guid) as leads_assigned,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c,
        AVG(SAFE_CAST(lead_value_cohort AS FLOAT64)) as avg_lvc,
        SUM(COALESCE(call_attempts, 0)) as call_attempts,
        COUNT(DISTINCT CASE 
            WHEN sf__contacted_guid IS NOT NULL 
            AND DATE(sf_contacted_date) >= DATE(lead_created_date)
            AND DATE(sf_contacted_date) <= DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            AND sent_to_sales_date IS NOT NULL
            THEN lendage_guid 
        END) as sf_contact_day7_count,
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as day7_eligible_sts,
        COUNT(DISTINCT CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as fas_day7_count,
        SUM(CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN e_loan_amount 
            ELSE 0 
        END) as fas_day7_dollars,
        AVG(CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN e_loan_amount 
        END) as avg_fas_day7_dollars,
        SUM(CASE 
            WHEN full_app_submit_datetime IS NOT NULL 
            THEN e_loan_amount 
            ELSE 0 
        END) as total_fas_dollars,
        COUNT(DISTINCT CASE WHEN funded_date IS NOT NULL THEN lendage_guid END) as funded_count
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2, 3
    """
    return run_query(query)

def get_ltpl_pricing_data(target_week):
    """Get LT PL pricing data for WoW comparison (last 12+ weeks dynamically)"""
    query = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
        COALESCE(CreditRange, 'Unknown') as credit_range,
        AVG(SAFE_CAST(avg_secured_competitor_apr AS FLOAT64)) as avg_secured_competitor_apr,
        AVG(SAFE_CAST(avg_achieve_loan_apr AS FLOAT64)) as avg_achieve_loan_apr,
        COUNT(DISTINCT lendage_guid) as lead_count
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 WEEK)
    AND sub_group = 'LT PL'
    GROUP BY 1, 2
    """
    return run_query(query)

def get_outreach_metrics(target_week):
    """Get outreach metrics (SMS, Calls) for target week vs previous 6 weeks"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=6)
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as week_start,
        mortgage_advisor,
        COUNT(DISTINCT lendage_guid) as total_leads,
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            THEN lendage_guid 
        END) as sts_eligible,
        COUNT(DISTINCT CASE 
            WHEN sent_to_sales_date IS NOT NULL 
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as day7_eligible_sts,
        COUNT(DISTINCT CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as fas_day7_count,
        SUM(CASE 
            WHEN DATE(full_app_submit_datetime) BETWEEN DATE(lead_created_date) AND DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            AND DATE(lead_created_date) <= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            THEN e_loan_amount 
            ELSE 0 
        END) as fas_day7_dollars,
        SUM(COALESCE(total_sms_outbound_count, 0)) as total_sms,
        SUM(COALESCE(total_sms_outbound_before_contact, 0)) as total_sms_before_contact,
        SUM(COALESCE(call_attempts, 0)) as total_call_attempts,
        AVG(COALESCE(total_sms_outbound_count, 0)) as avg_sms,
        AVG(COALESCE(total_sms_outbound_before_contact, 0)) as avg_sms_before_contact,
        AVG(COALESCE(call_attempts, 0)) as avg_call_attempts
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    AND mortgage_advisor IS NOT NULL
    GROUP BY 1, 2
    """
    return run_query(query)

# === IN-PERIOD DATA FUNCTIONS ===
# In-period logic: filter by lead_created_date but count events when they occurred

def get_inperiod_lvc_data(target_week):
    """Get in-period LVC group breakdown - simplified query"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    
    query = f"""
    SELECT
        CASE
            WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
            WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
            WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
            WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
            ELSE 'Other'
        END as lvc_group,
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as lead_week,
        COUNT(DISTINCT lendage_guid) as gross_leads,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sts_count,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contact_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN COALESCE(e_loan_amount, 0) ELSE 0 END) as fas_dollars,
        COUNT(DISTINCT CASE WHEN funded_date IS NOT NULL THEN lendage_guid END) as funded_count,
        SUM(CASE WHEN funded_date IS NOT NULL THEN COALESCE(funded_amount, 0) ELSE 0 END) as funded_dollars,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2
    """
    return run_query(query)

def get_inperiod_channel_data(target_week):
    """Get in-period channel breakdown - simplified query"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    
    query = f"""
    SELECT
        COALESCE(sub_group, 'Unknown') as channel,
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as lead_week,
        COUNT(DISTINCT lendage_guid) as gross_leads,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sts_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN COALESCE(e_loan_amount, 0) ELSE 0 END) as fas_dollars,
        COUNT(DISTINCT CASE WHEN funded_date IS NOT NULL THEN lendage_guid END) as funded_count,
        SUM(CASE WHEN funded_date IS NOT NULL THEN COALESCE(funded_amount, 0) ELSE 0 END) as funded_dollars,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c,
        AVG(SAFE_CAST(fico AS FLOAT64)) as avg_fico,
        AVG(SAFE_CAST(lead_value_cohort AS FLOAT64)) as avg_lvc
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2
    """
    return run_query(query)

def get_inperiod_ma_data(target_week):
    """Get in-period MA performance data - simplified query"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as lead_week,
        COALESCE(CAST(starring AS STRING), 'Unknown') as starring_group,
        mortgage_advisor,
        COUNT(DISTINCT lendage_guid) as leads_assigned,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sts_count,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contact_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN COALESCE(e_loan_amount, 0) ELSE 0 END) as fas_dollars,
        COUNT(DISTINCT CASE WHEN funded_date IS NOT NULL THEN lendage_guid END) as funded_count,
        SUM(CASE WHEN funded_date IS NOT NULL THEN COALESCE(funded_amount, 0) ELSE 0 END) as funded_dollars,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c,
        AVG(SAFE_CAST(lead_value_cohort AS FLOAT64)) as avg_lvc,
        SUM(COALESCE(call_attempts, 0)) as call_attempts
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    AND mortgage_advisor IS NOT NULL
    GROUP BY 1, 2, 3
    """
    return run_query(query)

# === Helper Functions ===
def create_collapsible_section(section_id, title, subtitle=None, content_id=None, default_collapsed=False):
    """Create a collapsible section with header and toggle button"""
    return html.Div([
        html.Div([
            html.Div([
                html.H3(title, style={
                    'color': COLORS['text'],
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'letterSpacing': '0.5px',
                    'marginBottom': '0',
                    'fontFamily': "'Roboto', 'Segoe UI', sans-serif",
                    'display': 'inline-block',
                }),
                html.Span(subtitle, style={
                    'color': COLORS['text_muted'],
                    'fontSize': '12px',
                    'marginLeft': '12px',
                }) if subtitle else None,
            ], style={'display': 'flex', 'alignItems': 'center', 'flex': '1'}),
            html.Button(
                html.I(className='collapse-icon', children='▼' if not default_collapsed else '▶'),
                id={'type': 'collapse-btn', 'index': section_id},
                n_clicks=0,
                style={
                    'backgroundColor': 'transparent',
                    'border': f"1px solid {COLORS['border']}",
                    'borderRadius': '4px',
                    'color': COLORS['text_muted'],
                    'cursor': 'pointer',
                    'padding': '4px 8px',
                    'fontSize': '10px',
                    'marginLeft': '12px',
                }
            ),
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'space-between',
            'marginBottom': '8px',
        }),
        html.Div(style={
            'width': '40px',
            'height': '2px',
            'backgroundColor': COLORS['primary'],
            'marginBottom': '16px',
        }),
        dbc.Collapse(
            html.Div(id=content_id if content_id else f'{section_id}-content'),
            id={'type': 'collapse-content', 'index': section_id},
            is_open=not default_collapsed,
        ),
    ], style={'marginTop': '32px'}, id=f'section-{section_id}')

def create_metric_card(title, value, delta=None, delta_suffix="", positive_is_good=True):
    """Create a metric card component with cyber/tech styling"""
    delta_color = COLORS['success'] if (delta and delta > 0 and positive_is_good) or (delta and delta < 0 and not positive_is_good) else COLORS['danger'] if delta else COLORS['text_muted']
    if delta is not None:
        if abs(delta) < 0.1 and delta != 0:
            delta_text = f"{delta:+.2f}{delta_suffix}"
        else:
            delta_text = f"{delta:+.1f}{delta_suffix}"
    else:
        delta_text = ""
    
    return html.Div([
        html.Div(style={
            'position': 'absolute',
            'top': '0',
            'left': '20%',
            'right': '20%',
            'height': '2px',
            'backgroundColor': COLORS['primary'],
            'borderRadius': '0 0 2px 2px',
        }),
        html.P(title, style={
            'color': COLORS['text'], 
            'fontSize': '13px', 
            'marginBottom': '10px', 
            'textTransform': 'uppercase',
            'letterSpacing': '0.5px',
            'fontWeight': '700',
        }),
        html.H3(value, style={
            'color': COLORS['primary'], 
            'margin': '0', 
            'fontSize': '32px', 
            'fontWeight': '400',
            'fontFamily': "'Segoe UI', 'Roboto', sans-serif",
        }),
        html.P(delta_text, style={
            'color': delta_color, 
            'fontSize': '13px', 
            'marginTop': '10px',
            'fontWeight': '500',
        }) if delta_text else html.Div()
    ], style=METRIC_CARD_STYLE)

def create_metric_card_with_sparkline(title, value, delta=None, delta_suffix="", positive_is_good=True, sparkline_data=None, sparkline_labels=None, value_format="number"):
    """Create a metric card with a small sparkline chart underneath"""
    delta_color = COLORS['success'] if (delta and delta > 0 and positive_is_good) or (delta and delta < 0 and not positive_is_good) else COLORS['danger'] if delta else COLORS['text_muted']
    
    if delta is not None:
        if abs(delta) < 0.1 and delta != 0:
            delta_text = f"{delta:+.2f}{delta_suffix}"
        else:
            delta_text = f"{delta:+.1f}{delta_suffix}"
    else:
        delta_text = ""
    
    sparkline_chart = html.Div()
    if sparkline_data is not None and len(sparkline_data) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sparkline_labels if sparkline_labels else list(range(len(sparkline_data))),
            y=sparkline_data,
            mode='lines',
            line=dict(color=COLORS['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)',
            showlegend=False,
            hoverinfo='none'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            height=40,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, fixedrange=True),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, fixedrange=True),
            hovermode=False
        )
        sparkline_chart = dcc.Graph(
            figure=fig,
            config={'displayModeBar': False, 'staticPlot': True},
            style={'height': '40px', 'marginTop': '6px'}
        )
    
    return html.Div([
        html.Div(style={
            'position': 'absolute',
            'top': '0',
            'left': '20%',
            'right': '20%',
            'height': '2px',
            'backgroundColor': COLORS['primary'],
            'borderRadius': '0 0 2px 2px',
        }),
        html.P(title, style={
            'color': COLORS['text'], 
            'fontSize': '12px', 
            'marginBottom': '6px', 
            'textTransform': 'uppercase',
            'letterSpacing': '0.5px',
            'fontWeight': '700',
        }),
        html.H3(value, style={
            'color': COLORS['primary'], 
            'margin': '0', 
            'fontSize': '24px', 
            'fontWeight': '400',
            'fontFamily': "'Segoe UI', 'Roboto', sans-serif",
        }),
        html.P(delta_text, style={
            'color': delta_color, 
            'fontSize': '11px', 
            'marginTop': '4px',
            'fontWeight': '500',
        }) if delta_text else html.Div(),
        sparkline_chart
    ], style={
        **METRIC_CARD_STYLE,
        'padding': '12px',
        'minHeight': '140px'
    })

def create_bar_chart_with_value(data, x_col, y_col, title, subtitle_value=None, color=None, y_format=None, formula_text=None, value_suffix="", value_prefix=""):
    """Create a bar chart with value displayed in title area"""
    if color is None:
        color = COLORS['chart_cyan']
    
    fig = go.Figure()
    
    def format_value(x):
        if not isinstance(x, (int, float)):
            return x
        if value_suffix == 'M':
            return f"${x:.2f}M"
        elif value_suffix == '%':
            return f"{x:.1f}%"
        elif value_prefix == '$':
            return f"${x:,.0f}"
        else:
            return f"{x:,.0f}{value_suffix}"
    
    fig.add_trace(go.Bar(
        x=data[x_col],
        y=data[y_col],
        marker_color=color,
        marker_line_color=color,
        marker_line_width=1,
        text=data[y_col].apply(format_value),
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=11)
    ))
    
    title_text = title
    if formula_text:
        title_text = f"{title} <span style='font-size:11px;color:{COLORS['text_muted']}'>{formula_text}</span>"
    
    y_tickformat = y_format if y_format else None
    y_ticksuffix = value_suffix if value_suffix and value_suffix != 'M' else ''
    y_tickprefix = value_prefix if value_prefix else ''
    
    y_max = data[y_col].max()
    y_range_max = y_max * 1.15
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(color=COLORS['text'], size=14, family="Segoe UI, Roboto, sans-serif"),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], family="Segoe UI, Roboto, sans-serif"),
        margin=dict(l=60, r=40, t=60, b=40),
        xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
        yaxis=dict(gridcolor=COLORS['border'], showgrid=True, tickformat=y_tickformat, ticksuffix=y_ticksuffix, tickprefix=y_tickprefix, gridwidth=1, range=[0, y_range_max]),
        height=350,
        bargap=0.2,
    )
    
    return fig

def create_grouped_bar_chart(data, x_col, y_cols, title, colors=None, y_format=None):
    """Create a grouped bar chart with cyber/tech styling"""
    fig = go.Figure()
    
    if colors is None:
        colors = [COLORS['chart_cyan'], COLORS['chart_gray']]
    
    for i, y_col in enumerate(y_cols):
        fig.add_trace(go.Bar(
            x=data[x_col],
            y=data[y_col],
            name=y_col,
            marker_color=colors[i % len(colors)],
            marker_line_color=colors[i % len(colors)],
            marker_line_width=1,
            text=data[y_col].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x),
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10)
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS['text'], size=13, family="Segoe UI, Roboto, sans-serif")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], family="Segoe UI, Roboto, sans-serif"),
        margin=dict(l=60, r=40, t=60, b=40),
        xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
        yaxis=dict(gridcolor=COLORS['border'], showgrid=True, tickformat=y_format, gridwidth=1),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
        barmode='group',
        height=350,
        bargap=0.3,
    )
    
    return fig

# === Layout Components ===
def create_header():
    """Create dashboard header with Achieve logo"""
    return html.Div([
        html.Div([
            html.H1("AHL WEEKLY PERFORMANCE", 
                    style={
                        'color': COLORS['primary'], 
                        'marginBottom': '4px', 
                        'fontSize': '28px',
                        'fontWeight': '300',
                        'letterSpacing': '4px',
                        'textTransform': 'uppercase',
                        'fontFamily': "'Segoe UI', 'Roboto', sans-serif",
                    }),
            html.Div(style={
                'width': '60px',
                'height': '2px',
                'backgroundColor': COLORS['primary'],
                'marginBottom': '12px',
            }),
            html.P("Weekly Performance Dashboard | Analysis by Lead Vintage",
                   style={
                       'color': COLORS['text_muted'], 
                       'marginBottom': '0',
                       'fontSize': '12px',
                       'letterSpacing': '1px',
                       'textTransform': 'uppercase',
                   }),
        ], style={'flex': '1'}),
        html.Div([
            html.Img(
                src='/assets/achieve_logo.png',
                style={
                    'width': '85px',
                    'height': 'auto',
                    'mixBlendMode': 'multiply',
                }
            )
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'flex-end',
            'paddingRight': '10px',
        })
    ], style={
        'padding': '24px 0 16px 0',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
    })

# === App Layout ===
app.layout = html.Div([
    create_header(),
    
    dcc.Tabs(id='main-tabs', value='tab-overview', children=[
        dcc.Tab(label='Overview', value='tab-overview', 
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label='Vintage Deep Dive', value='tab-vintage-deepdive',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label='In-Period Deep Dive', value='tab-inperiod-deepdive',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE),
    ], style={
        'marginBottom': '24px',
        'borderBottom': f"1px solid {COLORS['border']}",
    }),
    
    html.Div(id='tab-content'),
    
    dcc.Store(id='executive-data-store'),
    dcc.Store(id='deepdive-data-store'),
    
], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '20px 40px'})

# === Callbacks ===
@callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(tab):
    """Render content based on selected tab"""
    if tab == 'tab-overview':
        return create_overview_tab()
    elif tab == 'tab-vintage-deepdive':
        return create_vintage_deepdive_tab()
    elif tab == 'tab-inperiod-deepdive':
        return create_inperiod_deepdive_tab()
    return html.Div()

def create_overview_tab():
    """Create Overview tab content with FAS Day 7 metrics"""
    section_content_map = {
        'key-metrics': 'overview-metrics-row',
        'what-changed': 'what-changed-row',
        'weekly-trends': 'weekly-trends-row',
        'vintage-analysis': 'vintage-analysis-row',
        'daily-comparison': 'daily-comparison-row',
    }
    
    sections = []
    for config in OVERVIEW_LAYOUT_CONFIG:
        if config['visible']:
            sections.append(
                create_collapsible_section(
                    section_id=config['id'],
                    title=config['title'],
                    subtitle=config.get('subtitle'),
                    content_id=section_content_map.get(config['id']),
                    default_collapsed=config.get('collapsed', False)
                )
            )
    
    return html.Div([
        html.Div([
            html.Label("Target Week:", style={'color': COLORS['text'], 'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='overview-week-selector',
                options=[{'label': w, 'value': w} for w in get_available_weeks()],
                value=get_available_weeks()[0] if get_available_weeks() else None,
                style={'width': '200px', 'color': '#1a1a2e'},
                className='dash-dropdown-dark'
            ),
            html.Button(
                "Expand All",
                id='expand-all-overview',
                n_clicks=0,
                style={
                    'marginLeft': '20px',
                    'backgroundColor': COLORS['card_bg'],
                    'border': f"1px solid {COLORS['border']}",
                    'borderRadius': '6px',
                    'color': COLORS['text'],
                    'padding': '8px 16px',
                    'cursor': 'pointer',
                    'fontSize': '12px',
                }
            ),
            html.Button(
                "Collapse All",
                id='collapse-all-overview',
                n_clicks=0,
                style={
                    'marginLeft': '8px',
                    'backgroundColor': COLORS['card_bg'],
                    'border': f"1px solid {COLORS['border']}",
                    'borderRadius': '6px',
                    'color': COLORS['text'],
                    'padding': '8px 16px',
                    'cursor': 'pointer',
                    'fontSize': '12px',
                }
            ),
        ], style={'marginBottom': '24px', 'display': 'flex', 'alignItems': 'center'}),
        
        
        html.Div(sections),
    ])

def create_vintage_deepdive_tab():
    """Create Vintage Deeper Dive tab content with FAS Day 7 logic"""
    section_content_map = {
        'vintage-kpi': 'vintage-kpi-row',
        'vintage-lvc': 'vintage-lvc-row',
        'vintage-channel-v2': 'vintage-channel-v2-row',
        'vintage-channel': 'vintage-channel-row',
        'vintage-persona': 'vintage-persona-row',
        'vintage-outreach': 'vintage-outreach-row',
        'vintage-ma': 'vintage-ma-row',
    }
    
    sections = []
    for config in VINTAGE_DEEPDIVE_CONFIG:
        if config['visible']:
            sections.append(
                create_collapsible_section(
                    section_id=config['id'],
                    title=config['title'],
                    subtitle=config.get('subtitle'),
                    content_id=section_content_map.get(config['id']),
                    default_collapsed=config.get('collapsed', False)
                )
            )
    
    return html.Div([
        html.Div([
            html.Label("Select Week:", style={'color': COLORS['text'], 'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='vintage-week-selector',
                options=[{'label': w, 'value': w} for w in get_available_weeks()],
                value=get_available_weeks()[0] if get_available_weeks() else None,
                style={'width': '200px', 'color': '#1a1a2e'},
                className='dash-dropdown-dark'
            ),
            html.Button(
                "Expand All",
                id='expand-all-vintage',
                n_clicks=0,
                style={
                    'marginLeft': '20px',
                    'backgroundColor': COLORS['card_bg'],
                    'border': f"1px solid {COLORS['border']}",
                    'borderRadius': '6px',
                    'color': COLORS['text'],
                    'padding': '8px 16px',
                    'cursor': 'pointer',
                    'fontSize': '12px',
                }
            ),
            html.Button(
                "Collapse All",
                id='collapse-all-vintage',
                n_clicks=0,
                style={
                    'marginLeft': '8px',
                    'backgroundColor': COLORS['card_bg'],
                    'border': f"1px solid {COLORS['border']}",
                    'borderRadius': '6px',
                    'color': COLORS['text'],
                    'padding': '8px 16px',
                    'cursor': 'pointer',
                    'fontSize': '12px',
                }
            ),
        ], style={'marginBottom': '24px', 'display': 'flex', 'alignItems': 'center'}),
        html.Div(sections),
        dcc.Store(id='vintage-ma-export-store'),
        dcc.Download(id='vintage-ma-download'),
    ])

def create_inperiod_deepdive_tab():
    """Create In-Period Deeper Dive tab - metrics based on event occurrence dates"""
    import sys
    print("[IN-PERIOD] Creating In-Period tab layout...", flush=True)
    sys.stdout.flush()
    section_content_map = {
        'inperiod-kpi': 'inperiod-kpi-row',
        'inperiod-lvc': 'inperiod-lvc-row',
        'inperiod-channel': 'inperiod-channel-row',
        'inperiod-ma': 'inperiod-ma-row',
    }
    
    sections = []
    for config in INPERIOD_DEEPDIVE_CONFIG:
        if config['visible']:
            sections.append(
                create_collapsible_section(
                    section_id=config['id'],
                    title=config['title'],
                    subtitle=config.get('subtitle'),
                    content_id=section_content_map.get(config['id']),
                    default_collapsed=config.get('collapsed', False)
                )
            )
    
    return html.Div([
        html.Div([
            html.Label("Select Week:", style={'color': COLORS['text'], 'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='inperiod-week-selector',
                options=[{'label': w, 'value': w} for w in get_available_weeks()],
                value=get_available_weeks()[0] if get_available_weeks() else None,
                style={'width': '200px', 'color': '#1a1a2e'},
                className='dash-dropdown-dark'
            ),
        ], style={'marginBottom': '24px', 'display': 'flex', 'alignItems': 'center'}),
        html.Div(sections),
    ])

@callback(
    [Output('overview-metrics-row', 'children'),
     Output('what-changed-row', 'children'),
     Output('weekly-trends-row', 'children'),
     Output('vintage-analysis-row', 'children'),
     Output('daily-comparison-row', 'children')],
    Input('overview-week-selector', 'value')
)
def update_overview_tab(target_week):
    """Update all overview tab components with FAS Day 7 logic"""
    if not target_week:
        return [html.Div("Select a week", style={'color': COLORS['text']})] * 5
    
    try:
        df_summary = get_day7_summary_data(target_week)
        df_vintage = get_vintage_summary_data(target_week)
        df_daily = get_daily_comparison_data(target_week)
        df_changes = get_what_changed_data(target_week)
        
        target_date = pd.to_datetime(target_week)
        
        # === Key Metrics (FAS Day 7) ===
        metrics_row = html.Div("Loading metrics...", style={'color': COLORS['text']})
        if not df_summary.empty:
            df_summary['vintage_week'] = pd.to_datetime(df_summary['vintage_week'])
            df_summary['fas_day7_rate'] = df_summary['fas_day7_count'] / df_summary['day7_eligible_sts'].replace(0, 1) * 100
            df_summary['fas_day7_per_sts'] = df_summary['fas_day7_dollars'] / df_summary['sts_eligible'].replace(0, 1)
            
            target_data = df_summary[df_summary['vintage_week'] == target_date]
            prev_data = df_summary[(df_summary['vintage_week'] < target_date) & 
                                   (df_summary['vintage_week'] >= target_date - timedelta(weeks=4))]
            
            if len(target_data) > 0 and len(prev_data) > 0:
                target_row = target_data.iloc[0]
                prev_avg = prev_data[['fas_day7_count', 'fas_day7_dollars', 'avg_loan', 'sts_eligible', 'fas_day7_rate', 'fas_day7_per_sts']].mean()
                
                fas_delta = ((target_row['fas_day7_count'] - prev_avg['fas_day7_count']) / prev_avg['fas_day7_count'] * 100) if prev_avg['fas_day7_count'] > 0 else 0
                dollar_delta = target_row['fas_day7_dollars'] - prev_avg['fas_day7_dollars']
                rate_delta = target_row['fas_day7_rate'] - prev_avg['fas_day7_rate']
                avg_loan_delta = ((target_row['avg_loan'] - prev_avg['avg_loan']) / prev_avg['avg_loan'] * 100) if prev_avg['avg_loan'] > 0 else 0
                fas_per_sts_delta = ((target_row['fas_day7_per_sts'] - prev_avg['fas_day7_per_sts']) / prev_avg['fas_day7_per_sts'] * 100) if prev_avg['fas_day7_per_sts'] > 0 else 0
                sts_delta = ((target_row['sts_eligible'] - prev_avg['sts_eligible']) / prev_avg['sts_eligible'] * 100) if prev_avg['sts_eligible'] > 0 else 0
                
                metrics_row = dbc.Row([
                    dbc.Col(create_metric_card("FAS Day 7 Qty", f"{target_row['fas_day7_count']:,.0f}", fas_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("FAS Day 7 $", f"${target_row['fas_day7_dollars']/1000000:.2f}M", dollar_delta/1000000, "M vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("Avg Loan", f"${target_row['avg_loan']:,.0f}" if pd.notna(target_row['avg_loan']) else "N/A", avg_loan_delta if pd.notna(target_row['avg_loan']) else None, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("FAS Day 7 Rate", f"{target_row['fas_day7_rate']:.1f}%", rate_delta, "pp vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("FAS Day 7 $/StS", f"${target_row['fas_day7_per_sts']:,.0f}", fas_per_sts_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("StS (Eligible)", f"{target_row['sts_eligible']:,.0f}", sts_delta, "% vs 4wk avg"), width=2),
                ])
        
        # === What Changed Section ===
        what_changed_row = html.Div("Loading what changed...", style={'color': COLORS['text']})
        if not df_changes.empty:
            df_changes['vintage_week'] = pd.to_datetime(df_changes['vintage_week'])
            df_target = df_changes[df_changes['vintage_week'] == target_date]
            df_prev = df_changes[(df_changes['vintage_week'] < target_date) & 
                                (df_changes['vintage_week'] >= target_date - timedelta(weeks=4))]
            
            lvc_target_agg = df_target.groupby('lvc_group').agg({
                'fas_day7_count': 'sum',
                'fas_day7_dollars': 'sum'
            }).reset_index()
            lvc_prev_agg = df_prev.groupby('lvc_group').agg({
                'fas_day7_count': 'sum',
                'fas_day7_dollars': 'sum'
            }).reset_index()
            lvc_prev_agg['fas_day7_count'] = lvc_prev_agg['fas_day7_count'] / 4
            lvc_prev_agg['fas_day7_dollars'] = lvc_prev_agg['fas_day7_dollars'] / 4
            
            lvc_total_target = lvc_target_agg['fas_day7_count'].sum()
            lvc_total_prev = lvc_prev_agg['fas_day7_count'].sum()
            lvc_total_dollars_target = lvc_target_agg['fas_day7_dollars'].sum()
            lvc_total_dollars_prev = lvc_prev_agg['fas_day7_dollars'].sum()
            
            lvc_data = []
            for lvc in ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']:
                tgt_row = lvc_target_agg[lvc_target_agg['lvc_group'] == lvc]
                prv_row = lvc_prev_agg[lvc_prev_agg['lvc_group'] == lvc]
                
                tgt_fas = tgt_row['fas_day7_count'].values[0] if len(tgt_row) > 0 else 0
                prv_fas = prv_row['fas_day7_count'].values[0] if len(prv_row) > 0 else 0
                tgt_dollars = tgt_row['fas_day7_dollars'].values[0] if len(tgt_row) > 0 else 0
                prv_dollars = prv_row['fas_day7_dollars'].values[0] if len(prv_row) > 0 else 0
                
                tgt_pct = (tgt_fas / lvc_total_target * 100) if lvc_total_target > 0 else 0
                prv_pct = (prv_fas / lvc_total_prev * 100) if lvc_total_prev > 0 else 0
                tgt_avg = tgt_dollars / tgt_fas if tgt_fas > 0 else 0
                prv_avg = prv_dollars / prv_fas if prv_fas > 0 else 0
                
                lvc_data.append({
                    'LVC Group': lvc, 
                    'FAS Day 7': tgt_fas,
                    'Mix %': tgt_pct, 
                    '$FAS Day 7': tgt_dollars,
                    'Avg $FAS': tgt_avg,
                    'Δ Mix': tgt_pct - prv_pct,
                    'Δ Avg': ((tgt_avg - prv_avg) / prv_avg * 100) if prv_avg > 0 else 0
                })
            
            df_lvc = pd.DataFrame(lvc_data)
            
            avg_loan_target = lvc_total_dollars_target / lvc_total_target if lvc_total_target > 0 else 0
            avg_loan_prev = lvc_total_dollars_prev / lvc_total_prev if lvc_total_prev > 0 else 0
            avg_loan_change = ((avg_loan_target - avg_loan_prev) / avg_loan_prev * 100) if avg_loan_prev > 0 else 0
            
            what_changed_row = html.Div([
                html.Div([
                    html.H5("Week Summary & Insights", style={'color': COLORS['text'], 'marginBottom': '28px', 'fontSize': '24px', 'fontWeight': '600', 'textAlign': 'center'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("TARGET WEEK", style={'color': '#ffffff', 'fontSize': '14px', 'display': 'block', 'marginBottom': '12px', 'letterSpacing': '1px', 'fontWeight': '500'}),
                                html.Span(target_week, style={'color': COLORS['primary'], 'fontSize': '36px', 'fontWeight': '700'}),
                            ], style={'textAlign': 'center'})
                        ], width=2, className='d-flex align-items-center justify-content-center', style={'borderRight': f"1px solid {COLORS['border']}"}),
                        dbc.Col([
                            html.Div([
                                html.Span("CURRENT WEEK METRICS", style={'color': '#ffffff', 'fontSize': '14px', 'display': 'block', 'marginBottom': '16px', 'letterSpacing': '1px', 'fontWeight': '500', 'textAlign': 'center'}),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.Span("Total FAS Day 7", style={'color': COLORS['text_muted'], 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                                            html.Span(f"{lvc_total_target:,.0f}", style={'color': COLORS['primary'], 'fontSize': '22px', 'fontWeight': '600'}),
                                        ], style={'textAlign': 'center'})
                                    ], width=4),
                                    dbc.Col([
                                        html.Div([
                                            html.Span("Total $FAS Day 7", style={'color': COLORS['text_muted'], 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                                            html.Span(f"${lvc_total_dollars_target/1000000:.2f}M", style={'color': COLORS['primary'], 'fontSize': '22px', 'fontWeight': '600'}),
                                        ], style={'textAlign': 'center'})
                                    ], width=4),
                                    dbc.Col([
                                        html.Div([
                                            html.Span("Avg $FAS", style={'color': COLORS['text_muted'], 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                                            html.Span(f"${lvc_total_dollars_target/lvc_total_target:,.0f}" if lvc_total_target > 0 else "N/A", style={'color': COLORS['primary'], 'fontSize': '22px', 'fontWeight': '600'}),
                                        ], style={'textAlign': 'center'})
                                    ], width=4),
                                ])
                            ], style={'paddingLeft': '20px', 'paddingRight': '20px'})
                        ], width=5, style={'borderRight': f"1px solid {COLORS['border']}"}),
                        dbc.Col([
                            html.Div([
                                html.Span("VS PREVIOUS 4 WEEKS", style={'color': '#ffffff', 'fontSize': '14px', 'display': 'block', 'marginBottom': '16px', 'letterSpacing': '1px', 'fontWeight': '500', 'textAlign': 'center'}),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.Span("FAS Day 7 Δ", style={'color': COLORS['text_muted'], 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                                            html.Span(f"{lvc_total_target - lvc_total_prev:+,.0f} ({(lvc_total_target - lvc_total_prev)/lvc_total_prev*100:+.1f}%)" if lvc_total_prev > 0 else "N/A", 
                                                      style={'color': COLORS['success'] if lvc_total_target > lvc_total_prev else COLORS['danger'], 'fontSize': '18px', 'fontWeight': '600'}),
                                        ], style={'textAlign': 'center'})
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.Span("$FAS Day 7 Δ", style={'color': COLORS['text_muted'], 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                                            html.Span(f"${(lvc_total_dollars_target - lvc_total_dollars_prev)/1000000:+.2f}M" if lvc_total_dollars_prev > 0 else "N/A", 
                                                      style={'color': COLORS['success'] if lvc_total_dollars_target > lvc_total_dollars_prev else COLORS['danger'], 'fontSize': '18px', 'fontWeight': '600'}),
                                        ], style={'textAlign': 'center'})
                                    ], width=6),
                                ])
                            ], style={'paddingLeft': '20px', 'paddingRight': '20px'})
                        ], width=5),
                    ], className='align-items-center')
                ], style={**CARD_STYLE, 'marginBottom': '20px', 'padding': '28px'}),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H5("LVC Performance (FAS Day 7)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                            dash_table.DataTable(
                                data=df_lvc.to_dict('records'),
                                columns=[
                                    {'name': 'LVC', 'id': 'LVC Group'},
                                    {'name': 'FAS Day 7', 'id': 'FAS Day 7', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                                    {'name': 'Mix %', 'id': 'Mix %', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                    {'name': '$FAS Day 7', 'id': '$FAS Day 7', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
                                    {'name': 'Avg $FAS', 'id': 'Avg $FAS', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
                                    {'name': 'Δ Mix', 'id': 'Δ Mix', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                                    {'name': 'Δ Avg %', 'id': 'Δ Avg', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                                ],
                                style_table={'backgroundColor': COLORS['card_bg'], 'overflowX': 'auto'},
                                style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '8px', 'fontSize': '13px'},
                                style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '12px'},
                                style_data_conditional=[
                                    {'if': {'filter_query': '{Δ Mix} > 0', 'column_id': 'Δ Mix'}, 'color': COLORS['success']},
                                    {'if': {'filter_query': '{Δ Mix} < 0', 'column_id': 'Δ Mix'}, 'color': COLORS['danger']},
                                    {'if': {'filter_query': '{Δ Avg} > 0', 'column_id': 'Δ Avg'}, 'color': COLORS['success']},
                                    {'if': {'filter_query': '{Δ Avg} < 0', 'column_id': 'Δ Avg'}, 'color': COLORS['danger']},
                                ]
                            )
                        ], style=CARD_STYLE)
                    ], width=12),
                ])
            ])
        
        # === Weekly Trends ===
        weekly_trends_row = html.Div("Loading trends...", style={'color': COLORS['text']})
        if not df_summary.empty:
            df_summary_sorted = df_summary.sort_values('vintage_week')
            df_summary_sorted['week_label'] = df_summary_sorted['vintage_week'].dt.strftime('%-m/%d')
            df_summary_sorted['fas_day7_dollars_m'] = df_summary_sorted['fas_day7_dollars'] / 1000000
            
            fig_fas_count = create_bar_chart_with_value(
                df_summary_sorted, 'week_label', 'fas_day7_count',
                'FAS Day 7 Quantity by Week', color=COLORS['chart_cyan'], y_format=',.0f'
            )
            
            fig_fas_dollars = create_bar_chart_with_value(
                df_summary_sorted, 'week_label', 'fas_day7_dollars_m',
                'FAS Day 7 $ by Week (Millions)', color=COLORS['chart_green'], y_format='.2f', value_prefix='$', value_suffix='M'
            )
            
            fig_fas_rate = create_bar_chart_with_value(
                df_summary_sorted, 'week_label', 'fas_day7_rate',
                'FAS Day 7 Rate by Week', color=COLORS['chart_yellow'], y_format='.1f', value_suffix='%',
                formula_text='(FAS Day 7 / StS Eligible × 100)'
            )
            
            weekly_trends_row = dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_fas_count)], style=CARD_STYLE), width=4),
                dbc.Col(html.Div([dcc.Graph(figure=fig_fas_dollars)], style=CARD_STYLE), width=4),
                dbc.Col(html.Div([dcc.Graph(figure=fig_fas_rate)], style=CARD_STYLE), width=4),
            ])
        
        # === Vintage Analysis ===
        vintage_analysis_row = html.Div("Loading vintage analysis...", style={'color': COLORS['text']})
        if not df_vintage.empty:
            df_vintage['vintage_week'] = pd.to_datetime(df_vintage['vintage_week'])
            df_vintage['fas_day7_rate'] = df_vintage['fas_day7_count'] / df_vintage['day7_eligible_sts'].replace(0, 1) * 100
            df_vintage['week_label'] = df_vintage['vintage_week'].dt.strftime('%-m/%d')
            df_vintage_sorted = df_vintage.sort_values('vintage_week')
            
            target_vintage = df_vintage[df_vintage['vintage_week'] == target_date]
            prev_vintage = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                                      (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))]
            
            if len(target_vintage) > 0 and len(prev_vintage) > 0:
                tgt = target_vintage.iloc[0]
                prev_avg = prev_vintage[['total_leads', 'sts_eligible', 'fas_day7_count', 'fas_day7_dollars', 'avg_loan', 'fas_day7_rate', 'avg_lp2c']].mean()
                
                leads_delta = ((tgt['total_leads'] - prev_avg['total_leads']) / prev_avg['total_leads'] * 100) if prev_avg['total_leads'] > 0 else 0
                sts_delta = ((tgt['sts_eligible'] - prev_avg['sts_eligible']) / prev_avg['sts_eligible'] * 100) if prev_avg['sts_eligible'] > 0 else 0
                fas_delta = ((tgt['fas_day7_count'] - prev_avg['fas_day7_count']) / prev_avg['fas_day7_count'] * 100) if prev_avg['fas_day7_count'] > 0 else 0
                fas_rate_delta = tgt['fas_day7_rate'] - prev_avg['fas_day7_rate']
                fas_dollar_delta = ((tgt['fas_day7_dollars'] - prev_avg['fas_day7_dollars']) / prev_avg['fas_day7_dollars'] * 100) if prev_avg['fas_day7_dollars'] > 0 else 0
                avg_loan_delta = ((tgt['avg_loan'] - prev_avg['avg_loan']) / prev_avg['avg_loan'] * 100) if prev_avg['avg_loan'] > 0 and pd.notna(tgt['avg_loan']) else 0
                
                vintage_metrics = dbc.Row([
                    dbc.Col(create_metric_card("Vintage Leads", f"{tgt['total_leads']:,.0f}", leads_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("StS (Eligible)", f"{tgt['sts_eligible']:,.0f}", sts_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("FAS Day 7 Qty", f"{tgt['fas_day7_count']:,.0f}", fas_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("FAS Day 7 $", f"${tgt['fas_day7_dollars']/1000000:.2f}M", fas_dollar_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("FAS Day 7 Rate", f"{tgt['fas_day7_rate']:.1f}%", fas_rate_delta, "pp vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("Avg Loan", f"${tgt['avg_loan']:,.0f}" if pd.notna(tgt['avg_loan']) else "N/A", avg_loan_delta if pd.notna(tgt['avg_loan']) else None, "% vs 4wk avg"), width=2),
                ])
                
                fig_vintage_trend = create_bar_chart_with_value(
                    df_vintage_sorted, 'week_label', 'fas_day7_rate',
                    'Vintage FAS Day 7 Rate by Week',
                    color=COLORS['chart_cyan'],
                    y_format='.1f',
                    formula_text='(FAS Day 7 / StS Eligible × 100)',
                    value_suffix='%'
                )
                
                vintage_analysis_row = html.Div([
                    html.H5("Target Week Vintage vs Previous 4 Weeks", style={'color': COLORS['text'], 'marginBottom': '16px', 'fontSize': '16px', 'fontWeight': '600'}),
                    vintage_metrics,
                    html.Div(style={'marginTop': '24px'}),
                    dbc.Row([
                        dbc.Col(html.Div([dcc.Graph(figure=fig_vintage_trend)], style=CARD_STYLE), width=12),
                    ])
                ])
        
        # === Daily Comparison ===
        daily_comparison_row = html.Div("Loading daily comparison...", style={'color': COLORS['text']})
        if not df_daily.empty:
            df_daily['lead_date'] = pd.to_datetime(df_daily['lead_date'])
            df_daily['week_start'] = pd.to_datetime(df_daily['week_start'])
            df_daily['fas_day7_rate'] = df_daily['fas_day7_count'] / df_daily['day7_eligible_sts'].replace(0, 1) * 100
            
            target_daily = df_daily[df_daily['week_start'] == target_date].copy()
            prev_daily = df_daily[(df_daily['week_start'] < target_date) & 
                                 (df_daily['week_start'] >= target_date - timedelta(weeks=4))].copy()
            
            if len(target_daily) > 0 and len(prev_daily) > 0:
                prev_by_dow = prev_daily.groupby('day_of_week').agg({
                    'total_leads': 'mean',
                    'sts_eligible': 'mean',
                    'day7_eligible_sts': 'mean',
                    'fas_day7_count': 'mean',
                    'fas_day7_dollars': 'mean'
                }).reset_index()
                prev_by_dow['fas_day7_rate'] = prev_by_dow['fas_day7_count'] / prev_by_dow['day7_eligible_sts'].replace(0, 1) * 100
                
                target_daily['date_label'] = target_daily['lead_date'].dt.strftime('%a %m/%d')
                target_daily = target_daily.sort_values('day_of_week')
                
                comparison_df = target_daily.merge(
                    prev_by_dow[['day_of_week', 'fas_day7_count', 'fas_day7_rate']],
                    on='day_of_week',
                    suffixes=('', '_prev')
                )
                comparison_df['fas_delta'] = comparison_df['fas_day7_count'] - comparison_df['fas_day7_count_prev']
                
                fig_daily_fas = create_grouped_bar_chart(
                    comparison_df, 
                    'date_label', 
                    ['fas_day7_count', 'fas_day7_count_prev'], 
                    '📊 Daily FAS Day 7: Target Week vs Prev 4-Wk Avg',
                    colors=[COLORS['chart_cyan'], COLORS['chart_gray']]
                )
                fig_daily_fas.update_traces(name='Target Week', selector=dict(name='fas_day7_count'))
                fig_daily_fas.update_traces(name='Prev 4-Wk Avg', selector=dict(name='fas_day7_count_prev'))
                
                daily_comparison_row = dbc.Row([
                    dbc.Col(html.Div([dcc.Graph(figure=fig_daily_fas)], style=CARD_STYLE), width=8),
                    dbc.Col(html.Div([
                        html.H5("Daily Comparison", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                        dash_table.DataTable(
                            data=comparison_df[['date_label', 'fas_day7_count', 'fas_day7_count_prev', 'fas_delta']].round(1).to_dict('records'),
                            columns=[
                                {'name': 'Day', 'id': 'date_label'},
                                {'name': 'FAS Day 7', 'id': 'fas_day7_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                                {'name': 'Prev Avg', 'id': 'fas_day7_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
                                {'name': 'Δ', 'id': 'fas_delta', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                            ],
                            style_table={'backgroundColor': COLORS['card_bg']},
                            style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '8px'},
                            style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {'if': {'filter_query': '{fas_delta} > 0', 'column_id': 'fas_delta'}, 'color': COLORS['success']},
                                {'if': {'filter_query': '{fas_delta} < 0', 'column_id': 'fas_delta'}, 'color': COLORS['danger']},
                            ]
                        )
                    ], style=CARD_STYLE), width=4),
                ])
        
        return metrics_row, what_changed_row, weekly_trends_row, vintage_analysis_row, daily_comparison_row
    
    except Exception as e:
        error_div = html.Div([
            html.H4("Error loading data", style={'color': COLORS['danger']}),
            html.P(f"Error: {str(e)}", style={'color': COLORS['text_muted']})
        ], style=CARD_STYLE)
        return [error_div] * 5

# === Vintage Deep Dive Callback with FAS Day 7 ===
def create_wow_comparison_tables(df_in, grp_col, grp_label, is_channel=False):
    """Builds WoW Comparison tables for FAS Day 7%, FAS Day 7$, Avg LP2C, Avg LVC Mix, and Lead Volume."""
    df = df_in.copy()
    if 'vintage_week' not in df.columns:
        return html.Div()
    df['_vintage_date'] = pd.to_datetime(df['vintage_week']).dt.normalize()
    weeks = sorted(df['_vintage_date'].unique())[-6:]
    if len(weeks) < 2:
        return html.Div("Not enough weeks for WoW comparison.", style={'color': COLORS['text']})
        
    agg_dict = {
        'fas_day7_count': 'sum',
        'day7_eligible_sts': 'sum',
        'fas_day7_dollars': 'sum',
        'avg_lp2c': 'mean',
        'avg_lvc': 'mean'
    }
    if is_channel:
        agg_dict['sts_eligible'] = 'sum'
    else:
        agg_dict['leads_assigned'] = 'sum'
        
    by_week = df[df['_vintage_date'].isin(weeks)].groupby([grp_col, '_vintage_date']).agg(agg_dict).reset_index()
    
    # Calculate metrics
    by_week['fas_day7_pct'] = (by_week['fas_day7_count'] / by_week['day7_eligible_sts'].replace(0, 1) * 100).round(1)
    by_week['fas_day7_dollars'] = by_week['fas_day7_dollars'].round(0)
    by_week['avg_lp2c'] = by_week['avg_lp2c'].round(1)
    by_week['avg_lvc'] = by_week['avg_lvc'].round(1)
    if is_channel:
        by_week['lead_vol'] = by_week['sts_eligible']
    else:
        by_week['lead_vol'] = by_week['leads_assigned']
        
    # Determine Top 20 by latest week FAS $ to match tables
    latest_week = weeks[-1]
    top_grps = by_week[by_week['_vintage_date'] == latest_week].sort_values('fas_day7_dollars', ascending=False).head(20)[grp_col].tolist()
    by_week = by_week[by_week[grp_col].isin(top_grps)]
    
    metrics = [
        {'id': 'fas_day7_pct', 'name': 'FAS Day 7 %', 'format': {'specifier': '.1f'}},
        {'id': 'fas_day7_dollars', 'name': 'FAS Day 7 $', 'format': {'specifier': '$,.0f'}},
        {'id': 'avg_lp2c', 'name': 'Avg LP2C', 'format': {'specifier': '.1f'}},
        {'id': 'avg_lvc', 'name': 'Avg LVC Mix', 'format': {'specifier': '.1f'}},
        {'id': 'lead_vol', 'name': 'StS Eligible' if is_channel else 'Leads Assigned', 'format': {'specifier': ',.0f'}},
    ]
    
    week_strs = [pd.to_datetime(w).strftime('%-m/%-d') for w in weeks]
    
    tables = []
    for m in metrics:
        metric_id = m['id']
        m_name = m['name']
        
        pivot = by_week.pivot(index=grp_col, columns='_vintage_date', values=metric_id).reset_index()
        pivot[grp_col] = pd.Categorical(pivot[grp_col], categories=top_grps, ordered=True)
        pivot = pivot.sort_values(grp_col)
        
        cols = [{'name': [m_name, grp_label], 'id': 'grp'}]
        for w, w_str in zip(weeks, week_strs):
            cols.append({'name': [w_str, m_name], 'id': f'val_{w_str}', 'type': 'numeric', 'format': m['format']})
        
        cols.append({'name': ['', ''], 'id': 'spacer'})
        cols.append({'name': ['', grp_label], 'id': 'grp_repeat'})
        
        for i in range(1, len(weeks)):
            w_str = week_strs[i]
            cols.append({'name': [w_str, 'WoW Diff'], 'id': f'wow_{w_str}', 'type': 'numeric', 'format': {'specifier': '+.2%'}})
        
        rows = []
        for _, row in pivot.iterrows():
            r = {'grp': row[grp_col] if row[grp_col] else 'Unknown', 'spacer': '', 'grp_repeat': row[grp_col] if row[grp_col] else 'Unknown'}
            prev_val = None
            for i, (w, w_str) in enumerate(zip(weeks, week_strs)):
                val = row.get(w, None)
                r[f'val_{w_str}'] = val
                
                if i > 0:
                    if prev_val is not None and not pd.isna(prev_val) and val is not None and not pd.isna(val) and float(prev_val) != 0:
                        r[f'wow_{w_str}'] = (val - prev_val) / float(prev_val)
                    else:
                        r[f'wow_{w_str}'] = None
                prev_val = val
            rows.append(r)
            
        style_data_conditional = []
        # Positive changes are green, negative are red
        for i in range(1, len(weeks)):
            w_str = week_strs[i]
            style_data_conditional.extend([
                {'if': {'column_id': f'wow_{w_str}', 'filter_query': f'{{wow_{w_str}}} > 0'}, 'backgroundColor': 'rgba(40, 167, 69, 0.3)', 'color': '#FFFFFF', 'fontWeight': 'bold'},
                {'if': {'column_id': f'wow_{w_str}', 'filter_query': f'{{wow_{w_str}}} < 0'}, 'backgroundColor': 'rgba(220, 53, 69, 0.3)', 'color': '#FFFFFF', 'fontWeight': 'bold'},
                {'if': {'column_id': f'wow_{w_str}', 'filter_query': f'{{wow_{w_str}}} = 0'}, 'color': COLORS['text_muted']}
            ])
            
        table = html.Div([
            dash_table.DataTable(
                data=rows,
                columns=cols,
                style_table={'backgroundColor': COLORS['card_bg'], 'minWidth': '100%', 'marginBottom': '20px', 'overflowX': 'auto'},
                style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '8px', 'fontSize': '11px'},
                style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '11px', 'textAlign': 'center'},
                style_data_conditional=style_data_conditional,
                merge_duplicate_headers=True
            )
        ], style=CARD_STYLE)
        tables.append(table)
        
    return html.Div([
        html.H5("6-Week WoW Comparison", style={'color': COLORS['text'], 'marginBottom': '15px', 'marginTop': '20px', 'fontSize': '16px', 'fontWeight': '600'}),
        *tables
    ])

def create_ltpl_pricing_wow_table(df_in):
    """Builds WoW Comparison tables for LT PL Competitor Pricing by Credit Range."""
    df = df_in.copy()
    if 'vintage_week' not in df.columns or df.empty:
        return html.Div("No LT PL Pricing Data Available.", style={'color': COLORS['text_muted'], 'marginTop': '20px'})
        
    df['_vintage_date'] = pd.to_datetime(df['vintage_week']).dt.normalize()
    weeks = sorted(df['_vintage_date'].unique())[-12:]
    if len(weeks) < 2:
        return html.Div("Not enough weeks for LT PL Pricing WoW comparison.", style={'color': COLORS['text_muted'], 'marginTop': '20px'})
        
    # Filter out empty credit ranges and sort them properly (assuming string ranges like '620-639')
    df = df[df['credit_range'] != 'Unknown']
    credit_ranges = sorted(df['credit_range'].unique())
    
    metrics = [
        {'id': 'avg_secured_competitor_apr', 'name': 'Avg Competitor APR (Secured)'},
        {'id': 'avg_achieve_loan_apr', 'name': 'Avg AHL APR'}
    ]
    
    week_strs = [pd.to_datetime(w).strftime('%-m/%-d') for w in weeks]
    
    cols = [
        {'name': ['', 'Credit Range'], 'id': 'credit_range'},
        {'name': ['', 'Metric'], 'id': 'metric'}
    ]
    for w, w_str in zip(weeks, week_strs):
        cols.append({'name': [w_str, 'Value'], 'id': f'val_{w_str}', 'type': 'numeric', 'format': {'specifier': '.1f'}})
    
    cols.append({'name': ['', ''], 'id': 'spacer'})
    
    for i in range(1, len(weeks)):
        w_str = week_strs[i]
        cols.append({'name': [w_str, 'WoW Diff'], 'id': f'wow_{w_str}', 'type': 'numeric', 'format': {'specifier': '+.1f'}})
        
    rows = []
    
    for cr in credit_ranges:
        cr_data = df[df['credit_range'] == cr].set_index('_vintage_date')
        
        for m in metrics:
            metric_id = m['id']
            m_name = m['name']
            
            r = {'credit_range': cr, 'metric': m_name, 'spacer': ''}
            prev_val = None
            
            for i, (w, w_str) in enumerate(zip(weeks, week_strs)):
                val = cr_data.at[w, metric_id] if w in cr_data.index else None
                r[f'val_{w_str}'] = val
                
                if i > 0:
                    if prev_val is not None and not pd.isna(prev_val) and val is not None and not pd.isna(val):
                        # Calculate absolute difference (pp) for APRs
                        r[f'wow_{w_str}'] = val - prev_val
                    else:
                        r[f'wow_{w_str}'] = None
                prev_val = val
                
            rows.append(r)
            
    style_data_conditional = []
    # For APRs, lower is generally better for the customer, but let's just color code jumps
    # Assuming positive jump is red (higher APR), negative is green (lower APR)
    for i in range(1, len(weeks)):
        w_str = week_strs[i]
        style_data_conditional.extend([
            {'if': {'column_id': f'wow_{w_str}', 'filter_query': f'{{wow_{w_str}}} < 0'}, 'backgroundColor': 'rgba(40, 167, 69, 0.3)', 'color': '#FFFFFF', 'fontWeight': 'bold'},
            {'if': {'column_id': f'wow_{w_str}', 'filter_query': f'{{wow_{w_str}}} > 0'}, 'backgroundColor': 'rgba(220, 53, 69, 0.3)', 'color': '#FFFFFF', 'fontWeight': 'bold'},
            {'if': {'column_id': f'wow_{w_str}', 'filter_query': f'{{wow_{w_str}}} = 0'}, 'color': COLORS['text_muted']}
        ])
        
    table = html.Div([
        dash_table.DataTable(
            data=rows,
            columns=cols,
            style_table={'backgroundColor': COLORS['card_bg'], 'minWidth': '100%', 'marginBottom': '20px', 'overflowX': 'auto'},
            style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '8px', 'fontSize': '11px'},
            style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '11px', 'textAlign': 'center'},
            style_data_conditional=style_data_conditional,
            merge_duplicate_headers=True,
            style_cell_conditional=[
                {'if': {'column_id': 'credit_range'}, 'textAlign': 'left', 'fontWeight': 'bold'},
                {'if': {'column_id': 'metric'}, 'textAlign': 'left'}
            ]
        )
    ], style=CARD_STYLE)

    return html.Div([
        html.H5("LT PL Channel: Competitor Pricing by Credit Range (12-Week Trend)", style={'color': COLORS['text'], 'marginBottom': '15px', 'marginTop': '20px', 'fontSize': '16px', 'fontWeight': '600'}),
        table
    ])

@callback(
    [Output('vintage-kpi-row', 'children'),
     Output('vintage-lvc-row', 'children'),
     Output('vintage-channel-v2-row', 'children'),
     Output('vintage-ma-row', 'children'),
     Output('vintage-ma-export-store', 'data')],
    Input('vintage-week-selector', 'value')
)
def update_vintage_deepdive_tab(target_week):
    """Update all vintage deep dive tab components with FAS Day 7 logic"""
    if not target_week:
        return [html.Div("Select a week")] * 4 + [None]
    
    target_date = pd.to_datetime(target_week)
    
    df_vintage = get_vintage_deep_dive_data(target_week)
    
    # === KPI Summary ===
    kpi_row = html.Div("Loading KPIs...")
    if not df_vintage.empty:
        df_vintage['vintage_week'] = pd.to_datetime(df_vintage['vintage_week'])
        df_vintage['fas_day7_rate'] = df_vintage['fas_day7_count'] / df_vintage['day7_eligible_sts'].replace(0, 1) * 100
        
        target_data = df_vintage[df_vintage['vintage_week'] == target_date]
        prev_data = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                               (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))]
        
        tgt_leads = target_data['lead_count'].sum()
        tgt_sts = target_data['sts_eligible'].sum()
        tgt_day7_sts = target_data['day7_eligible_sts'].sum()  # Use day7 eligible for rate calc
        tgt_sf_contact = target_data['sf_contact_day7_count'].sum()
        tgt_sf_contact_pct = (tgt_sf_contact / tgt_day7_sts * 100) if tgt_day7_sts > 0 else 0
        tgt_fas = target_data['fas_day7_count'].sum()
        tgt_dollars = target_data['fas_day7_dollars'].sum()
        tgt_fas_rate = (tgt_fas / tgt_day7_sts * 100) if tgt_day7_sts > 0 else 0  # Use day7 eligible
        tgt_avg_loan = (tgt_dollars / tgt_fas) if tgt_fas > 0 else 0
        
        prev_leads = prev_data['lead_count'].sum() / 4
        prev_sts = prev_data['sts_eligible'].sum() / 4
        prev_day7_sts = prev_data['day7_eligible_sts'].sum() / 4  # Use day7 eligible for rate calc
        prev_sf_contact = prev_data['sf_contact_day7_count'].sum() / 4
        prev_sf_contact_pct = (prev_sf_contact / prev_day7_sts * 100) if prev_day7_sts > 0 else 0
        prev_fas = prev_data['fas_day7_count'].sum() / 4
        prev_dollars = prev_data['fas_day7_dollars'].sum() / 4
        prev_fas_rate = (prev_fas / prev_day7_sts * 100) if prev_day7_sts > 0 else 0  # Use day7 eligible
        prev_avg_loan = (prev_dollars / prev_fas) if prev_fas > 0 else 0
        
        fas_delta = ((tgt_fas - prev_fas) / prev_fas * 100) if prev_fas > 0 else 0
        dollars_delta = ((tgt_dollars - prev_dollars) / prev_dollars * 100) if prev_dollars > 0 else 0
        rate_delta = tgt_fas_rate - prev_fas_rate
        sf_contact_delta = tgt_sf_contact_pct - prev_sf_contact_pct  # pp delta for SF Contact %
        avg_loan_delta = ((tgt_avg_loan - prev_avg_loan) / prev_avg_loan * 100) if prev_avg_loan > 0 else 0
        leads_delta = ((tgt_leads - prev_leads) / prev_leads * 100) if prev_leads > 0 else 0
        sts_delta = ((tgt_sts - prev_sts) / prev_sts * 100) if prev_sts > 0 else 0
        
        # === Calculate Key Insights for Summary ===
        # LVC Group insights (using pct_of_total to match LVC Group Analysis section)
        lvc_weekly = df_vintage.groupby(['vintage_week', 'lvc_group']).agg({'lead_count': 'sum'}).reset_index()
        week_totals = lvc_weekly.groupby('vintage_week')['lead_count'].sum().reset_index()
        week_totals.columns = ['vintage_week', 'week_total']
        lvc_weekly = lvc_weekly.merge(week_totals, on='vintage_week')
        lvc_weekly['pct_of_total'] = (lvc_weekly['lead_count'] / lvc_weekly['week_total'] * 100).round(1)
        lvc_weekly['wow_change'] = lvc_weekly.groupby('lvc_group')['pct_of_total'].diff()
        lvc_insights = []
        for group in ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer']:
            group_data = lvc_weekly[lvc_weekly['lvc_group'] == group].sort_values('vintage_week').tail(6)
            if len(group_data) >= 2:
                wow = group_data['wow_change'].dropna()
                if len(wow) > 1:
                    mean_c, std_c = wow.mean(), wow.std()
                    latest_c = wow.iloc[-1] if len(wow) > 0 else 0
                    z = (latest_c - mean_c) / std_c if std_c > 0 else 0
                    if abs(z) >= 1.495:
                        lvc_insights.append({'group': group, 'change': latest_c, 'z': z, 'dir': '↑' if latest_c > 0 else '↓', 'marginal': False, 'metric': 'Gross Lead %'})
                    elif abs(z) >= 1.0:
                        lvc_insights.append({'group': group, 'change': latest_c, 'z': z, 'dir': '↑' if latest_c > 0 else '↓', 'marginal': True, 'metric': 'Gross Lead %'})
        
        # Channel insights (top channels by StS)
        six_wk_ago = target_date - timedelta(weeks=6)
        ch_sts = df_vintage[(df_vintage['vintage_week'] >= six_wk_ago)].groupby('channel')['sts_eligible'].sum().nlargest(10).index.tolist()
        ch_weekly = df_vintage[df_vintage['channel'].isin(ch_sts)].groupby(['vintage_week', 'channel']).agg({'fas_day7_dollars': 'sum'}).reset_index()
        ch_weekly['wow_change'] = ch_weekly.groupby('channel')['fas_day7_dollars'].diff()
        ch_insights = []
        for ch in ch_sts[:10]:
            ch_data = ch_weekly[ch_weekly['channel'] == ch].sort_values('vintage_week').tail(6)
            if len(ch_data) >= 2:
                wow = ch_data['wow_change'].dropna()
                if len(wow) > 1:
                    mean_c, std_c = wow.mean(), wow.std()
                    latest_c = wow.iloc[-1] if len(wow) > 0 else 0
                    z = (latest_c - mean_c) / std_c if std_c > 0 else 0
                    if abs(z) >= 1.495:
                        ch_insights.append({'group': ch[:20], 'change': latest_c, 'z': z, 'dir': '↑' if latest_c > 0 else '↓', 'marginal': False})
                    elif abs(z) >= 1.0:
                        ch_insights.append({'group': ch[:20], 'change': latest_c, 'z': z, 'dir': '↑' if latest_c > 0 else '↓', 'marginal': True})
        
        # Get MA insights from separate query
        df_ma_summary = get_ma_performance_data(target_week)
        ma_insights = []
        starring_insights = []
        if not df_ma_summary.empty:
            df_ma_summary['vintage_week'] = pd.to_datetime(df_ma_summary['vintage_week'])
            # Starring insights
            star_weekly = df_ma_summary.groupby(['vintage_week', 'starring_group']).agg({'fas_day7_dollars': 'sum'}).reset_index()
            star_weekly['wow_change'] = star_weekly.groupby('starring_group')['fas_day7_dollars'].diff()
            for star in star_weekly['starring_group'].unique():
                star_data = star_weekly[star_weekly['starring_group'] == star].sort_values('vintage_week').tail(6)
                if len(star_data) >= 2:
                    wow = star_data['wow_change'].dropna()
                    if len(wow) > 1:
                        mean_c, std_c = wow.mean(), wow.std()
                        latest_c = wow.iloc[-1] if len(wow) > 0 else 0
                        z = (latest_c - mean_c) / std_c if std_c > 0 else 0
                        if abs(z) >= 1.495:
                            starring_insights.append({'group': f"Star {star}", 'change': latest_c, 'z': z, 'dir': '↑' if latest_c > 0 else '↓', 'marginal': False})
                        elif abs(z) >= 1.0:
                            starring_insights.append({'group': f"Star {star}", 'change': latest_c, 'z': z, 'dir': '↑' if latest_c > 0 else '↓', 'marginal': True})
            # Top MA insights
            top_mas = df_ma_summary.groupby('mortgage_advisor')['fas_day7_dollars'].sum().nlargest(15).index.tolist()
            ma_weekly = df_ma_summary[df_ma_summary['mortgage_advisor'].isin(top_mas)].groupby(['vintage_week', 'mortgage_advisor']).agg({'fas_day7_dollars': 'sum'}).reset_index()
            ma_weekly['wow_change'] = ma_weekly.groupby('mortgage_advisor')['fas_day7_dollars'].diff()
            for ma in top_mas[:10]:
                ma_data = ma_weekly[ma_weekly['mortgage_advisor'] == ma].sort_values('vintage_week').tail(6)
                if len(ma_data) >= 2:
                    wow = ma_data['wow_change'].dropna()
                    if len(wow) > 1:
                        mean_c, std_c = wow.mean(), wow.std()
                        latest_c = wow.iloc[-1] if len(wow) > 0 else 0
                        z = (latest_c - mean_c) / std_c if std_c > 0 else 0
                        if abs(z) >= 1.495:
                            ma_insights.append({'group': ma[:18], 'change': latest_c, 'z': z, 'dir': '↑' if latest_c > 0 else '↓', 'marginal': False})
                        elif abs(z) >= 1.0:
                            ma_insights.append({'group': ma[:18], 'change': latest_c, 'z': z, 'dir': '↑' if latest_c > 0 else '↓', 'marginal': True})
        
        # Sort by absolute z-score
        lvc_insights.sort(key=lambda x: abs(x['z']), reverse=True)
        ch_insights.sort(key=lambda x: abs(x['z']), reverse=True)
        starring_insights.sort(key=lambda x: abs(x['z']), reverse=True)
        ma_insights.sort(key=lambda x: abs(x['z']), reverse=True)
        
        # === Weekly Metrics Insights (4 charts) ===
        weekly_metrics_agg = df_vintage.groupby('vintage_week').agg({
            'sf_contact_day7_count': 'sum',
            'day7_eligible_sts': 'sum',
            'fas_day7_count': 'sum',
            'fas_day7_dollars': 'sum'
        }).reset_index()
        weekly_metrics_agg = weekly_metrics_agg.sort_values('vintage_week').tail(7)  # Need 7 for Z-score calc
        weekly_metrics_agg['sf_contact_day7_pct'] = (weekly_metrics_agg['sf_contact_day7_count'] / weekly_metrics_agg['day7_eligible_sts'].replace(0, 1) * 100).round(2)
        weekly_metrics_agg['contact_to_fas_pct'] = (weekly_metrics_agg['fas_day7_count'] / weekly_metrics_agg['sf_contact_day7_count'].replace(0, 1) * 100).round(2)
        weekly_metrics_agg['fas_day7_pct'] = (weekly_metrics_agg['fas_day7_count'] / weekly_metrics_agg['day7_eligible_sts'].replace(0, 1) * 100).round(2)
        
        weekly_metrics_insights = []
        metrics_config = [
            ('sf_contact_day7_pct', 'SF Contact Day 7 %', True),
            ('contact_to_fas_pct', 'Contact to FAS %', True),
            ('fas_day7_pct', 'FAS Day 7 %', True),
            ('fas_day7_dollars', 'FAS Day 7 $', False),
        ]
        for col, label, is_pct in metrics_config:
            if len(weekly_metrics_agg) >= 7:
                last_6 = weekly_metrics_agg[col].tail(7).head(6)
                latest = weekly_metrics_agg[col].iloc[-1]
                mean_val = last_6.mean()
                std_val = last_6.std()
                z = (latest - mean_val) / std_val if std_val > 0 else 0
                change = latest - mean_val
                if abs(z) >= 1.495:
                    weekly_metrics_insights.append({'group': label, 'change': change, 'z': z, 'dir': '↑' if change > 0 else '↓', 'marginal': False, 'is_pct': is_pct})
                elif abs(z) >= 1.0:
                    weekly_metrics_insights.append({'group': label, 'change': change, 'z': z, 'dir': '↑' if change > 0 else '↓', 'marginal': True, 'is_pct': is_pct})
        weekly_metrics_insights.sort(key=lambda x: abs(x['z']), reverse=True)
        
        # Build insight bullets for each section with more context
        def format_insight_bullets(insights, section_type, max_items=4):
            if not insights:
                return [html.Div("• No significant triggers", style={'color': COLORS['text_muted'], 'fontSize': '14px', 'marginBottom': '6px', 'textAlign': 'center'})]
            items = []
            for item in insights[:max_items]:
                color = COLORS['success'] if item['change'] > 0 else COLORS['danger']
                direction = "up" if item['change'] > 0 else "down"
                metric_label = item.get('metric', 'FAS Day 7 $')
                marginal_tag = " (marginal)" if item.get('marginal', False) else ""
                
                # Format value based on metric type
                if metric_label == 'Gross Lead %':
                    value_str = f"{abs(item['change']):.1f}pp"
                else:
                    value_str = f"${abs(item['change'])/1000:,.0f}K"
                
                items.append(html.Div([
                    html.Span(f"• {item['dir']} ", style={'color': color, 'fontWeight': 'bold'}),
                    html.Span(f"{item['group']}: ", style={'color': COLORS['text'], 'fontWeight': '500'}),
                    html.Span(f"{metric_label} {direction} ", style={'color': COLORS['text']}),
                    html.Span(value_str, style={'color': color, 'fontWeight': 'bold'}),
                    html.Span(f" (Z={item['z']:.1f}){marginal_tag}", style={'color': COLORS['chart_yellow'] if item.get('marginal') else COLORS['text_muted'], 'fontSize': '12px'}),
                ], style={'fontSize': '14px', 'marginBottom': '6px', 'textAlign': 'center'}))
            return items
        
        lvc_bullets = format_insight_bullets(lvc_insights, 'lvc')
        ch_bullets = format_insight_bullets(ch_insights, 'channel')
        ma_bullets = format_insight_bullets(starring_insights + ma_insights, 'ma', 4)
        
        # Format weekly metrics bullets
        def format_weekly_metrics_bullets(insights):
            if not insights:
                return [html.Div("• No significant changes", style={'color': COLORS['text_muted'], 'fontSize': '14px', 'marginBottom': '6px', 'textAlign': 'center'})]
            items = []
            for item in insights[:4]:
                color = COLORS['success'] if item['change'] > 0 else COLORS['danger']
                direction = "up" if item['change'] > 0 else "down"
                marginal_tag = " (marginal)" if item.get('marginal', False) else ""
                if item.get('is_pct', True):
                    value_str = f"{abs(item['change']):.2f}pp"
                else:
                    value_str = f"${abs(item['change'])/1000:,.0f}K"
                items.append(html.Div([
                    html.Span(f"• {item['dir']} ", style={'color': color, 'fontWeight': 'bold'}),
                    html.Span(f"{item['group']}: ", style={'color': COLORS['text'], 'fontWeight': '500'}),
                    html.Span(f"{direction} ", style={'color': COLORS['text']}),
                    html.Span(value_str, style={'color': color, 'fontWeight': 'bold'}),
                    html.Span(f" (Z={item['z']:.1f}){marginal_tag}", style={'color': COLORS['chart_yellow'] if item.get('marginal') else COLORS['text_muted'], 'fontSize': '12px'}),
                ], style={'fontSize': '14px', 'marginBottom': '6px', 'textAlign': 'center'}))
            return items
        
        weekly_metrics_bullets = format_weekly_metrics_bullets(weekly_metrics_insights)
        
        # Generate week summary bullet points - split into positive and negative
        positive_bullets = []
        negative_bullets = []
        
        # FAS $ performance
        if dollars_delta > 5:
            positive_bullets.append(f"• FAS Day 7 $ up {dollars_delta:.0f}% vs 4-wk avg (${tgt_dollars/1000000:.2f}M)")
        elif dollars_delta < -5:
            negative_bullets.append(f"• FAS Day 7 $ down {abs(dollars_delta):.0f}% vs 4-wk avg (${tgt_dollars/1000000:.2f}M)")
        
        # Conversion rate
        if rate_delta > 0.5:
            positive_bullets.append(f"• Conversion rate +{rate_delta:.1f}pp vs 4-wk avg ({tgt_fas_rate:.1f}%)")
        elif rate_delta < -0.5:
            negative_bullets.append(f"• Conversion rate {rate_delta:.1f}pp vs 4-wk avg ({tgt_fas_rate:.1f}%)")
        
        # Lead volume
        if leads_delta > 5:
            positive_bullets.append(f"• Lead volume up {leads_delta:.0f}% vs 4-wk avg ({tgt_leads:,.0f})")
        elif leads_delta < -5:
            negative_bullets.append(f"• Lead volume down {abs(leads_delta):.0f}% vs 4-wk avg ({tgt_leads:,.0f})")
        
        # StS volume
        if sts_delta > 5:
            positive_bullets.append(f"• StS volume up {sts_delta:.0f}% vs 4-wk avg ({tgt_sts:,.0f})")
        elif sts_delta < -5:
            negative_bullets.append(f"• StS volume down {abs(sts_delta):.0f}% vs 4-wk avg ({tgt_sts:,.0f})")
        
        # Avg loan size
        if avg_loan_delta > 5:
            positive_bullets.append(f"• Avg loan size up {avg_loan_delta:.0f}% vs 4-wk avg (${tgt_avg_loan:,.0f})")
        elif avg_loan_delta < -5:
            negative_bullets.append(f"• Avg loan size down {abs(avg_loan_delta):.0f}% vs 4-wk avg (${tgt_avg_loan:,.0f})")
        
        # FAS count
        if fas_delta > 5:
            positive_bullets.append(f"• FAS Day 7 qty up {fas_delta:.0f}% vs 4-wk avg ({tgt_fas:,.0f})")
        elif fas_delta < -5:
            negative_bullets.append(f"• FAS Day 7 qty down {abs(fas_delta):.0f}% vs 4-wk avg ({tgt_fas:,.0f})")
        
        # Top triggers from insights (include weekly metrics)
        all_triggers = lvc_insights + ch_insights + starring_insights + ma_insights
        all_triggers.sort(key=lambda x: abs(x['z']), reverse=True)
        for trigger in all_triggers[:2]:
            if trigger['change'] > 0:
                positive_bullets.append(f"• {trigger['group']} surge +${trigger['change']/1000:,.0f}K vs prior wk")
            else:
                negative_bullets.append(f"• {trigger['group']} decline ${trigger['change']/1000:,.0f}K vs prior wk")
        
        # Add weekly metrics triggers
        for wm in weekly_metrics_insights[:2]:
            if wm.get('is_pct', True):
                change_str = f"{abs(wm['change']):.1f}pp"
            else:
                change_str = f"${abs(wm['change'])/1000:,.0f}K"
            if wm['change'] > 0:
                positive_bullets.append(f"• {wm['group']} up {change_str} vs 6-wk avg")
            else:
                negative_bullets.append(f"• {wm['group']} down {change_str} vs 6-wk avg")
        
        # Build the two-column layout
        positive_col = html.Div([
            html.Div(bullet, style={'marginBottom': '5px', 'fontSize': '13px', 'color': COLORS['success']}) 
            for bullet in positive_bullets[:5]
        ] if positive_bullets else [html.Div("• No significant gains", style={'fontSize': '13px', 'color': COLORS['text_muted']})], 
        style={'textAlign': 'left', 'paddingRight': '10px'})
        
        negative_col = html.Div([
            html.Div(bullet, style={'marginBottom': '5px', 'fontSize': '13px', 'color': COLORS['danger']}) 
            for bullet in negative_bullets[:5]
        ] if negative_bullets else [html.Div("• No significant declines", style={'fontSize': '13px', 'color': COLORS['text_muted']})],
        style={'textAlign': 'left', 'paddingLeft': '10px'})
        
        week_summary_html = dbc.Row([
            dbc.Col(positive_col, width=6),
            dbc.Col(negative_col, width=6),
        ], style={'margin': '0'})
        
        # Column style
        col_style = {'borderRight': f"1px solid {COLORS['border']}", 'padding': '15px 20px', 'textAlign': 'center'}
        col_style_week = {'padding': '15px 20px', 'textAlign': 'center'}
        header_style = {'color': '#FFFFFF', 'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '12px', 'textAlign': 'center'}
        header_style_week = {'color': '#FFFFFF', 'fontSize': '21px', 'fontWeight': 'bold', 'marginBottom': '12px', 'textAlign': 'center'}
        
        # Collapsible Statistical Insights block
        insights_content = dbc.Row([
            # LVC Group Column
            dbc.Col(html.Div([
                html.Div("LVC Group Analysis", style=header_style),
                html.Div(lvc_bullets)
            ], style=col_style), width=2),
            # Channel Mix Column
            dbc.Col(html.Div([
                html.Div("Channel Mix v2", style=header_style),
                html.Div(ch_bullets)
            ], style=col_style), width=2),
            # Weekly Metrics Column (4 new charts)
            dbc.Col(html.Div([
                html.Div("Weekly Metrics", style=header_style),
                html.Div(weekly_metrics_bullets)
            ], style=col_style), width=2),
            # MA Performance Column
            dbc.Col(html.Div([
                html.Div("MA Performance", style=header_style),
                html.Div(ma_bullets)
            ], style=col_style), width=2),
            # Week Summary Column
            dbc.Col(html.Div([
                html.Div("Week Summary", style=header_style_week),
                week_summary_html
            ], style=col_style_week), width=4),
        ], style={'margin': '0'})
        
        insights_block = html.Div([
            html.Button([
                html.Span("📊 Statistical Insights & Week Summary", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                html.Span(" (Click to expand/collapse)", style={'fontSize': '12px', 'color': COLORS['text_muted'], 'marginLeft': '10px'}),
            ], id='insights-toggle-btn', n_clicks=0, style={
                'width': '100%', 'padding': '15px 20px', 'border': 'none', 'borderRadius': '12px 12px 0 0',
                'backgroundColor': 'rgba(255,255,255,0.03)', 'color': '#FFFFFF', 'cursor': 'pointer', 
                'textAlign': 'center', 'borderBottom': f"1px solid {COLORS['border']}"
            }),
            dbc.Collapse(
                insights_content,
                id='insights-collapse',
                is_open=True
            ),
        ], style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'border': f"1px solid {COLORS['border']}", 
                 'marginTop': '16px', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # === THE BUTTON - Executive Summary ===
        # Determine overall week performance sentiment
        fas_dollar_change = tgt_dollars - prev_dollars
        if dollars_delta >= 10:
            week_sentiment = "STRONG WEEK"
            sentiment_color = COLORS['success']
            sentiment_emoji = "🚀"
        elif dollars_delta >= 3:
            week_sentiment = "SOLID WEEK"
            sentiment_color = COLORS['success']
            sentiment_emoji = "✅"
        elif dollars_delta >= -3:
            week_sentiment = "FLAT WEEK"
            sentiment_color = COLORS['chart_yellow']
            sentiment_emoji = "➡️"
        elif dollars_delta >= -10:
            week_sentiment = "SOFT WEEK"
            sentiment_color = COLORS['danger']
            sentiment_emoji = "⚠️"
        else:
            week_sentiment = "CHALLENGING WEEK"
            sentiment_color = COLORS['danger']
            sentiment_emoji = "🔴"
        
        # Build comprehensive driver analysis
        # 1. CHANNEL ANALYSIS - detailed breakdown
        channel_analysis = []
        ch_target_agg = df_vintage[df_vintage['vintage_week'] == target_date].groupby('channel').agg({
            'sts_eligible': 'sum', 'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum', 'lead_count': 'sum'
        }).reset_index()
        ch_prev_agg = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                                 (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))].groupby('channel').agg({
            'sts_eligible': 'sum', 'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum', 'lead_count': 'sum'
        }).reset_index()
        ch_prev_agg['sts_eligible'] = ch_prev_agg['sts_eligible'] / 4
        ch_prev_agg['fas_day7_count'] = ch_prev_agg['fas_day7_count'] / 4
        ch_prev_agg['fas_day7_dollars'] = ch_prev_agg['fas_day7_dollars'] / 4
        ch_prev_agg['lead_count'] = ch_prev_agg['lead_count'] / 4
        ch_merged = ch_target_agg.merge(ch_prev_agg, on='channel', suffixes=('', '_prev'), how='outer').fillna(0)
        ch_merged['dollar_delta'] = ch_merged['fas_day7_dollars'] - ch_merged['fas_day7_dollars_prev']
        ch_merged['fas_delta'] = ch_merged['fas_day7_count'] - ch_merged['fas_day7_count_prev']
        ch_merged['fas_rate'] = (ch_merged['fas_day7_count'] / ch_merged['sts_eligible'].replace(0,1) * 100)
        ch_merged['fas_rate_prev'] = (ch_merged['fas_day7_count_prev'] / ch_merged['sts_eligible_prev'].replace(0,1) * 100)
        ch_merged['rate_delta'] = ch_merged['fas_rate'] - ch_merged['fas_rate_prev']
        ch_merged = ch_merged.sort_values('dollar_delta', ascending=False)
        
        top_channel_gainers = ch_merged.head(3)
        top_channel_losers = ch_merged.tail(3).sort_values('dollar_delta')
        
        channel_wins = []
        channel_concerns = []
        for _, row in top_channel_gainers.iterrows():
            if row['dollar_delta'] > 5000:
                sig_tag = ""
                for ins in ch_insights:
                    if ins['group'] in row['channel']:
                        sig_tag = f" (Z={ins['z']:.1f})" if abs(ins['z']) >= 1.0 else ""
                        break
                channel_wins.append({
                    'name': row['channel'][:25],
                    'dollar_change': row['dollar_delta'],
                    'fas_change': row['fas_delta'],
                    'rate': row['fas_rate'],
                    'rate_delta': row['rate_delta'],
                    'sig': sig_tag
                })
        for _, row in top_channel_losers.iterrows():
            if row['dollar_delta'] < -5000:
                sig_tag = ""
                for ins in ch_insights:
                    if ins['group'] in row['channel']:
                        sig_tag = f" (Z={ins['z']:.1f})" if abs(ins['z']) >= 1.0 else ""
                        break
                channel_concerns.append({
                    'name': row['channel'][:25],
                    'dollar_change': row['dollar_delta'],
                    'fas_change': row['fas_delta'],
                    'rate': row['fas_rate'],
                    'rate_delta': row['rate_delta'],
                    'sig': sig_tag
                })
        
        # 2. LVC GROUP ANALYSIS - lead quality breakdown
        lvc_target_agg = df_vintage[df_vintage['vintage_week'] == target_date].groupby('lvc_group').agg({
            'lead_count': 'sum', 'sts_eligible': 'sum', 'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum'
        }).reset_index()
        lvc_prev_agg = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                                  (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))].groupby('lvc_group').agg({
            'lead_count': 'sum', 'sts_eligible': 'sum', 'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum'
        }).reset_index()
        lvc_prev_agg['lead_count'] = lvc_prev_agg['lead_count'] / 4
        lvc_prev_agg['sts_eligible'] = lvc_prev_agg['sts_eligible'] / 4
        lvc_prev_agg['fas_day7_count'] = lvc_prev_agg['fas_day7_count'] / 4
        lvc_prev_agg['fas_day7_dollars'] = lvc_prev_agg['fas_day7_dollars'] / 4
        lvc_merged = lvc_target_agg.merge(lvc_prev_agg, on='lvc_group', suffixes=('', '_prev'), how='outer').fillna(0)
        lvc_merged['dollar_delta'] = lvc_merged['fas_day7_dollars'] - lvc_merged['fas_day7_dollars_prev']
        lvc_merged['lead_delta'] = lvc_merged['lead_count'] - lvc_merged['lead_count_prev']
        lvc_merged['lead_pct'] = lvc_merged['lead_count'] / lvc_merged['lead_count'].sum() * 100
        lvc_merged['lead_pct_prev'] = lvc_merged['lead_count_prev'] / lvc_merged['lead_count_prev'].sum() * 100
        lvc_merged['mix_shift'] = lvc_merged['lead_pct'] - lvc_merged['lead_pct_prev']
        lvc_merged['fas_rate'] = lvc_merged['fas_day7_count'] / lvc_merged['sts_eligible'].replace(0,1) * 100
        
        lvc_quality_groups = {'High Quality': ['LVC 1-2'], 'Medium Quality': ['LVC 3-8'], 'Low Quality': ['LVC 9-10', 'PHX Transfer']}
        lvc_quality_summary = []
        for quality, groups in lvc_quality_groups.items():
            group_data = lvc_merged[lvc_merged['lvc_group'].isin(groups)]
            if not group_data.empty:
                total_lead_pct = group_data['lead_pct'].sum()
                total_lead_pct_prev = group_data['lead_pct_prev'].sum()
                total_dollar_delta = group_data['dollar_delta'].sum()
                avg_fas_rate = (group_data['fas_day7_count'].sum() / group_data['sts_eligible'].sum() * 100) if group_data['sts_eligible'].sum() > 0 else 0
                lvc_quality_summary.append({
                    'quality': quality,
                    'lead_pct': total_lead_pct,
                    'mix_shift': total_lead_pct - total_lead_pct_prev,
                    'dollar_delta': total_dollar_delta,
                    'fas_rate': avg_fas_rate
                })
        
        # 3. MA PERFORMANCE ANALYSIS
        ma_wins = []
        ma_concerns = []
        starring_summary = []
        
        if not df_ma_summary.empty and 'day7_eligible_sts' in df_ma_summary.columns:
            # Starring group performance
            star_target = df_ma_summary[df_ma_summary['vintage_week'] == target_date].groupby('starring_group').agg({
                'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum', 'leads_assigned': 'sum', 'sf_contact_day7_count': 'sum', 'day7_eligible_sts': 'sum'
            }).reset_index()
            star_prev = df_ma_summary[(df_ma_summary['vintage_week'] < target_date) & 
                                      (df_ma_summary['vintage_week'] >= target_date - timedelta(weeks=4))].groupby('starring_group').agg({
                'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum', 'leads_assigned': 'sum', 'sf_contact_day7_count': 'sum', 'day7_eligible_sts': 'sum'
            }).reset_index()
            for col in ['fas_day7_count', 'fas_day7_dollars', 'leads_assigned', 'sf_contact_day7_count', 'day7_eligible_sts']:
                star_prev[col] = star_prev[col] / 4
            star_merged = star_target.merge(star_prev, on='starring_group', suffixes=('', '_prev'), how='outer').fillna(0)
            star_merged['dollar_delta'] = star_merged['fas_day7_dollars'] - star_merged['fas_day7_dollars_prev']
            star_merged['fas_rate'] = star_merged['fas_day7_count'] / star_merged['day7_eligible_sts'].replace(0,1) * 100
            star_merged['fas_rate_prev'] = star_merged['fas_day7_count_prev'] / star_merged['day7_eligible_sts_prev'].replace(0,1) * 100
            star_merged['contact_rate'] = star_merged['sf_contact_day7_count'] / star_merged['day7_eligible_sts'].replace(0,1) * 100
            
            for _, row in star_merged.iterrows():
                starring_summary.append({
                    'group': f"Star {int(row['starring_group'])}" if pd.notna(row['starring_group']) and row['starring_group'] != 'Unknown' else "Unassigned",
                    'dollar_delta': row['dollar_delta'],
                    'fas_rate': row['fas_rate'],
                    'rate_delta': row['fas_rate'] - row['fas_rate_prev'],
                    'contact_rate': row['contact_rate'],
                    'leads': row['leads_assigned']
                })
            
            # Top MA performance (Using FAS D7 / LP2C)
            ma_target = df_ma_summary[df_ma_summary['vintage_week'] == target_date].groupby('mortgage_advisor').agg({
                'starring_group': 'first',
                'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum', 'leads_assigned': 'sum', 'day7_eligible_sts': 'sum',
                'avg_lp2c': 'mean'
            }).reset_index()
            ma_prev = df_ma_summary[(df_ma_summary['vintage_week'] < target_date) & 
                                    (df_ma_summary['vintage_week'] >= target_date - timedelta(weeks=6))].groupby('mortgage_advisor').agg({
                'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum', 'leads_assigned': 'sum', 'day7_eligible_sts': 'sum',
                'avg_lp2c': 'mean'
            }).reset_index()
            for col in ['fas_day7_count', 'fas_day7_dollars', 'leads_assigned', 'day7_eligible_sts']:
                ma_prev[col] = ma_prev[col] / 6
            ma_merged = ma_target.merge(ma_prev, on='mortgage_advisor', suffixes=('', '_prev'), how='outer').fillna(0)
            
            ma_merged['dollar_delta'] = ma_merged['fas_day7_dollars'] - ma_merged['fas_day7_dollars_prev']
            ma_merged['fas_rate'] = ma_merged['fas_day7_count'] / ma_merged['day7_eligible_sts'].replace(0,1) * 100
            ma_merged['fas_rate_prev'] = ma_merged['fas_day7_count_prev'] / ma_merged['day7_eligible_sts_prev'].replace(0,1) * 100
            
            ma_merged['fas_div_lp2c'] = ma_merged['fas_rate'] / ma_merged['avg_lp2c'].replace(0,1)
            ma_merged['fas_div_lp2c_prev'] = ma_merged['fas_rate_prev'] / ma_merged['avg_lp2c_prev'].replace(0,1)
            ma_merged['ratio_delta'] = ma_merged['fas_div_lp2c'] - ma_merged['fas_div_lp2c_prev']
            
            ma_merged = ma_merged[ma_merged['leads_assigned'] >= 10]  # Only MAs with meaningful volume
            
            ma_top = ma_merged[ma_merged['ratio_delta'] > 0.05].sort_values('ratio_delta', ascending=False)
            
            # Sort struggling: prioritize Star 5, then by worst ratio_delta
            ma_bottom_df = ma_merged[ma_merged['ratio_delta'] < -0.05].copy()
            if not ma_bottom_df.empty:
                ma_bottom_df['is_star_5'] = ma_bottom_df['starring_group'].astype(str) == '5'
                ma_bottom = ma_bottom_df.sort_values(['is_star_5', 'ratio_delta'], ascending=[False, True])
            else:
                ma_bottom = pd.DataFrame()
            
            for _, row in ma_top.head(3).iterrows():
                ma_wins.append({
                    'name': row['mortgage_advisor'][:20],
                    'ratio_delta': row['ratio_delta'],
                    'fas_div_lp2c': row['fas_div_lp2c'],
                    'fas_rate': row['fas_rate'],
                    'fas_dollars': row['fas_day7_dollars'],
                    'leads': row['leads_assigned']
                })
            for _, row in ma_bottom.head(4).iterrows():
                ma_concerns.append({
                    'name': row['mortgage_advisor'][:20],
                    'ratio_delta': row['ratio_delta'],
                    'fas_div_lp2c': row['fas_div_lp2c'],
                    'fas_rate': row['fas_rate'],
                    'fas_dollars': row['fas_day7_dollars'],
                    'leads': row['leads_assigned'],
                    'starring_group': str(row.get('starring_group', ''))
                })
        
        # === BUILD THE BUTTON UI ===
        # Header with sentiment
        button_header = html.Div([
            html.Div([
                html.Span(sentiment_emoji, style={'fontSize': '36px', 'marginRight': '15px'}),
                html.Span(week_sentiment, style={'fontSize': '32px', 'fontWeight': 'bold', 'color': sentiment_color}),
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div([
                html.Span(f"FAS Day 7 $: ", style={'fontSize': '20px', 'color': COLORS['text']}),
                html.Span(f"${tgt_dollars/1000000:.2f}M ", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': COLORS['text']}),
                html.Span(f"({'+' if dollars_delta >= 0 else ''}{dollars_delta:.0f}% vs 4-wk avg)", 
                         style={'fontSize': '18px', 'color': sentiment_color, 'fontWeight': 'bold'}),
            ], style={'textAlign': 'center', 'marginBottom': '5px'}),
            html.Div([
                html.Span(f"FAS Count: {tgt_fas:,.0f} ({'+' if fas_delta >= 0 else ''}{fas_delta:.0f}%) • ", style={'fontSize': '14px', 'color': COLORS['text_muted']}),
                html.Span(f"Conversion: {tgt_fas_rate:.1f}% ({'+' if rate_delta >= 0 else ''}{rate_delta:.1f}pp) • ", style={'fontSize': '14px', 'color': COLORS['text_muted']}),
                html.Span(f"Avg Loan: ${tgt_avg_loan:,.0f}", style={'fontSize': '14px', 'color': COLORS['text_muted']}),
            ], style={'textAlign': 'center'}),
        ], style={'padding': '20px', 'borderBottom': f"2px solid {COLORS['border']}", 'backgroundColor': 'rgba(255,255,255,0.02)'})
        
        # Main narrative - WHAT DROVE THE CHANGE
        driver_narrative = []
        
        # Determine primary driver
        total_channel_impact = sum([c['dollar_change'] for c in channel_wins]) + sum([c['dollar_change'] for c in channel_concerns])
        total_lvc_impact = sum([l['dollar_delta'] for l in lvc_quality_summary])
        total_ma_impact = sum([s['dollar_delta'] for s in starring_summary])
        
        if abs(total_channel_impact) >= abs(total_lvc_impact) and abs(total_channel_impact) >= abs(total_ma_impact):
            primary_driver = "Channel Mix"
        elif abs(total_lvc_impact) >= abs(total_ma_impact):
            primary_driver = "Lead Quality"
        else:
            primary_driver = "MA Performance"
        
        if fas_dollar_change > 0:
            driver_narrative.append(f"This week's ${abs(fas_dollar_change)/1000:,.0f}K increase was primarily driven by {primary_driver}.")
        else:
            driver_narrative.append(f"This week's ${abs(fas_dollar_change)/1000:,.0f}K decrease was primarily driven by {primary_driver}.")
        
        # Channel section - build items first
        channel_win_items = [html.Div([
            html.Span(f"• {c['name']}: ", style={'fontWeight': '500'}),
            html.Span(f"+${c['dollar_change']/1000:,.0f}K", style={'color': COLORS['success'], 'fontWeight': 'bold'}),
            html.Span(f" ({c['rate']:.1f}% conv, {'+' if c['rate_delta'] >= 0 else ''}{c['rate_delta']:.1f}pp){c['sig']}", style={'fontSize': '11px', 'color': COLORS['text_muted']}),
        ], style={'fontSize': '12px', 'marginBottom': '3px'}) for c in channel_wins[:3]] if channel_wins else [html.Div("• No standout performers", style={'fontSize': '12px', 'color': COLORS['text_muted']})]
        
        channel_concern_items = [html.Div([
            html.Span(f"• {c['name']}: ", style={'fontWeight': '500'}),
            html.Span(f"${c['dollar_change']/1000:,.0f}K", style={'color': COLORS['danger'], 'fontWeight': 'bold'}),
            html.Span(f" ({c['rate']:.1f}% conv, {c['rate_delta']:.1f}pp){c['sig']}", style={'fontSize': '11px', 'color': COLORS['text_muted']}),
        ], style={'fontSize': '12px', 'marginBottom': '3px'}) for c in channel_concerns[:3]] if channel_concerns else [html.Div("• No major concerns", style={'fontSize': '12px', 'color': COLORS['text_muted']})]
        
        # LVC section items
        lvc_quality_items = [html.Div([
            html.Div([
                html.Span(f"{l['quality']}: ", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                html.Span(f"{l['lead_pct']:.1f}% of leads", style={'fontSize': '12px'}),
            ]),
            html.Div([
                html.Span(f"Mix: {'+' if l['mix_shift'] >= 0 else ''}{l['mix_shift']:.1f}pp • ", style={'fontSize': '11px', 'color': COLORS['success'] if l['mix_shift'] > 0 and 'High' in l['quality'] else (COLORS['danger'] if l['mix_shift'] < 0 and 'High' in l['quality'] else COLORS['text_muted'])}),
                html.Span(f"FAS $: {'+' if l['dollar_delta'] >= 0 else ''}${l['dollar_delta']/1000:,.0f}K • ", style={'fontSize': '11px', 'color': COLORS['success'] if l['dollar_delta'] > 0 else COLORS['danger']}),
                html.Span(f"Conv: {l['fas_rate']:.1f}%", style={'fontSize': '11px', 'color': COLORS['text_muted']}),
            ]),
        ], style={'marginBottom': '8px', 'padding': '5px', 'backgroundColor': 'rgba(255,255,255,0.02)', 'borderRadius': '4px'}) for l in lvc_quality_summary]
        
        lvc_sig_items = [html.Div(f"• {ins['group']}: {ins['dir']} {abs(ins['change']):.1f}pp (Z={ins['z']:.1f})", 
                      style={'fontSize': '11px', 'color': COLORS['success'] if ins['change'] > 0 else COLORS['danger']}) for ins in lvc_insights[:2]] if lvc_insights else [html.Div("• No stat sig changes", style={'fontSize': '11px', 'color': COLORS['text_muted']})]
        
        # MA section items
        starring_items = [html.Div([
            html.Span(f"• {s['group']}: ", style={'fontWeight': '500', 'fontSize': '11px'}),
            html.Span(f"{'+' if s['dollar_delta'] >= 0 else ''}${s['dollar_delta']/1000:,.0f}K ", style={'fontSize': '11px', 'color': COLORS['success'] if s['dollar_delta'] > 0 else COLORS['danger'], 'fontWeight': 'bold'}),
            html.Span(f"({s['fas_rate']:.1f}% conv)", style={'fontSize': '10px', 'color': COLORS['text_muted']}),
        ], style={'marginBottom': '2px'}) for s in sorted(starring_summary, key=lambda x: x['dollar_delta'], reverse=True)[:4]] if starring_summary else []
        
        ma_win_items = [html.Div([
            html.Span(f"• {m['name']}: ", style={'fontWeight': '500'}),
            html.Span(f"Δ +{m['ratio_delta']:.2f}", style={'color': COLORS['success'], 'fontWeight': 'bold'}),
            html.Span(f" (F/L: {m['fas_div_lp2c']:.2f}, {m['fas_rate']:.1f}%, ${m['fas_dollars']/1000:,.0f}K)", style={'fontSize': '10px', 'color': COLORS['text_muted']})
        ], style={'fontSize': '11px', 'marginBottom': '3px'}) for m in ma_wins[:3]] if ma_wins else [html.Div("• No standouts", style={'fontSize': '11px', 'color': COLORS['text_muted']})]
        
        ma_concern_items = [html.Div([
            html.Span(f"• {m['name']}", style={'fontWeight': '500'}),
            html.Span(f" (Star {m.get('starring_group', '?')})" if m.get('starring_group') and m.get('starring_group') != 'nan' else "", style={'fontSize': '9px', 'color': COLORS['chart_purple'], 'fontWeight': 'bold'}),
            html.Span(": "),
            html.Span(f"Δ {m['ratio_delta']:.2f}", style={'color': COLORS['danger'], 'fontWeight': 'bold'}),
            html.Span(f" (F/L: {m['fas_div_lp2c']:.2f}, {m['fas_rate']:.1f}%, ${m['fas_dollars']/1000:,.0f}K)", style={'fontSize': '10px', 'color': COLORS['text_muted']})
        ], style={'fontSize': '11px', 'marginBottom': '3px'}) for m in ma_concerns[:4]] if ma_concerns else [html.Div("• No major concerns", style={'fontSize': '11px', 'color': COLORS['text_muted']})]
        
        # Key takeaway
        key_takeaway_items = []
        if channel_wins:
            key_takeaway_items.append(f"Channel wins from {channel_wins[0]['name']} (+${channel_wins[0]['dollar_change']/1000:,.0f}K)")
        if channel_concerns:
            key_takeaway_items.append(f"Watch {channel_concerns[0]['name']} (${channel_concerns[0]['dollar_change']/1000:,.0f}K)")
        high_quality = next((l for l in lvc_quality_summary if l['quality'] == 'High Quality'), None)
        if high_quality and abs(high_quality['mix_shift']) > 0.5:
            if high_quality['mix_shift'] > 0:
                key_takeaway_items.append(f"Lead quality improving (+{high_quality['mix_shift']:.1f}pp high quality)")
            else:
                key_takeaway_items.append(f"Lead quality declining ({high_quality['mix_shift']:.1f}pp high quality)")
        if starring_summary:
            best_star = max(starring_summary, key=lambda x: x['dollar_delta'])
            worst_star = min(starring_summary, key=lambda x: x['dollar_delta'])
            if best_star['dollar_delta'] > 10000:
                key_takeaway_items.append(f"{best_star['group']} leading (+${best_star['dollar_delta']/1000:,.0f}K)")
            if worst_star['dollar_delta'] < -10000:
                key_takeaway_items.append(f"{worst_star['group']} struggling (${worst_star['dollar_delta']/1000:,.0f}K)")
        
        takeaway_items = [html.Div(f"• {item}", style={'fontSize': '16px', 'marginBottom': '10px', 'lineHeight': '1.5'}) for item in key_takeaway_items[:5]] if key_takeaway_items else [html.Div("• Week tracking close to expectations", style={'fontSize': '16px', 'color': COLORS['text_muted']})]
        
        # Key Takeaways as its own column (first/left side) - CENTERED
        key_takeaway_col = html.Div([
            html.Div("🎯 KEY TAKEAWAYS", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': '#FFFFFF', 'marginBottom': '15px', 'textAlign': 'center'}),
            html.Div(takeaway_items, style={'textAlign': 'center'}),
        ], style={'padding': '20px', 'borderRight': f"2px solid {sentiment_color}", 'height': '100%', 'textAlign': 'center'})
        
        # Analysis sections (Channel, LVC, MA) - centered
        analysis_section_style = {'padding': '20px', 'textAlign': 'center'}
        
        # Content for THE BUTTON (collapsible)
        the_button_content = html.Div([
            button_header,
            dbc.Row([
                # Key Takeaways FIRST (left column) - CENTERED
                dbc.Col(key_takeaway_col, width=3),
                # Channel section - CENTERED
                dbc.Col(html.Div([
                    html.Div("📢 CHANNEL", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': COLORS['accent3'], 'marginBottom': '12px', 'textAlign': 'center'}),
                    html.Div([
                        html.Div("✅ Top Performers", style={'fontSize': '14px', 'fontWeight': 'bold', 'color': COLORS['success'], 'marginBottom': '8px', 'textAlign': 'center'}),
                        *channel_win_items
                    ], style={'marginBottom': '15px', 'textAlign': 'center'}),
                    html.Div([
                        html.Div("⚠️ Concerns", style={'fontSize': '14px', 'fontWeight': 'bold', 'color': COLORS['danger'], 'marginBottom': '8px', 'textAlign': 'center'}),
                        *channel_concern_items
                    ], style={'textAlign': 'center'}),
                ], style=analysis_section_style), width=3),
                # LVC section - CENTERED
                dbc.Col(html.Div([
                    html.Div("📊 LEAD QUALITY", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': COLORS['chart_yellow'], 'marginBottom': '12px', 'textAlign': 'center'}),
                    *lvc_quality_items,
                    html.Div([
                        html.Div("📈 Stat Sig Shifts:", style={'fontSize': '13px', 'fontWeight': 'bold', 'color': COLORS['text'], 'marginTop': '10px', 'marginBottom': '5px', 'textAlign': 'center'}),
                        *lvc_sig_items
                    ], style={'textAlign': 'center'}),
                ], style=analysis_section_style), width=3),
                # MA section - CENTERED
                dbc.Col(html.Div([
                    html.Div("👥 MA PERFORMANCE", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': COLORS['chart_purple'], 'marginBottom': '12px', 'textAlign': 'center'}),
                    html.Div([
                        html.Div("By Starring:", style={'fontSize': '13px', 'fontWeight': 'bold', 'marginBottom': '5px', 'textAlign': 'center'}),
                        *starring_items
                    ], style={'marginBottom': '12px', 'textAlign': 'center'}),
                    html.Div([
                        html.Div("✅ Top MAs:", style={'fontSize': '13px', 'fontWeight': 'bold', 'color': COLORS['success'], 'marginBottom': '5px', 'textAlign': 'center'}),
                        *ma_win_items
                    ], style={'marginBottom': '8px', 'textAlign': 'center'}),
                    html.Div([
                        html.Div("⚠️ Struggling:", style={'fontSize': '13px', 'fontWeight': 'bold', 'color': COLORS['danger'], 'marginBottom': '5px', 'textAlign': 'center'}),
                        *ma_concern_items
                    ], style={'textAlign': 'center'}),
                ], style=analysis_section_style), width=3),
            ], style={'margin': '0'}),
        ])
        
        # THE BUTTON - now a clickable button that expands/collapses
        the_button = html.Div([
            # Clickable Button Header
            html.Button([
                html.Span("🔘 ", style={'fontSize': '28px'}),
                html.Span("THE BUTTON", style={'fontSize': '28px', 'fontWeight': 'bold', 'color': '#FFFFFF'}),
                html.Span(f" - {week_sentiment}", style={'fontSize': '18px', 'color': sentiment_color, 'marginLeft': '15px', 'fontWeight': 'bold'}),
                html.Span(" (Click to expand)", style={'fontSize': '14px', 'color': COLORS['text_muted'], 'marginLeft': '15px'}),
            ], id='the-button-toggle', n_clicks=0, style={
                'width': '100%', 'padding': '20px 30px', 'border': 'none', 'borderRadius': '12px',
                'backgroundColor': COLORS['card_bg'], 'cursor': 'pointer', 'textAlign': 'center',
                'border': f"3px solid {sentiment_color}", 'boxShadow': f'0 4px 25px rgba(0, 0, 0, 0.4)',
                'transition': 'all 0.3s ease'
            }),
            # Collapsible Content
            dbc.Collapse(
                html.Div([the_button_content], style={
                    'backgroundColor': COLORS['card_bg'], 'borderRadius': '0 0 12px 12px', 
                    'border': f"2px solid {sentiment_color}", 'borderTop': 'none',
                    'marginTop': '-5px'
                }),
                id='the-button-collapse',
                is_open=False
            ),
        ], style={'marginTop': '20px'})
        
        # === Weekly Metrics Trending Charts (12 weeks) ===
        # Handle missing columns with defaults
        if 'funded_count' not in df_vintage.columns:
            df_vintage['funded_count'] = 0
        if 'funded_dollars' not in df_vintage.columns:
            df_vintage['funded_dollars'] = 0
        if 'avg_lp2c' not in df_vintage.columns:
            df_vintage['avg_lp2c'] = 0
        if 'avg_lvc' not in df_vintage.columns:
            df_vintage['avg_lvc'] = 0
        if 'avg_loan' not in df_vintage.columns:
            df_vintage['avg_loan'] = 0
        
        # Calculate weighted sums for proper averaging (weight by lead_count)
        # Only count leads with valid data to avoid NaN dilution
        df_vintage['lp2c_weighted'] = df_vintage['avg_lp2c'].fillna(0) * df_vintage['lead_count']
        df_vintage['lp2c_lead_count'] = df_vintage['lead_count'].where(df_vintage['avg_lp2c'].notna(), 0)
        df_vintage['lvc_weighted'] = df_vintage['avg_lvc'].fillna(0) * df_vintage['lead_count']
        df_vintage['lvc_lead_count'] = df_vintage['lead_count'].where(df_vintage['avg_lvc'].notna(), 0)
            
        weekly_agg = df_vintage.groupby('vintage_week').agg({
            'sf_contact_day7_count': 'sum',
            'day7_eligible_sts': 'sum',
            'fas_day7_count': 'sum',
            'fas_day7_dollars': 'sum',
            'sts_eligible': 'sum',
            'lead_count': 'sum',
            'lp2c_weighted': 'sum',
            'lp2c_lead_count': 'sum',
            'lvc_weighted': 'sum',
            'lvc_lead_count': 'sum',
            'funded_count': 'sum',
            'funded_dollars': 'sum'
        }).reset_index()
        weekly_agg = weekly_agg.sort_values('vintage_week').tail(12)
        weekly_agg['week_label'] = weekly_agg['vintage_week'].dt.strftime('%-m/%d')
        # Calculate proper weighted averages (only using leads with valid data)
        weekly_agg['avg_lp2c'] = (weekly_agg['lp2c_weighted'] / weekly_agg['lp2c_lead_count'].replace(0, 1)).round(2)
        weekly_agg['avg_lvc'] = (weekly_agg['lvc_weighted'] / weekly_agg['lvc_lead_count'].replace(0, 1)).round(2)
        weekly_agg['sf_contact_day7_pct'] = (weekly_agg['sf_contact_day7_count'] / weekly_agg['day7_eligible_sts'].replace(0, 1) * 100).round(2)
        weekly_agg['contact_to_fas_pct'] = (weekly_agg['fas_day7_count'] / weekly_agg['sf_contact_day7_count'].replace(0, 1) * 100).round(2)
        weekly_agg['fas_day7_pct'] = (weekly_agg['fas_day7_count'] / weekly_agg['day7_eligible_sts'].replace(0, 1) * 100).round(2)
        weekly_agg['close_rate'] = (weekly_agg['funded_count'] / weekly_agg['lead_count'].replace(0, 1) * 100).round(2)
        weekly_agg['avg_fas_day7_dollars'] = (weekly_agg['fas_day7_dollars'] / weekly_agg['fas_day7_count'].replace(0, 1)).round(0)
        
        # Calculate Z-scores for statistical significance (last 6 weeks)
        def calc_significance(series, metric_name, is_pct=True):
            if len(series) < 7:
                return html.Div("Insufficient data", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'textAlign': 'center'})
            last_6 = series.tail(7).head(6)  # 6 weeks before the latest
            latest = series.iloc[-1]
            mean_val = last_6.mean()
            std_val = last_6.std()
            z_score = (latest - mean_val) / std_val if std_val > 0 else 0
            change = latest - mean_val
            direction = "up" if change > 0 else "down"
            
            if is_pct:
                change_str = f"{abs(change):.1f}pp {direction}"
            else:
                change_str = f"${abs(change)/1000:,.0f}K {direction}"
            
            if abs(z_score) >= 1.495:
                color = COLORS['success'] if change > 0 else COLORS['danger']
                return html.Div([
                    html.Span("⚠️ Significant Change: ", style={'fontWeight': 'bold', 'color': color}),
                    html.Span(f"{change_str} (Z={z_score:.1f})", style={'color': color})
                ], style={'fontSize': '11px', 'textAlign': 'center', 'padding': '4px'})
            elif abs(z_score) >= 1.0:
                color = COLORS['chart_yellow']
                return html.Div([
                    html.Span("📊 Minor Change: ", style={'fontWeight': 'bold', 'color': color}),
                    html.Span(f"{change_str} (Z={z_score:.1f})", style={'color': color})
                ], style={'fontSize': '11px', 'textAlign': 'center', 'padding': '4px'})
            else:
                return html.Div("No Significant Change", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'textAlign': 'center', 'padding': '4px'})
        
        sf_contact_sig = calc_significance(weekly_agg['sf_contact_day7_pct'], 'SF Contact %', is_pct=True)
        contact_fas_sig = calc_significance(weekly_agg['contact_to_fas_pct'], 'Contact to FAS %', is_pct=True)
        fas_pct_sig = calc_significance(weekly_agg['fas_day7_pct'], 'FAS Day 7 %', is_pct=True)
        fas_dollar_sig = calc_significance(weekly_agg['fas_day7_dollars'], 'FAS Day 7 $', is_pct=False)
        
        # Chart 1: SF Contact Day 7 % (combo - emphasis on %)
        fig_sf_contact_wk = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sf_contact_wk.add_trace(go.Bar(x=weekly_agg['week_label'], y=weekly_agg['sf_contact_day7_count'],
            marker_color='rgba(52, 152, 219, 0.4)', name='Contact Qty', showlegend=False), secondary_y=True)
        fig_sf_contact_wk.add_trace(go.Scatter(x=weekly_agg['week_label'], y=weekly_agg['sf_contact_day7_pct'],
            mode='lines+markers+text', line=dict(color=COLORS['chart_yellow'], width=3),
            marker=dict(size=8), text=weekly_agg['sf_contact_day7_pct'].apply(lambda x: f"{x:.1f}%"),
            textposition='top center', textfont=dict(size=10, color='#FFFFFF', family='Arial Black'),
            name='Contact %', showlegend=False), secondary_y=False)
        sf_pct_max = weekly_agg['sf_contact_day7_pct'].max() * 1.25  # 25% headroom for labels
        sf_count_max = weekly_agg['sf_contact_day7_count'].max() * 1.25
        fig_sf_contact_wk.update_layout(title=dict(text='<b>SF Contact Day 7 %</b>', font=dict(color=COLORS['text'], size=13), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=50, r=50, t=50, b=30), height=250, showlegend=False, bargap=0.3)
        fig_sf_contact_wk.update_yaxes(title_text='', tickfont=dict(size=9, color=COLORS['chart_yellow']), showgrid=False, ticksuffix='%', range=[0, sf_pct_max], secondary_y=False)
        fig_sf_contact_wk.update_yaxes(title_text='', tickfont=dict(size=9, color=COLORS['text']), showgrid=True, gridcolor=COLORS['border'], range=[0, sf_count_max], secondary_y=True)
        fig_sf_contact_wk.update_xaxes(tickfont=dict(size=9, color=COLORS['text']), showgrid=False)
        
        # Chart 2: Contact to FAS % (combo - emphasis on %)
        fig_contact_fas_wk = make_subplots(specs=[[{"secondary_y": True}]])
        fig_contact_fas_wk.add_trace(go.Bar(x=weekly_agg['week_label'], y=weekly_agg['fas_day7_count'],
            marker_color='rgba(155, 89, 182, 0.4)', name='FAS Qty', showlegend=False), secondary_y=True)
        fig_contact_fas_wk.add_trace(go.Scatter(x=weekly_agg['week_label'], y=weekly_agg['contact_to_fas_pct'],
            mode='lines+markers+text', line=dict(color=COLORS['chart_yellow'], width=3),
            marker=dict(size=8), text=weekly_agg['contact_to_fas_pct'].apply(lambda x: f"{x:.1f}%"),
            textposition='top center', textfont=dict(size=10, color='#FFFFFF', family='Arial Black'),
            name='Contact to FAS %', showlegend=False), secondary_y=False)
        contact_fas_pct_max = weekly_agg['contact_to_fas_pct'].max() * 1.25  # 25% headroom for labels
        contact_fas_count_max = weekly_agg['fas_day7_count'].max() * 1.25
        fig_contact_fas_wk.update_layout(title=dict(text='<b>Contact to FAS %</b>', font=dict(color=COLORS['text'], size=13), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=50, r=50, t=50, b=30), height=250, showlegend=False, bargap=0.3)
        fig_contact_fas_wk.update_yaxes(title_text='', tickfont=dict(size=9, color=COLORS['chart_yellow']), showgrid=False, ticksuffix='%', range=[0, contact_fas_pct_max], secondary_y=False)
        fig_contact_fas_wk.update_yaxes(title_text='', tickfont=dict(size=9, color=COLORS['text']), showgrid=True, gridcolor=COLORS['border'], range=[0, contact_fas_count_max], secondary_y=True)
        fig_contact_fas_wk.update_xaxes(tickfont=dict(size=9, color=COLORS['text']), showgrid=False)
        
        # Chart 3: FAS Day 7 % (combo - emphasis on %)
        fig_fas_pct_wk = make_subplots(specs=[[{"secondary_y": True}]])
        fig_fas_pct_wk.add_trace(go.Bar(x=weekly_agg['week_label'], y=weekly_agg['fas_day7_count'],
            marker_color='rgba(46, 204, 113, 0.4)', name='FAS Qty', showlegend=False), secondary_y=True)
        fig_fas_pct_wk.add_trace(go.Scatter(x=weekly_agg['week_label'], y=weekly_agg['fas_day7_pct'],
            mode='lines+markers+text', line=dict(color=COLORS['chart_yellow'], width=3),
            marker=dict(size=8), text=weekly_agg['fas_day7_pct'].apply(lambda x: f"{x:.1f}%"),
            textposition='top center', textfont=dict(size=10, color='#FFFFFF', family='Arial Black'),
            name='FAS %', showlegend=False), secondary_y=False)
        fas_pct_max = weekly_agg['fas_day7_pct'].max() * 1.25  # 25% headroom for labels
        fas_count_max = weekly_agg['fas_day7_count'].max() * 1.25
        fig_fas_pct_wk.update_layout(title=dict(text='<b>FAS Day 7 %</b>', font=dict(color=COLORS['text'], size=13), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=50, r=50, t=50, b=30), height=250, showlegend=False, bargap=0.3)
        fig_fas_pct_wk.update_yaxes(title_text='', tickfont=dict(size=9, color=COLORS['chart_yellow']), showgrid=False, ticksuffix='%', range=[0, fas_pct_max], secondary_y=False)
        fig_fas_pct_wk.update_yaxes(title_text='', tickfont=dict(size=9, color=COLORS['text']), showgrid=True, gridcolor=COLORS['border'], range=[0, fas_count_max], secondary_y=True)
        fig_fas_pct_wk.update_xaxes(tickfont=dict(size=9, color=COLORS['text']), showgrid=False)
        
        # Chart 4: FAS Day 7 $ (bar chart only)
        fig_fas_dollar_wk = go.Figure()
        fig_fas_dollar_wk.add_trace(go.Bar(x=weekly_agg['week_label'], y=weekly_agg['fas_day7_dollars'],
            marker_color=COLORS['primary'], text=weekly_agg['fas_day7_dollars'].apply(lambda x: f"${x/1000:,.0f}K"),
            textposition='outside', textfont=dict(size=9, color=COLORS['text']), name='FAS $', showlegend=False))
        fas_dollar_max = weekly_agg['fas_day7_dollars'].max() * 1.25  # 25% headroom for labels
        fig_fas_dollar_wk.update_layout(title=dict(text='<b>FAS Day 7 $</b>', font=dict(color=COLORS['text'], size=13), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=50, r=50, t=50, b=30), height=250, showlegend=False, bargap=0.3)
        fig_fas_dollar_wk.update_yaxes(tickfont=dict(size=9, color=COLORS['text']), showgrid=True, gridcolor=COLORS['border'], range=[0, fas_dollar_max])
        fig_fas_dollar_wk.update_xaxes(tickfont=dict(size=9, color=COLORS['text']), showgrid=False)
        
        # === 4 ADDITIONAL CHARTS ===
        
        # Chart 5: Avg LP2C % 
        lp2c_sig = calc_significance(weekly_agg['avg_lp2c'], 'Avg LP2C %', is_pct=True)
        fig_lp2c_wk = go.Figure()
        fig_lp2c_wk.add_trace(go.Scatter(x=weekly_agg['week_label'], y=weekly_agg['avg_lp2c'],
            mode='lines+markers+text', line=dict(color=COLORS['accent3'], width=3),
            marker=dict(size=8), text=weekly_agg['avg_lp2c'].apply(lambda x: f"{x:.1f}%"),
            textposition='top center', textfont=dict(size=10, color='#FFFFFF', family='Arial Black'),
            name='LP2C %', showlegend=False))
        lp2c_max = weekly_agg['avg_lp2c'].max() * 1.25 if weekly_agg['avg_lp2c'].max() > 0 else 10
        fig_lp2c_wk.update_layout(title=dict(text='<b>Avg LP2C %</b>', font=dict(color=COLORS['text'], size=13), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=50, r=50, t=50, b=30), height=250, showlegend=False)
        fig_lp2c_wk.update_yaxes(tickfont=dict(size=9, color=COLORS['text']), showgrid=True, gridcolor=COLORS['border'], ticksuffix='%', range=[0, lp2c_max])
        fig_lp2c_wk.update_xaxes(tickfont=dict(size=9, color=COLORS['text']), showgrid=False)
        
        # Chart 6: Avg LVC 
        lvc_sig = calc_significance(weekly_agg['avg_lvc'], 'Avg LVC', is_pct=False)
        fig_lvc_wk = go.Figure()
        fig_lvc_wk.add_trace(go.Scatter(x=weekly_agg['week_label'], y=weekly_agg['avg_lvc'],
            mode='lines+markers+text', line=dict(color=COLORS['chart_purple'], width=3),
            marker=dict(size=8), text=weekly_agg['avg_lvc'].apply(lambda x: f"{x:.1f}"),
            textposition='top center', textfont=dict(size=10, color='#FFFFFF', family='Arial Black'),
            name='LVC', showlegend=False))
        lvc_max = weekly_agg['avg_lvc'].max() * 1.25 if weekly_agg['avg_lvc'].max() > 0 else 10
        lvc_min = weekly_agg['avg_lvc'].min() * 0.9 if weekly_agg['avg_lvc'].min() > 0 else 0
        fig_lvc_wk.update_layout(title=dict(text='<b>Avg LVC</b>', font=dict(color=COLORS['text'], size=13), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=50, r=50, t=50, b=30), height=250, showlegend=False)
        fig_lvc_wk.update_yaxes(tickfont=dict(size=9, color=COLORS['text']), showgrid=True, gridcolor=COLORS['border'], range=[lvc_min, lvc_max])
        fig_lvc_wk.update_xaxes(tickfont=dict(size=9, color=COLORS['text']), showgrid=False)
        
        # Chart 7: Avg FAS $ Day 7 
        avg_fas_sig = calc_significance(weekly_agg['avg_fas_day7_dollars'], 'Avg FAS $ Day 7', is_pct=False)
        fig_avg_fas_wk = go.Figure()
        fig_avg_fas_wk.add_trace(go.Bar(x=weekly_agg['week_label'], y=weekly_agg['avg_fas_day7_dollars'],
            marker_color=COLORS['success'], text=weekly_agg['avg_fas_day7_dollars'].apply(lambda x: f"${x/1000:,.0f}K" if x > 0 else "N/A"),
            textposition='outside', textfont=dict(size=9, color=COLORS['text']), name='Avg FAS $', showlegend=False))
        avg_fas_max = weekly_agg['avg_fas_day7_dollars'].max() * 1.25 if weekly_agg['avg_fas_day7_dollars'].max() > 0 else 100000
        fig_avg_fas_wk.update_layout(title=dict(text='<b>Avg FAS $ Day 7</b>', font=dict(color=COLORS['text'], size=13), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=50, r=50, t=50, b=30), height=250, showlegend=False, bargap=0.3)
        fig_avg_fas_wk.update_yaxes(tickfont=dict(size=9, color=COLORS['text']), showgrid=True, gridcolor=COLORS['border'], range=[0, avg_fas_max])
        fig_avg_fas_wk.update_xaxes(tickfont=dict(size=9, color=COLORS['text']), showgrid=False)
        
        # Chart 8: Close Rate % (with funded $ as bars)
        close_sig = calc_significance(weekly_agg['close_rate'], 'Close Rate %', is_pct=True)
        fig_close_wk = make_subplots(specs=[[{"secondary_y": True}]])
        fig_close_wk.add_trace(go.Bar(x=weekly_agg['week_label'], y=weekly_agg['funded_dollars'],
            marker_color='rgba(241, 196, 15, 0.4)', name='Funded $', showlegend=False), secondary_y=True)
        fig_close_wk.add_trace(go.Scatter(x=weekly_agg['week_label'], y=weekly_agg['close_rate'],
            mode='lines+markers+text', line=dict(color=COLORS['chart_yellow'], width=3),
            marker=dict(size=8), text=weekly_agg['close_rate'].apply(lambda x: f"{x:.1f}%"),
            textposition='top center', textfont=dict(size=10, color='#FFFFFF', family='Arial Black'),
            name='Close %', showlegend=False), secondary_y=False)
        close_pct_max = weekly_agg['close_rate'].max() * 1.25 if weekly_agg['close_rate'].max() > 0 else 10
        funded_max = weekly_agg['funded_dollars'].max() * 1.25 if weekly_agg['funded_dollars'].max() > 0 else 1000000
        fig_close_wk.update_layout(title=dict(text='<b>Close Rate %</b>', font=dict(color=COLORS['text'], size=13), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=50, r=50, t=50, b=30), height=250, showlegend=False, bargap=0.3)
        fig_close_wk.update_yaxes(title_text='', tickfont=dict(size=9, color=COLORS['chart_yellow']), showgrid=False, ticksuffix='%', range=[0, close_pct_max], secondary_y=False)
        fig_close_wk.update_yaxes(title_text='', tickfont=dict(size=9, color=COLORS['text']), showgrid=True, gridcolor=COLORS['border'], range=[0, funded_max], secondary_y=True)
        fig_close_wk.update_xaxes(tickfont=dict(size=9, color=COLORS['text']), showgrid=False)
        
        # Create charts rows with significance indicators (2 rows of 4)
        weekly_charts_row = html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    dcc.Graph(figure=fig_sf_contact_wk, config={'displayModeBar': False}),
                    sf_contact_sig
                ]), width=3),
                dbc.Col(html.Div([
                    dcc.Graph(figure=fig_contact_fas_wk, config={'displayModeBar': False}),
                    contact_fas_sig
                ]), width=3),
                dbc.Col(html.Div([
                    dcc.Graph(figure=fig_fas_pct_wk, config={'displayModeBar': False}),
                    fas_pct_sig
                ]), width=3),
                dbc.Col(html.Div([
                    dcc.Graph(figure=fig_fas_dollar_wk, config={'displayModeBar': False}),
                    fas_dollar_sig
                ]), width=3),
            ], style={'marginTop': '16px'}),
            dbc.Row([
                dbc.Col(html.Div([
                    dcc.Graph(figure=fig_lp2c_wk, config={'displayModeBar': False}),
                    lp2c_sig
                ]), width=3),
                dbc.Col(html.Div([
                    dcc.Graph(figure=fig_lvc_wk, config={'displayModeBar': False}),
                    lvc_sig
                ]), width=3),
                dbc.Col(html.Div([
                    dcc.Graph(figure=fig_avg_fas_wk, config={'displayModeBar': False}),
                    avg_fas_sig
                ]), width=3),
                dbc.Col(html.Div([
                    dcc.Graph(figure=fig_close_wk, config={'displayModeBar': False}),
                    close_sig
                ]), width=3),
            ], style={'marginTop': '8px'}),
        ])
        
        kpi_row = html.Div([
            html.Div([
                html.Div(create_metric_card("Gross Qty", f"{tgt_leads:,.0f}", leads_delta, "% vs 4wk avg"), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
                html.Div(create_metric_card("StS (Eligible)", f"{tgt_sts:,.0f}", sts_delta, "% vs 4wk avg"), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
                html.Div(create_metric_card("SF Contact Day 7 %", f"{tgt_sf_contact_pct:.1f}%", sf_contact_delta, "pp vs 4wk avg"), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
                html.Div(create_metric_card("FAS Day 7 Qty", f"{tgt_fas:,.0f}", fas_delta, "% vs 4wk avg"), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
                html.Div(create_metric_card("FAS Day 7 $", f"${tgt_dollars/1000:,.0f}K", dollars_delta, "% vs 4wk avg"), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
                html.Div(create_metric_card("FAS Day 7 Rate", f"{tgt_fas_rate:.1f}%", rate_delta, "pp vs 4wk avg"), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
                html.Div(create_metric_card("Avg FAS Day 7 $", f"${tgt_avg_loan:,.0f}" if tgt_avg_loan > 0 else "N/A", avg_loan_delta if tgt_avg_loan > 0 else None, "% vs 4wk avg"), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
            ], style={'display': 'flex', 'gap': '0px', 'width': '100%'}),
            insights_block,
            the_button,
            weekly_charts_row,
        ])
    
    # === LVC Analysis ===
    lvc_row = html.Div("Loading LVC analysis...")
    channel_v2_row = html.Div("Loading Channel Mix v2...")
    if not df_vintage.empty:
        lvc_target = df_vintage[df_vintage['vintage_week'] == target_date].groupby('lvc_group').agg({
            'lead_count': 'sum',
            'sts_eligible': 'sum',
            'fas_day7_count': 'sum',
            'fas_day7_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        lvc_target = lvc_target.rename(columns={'avg_lp2c': 'lp2c'})
        
        lvc_prev = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                             (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))].groupby('lvc_group').agg({
            'lead_count': 'sum',
            'sts_eligible': 'sum',
            'fas_day7_count': 'sum',
            'fas_day7_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        lvc_prev['lead_count'] = lvc_prev['lead_count'] / 4
        lvc_prev['sts_eligible'] = lvc_prev['sts_eligible'] / 4
        lvc_prev['fas_day7_count'] = lvc_prev['fas_day7_count'] / 4
        lvc_prev['fas_day7_dollars'] = lvc_prev['fas_day7_dollars'] / 4
        lvc_prev = lvc_prev.rename(columns={'avg_lp2c': 'lp2c_prev'})
        
        # Create pivot tables for charts
        lvc_target['period'] = 'This Week'
        lvc_prev['period'] = '4-Wk Avg'
        lvc_combined = pd.concat([lvc_target, lvc_prev])
        
        lvc_pivot_count = lvc_combined.pivot(index='lvc_group', columns='period', values='fas_day7_count').reset_index().fillna(0)
        lvc_pivot_dollars = lvc_combined.pivot(index='lvc_group', columns='period', values='fas_day7_dollars').reset_index().fillna(0)
        
        # Calculate % change
        if 'This Week' in lvc_pivot_count.columns and '4-Wk Avg' in lvc_pivot_count.columns:
            lvc_pivot_count['pct_change'] = ((lvc_pivot_count['This Week'] - lvc_pivot_count['4-Wk Avg']) / lvc_pivot_count['4-Wk Avg'].replace(0, 1) * 100)
            lvc_pivot_dollars['pct_change'] = ((lvc_pivot_dollars['This Week'] - lvc_pivot_dollars['4-Wk Avg']) / lvc_pivot_dollars['4-Wk Avg'].replace(0, 1) * 100)
        else:
            lvc_pivot_count['pct_change'] = 0
            lvc_pivot_dollars['pct_change'] = 0
        
        # FAS Day 7 Count chart
        fig_lvc_count = go.Figure()
        if '4-Wk Avg' in lvc_pivot_count.columns:
            fig_lvc_count.add_trace(go.Bar(
                x=lvc_pivot_count['lvc_group'],
                y=lvc_pivot_count['4-Wk Avg'],
                name='4-Wk Avg',
                marker_color=COLORS['chart_gray'],
                text=lvc_pivot_count['4-Wk Avg'].apply(lambda x: f"{x:,.0f}"),
                textposition='outside',
                textfont=dict(color=COLORS['text'], size=10)
            ))
        if 'This Week' in lvc_pivot_count.columns:
            fig_lvc_count.add_trace(go.Bar(
                x=lvc_pivot_count['lvc_group'],
                y=lvc_pivot_count['This Week'],
                name='This Week',
                marker_color=COLORS['chart_yellow'],
                text=lvc_pivot_count['This Week'].apply(lambda x: f"{x:,.0f}"),
                textposition='outside',
                textfont=dict(color=COLORS['text'], size=10)
            ))
        for i, row in lvc_pivot_count.iterrows():
            pct = row['pct_change']
            color = COLORS['success'] if pct >= 0 else COLORS['danger']
            fig_lvc_count.add_annotation(
                x=row['lvc_group'], y=0, text=f"{pct:+.1f}%",
                showarrow=False, font=dict(color=color, size=11, weight='bold'), yshift=-18
            )
        fig_lvc_count.update_layout(
            title=dict(text='📊 FAS Day 7 Count by LVC Group (Vintage)', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']), margin=dict(l=60, r=40, t=60, b=50),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
            barmode='group', height=380, bargap=0.3,
        )
        
        # FAS Day 7 $ chart
        fig_lvc_dollars = go.Figure()
        if '4-Wk Avg' in lvc_pivot_dollars.columns:
            fig_lvc_dollars.add_trace(go.Bar(
                x=lvc_pivot_dollars['lvc_group'],
                y=lvc_pivot_dollars['4-Wk Avg'],
                name='4-Wk Avg',
                marker_color=COLORS['chart_gray'],
                text=lvc_pivot_dollars['4-Wk Avg'].apply(lambda x: f"${x/1000000:.1f}M" if x >= 1000000 else f"${x/1000:.0f}K"),
                textposition='outside',
                textfont=dict(color=COLORS['text'], size=10)
            ))
        if 'This Week' in lvc_pivot_dollars.columns:
            fig_lvc_dollars.add_trace(go.Bar(
                x=lvc_pivot_dollars['lvc_group'],
                y=lvc_pivot_dollars['This Week'],
                name='This Week',
                marker_color=COLORS['chart_yellow'],
                text=lvc_pivot_dollars['This Week'].apply(lambda x: f"${x/1000000:.1f}M" if x >= 1000000 else f"${x/1000:.0f}K"),
                textposition='outside',
                textfont=dict(color=COLORS['text'], size=10)
            ))
        for i, row in lvc_pivot_dollars.iterrows():
            pct = row['pct_change']
            color = COLORS['success'] if pct >= 0 else COLORS['danger']
            fig_lvc_dollars.add_annotation(
                x=row['lvc_group'], y=0, text=f"{pct:+.1f}%",
                showarrow=False, font=dict(color=color, size=11, weight='bold'), yshift=-18
            )
        fig_lvc_dollars.update_layout(
            title=dict(text='💰 FAS Day 7 $ by LVC Group (Vintage)', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']), margin=dict(l=60, r=40, t=60, b=50),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
            barmode='group', height=380, bargap=0.3,
        )
        
        # === Channel Mix Analysis v2 (Top 10 Channels + Other) ===
        # Get last 6 weeks of data for ranking channels by StS volume
        six_weeks_ago = target_date - timedelta(weeks=6)
        channel_sts_6wk = df_vintage[(df_vintage['vintage_week'] >= six_weeks_ago) & 
                                      (df_vintage['vintage_week'] <= target_date)].groupby('channel').agg({
            'sts_eligible': 'sum'
        }).reset_index().sort_values('sts_eligible', ascending=False)
        
        # Get top 10 channels
        top_10_channels = channel_sts_6wk.head(10)['channel'].tolist()
        
        # Create channel grouping (Top 10 individual, rest as 'Other')
        df_vintage['channel_grouped'] = df_vintage['channel'].apply(lambda x: x if x in top_10_channels else 'Other')
        
        # Aggregate by channel_grouped for target week
        channel_target = df_vintage[df_vintage['vintage_week'] == target_date].groupby('channel_grouped').agg({
            'lead_count': 'sum',
            'sts_eligible': 'sum',
            'fas_day7_count': 'sum',
            'fas_day7_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        channel_target = channel_target.rename(columns={'avg_lp2c': 'lp2c'})
        
        # Aggregate for 4-week average
        channel_prev = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                                  (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))].groupby('channel_grouped').agg({
            'lead_count': 'sum',
            'sts_eligible': 'sum',
            'fas_day7_count': 'sum',
            'fas_day7_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        channel_prev['lead_count'] = channel_prev['lead_count'] / 4
        channel_prev['sts_eligible'] = channel_prev['sts_eligible'] / 4
        channel_prev['fas_day7_count'] = channel_prev['fas_day7_count'] / 4
        channel_prev['fas_day7_dollars'] = channel_prev['fas_day7_dollars'] / 4
        channel_prev = channel_prev.rename(columns={'avg_lp2c': 'lp2c_prev'})
        
        # Create pivot tables for channel charts
        channel_target['period'] = 'This Week'
        channel_prev['period'] = '4-Wk Avg'
        channel_combined = pd.concat([channel_target, channel_prev])
        
        channel_pivot_count = channel_combined.pivot(index='channel_grouped', columns='period', values='fas_day7_count').reset_index().fillna(0)
        channel_pivot_dollars = channel_combined.pivot(index='channel_grouped', columns='period', values='fas_day7_dollars').reset_index().fillna(0)
        
        # Sort by This Week volume descending, but keep 'Other' at end
        if 'This Week' in channel_pivot_count.columns:
            other_row_count = channel_pivot_count[channel_pivot_count['channel_grouped'] == 'Other']
            non_other_count = channel_pivot_count[channel_pivot_count['channel_grouped'] != 'Other'].sort_values('This Week', ascending=False)
            channel_pivot_count = pd.concat([non_other_count, other_row_count]).reset_index(drop=True)
            
            other_row_dollars = channel_pivot_dollars[channel_pivot_dollars['channel_grouped'] == 'Other']
            non_other_dollars = channel_pivot_dollars[channel_pivot_dollars['channel_grouped'] != 'Other'].sort_values('This Week', ascending=False)
            channel_pivot_dollars = pd.concat([non_other_dollars, other_row_dollars]).reset_index(drop=True)
        
        # Calculate % change for channels
        if 'This Week' in channel_pivot_count.columns and '4-Wk Avg' in channel_pivot_count.columns:
            channel_pivot_count['pct_change'] = ((channel_pivot_count['This Week'] - channel_pivot_count['4-Wk Avg']) / channel_pivot_count['4-Wk Avg'].replace(0, 1) * 100)
            channel_pivot_dollars['pct_change'] = ((channel_pivot_dollars['This Week'] - channel_pivot_dollars['4-Wk Avg']) / channel_pivot_dollars['4-Wk Avg'].replace(0, 1) * 100)
        else:
            channel_pivot_count['pct_change'] = 0
            channel_pivot_dollars['pct_change'] = 0
        
        # FAS Day 7 Count by Channel chart
        fig_channel_count = go.Figure()
        if '4-Wk Avg' in channel_pivot_count.columns:
            fig_channel_count.add_trace(go.Bar(
                x=channel_pivot_count['channel_grouped'],
                y=channel_pivot_count['4-Wk Avg'],
                name='4-Wk Avg',
                marker_color=COLORS['chart_gray'],
                text=channel_pivot_count['4-Wk Avg'].apply(lambda x: f"{x:,.0f}"),
                textposition='outside',
                textfont=dict(color=COLORS['text'], size=10)
            ))
        if 'This Week' in channel_pivot_count.columns:
            fig_channel_count.add_trace(go.Bar(
                x=channel_pivot_count['channel_grouped'],
                y=channel_pivot_count['This Week'],
                name='This Week',
                marker_color=COLORS['chart_yellow'],
                text=channel_pivot_count['This Week'].apply(lambda x: f"{x:,.0f}"),
                textposition='outside',
                textfont=dict(color=COLORS['text'], size=10)
            ))
        for i, row in channel_pivot_count.iterrows():
            pct = row['pct_change']
            color = COLORS['success'] if pct >= 0 else COLORS['danger']
            fig_channel_count.add_annotation(
                x=row['channel_grouped'], y=0, text=f"{pct:+.1f}%",
                showarrow=False, font=dict(color=color, size=11, weight='bold'), yshift=-18
            )
        fig_channel_count.update_layout(
            title=dict(text='📊 FAS Day 7 Count by Channel (Top 10 + Other)', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']), margin=dict(l=60, r=40, t=60, b=80),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False, tickangle=-45),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
            barmode='group', height=420, bargap=0.3,
        )
        
        # FAS Day 7 $ by Channel chart
        fig_channel_dollars = go.Figure()
        if '4-Wk Avg' in channel_pivot_dollars.columns:
            fig_channel_dollars.add_trace(go.Bar(
                x=channel_pivot_dollars['channel_grouped'],
                y=channel_pivot_dollars['4-Wk Avg'],
                name='4-Wk Avg',
                marker_color=COLORS['chart_gray'],
                text=channel_pivot_dollars['4-Wk Avg'].apply(lambda x: f"${x/1000000:.1f}M" if x >= 1000000 else f"${x/1000:.0f}K"),
                textposition='outside',
                textfont=dict(color=COLORS['text'], size=10)
            ))
        if 'This Week' in channel_pivot_dollars.columns:
            fig_channel_dollars.add_trace(go.Bar(
                x=channel_pivot_dollars['channel_grouped'],
                y=channel_pivot_dollars['This Week'],
                name='This Week',
                marker_color=COLORS['chart_yellow'],
                text=channel_pivot_dollars['This Week'].apply(lambda x: f"${x/1000000:.1f}M" if x >= 1000000 else f"${x/1000:.0f}K"),
                textposition='outside',
                textfont=dict(color=COLORS['text'], size=10)
            ))
        for i, row in channel_pivot_dollars.iterrows():
            pct = row['pct_change']
            color = COLORS['success'] if pct >= 0 else COLORS['danger']
            fig_channel_dollars.add_annotation(
                x=row['channel_grouped'], y=0, text=f"{pct:+.1f}%",
                showarrow=False, font=dict(color=color, size=11, weight='bold'), yshift=-18
            )
        fig_channel_dollars.update_layout(
            title=dict(text='💰 FAS Day 7 $ by Channel (Top 10 + Other)', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']), margin=dict(l=60, r=40, t=60, b=80),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False, tickangle=-45),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
            barmode='group', height=420, bargap=0.3,
        )
        
        # === Avg FICO by Channel (Top 10 + Other) ===
        channel_fico_target = df_vintage[df_vintage['vintage_week'] == target_date].groupby('channel_grouped').agg({
            'avg_fico': 'mean'
        }).reset_index()
        channel_fico_prev = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                                       (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))].groupby('channel_grouped').agg({
            'avg_fico': 'mean'
        }).reset_index()
        channel_fico_prev = channel_fico_prev.rename(columns={'avg_fico': 'avg_fico_prev'})
        
        channel_fico_target['period'] = 'This Week'
        channel_fico_prev['period'] = '4-Wk Avg'
        channel_fico_combined = channel_fico_target.merge(channel_fico_prev[['channel_grouped', 'avg_fico_prev']], on='channel_grouped', how='left').fillna(0)
        
        # Sort by This Week value, keep Other at end
        other_row_fico = channel_fico_combined[channel_fico_combined['channel_grouped'] == 'Other']
        non_other_fico = channel_fico_combined[channel_fico_combined['channel_grouped'] != 'Other'].sort_values('avg_fico', ascending=False)
        channel_fico_combined = pd.concat([non_other_fico, other_row_fico]).reset_index(drop=True)
        channel_fico_combined['pct_change'] = ((channel_fico_combined['avg_fico'] - channel_fico_combined['avg_fico_prev']) / channel_fico_combined['avg_fico_prev'].replace(0, 1) * 100)
        
        fig_channel_fico = go.Figure()
        fig_channel_fico.add_trace(go.Bar(
            x=channel_fico_combined['channel_grouped'],
            y=channel_fico_combined['avg_fico_prev'],
            name='4-Wk Avg',
            marker_color=COLORS['chart_gray'],
            text=channel_fico_combined['avg_fico_prev'].apply(lambda x: f"{x:.0f}" if pd.notna(x) and x > 0 else ""),
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10)
        ))
        fig_channel_fico.add_trace(go.Bar(
            x=channel_fico_combined['channel_grouped'],
            y=channel_fico_combined['avg_fico'],
            name='This Week',
            marker_color=COLORS['chart_yellow'],
            text=channel_fico_combined['avg_fico'].apply(lambda x: f"{x:.0f}" if pd.notna(x) and x > 0 else ""),
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10)
        ))
        for i, row in channel_fico_combined.iterrows():
            pct = row['pct_change']
            color = COLORS['success'] if pct >= 0 else COLORS['danger']
            fig_channel_fico.add_annotation(
                x=row['channel_grouped'], y=0, text=f"{pct:+.1f}%",
                showarrow=False, font=dict(color=color, size=11, weight='bold'), yshift=-18
            )
        fig_channel_fico.update_layout(
            title=dict(text='📊 Avg. FICO by Channel (Top 10 + Other)', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']), margin=dict(l=60, r=40, t=60, b=80),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False, tickangle=-45),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
            barmode='group', height=420, bargap=0.3,
        )
        
        # === Avg LVC by Channel (Top 10 + Other) ===
        channel_lvc_target = df_vintage[df_vintage['vintage_week'] == target_date].groupby('channel_grouped').agg({
            'avg_lvc': 'mean'
        }).reset_index()
        channel_lvc_prev = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                                      (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))].groupby('channel_grouped').agg({
            'avg_lvc': 'mean'
        }).reset_index()
        channel_lvc_prev = channel_lvc_prev.rename(columns={'avg_lvc': 'avg_lvc_prev'})
        
        channel_lvc_target['period'] = 'This Week'
        channel_lvc_prev['period'] = '4-Wk Avg'
        channel_lvc_combined = channel_lvc_target.merge(channel_lvc_prev[['channel_grouped', 'avg_lvc_prev']], on='channel_grouped', how='left').fillna(0)
        
        # Sort by This Week value, keep Other at end
        other_row_lvc = channel_lvc_combined[channel_lvc_combined['channel_grouped'] == 'Other']
        non_other_lvc = channel_lvc_combined[channel_lvc_combined['channel_grouped'] != 'Other'].sort_values('avg_lvc', ascending=False)
        channel_lvc_combined = pd.concat([non_other_lvc, other_row_lvc]).reset_index(drop=True)
        channel_lvc_combined['pct_change'] = ((channel_lvc_combined['avg_lvc'] - channel_lvc_combined['avg_lvc_prev']) / channel_lvc_combined['avg_lvc_prev'].replace(0, 1) * 100)
        
        fig_channel_lvc = go.Figure()
        fig_channel_lvc.add_trace(go.Bar(
            x=channel_lvc_combined['channel_grouped'],
            y=channel_lvc_combined['avg_lvc_prev'],
            name='4-Wk Avg',
            marker_color=COLORS['chart_gray'],
            text=channel_lvc_combined['avg_lvc_prev'].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x > 0 else ""),
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10)
        ))
        fig_channel_lvc.add_trace(go.Bar(
            x=channel_lvc_combined['channel_grouped'],
            y=channel_lvc_combined['avg_lvc'],
            name='This Week',
            marker_color=COLORS['chart_yellow'],
            text=channel_lvc_combined['avg_lvc'].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x > 0 else ""),
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10)
        ))
        for i, row in channel_lvc_combined.iterrows():
            pct = row['pct_change']
            color = COLORS['success'] if pct >= 0 else COLORS['danger']
            fig_channel_lvc.add_annotation(
                x=row['channel_grouped'], y=0, text=f"{pct:+.1f}%",
                showarrow=False, font=dict(color=color, size=11, weight='bold'), yshift=-18
            )
        fig_channel_lvc.update_layout(
            title=dict(text='📊 Avg. Lead Value Cohort by Channel (Top 10 + Other)', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']), margin=dict(l=60, r=40, t=60, b=80),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False, tickangle=-45),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
            barmode='group', height=420, bargap=0.3,
        )
        
        # === Gross Leads Trending by Channel (Top 10 + Other) ===
        channel_trending = df_vintage.groupby(['vintage_week', 'channel_grouped']).agg({
            'lead_count': 'sum'
        }).reset_index()
        channel_trending = channel_trending.sort_values(['channel_grouped', 'vintage_week'])
        channel_trending['vintage_week'] = pd.to_datetime(channel_trending['vintage_week'])
        
        # Get sorted weeks and filter to last 6 weeks before target
        all_weeks_ch = sorted(channel_trending['vintage_week'].unique())
        target_idx_ch = all_weeks_ch.index(pd.to_datetime(target_date)) if pd.to_datetime(target_date) in all_weeks_ch else len(all_weeks_ch) - 1
        last_6_weeks_ch = all_weeks_ch[max(0, target_idx_ch - 6):target_idx_ch + 1]
        
        channel_trending = channel_trending[channel_trending['vintage_week'].isin(last_6_weeks_ch)]
        channel_trending['week_label'] = channel_trending['vintage_week'].dt.strftime('%b %-d, %y')
        
        # Get channel groups excluding 'Other', sorted by total volume
        channel_groups = channel_trending[channel_trending['channel_grouped'] != 'Other'].groupby('channel_grouped')['lead_count'].sum().sort_values(ascending=False).index.tolist()
        
        # Calculate % of total for each week
        week_totals_ch = channel_trending.groupby('vintage_week')['lead_count'].sum().reset_index()
        week_totals_ch.columns = ['vintage_week', 'week_total']
        channel_trending = channel_trending.merge(week_totals_ch, on='vintage_week')
        channel_trending['pct_of_total'] = (channel_trending['lead_count'] / channel_trending['week_total'] * 100).round(1)
        channel_trending['wow_change'] = channel_trending.groupby('channel_grouped')['pct_of_total'].diff()
        
        n_channel_groups = len(channel_groups)
        fig_channel_gross_leads = make_subplots(rows=n_channel_groups, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[1] * n_channel_groups, specs=[[{"secondary_y": True}] for _ in range(n_channel_groups)])
        
        insights_data_ch = []
        for i, group in enumerate(channel_groups):
            group_data = channel_trending[channel_trending['channel_grouped'] == group].sort_values('vintage_week')
            if group_data.empty:
                continue
            avg_val = group_data['lead_count'].mean()
            avg_pct = group_data['pct_of_total'].mean()
            last_6 = group_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change = wow_changes.mean()
                std_change = wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                significance = ""
                if abs(z_score) >= 2.58: significance = "⚠️ Highly Significant (99%)"
                elif abs(z_score) >= 1.96: significance = "📊 Significant (95%)"
                elif abs(z_score) >= 1.645: significance = "📈 Marginally Significant (90%)"
                elif abs(z_score) >= 1.495: significance = "📉 Notable movement"
                insights_data_ch.append({'group': group, 'latest_val': group_data['lead_count'].iloc[-1],
                    'latest_pct': group_data['pct_of_total'].iloc[-1], 'change': latest_change,
                    'z_score': z_score, 'significance': significance, 'avg_val': avg_val, 'avg_pct': avg_pct})
            
            fig_channel_gross_leads.add_trace(go.Bar(x=group_data['week_label'], y=group_data['lead_count'],
                marker_color='#3498db', text=group_data['lead_count'].apply(lambda x: f"{x:,.0f}"),
                textposition='outside', textfont=dict(size=9, color=COLORS['text']), showlegend=False),
                row=i+1, col=1, secondary_y=False)
            fig_channel_gross_leads.add_trace(go.Scatter(x=group_data['week_label'], y=group_data['pct_of_total'],
                mode='lines+markers', line=dict(color=COLORS['chart_yellow'], width=2),
                marker=dict(size=5), showlegend=False), row=i+1, col=1, secondary_y=True)
            
            y_max = group_data['lead_count'].max() * 1.3
            yref_str_ch = 'y domain' if i == 0 else f'y{i*2+1} domain'
            fig_channel_gross_leads.update_yaxes(title_text='', range=[0, y_max], tickfont=dict(size=9, color=COLORS['text']),
                showgrid=True, gridcolor=COLORS['border'], row=i+1, col=1, secondary_y=False)
            fig_channel_gross_leads.update_yaxes(range=[0, 50], tickfont=dict(size=9, color=COLORS['chart_yellow']),
                showgrid=False, ticksuffix='%', row=i+1, col=1, secondary_y=True)
            
            # Truncate long channel names for label
            display_name = group[:15] + '...' if len(group) > 15 else group
            fig_channel_gross_leads.add_annotation(x=-0.03, y=0.5, xref='paper', yref=yref_str_ch, text=f'<b>{display_name}</b>',
                showarrow=False, font=dict(size=10, color=COLORS['chart_yellow'], family='Arial Black'),
                xanchor='right', yanchor='middle', textangle=0)
            fig_channel_gross_leads.add_annotation(x=1.005, y=0.5, xref='paper', yref=yref_str_ch,
                text=f"<b>Avg:</b><br><span style='color:#f0b429'>{avg_pct:.1f}%</span><br><span style='color:#3498db'>{avg_val:,.0f}</span>",
                showarrow=False, font=dict(size=16, color=COLORS['text']), xanchor='left', yanchor='middle', align='left')
        
        for i in range(1, n_channel_groups + 1):
            fig_channel_gross_leads.update_xaxes(showgrid=False, showticklabels=(i == n_channel_groups), tickangle=0,
                tickfont=dict(size=10, color=COLORS['text']), row=i, col=1)
        fig_channel_gross_leads.update_layout(title=dict(text='<b>Gross Lead Qty by Channel (6 Weeks Prior)</b>', font=dict(color=COLORS['text'], size=14), x=0, xanchor='left'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=160, r=100, t=50, b=50), height=550, showlegend=False, bargap=0.3)
        
        # Channel Gross Leads Insights box
        insights_items_ch = [html.H5("📊 Statistical Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}),
            html.P("6-Week WoW Z-Score Analysis", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})]
        for item in insights_data_ch:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            display_name = item['group'][:12] + '...' if len(item['group']) > 12 else item['group']
            insights_items_ch.append(html.Div([
                html.Div([html.Span(f"{display_name}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '11px'}),
                    html.Span(f" {item['latest_val']:,.0f}", style={'color': COLORS['primary'], 'fontSize': '11px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div([html.Span(f"{change_symbol} {abs(item['change']):.2f}pp", style={'color': change_color, 'fontSize': '10px'}),
                    html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'}) if item['significance'] else html.Div("No significant change", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'}),
            ], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '6px', 'marginBottom': '6px'}))
        insights_box_ch = html.Div(insights_items_ch, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '15px',
            'border': f"1px solid {COLORS['border']}", 'height': '590px', 'overflowY': 'auto', 'marginBottom': '16px', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # === Channel Sent to Sales Trending ===
        ch_sts_agg = df_vintage.groupby(['vintage_week', 'channel_grouped']).agg({'sts_eligible': 'sum'}).reset_index()
        ch_sts_trending = ch_sts_agg[ch_sts_agg['channel_grouped'].isin(channel_groups)].copy()
        ch_sts_trending['vintage_week'] = pd.to_datetime(ch_sts_trending['vintage_week'])
        ch_sts_trending = ch_sts_trending[ch_sts_trending['vintage_week'].isin(last_6_weeks_ch)]
        ch_sts_trending['week_label'] = ch_sts_trending['vintage_week'].dt.strftime('%b %-d, %y')
        
        week_totals_ch_sts = ch_sts_trending.groupby('vintage_week')['sts_eligible'].sum().reset_index()
        week_totals_ch_sts.columns = ['vintage_week', 'week_total']
        ch_sts_trending = ch_sts_trending.merge(week_totals_ch_sts, on='vintage_week')
        ch_sts_trending['pct_of_total'] = (ch_sts_trending['sts_eligible'] / ch_sts_trending['week_total'] * 100).round(1)
        ch_sts_trending['wow_change'] = ch_sts_trending.groupby('channel_grouped')['pct_of_total'].diff()
        
        fig_ch_sts = make_subplots(rows=n_channel_groups, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[1] * n_channel_groups, specs=[[{"secondary_y": True}] for _ in range(n_channel_groups)])
        
        insights_data_ch_sts = []
        for i, group in enumerate(channel_groups):
            group_data = ch_sts_trending[ch_sts_trending['channel_grouped'] == group].sort_values('vintage_week')
            if group_data.empty:
                continue
            avg_val = group_data['sts_eligible'].mean()
            avg_pct = group_data['pct_of_total'].mean()
            last_6 = group_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change, std_change = wow_changes.mean(), wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                significance = ""
                if abs(z_score) >= 2.58: significance = "⚠️ Highly Significant (99%)"
                elif abs(z_score) >= 1.96: significance = "📊 Significant (95%)"
                elif abs(z_score) >= 1.645: significance = "📈 Marginally Significant (90%)"
                elif abs(z_score) >= 1.495: significance = "📉 Notable movement"
                insights_data_ch_sts.append({'group': group, 'latest_val': group_data['sts_eligible'].iloc[-1],
                    'change': latest_change, 'z_score': z_score, 'significance': significance, 'avg_val': avg_val, 'avg_pct': avg_pct})
            
            fig_ch_sts.add_trace(go.Bar(x=group_data['week_label'], y=group_data['sts_eligible'],
                marker_color='#9b59b6', text=group_data['sts_eligible'].apply(lambda x: f"{x:,.0f}"),
                textposition='outside', textfont=dict(size=9, color=COLORS['text']), showlegend=False), row=i+1, col=1, secondary_y=False)
            fig_ch_sts.add_trace(go.Scatter(x=group_data['week_label'], y=group_data['pct_of_total'],
                mode='lines+markers', line=dict(color=COLORS['chart_yellow'], width=2), marker=dict(size=5), showlegend=False), row=i+1, col=1, secondary_y=True)
            
            y_max = group_data['sts_eligible'].max() * 1.3
            yref_str = 'y domain' if i == 0 else f'y{i*2+1} domain'
            fig_ch_sts.update_yaxes(range=[0, y_max], tickfont=dict(size=9, color=COLORS['text']), showgrid=True, gridcolor=COLORS['border'], row=i+1, col=1, secondary_y=False)
            fig_ch_sts.update_yaxes(range=[0, 50], tickfont=dict(size=9, color=COLORS['chart_yellow']), showgrid=False, ticksuffix='%', row=i+1, col=1, secondary_y=True)
            display_name = group[:15] + '...' if len(group) > 15 else group
            fig_ch_sts.add_annotation(x=-0.03, y=0.5, xref='paper', yref=yref_str, text=f'<b>{display_name}</b>', showarrow=False, font=dict(size=10, color=COLORS['chart_yellow'], family='Arial Black'), xanchor='right', yanchor='middle', textangle=0)
            fig_ch_sts.add_annotation(x=1.005, y=0.5, xref='paper', yref=yref_str, text=f"<b>Avg:</b><br><span style='color:#f0b429'>{avg_pct:.1f}%</span><br><span style='color:#9b59b6'>{avg_val:,.0f}</span>", showarrow=False, font=dict(size=16, color=COLORS['text']), xanchor='left', yanchor='middle', align='left')
        
        for i in range(1, n_channel_groups + 1):
            fig_ch_sts.update_xaxes(showgrid=False, showticklabels=(i == n_channel_groups), tickangle=0, tickfont=dict(size=10, color=COLORS['text']), row=i, col=1)
        fig_ch_sts.update_layout(title=dict(text='<b>Sent to Sales by Channel (6 Weeks Prior)</b>', font=dict(color=COLORS['text'], size=14), x=0, xanchor='left'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), margin=dict(l=160, r=100, t=50, b=50), height=150 * n_channel_groups, showlegend=False, bargap=0.3)
        
        insights_items_ch_sts = [html.H5("📊 Statistical Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}), html.P("6-Week WoW Z-Score Analysis", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})]
        for item in insights_data_ch_sts:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            display_name = item['group'][:12] + '...' if len(item['group']) > 12 else item['group']
            insights_items_ch_sts.append(html.Div([html.Div([html.Span(f"{display_name}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '11px'}), html.Span(f" {item['latest_val']:,.0f}", style={'color': COLORS['primary'], 'fontSize': '11px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}), html.Div([html.Span(f"{change_symbol} {abs(item['change']):.2f}pp", style={'color': change_color, 'fontSize': '10px'}), html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}), html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'}) if item['significance'] else html.Div("No significant change", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'})], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '6px', 'marginBottom': '6px'}))
        insights_box_ch_sts = html.Div(insights_items_ch_sts, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '15px', 'border': f"1px solid {COLORS['border']}", 'height': '590px', 'overflowY': 'auto', 'marginBottom': '16px', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # === Channel Avg LP2C Trending ===
        ch_lp2c_agg = df_vintage.groupby(['vintage_week', 'channel_grouped']).agg({'avg_lp2c': 'mean'}).reset_index()
        ch_lp2c_trending = ch_lp2c_agg[ch_lp2c_agg['channel_grouped'].isin(channel_groups)].copy()
        ch_lp2c_trending['vintage_week'] = pd.to_datetime(ch_lp2c_trending['vintage_week'])
        ch_lp2c_trending = ch_lp2c_trending[ch_lp2c_trending['vintage_week'].isin(last_6_weeks_ch)]
        ch_lp2c_trending['week_label'] = ch_lp2c_trending['vintage_week'].dt.strftime('%b %-d, %y')
        ch_lp2c_trending['wow_change'] = ch_lp2c_trending.groupby('channel_grouped')['avg_lp2c'].diff()
        
        fig_ch_lp2c = make_subplots(rows=n_channel_groups, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[1] * n_channel_groups, specs=[[{"secondary_y": False}] for _ in range(n_channel_groups)])
        
        insights_data_ch_lp2c = []
        for i, group in enumerate(channel_groups):
            group_data = ch_lp2c_trending[ch_lp2c_trending['channel_grouped'] == group].sort_values('vintage_week')
            if group_data.empty:
                continue
            avg_val = group_data['avg_lp2c'].mean()
            last_6 = group_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change, std_change = wow_changes.mean(), wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                significance = ""
                if abs(z_score) >= 2.58: significance = "⚠️ Highly Significant (99%)"
                elif abs(z_score) >= 1.96: significance = "📊 Significant (95%)"
                elif abs(z_score) >= 1.645: significance = "📈 Marginally Significant (90%)"
                elif abs(z_score) >= 1.495: significance = "📉 Notable movement"
                insights_data_ch_lp2c.append({'group': group, 'latest_val': group_data['avg_lp2c'].iloc[-1],
                    'change': latest_change, 'z_score': z_score, 'significance': significance, 'avg_val': avg_val})
            
            fig_ch_lp2c.add_trace(go.Bar(x=group_data['week_label'], y=group_data['avg_lp2c'],
                marker_color='#3498db', text=group_data['avg_lp2c'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside', textfont=dict(size=9, color=COLORS['text']), showlegend=False), row=i+1, col=1)
            
            y_max = group_data['avg_lp2c'].max() * 1.3
            yref_str = 'y domain' if i == 0 else f'y{i+1} domain'
            fig_ch_lp2c.update_yaxes(range=[0, y_max], tickfont=dict(size=9, color=COLORS['text']), showgrid=True, gridcolor=COLORS['border'], row=i+1, col=1)
            display_name = group[:15] + '...' if len(group) > 15 else group
            fig_ch_lp2c.add_annotation(x=-0.03, y=0.5, xref='paper', yref=yref_str, text=f'<b>{display_name}</b>', showarrow=False, font=dict(size=10, color=COLORS['chart_yellow'], family='Arial Black'), xanchor='right', yanchor='middle', textangle=0)
            fig_ch_lp2c.add_annotation(x=1.005, y=0.5, xref='paper', yref=yref_str, text=f"<b>Avg:</b><br><span style='color:#3498db'>{avg_val:.1f}%</span>", showarrow=False, font=dict(size=16, color=COLORS['text']), xanchor='left', yanchor='middle', align='left')
        
        for i in range(1, n_channel_groups + 1):
            fig_ch_lp2c.update_xaxes(showgrid=False, showticklabels=(i == n_channel_groups), tickangle=0, tickfont=dict(size=10, color=COLORS['text']), row=i, col=1)
        fig_ch_lp2c.update_layout(title=dict(text='<b>Avg. LP2C by Channel (6 Weeks Prior)</b>', font=dict(color=COLORS['text'], size=14), x=0, xanchor='left'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), margin=dict(l=160, r=100, t=50, b=50), height=150 * n_channel_groups, showlegend=False, bargap=0.3)
        
        insights_items_ch_lp2c = [html.H5("📊 Statistical Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}), html.P("6-Week WoW Z-Score Analysis", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})]
        for item in insights_data_ch_lp2c:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            display_name = item['group'][:12] + '...' if len(item['group']) > 12 else item['group']
            insights_items_ch_lp2c.append(html.Div([html.Div([html.Span(f"{display_name}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '11px'}), html.Span(f" {item['latest_val']:.1f}%", style={'color': COLORS['primary'], 'fontSize': '11px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}), html.Div([html.Span(f"{change_symbol} {abs(item['change']):.2f}pp", style={'color': change_color, 'fontSize': '10px'}), html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}), html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'}) if item['significance'] else html.Div("No significant change", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'})], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '6px', 'marginBottom': '6px'}))
        insights_box_ch_lp2c = html.Div(insights_items_ch_lp2c, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '15px', 'border': f"1px solid {COLORS['border']}", 'height': '590px', 'overflowY': 'auto', 'marginBottom': '16px', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # === Channel SF Contact % Day 7 Trending ===
        ch_sf_agg = df_vintage.groupby(['vintage_week', 'channel_grouped']).agg({'sf_contact_day7_count': 'sum', 'day7_eligible_sts': 'sum'}).reset_index()
        ch_sf_trending = ch_sf_agg[ch_sf_agg['channel_grouped'].isin(channel_groups)].copy()
        ch_sf_trending['vintage_week'] = pd.to_datetime(ch_sf_trending['vintage_week'])
        ch_sf_trending = ch_sf_trending[ch_sf_trending['vintage_week'].isin(last_6_weeks_ch)]
        ch_sf_trending['week_label'] = ch_sf_trending['vintage_week'].dt.strftime('%b %-d, %y')
        ch_sf_trending['sf_contact_rate'] = (ch_sf_trending['sf_contact_day7_count'] / ch_sf_trending['day7_eligible_sts'].replace(0, 1) * 100).round(2)
        ch_sf_trending['wow_change'] = ch_sf_trending.groupby('channel_grouped')['sf_contact_rate'].diff()
        
        fig_ch_sf = make_subplots(rows=n_channel_groups, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[1] * n_channel_groups, specs=[[{"secondary_y": True}] for _ in range(n_channel_groups)])
        
        insights_data_ch_sf = []
        for i, group in enumerate(channel_groups):
            group_data = ch_sf_trending[ch_sf_trending['channel_grouped'] == group].sort_values('vintage_week')
            if group_data.empty:
                continue
            avg_rate = group_data['sf_contact_rate'].mean()
            avg_count = group_data['sf_contact_day7_count'].mean()
            last_6 = group_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change, std_change = wow_changes.mean(), wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                significance = ""
                if abs(z_score) >= 2.58: significance = "⚠️ Highly Significant (99%)"
                elif abs(z_score) >= 1.96: significance = "📊 Significant (95%)"
                elif abs(z_score) >= 1.645: significance = "📈 Marginally Significant (90%)"
                elif abs(z_score) >= 1.495: significance = "📉 Notable movement"
                insights_data_ch_sf.append({'group': group, 'latest_rate': group_data['sf_contact_rate'].iloc[-1],
                    'change': latest_change, 'z_score': z_score, 'significance': significance, 'avg_rate': avg_rate})
            
            fig_ch_sf.add_trace(go.Bar(x=group_data['week_label'], y=group_data['sf_contact_day7_count'],
                marker_color='#e74c3c', marker_opacity=0.25, showlegend=False), row=i+1, col=1, secondary_y=True)
            fig_ch_sf.add_trace(go.Scatter(x=group_data['week_label'], y=group_data['sf_contact_rate'],
                mode='lines+markers+text', line=dict(color=COLORS['chart_yellow'], width=4), marker=dict(size=8, color=COLORS['chart_yellow']),
                text=group_data['sf_contact_rate'].apply(lambda x: f"{x:.1f}%"), textposition='top center', textfont=dict(size=10, color='#FFFFFF', family='Arial Black'), showlegend=False), row=i+1, col=1, secondary_y=False)
            
            y_max_count = group_data['sf_contact_day7_count'].max() * 1.3
            y_max_rate = group_data['sf_contact_rate'].max() * 1.3
            yref_str = 'y domain' if i == 0 else f'y{i*2+1} domain'
            fig_ch_sf.update_yaxes(range=[0, max(y_max_rate, 100)], tickfont=dict(size=9, color=COLORS['chart_yellow']), showgrid=True, gridcolor=COLORS['border'], ticksuffix='%', row=i+1, col=1, secondary_y=False)
            fig_ch_sf.update_yaxes(range=[0, y_max_count], tickfont=dict(size=9, color=COLORS['text']), showgrid=False, row=i+1, col=1, secondary_y=True)
            display_name = group[:15] + '...' if len(group) > 15 else group
            fig_ch_sf.add_annotation(x=-0.03, y=0.5, xref='paper', yref=yref_str, text=f'<b>{display_name}</b>', showarrow=False, font=dict(size=10, color=COLORS['chart_yellow'], family='Arial Black'), xanchor='right', yanchor='middle', textangle=0)
            fig_ch_sf.add_annotation(x=1.005, y=0.5, xref='paper', yref=yref_str, text=f"<b>Avg:</b><br><span style='color:#f0b429'>{avg_rate:.1f}%</span>", showarrow=False, font=dict(size=16, color=COLORS['text']), xanchor='left', yanchor='middle', align='left')
        
        for i in range(1, n_channel_groups + 1):
            fig_ch_sf.update_xaxes(showgrid=False, showticklabels=(i == n_channel_groups), tickangle=0, tickfont=dict(size=10, color=COLORS['text']), row=i, col=1)
        fig_ch_sf.update_layout(title=dict(text='<b>SF Contact % Day 7 by Channel (6 Weeks Prior)</b>', font=dict(color=COLORS['text'], size=14), x=0, xanchor='left'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), margin=dict(l=160, r=100, t=50, b=50), height=150 * n_channel_groups, showlegend=False, bargap=0.3)
        
        insights_items_ch_sf = [html.H5("📊 Statistical Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}), html.P("6-Week WoW Z-Score Analysis", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})]
        for item in insights_data_ch_sf:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            display_name = item['group'][:12] + '...' if len(item['group']) > 12 else item['group']
            insights_items_ch_sf.append(html.Div([html.Div([html.Span(f"{display_name}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '11px'}), html.Span(f" {item['latest_rate']:.1f}%", style={'color': COLORS['primary'], 'fontSize': '11px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}), html.Div([html.Span(f"{change_symbol} {abs(item['change']):.2f}pp", style={'color': change_color, 'fontSize': '10px'}), html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}), html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'}) if item['significance'] else html.Div("No significant change", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'})], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '6px', 'marginBottom': '6px'}))
        insights_box_ch_sf = html.Div(insights_items_ch_sf, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '15px', 'border': f"1px solid {COLORS['border']}", 'height': '590px', 'overflowY': 'auto', 'marginBottom': '16px', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # === Channel FAS Day 7 % Trending ===
        ch_fas_pct_agg = df_vintage.groupby(['vintage_week', 'channel_grouped']).agg({'fas_day7_count': 'sum', 'day7_eligible_sts': 'sum'}).reset_index()
        ch_fas_pct_trending = ch_fas_pct_agg[ch_fas_pct_agg['channel_grouped'].isin(channel_groups)].copy()
        ch_fas_pct_trending['vintage_week'] = pd.to_datetime(ch_fas_pct_trending['vintage_week'])
        ch_fas_pct_trending = ch_fas_pct_trending[ch_fas_pct_trending['vintage_week'].isin(last_6_weeks_ch)]
        ch_fas_pct_trending['week_label'] = ch_fas_pct_trending['vintage_week'].dt.strftime('%b %-d, %y')
        ch_fas_pct_trending['fas_day7_rate'] = (ch_fas_pct_trending['fas_day7_count'] / ch_fas_pct_trending['day7_eligible_sts'].replace(0, 1) * 100).round(2)
        ch_fas_pct_trending['wow_change'] = ch_fas_pct_trending.groupby('channel_grouped')['fas_day7_rate'].diff()
        
        fig_ch_fas_pct = make_subplots(rows=n_channel_groups, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[1] * n_channel_groups, specs=[[{"secondary_y": True}] for _ in range(n_channel_groups)])
        
        insights_data_ch_fas_pct = []
        for i, group in enumerate(channel_groups):
            group_data = ch_fas_pct_trending[ch_fas_pct_trending['channel_grouped'] == group].sort_values('vintage_week')
            if group_data.empty:
                continue
            avg_rate = group_data['fas_day7_rate'].mean()
            avg_count = group_data['fas_day7_count'].mean()
            last_6 = group_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change, std_change = wow_changes.mean(), wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                significance = ""
                if abs(z_score) >= 2.58: significance = "⚠️ Highly Significant (99%)"
                elif abs(z_score) >= 1.96: significance = "📊 Significant (95%)"
                elif abs(z_score) >= 1.645: significance = "📈 Marginally Significant (90%)"
                elif abs(z_score) >= 1.495: significance = "📉 Notable movement"
                insights_data_ch_fas_pct.append({'group': group, 'latest_rate': group_data['fas_day7_rate'].iloc[-1],
                    'change': latest_change, 'z_score': z_score, 'significance': significance, 'avg_rate': avg_rate})
            
            fig_ch_fas_pct.add_trace(go.Bar(x=group_data['week_label'], y=group_data['fas_day7_count'],
                marker_color='#1abc9c', marker_opacity=0.25, showlegend=False), row=i+1, col=1, secondary_y=True)
            fig_ch_fas_pct.add_trace(go.Scatter(x=group_data['week_label'], y=group_data['fas_day7_rate'],
                mode='lines+markers+text', line=dict(color=COLORS['chart_yellow'], width=4), marker=dict(size=8, color=COLORS['chart_yellow']),
                text=group_data['fas_day7_rate'].apply(lambda x: f"{x:.1f}%"), textposition='top center', textfont=dict(size=10, color='#FFFFFF', family='Arial Black'), showlegend=False), row=i+1, col=1, secondary_y=False)
            
            y_max_count = group_data['fas_day7_count'].max() * 1.3
            y_max_rate = group_data['fas_day7_rate'].max() * 1.3
            yref_str = 'y domain' if i == 0 else f'y{i*2+1} domain'
            fig_ch_fas_pct.update_yaxes(range=[0, max(y_max_rate, 50)], tickfont=dict(size=9, color=COLORS['chart_yellow']), showgrid=True, gridcolor=COLORS['border'], ticksuffix='%', row=i+1, col=1, secondary_y=False)
            fig_ch_fas_pct.update_yaxes(range=[0, y_max_count], tickfont=dict(size=9, color=COLORS['text']), showgrid=False, row=i+1, col=1, secondary_y=True)
            display_name = group[:15] + '...' if len(group) > 15 else group
            fig_ch_fas_pct.add_annotation(x=-0.03, y=0.5, xref='paper', yref=yref_str, text=f'<b>{display_name}</b>', showarrow=False, font=dict(size=10, color=COLORS['chart_yellow'], family='Arial Black'), xanchor='right', yanchor='middle', textangle=0)
            fig_ch_fas_pct.add_annotation(x=1.005, y=0.5, xref='paper', yref=yref_str, text=f"<b>Avg:</b><br><span style='color:#f0b429'>{avg_rate:.1f}%</span>", showarrow=False, font=dict(size=16, color=COLORS['text']), xanchor='left', yanchor='middle', align='left')
        
        for i in range(1, n_channel_groups + 1):
            fig_ch_fas_pct.update_xaxes(showgrid=False, showticklabels=(i == n_channel_groups), tickangle=0, tickfont=dict(size=10, color=COLORS['text']), row=i, col=1)
        fig_ch_fas_pct.update_layout(title=dict(text='<b>FAS Day 7 % by Channel (6 Weeks Prior)</b>', font=dict(color=COLORS['text'], size=14), x=0, xanchor='left'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), margin=dict(l=160, r=100, t=50, b=50), height=150 * n_channel_groups, showlegend=False, bargap=0.3)
        
        insights_items_ch_fas_pct = [html.H5("📊 Statistical Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}), html.P("6-Week WoW Z-Score Analysis", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})]
        for item in insights_data_ch_fas_pct:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            display_name = item['group'][:12] + '...' if len(item['group']) > 12 else item['group']
            insights_items_ch_fas_pct.append(html.Div([html.Div([html.Span(f"{display_name}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '11px'}), html.Span(f" {item['latest_rate']:.1f}%", style={'color': COLORS['primary'], 'fontSize': '11px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}), html.Div([html.Span(f"{change_symbol} {abs(item['change']):.2f}pp", style={'color': change_color, 'fontSize': '10px'}), html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}), html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'}) if item['significance'] else html.Div("No significant change", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'})], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '6px', 'marginBottom': '6px'}))
        insights_box_ch_fas_pct = html.Div(insights_items_ch_fas_pct, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '15px', 'border': f"1px solid {COLORS['border']}", 'height': '590px', 'overflowY': 'auto', 'marginBottom': '16px', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # === Channel FAS Day 7 $ Trending ===
        ch_fas_dollar_agg = df_vintage.groupby(['vintage_week', 'channel_grouped']).agg({'fas_day7_dollars': 'sum', 'fas_day7_count': 'sum'}).reset_index()
        ch_fas_dollar_trending = ch_fas_dollar_agg[ch_fas_dollar_agg['channel_grouped'].isin(channel_groups)].copy()
        ch_fas_dollar_trending['vintage_week'] = pd.to_datetime(ch_fas_dollar_trending['vintage_week'])
        ch_fas_dollar_trending = ch_fas_dollar_trending[ch_fas_dollar_trending['vintage_week'].isin(last_6_weeks_ch)]
        ch_fas_dollar_trending['week_label'] = ch_fas_dollar_trending['vintage_week'].dt.strftime('%b %-d, %y')
        ch_fas_dollar_trending['avg_fas_dollars'] = (ch_fas_dollar_trending['fas_day7_dollars'] / ch_fas_dollar_trending['fas_day7_count'].replace(0, 1)).round(0)
        ch_fas_dollar_trending['wow_change'] = ch_fas_dollar_trending.groupby('channel_grouped')['fas_day7_dollars'].diff()
        
        fig_ch_fas_dollar = make_subplots(rows=n_channel_groups, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[1] * n_channel_groups, specs=[[{"secondary_y": True}] for _ in range(n_channel_groups)])
        
        insights_data_ch_fas_dollar = []
        for i, group in enumerate(channel_groups):
            group_data = ch_fas_dollar_trending[ch_fas_dollar_trending['channel_grouped'] == group].sort_values('vintage_week')
            if group_data.empty:
                continue
            avg_total = group_data['fas_day7_dollars'].mean()
            avg_per_fas = group_data['avg_fas_dollars'].mean()
            last_6 = group_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change, std_change = wow_changes.mean(), wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                significance = ""
                if abs(z_score) >= 2.58: significance = "⚠️ Highly Significant (99%)"
                elif abs(z_score) >= 1.96: significance = "📊 Significant (95%)"
                elif abs(z_score) >= 1.645: significance = "📈 Marginally Significant (90%)"
                elif abs(z_score) >= 1.495: significance = "📉 Notable movement"
                insights_data_ch_fas_dollar.append({'group': group, 'latest_total': group_data['fas_day7_dollars'].iloc[-1],
                    'change': latest_change, 'z_score': z_score, 'significance': significance, 'avg_total': avg_total, 'avg_per_fas': avg_per_fas})
            
            fig_ch_fas_dollar.add_trace(go.Bar(x=group_data['week_label'], y=group_data['fas_day7_dollars'],
                marker_color='#3498db', text=group_data['fas_day7_dollars'].apply(lambda x: f"${x/1000:,.0f}K"),
                textposition='outside', textfont=dict(size=9, color=COLORS['text']), showlegend=False), row=i+1, col=1, secondary_y=False)
            fig_ch_fas_dollar.add_trace(go.Scatter(x=group_data['week_label'], y=group_data['avg_fas_dollars'],
                mode='lines+markers', line=dict(color=COLORS['chart_yellow'], width=2), marker=dict(size=5), showlegend=False), row=i+1, col=1, secondary_y=True)
            
            y_max = group_data['fas_day7_dollars'].max() * 1.3
            y2_max = group_data['avg_fas_dollars'].max() * 1.3
            yref_str = 'y domain' if i == 0 else f'y{i*2+1} domain'
            fig_ch_fas_dollar.update_yaxes(range=[0, y_max], tickfont=dict(size=9, color=COLORS['text']), showgrid=True, gridcolor=COLORS['border'], row=i+1, col=1, secondary_y=False)
            fig_ch_fas_dollar.update_yaxes(range=[0, y2_max], tickfont=dict(size=9, color=COLORS['chart_yellow']), showgrid=False, tickprefix='$', row=i+1, col=1, secondary_y=True)
            display_name = group[:15] + '...' if len(group) > 15 else group
            fig_ch_fas_dollar.add_annotation(x=-0.03, y=0.5, xref='paper', yref=yref_str, text=f'<b>{display_name}</b>', showarrow=False, font=dict(size=10, color=COLORS['chart_yellow'], family='Arial Black'), xanchor='right', yanchor='middle', textangle=0)
            fig_ch_fas_dollar.add_annotation(x=1.005, y=0.5, xref='paper', yref=yref_str, text=f"<b>Avg:</b><br><span style='color:#f0b429'>${avg_per_fas:,.0f}</span><br><span style='color:#3498db'>${avg_total/1000:,.0f}K</span>", showarrow=False, font=dict(size=16, color=COLORS['text']), xanchor='left', yanchor='middle', align='left')
        
        for i in range(1, n_channel_groups + 1):
            fig_ch_fas_dollar.update_xaxes(showgrid=False, showticklabels=(i == n_channel_groups), tickangle=0, tickfont=dict(size=10, color=COLORS['text']), row=i, col=1)
        fig_ch_fas_dollar.update_layout(title=dict(text='<b>FAS Day 7 $ by Channel (6 Weeks Prior)</b>', font=dict(color=COLORS['text'], size=14), x=0, xanchor='left'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), margin=dict(l=160, r=100, t=50, b=50), height=150 * n_channel_groups, showlegend=False, bargap=0.3)
        
        insights_items_ch_fas_dollar = [html.H5("📊 Statistical Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}), html.P("6-Week WoW Z-Score Analysis", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})]
        for item in insights_data_ch_fas_dollar:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            display_name = item['group'][:12] + '...' if len(item['group']) > 12 else item['group']
            insights_items_ch_fas_dollar.append(html.Div([html.Div([html.Span(f"{display_name}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '11px'}), html.Span(f" ${item['latest_total']/1000:,.0f}K", style={'color': COLORS['primary'], 'fontSize': '11px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}), html.Div([html.Span(f"{change_symbol} ${abs(item['change'])/1000:,.0f}K", style={'color': change_color, 'fontSize': '10px'}), html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}), html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'}) if item['significance'] else html.Div("No significant change", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'})], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '6px', 'marginBottom': '6px'}))
        insights_box_ch_fas_dollar = html.Div(insights_items_ch_fas_dollar, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '15px', 'border': f"1px solid {COLORS['border']}", 'height': '590px', 'overflowY': 'auto', 'marginBottom': '16px', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # === Gross Leads % of Total Trending by LVC Group ===
        # Get all weeks data for trending
        lvc_trending = df_vintage.groupby(['vintage_week', 'lvc_group']).agg({
            'lead_count': 'sum'
        }).reset_index()
        lvc_trending = lvc_trending.sort_values(['lvc_group', 'vintage_week'])
        
        # Filter to 6 weeks prior to selected week (inclusive of selected week)
        all_weeks = sorted(lvc_trending['vintage_week'].unique())
        # Find weeks up to and including target_date, then take last 7 (selected + 6 prior)
        weeks_up_to_target = [w for w in all_weeks if w <= target_date]
        last_6_weeks = weeks_up_to_target[-7:] if len(weeks_up_to_target) >= 7 else weeks_up_to_target
        lvc_trending = lvc_trending[lvc_trending['vintage_week'].isin(last_6_weeks)]
        
        # Date format like "Sep 5, 25"
        lvc_trending['week_label'] = lvc_trending['vintage_week'].dt.strftime('%b %-d, %y')
        
        # Calculate % of Total for each week
        week_totals = lvc_trending.groupby('vintage_week')['lead_count'].sum().reset_index()
        week_totals.columns = ['vintage_week', 'week_total']
        lvc_trending = lvc_trending.merge(week_totals, on='vintage_week')
        lvc_trending['pct_of_total'] = (lvc_trending['lead_count'] / lvc_trending['week_total'] * 100).round(1)
        
        # Calculate 6-week rolling average
        lvc_trending = lvc_trending.sort_values(['lvc_group', 'vintage_week'])
        lvc_trending['rolling_6wk_avg'] = lvc_trending.groupby('lvc_group')['pct_of_total'].transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        ).round(1)
        
        # Calculate week-over-week change for Z-score analysis
        lvc_trending['wow_change'] = lvc_trending.groupby('lvc_group')['pct_of_total'].diff()
        
        # Get unique LVC groups and sort them (exclude "Other")
        lvc_groups = sorted([g for g in lvc_trending['lvc_group'].unique() if g != 'Other'])
        n_groups = len(lvc_groups)
        
        # Create Gross Leads chart with Qty bars and % of Total line - dual Y-axis
        fig_gross_leads_pct = make_subplots(
            rows=n_groups, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[1] * n_groups,
            specs=[[{"secondary_y": True}] for _ in range(n_groups)]
        )
        
        # Z-score insights data collection (based on last 6 weeks)
        insights_data = []
        
        for i, group in enumerate(lvc_groups):
            group_data = lvc_trending[lvc_trending['lvc_group'] == group].sort_values('vintage_week')
            avg_pct = group_data['pct_of_total'].mean()
            avg_qty = group_data['lead_count'].mean()
            
            # Get last 6 weeks of data for Z-score analysis
            last_6_weeks = group_data.tail(6)
            
            # Calculate Z-score for latest week's change (over past 6 weeks)
            wow_changes = last_6_weeks['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change = wow_changes.mean()
                std_change = wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                
                # Determine significance
                significance = ""
                if abs(z_score) >= 2.58:
                    significance = "⚠️ Highly Significant (99%)"
                elif abs(z_score) >= 1.96:
                    significance = "📊 Significant (95%)"
                elif abs(z_score) >= 1.645:
                    significance = "📈 Marginally Significant (90%)"
                elif abs(z_score) >= 1.495:
                    significance = "📉 Notable movement"
                
                insights_data.append({
                    'group': group,
                    'latest_pct': group_data['pct_of_total'].iloc[-1],
                    'prev_pct': group_data['pct_of_total'].iloc[-2] if len(group_data) > 1 else 0,
                    'change': latest_change,
                    'z_score': z_score,
                    'significance': significance,
                    'avg': avg_pct
                })
            
            # Bar trace for Gross Lead Qty (primary y-axis)
            fig_gross_leads_pct.add_trace(
                go.Bar(
                    x=group_data['week_label'],
                    y=group_data['lead_count'],
                    marker_color='#2a9d8f',
                    text=group_data['lead_count'].apply(lambda x: f"{x:,.0f}"),
                    textposition='outside',
                    textfont=dict(size=9, color=COLORS['text']),
                    showlegend=False,
                    name='Gross Qty'
                ),
                row=i+1, col=1, secondary_y=False
            )
            
            # Line trace for % of Total (secondary y-axis)
            fig_gross_leads_pct.add_trace(
                go.Scatter(
                    x=group_data['week_label'],
                    y=group_data['pct_of_total'],
                    mode='lines+markers',
                    line=dict(color=COLORS['chart_yellow'], width=2),
                    marker=dict(size=5),
                    showlegend=False,
                    name='% of Total'
                ),
                row=i+1, col=1, secondary_y=True
            )
            
            # Primary Y-axis (Qty) - left side
            y_max_qty = group_data['lead_count'].max() * 1.3
            # Format LVC group label for horizontal annotation
            if group == 'LVC 1-2':
                group_label = '<b>LVC 1-2</b>'
            elif group == 'LVC 3-5':
                group_label = '<b>LVC 3-5</b>'
            elif group == 'LVC 6-10':
                group_label = '<b>LVC 6-10</b>'
            elif group == 'PHX Transfer':
                group_label = '<b>PHX Transfer</b>'
            else:
                group_label = f'<b>{group}</b>'
            
            fig_gross_leads_pct.update_yaxes(
                title_text='',
                tickformat=',.0f',
                tickfont=dict(size=9, color=COLORS['text']),
                showgrid=True,
                gridcolor=COLORS['border'],
                range=[0, y_max_qty],
                row=i+1, col=1, secondary_y=False
            )
            
            # Add horizontal LVC group label on the left
            fig_gross_leads_pct.add_annotation(
                xref='paper', yref='paper',
                x=-0.03, y=1 - ((i + 0.5) / n_groups),
                text=group_label,
                showarrow=False,
                font=dict(size=13, color=COLORS['chart_yellow']),
                xanchor='right', yanchor='middle'
            )
            
            # Secondary Y-axis (%) - right side
            y_max_pct = max(group_data['pct_of_total'].max() * 1.3, 15)
            fig_gross_leads_pct.update_yaxes(
                tickformat='.0f',
                ticksuffix='%',
                tickfont=dict(size=9, color=COLORS['chart_yellow']),
                showgrid=False,
                range=[0, y_max_pct],
                row=i+1, col=1, secondary_y=True
            )
            
            # Add Avg annotation on the right (% in yellow, # in teal)
            avg_qty_formatted = f"{avg_qty/1000:.1f}K" if avg_qty >= 1000 else f"{avg_qty:.0f}"
            fig_gross_leads_pct.add_annotation(
                xref='paper', yref='paper',
                x=1.005, y=1 - ((i + 0.5) / n_groups),
                text=f"<b>Avg:</b><br><span style='color:#f0b429'>{avg_pct:.1f}%</span><br><span style='color:#2a9d8f'>{avg_qty_formatted}</span>",
                showarrow=False,
                font=dict(size=18, color=COLORS['text']),
                xanchor='left', yanchor='middle',
                align='left'
            )
        
        # X-axis - horizontal labels on bottom row only
        for i in range(1, n_groups + 1):
            fig_gross_leads_pct.update_xaxes(
                showgrid=False,
                showticklabels=(i == n_groups),
                tickangle=0,
                tickfont=dict(size=10, color=COLORS['text']),
                row=i, col=1
            )
        
        fig_gross_leads_pct.update_layout(
            title=dict(text='<b>Gross Lead Qty & % of Total by LVC Group (6 Weeks Prior)</b>', font=dict(color=COLORS['text'], size=14), x=0, xanchor='left'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']),
            margin=dict(l=160, r=100, t=50, b=50),
            height=550,
            showlegend=False,
            bargap=0.3
        )
        
        # Build insights box content
        insights_items = []
        insights_items.append(html.H5("📊 Statistical Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}))
        insights_items.append(html.P("6-Week WoW Z-Score Analysis", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'}))
        
        for item in insights_data:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            
            insight_row = html.Div([
                html.Div([
                    html.Span(f"{item['group']}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '12px'}),
                    html.Span(f" {item['latest_pct']:.1f}%", style={'color': COLORS['primary'], 'fontSize': '12px', 'marginLeft': '5px'}),
                ], style={'marginBottom': '3px'}),
                html.Div([
                    html.Span(f"{change_symbol} {abs(item['change']):.2f}pp", style={'color': change_color, 'fontSize': '11px'}),
                    html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginLeft': '5px'}),
                ], style={'marginBottom': '3px'}),
                html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '10px'}) if item['significance'] else html.Div("No significant change", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '10px'}),
            ], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '8px', 'marginBottom': '8px'})
            insights_items.append(insight_row)
        
        # Add legend/key
        insights_items.append(html.Div([
            html.Hr(style={'borderColor': COLORS['border'], 'margin': '10px 0'}),
            html.P("Z-Score Thresholds:", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'fontWeight': '600', 'marginBottom': '5px'}),
            html.P("• |Z| ≥ 2.58 → 99% confidence", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '2px'}),
            html.P("• |Z| ≥ 1.96 → 95% confidence", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '2px'}),
            html.P("• |Z| ≥ 1.645 → 90% confidence", style={'color': COLORS['text_muted'], 'fontSize': '9px'}),
        ]))
        
        insights_box = html.Div(insights_items, style={
            'backgroundColor': COLORS['card_bg'],
            'borderRadius': '12px',
            'padding': '20px',
            'border': f"1px solid {COLORS['border']}",
            'height': '590px',
            'overflowY': 'auto',
            'marginBottom': '16px',
            'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'
        })
        
        # === Additional Trending Charts ===
        # 1. Avg LP2C Trending Chart - derive from same df_vintage, using same week filter
        lp2c_agg = df_vintage.groupby(['vintage_week', 'lvc_group']).agg({
            'avg_lp2c': 'mean'
        }).reset_index()
        # Use same filtering approach as lvc_trending
        lp2c_trending = lp2c_agg[lp2c_agg['lvc_group'].isin(lvc_groups)].copy()
        lp2c_trending = lp2c_trending[lp2c_trending['vintage_week'].isin(lvc_trending['vintage_week'].unique())]
        lp2c_trending['week_label'] = lp2c_trending['vintage_week'].dt.strftime('%b %-d, %y')
        
        # Calculate % of Total for LP2C
        week_totals_lp2c = lp2c_trending.groupby('vintage_week')['avg_lp2c'].sum().reset_index()
        week_totals_lp2c.columns = ['vintage_week', 'week_total']
        lp2c_trending = lp2c_trending.merge(week_totals_lp2c, on='vintage_week')
        lp2c_trending['pct_of_total'] = (lp2c_trending['avg_lp2c'] / lp2c_trending['week_total'] * 100).round(1)
        lp2c_trending['wow_change'] = lp2c_trending.groupby('lvc_group')['pct_of_total'].diff()
        
        fig_lp2c = make_subplots(
            rows=n_groups, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[1] * n_groups,
            specs=[[{"secondary_y": True}] for _ in range(n_groups)]
        )
        
        insights_data_lp2c = []
        for i, group in enumerate(lvc_groups):
            group_data = lp2c_trending[lp2c_trending['lvc_group'] == group].sort_values('vintage_week')
            if group_data.empty:
                continue
            avg_val = group_data['avg_lp2c'].mean()
            avg_pct = group_data['pct_of_total'].mean()
            
            last_6 = group_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change = wow_changes.mean()
                std_change = wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                significance = ""
                if abs(z_score) >= 2.58:
                    significance = "⚠️ Highly Significant (99%)"
                elif abs(z_score) >= 1.96:
                    significance = "📊 Significant (95%)"
                elif abs(z_score) >= 1.645:
                    significance = "📈 Marginally Significant (90%)"
                elif abs(z_score) >= 1.495:
                    significance = "📉 Notable movement"
                insights_data_lp2c.append({
                    'group': group, 'latest_val': group_data['avg_lp2c'].iloc[-1],
                    'latest_pct': group_data['pct_of_total'].iloc[-1], 'change': latest_change,
                    'z_score': z_score, 'significance': significance, 'avg_val': avg_val, 'avg_pct': avg_pct
                })
            
            fig_lp2c.add_trace(go.Bar(x=group_data['week_label'], y=group_data['avg_lp2c'],
                marker_color='#3498db', text=group_data['avg_lp2c'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside', textfont=dict(size=9, color=COLORS['text']), showlegend=False),
                row=i+1, col=1, secondary_y=False)
            
            y_max = group_data['avg_lp2c'].max() * 1.3
            group_label = f'<b>{group}</b>'
            fig_lp2c.update_yaxes(title_text='', range=[0, y_max], tickfont=dict(size=9, color=COLORS['text']),
                showgrid=True, gridcolor=COLORS['border'], row=i+1, col=1, secondary_y=False)
            yref_str = 'y domain' if i == 0 else f'y{i*2+1} domain'
            fig_lp2c.add_annotation(x=-0.03, y=0.5, xref='paper', yref=yref_str, text=group_label,
                showarrow=False, font=dict(size=12, color=COLORS['chart_yellow'], family='Arial Black'),
                xanchor='right', yanchor='middle', textangle=0)
            fig_lp2c.add_annotation(x=1.005, y=0.5, xref='paper', yref=yref_str,
                text=f"<b>Avg:</b><br><span style='color:#3498db'>{avg_val:.1f}%</span>",
                showarrow=False, font=dict(size=18, color=COLORS['text']), xanchor='left', yanchor='middle', align='left')
        
        for i in range(1, n_groups + 1):
            fig_lp2c.update_xaxes(showgrid=False, showticklabels=(i == n_groups), tickangle=0,
                tickfont=dict(size=10, color=COLORS['text']), row=i, col=1)
        fig_lp2c.update_layout(title=dict(text='<b>Avg. LP2C by LVC Group (6 Weeks Prior)</b>', font=dict(color=COLORS['text'], size=14), x=0, xanchor='left'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=160, r=100, t=50, b=50), height=550, showlegend=False, bargap=0.3)
        
        # LP2C Insights box
        insights_items_lp2c = [html.H5("📊 Statistical Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}),
            html.P("6-Week WoW Z-Score Analysis", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})]
        for item in insights_data_lp2c:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            insights_items_lp2c.append(html.Div([
                html.Div([html.Span(f"{item['group']}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '12px'}),
                    html.Span(f" {item['latest_val']:.1f}%", style={'color': COLORS['primary'], 'fontSize': '12px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div([html.Span(f"{change_symbol} {abs(item['change']):.2f}pp", style={'color': change_color, 'fontSize': '11px'}),
                    html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '10px'}) if item['significance'] else html.Div("No significant change", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '10px'}),
            ], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '8px', 'marginBottom': '8px'}))
        insights_box_lp2c = html.Div(insights_items_lp2c, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '20px',
            'border': f"1px solid {COLORS['border']}", 'height': '590px', 'overflowY': 'auto', 'marginBottom': '16px', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # 2. SF Contact % Day 7 Trending Chart - derive from same df_vintage
        sf_agg = df_vintage.groupby(['vintage_week', 'lvc_group']).agg({
            'sf_contact_day7_count': 'sum',
            'day7_eligible_sts': 'sum'
        }).reset_index()
        sf_contact_trending = sf_agg[sf_agg['lvc_group'].isin(lvc_groups)].copy()
        sf_contact_trending = sf_contact_trending[sf_contact_trending['vintage_week'].isin(lvc_trending['vintage_week'].unique())]
        sf_contact_trending['week_label'] = sf_contact_trending['vintage_week'].dt.strftime('%b %-d, %y')
        sf_contact_trending['sf_contact_day7_rate'] = (sf_contact_trending['sf_contact_day7_count'] / sf_contact_trending['day7_eligible_sts'].replace(0, 1) * 100).round(2)
        
        # Calculate % of Total for SF Contact counts
        week_totals_sf = sf_contact_trending.groupby('vintage_week')['sf_contact_day7_count'].sum().reset_index()
        week_totals_sf.columns = ['vintage_week', 'week_total']
        sf_contact_trending = sf_contact_trending.merge(week_totals_sf, on='vintage_week')
        sf_contact_trending['pct_of_total'] = (sf_contact_trending['sf_contact_day7_count'] / sf_contact_trending['week_total'].replace(0, 1) * 100).round(1)
        sf_contact_trending['wow_change'] = sf_contact_trending.groupby('lvc_group')['sf_contact_day7_rate'].diff()
        
        fig_sf_contact = make_subplots(
            rows=n_groups, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[1] * n_groups,
            specs=[[{"secondary_y": True}] for _ in range(n_groups)]
        )
        
        insights_data_sf = []
        for i, group in enumerate(lvc_groups):
            group_data = sf_contact_trending[sf_contact_trending['lvc_group'] == group].sort_values('vintage_week')
            if group_data.empty:
                continue
            avg_rate = group_data['sf_contact_day7_rate'].mean()
            avg_count = group_data['sf_contact_day7_count'].mean()
            
            last_6 = group_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change = wow_changes.mean()
                std_change = wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                significance = ""
                if abs(z_score) >= 2.58:
                    significance = "⚠️ Highly Significant (99%)"
                elif abs(z_score) >= 1.96:
                    significance = "📊 Significant (95%)"
                elif abs(z_score) >= 1.645:
                    significance = "📈 Marginally Significant (90%)"
                elif abs(z_score) >= 1.495:
                    significance = "📉 Notable movement"
                insights_data_sf.append({
                    'group': group, 'latest_rate': group_data['sf_contact_day7_rate'].iloc[-1],
                    'latest_count': group_data['sf_contact_day7_count'].iloc[-1], 'change': latest_change,
                    'z_score': z_score, 'significance': significance, 'avg_rate': avg_rate, 'avg_count': avg_count
                })
            
            # Bar = count on secondary Y (right), no text - added first so line is on top
            fig_sf_contact.add_trace(go.Bar(x=group_data['week_label'], y=group_data['sf_contact_day7_count'],
                marker_color='#e74c3c', marker_opacity=0.25, showlegend=False),
                row=i+1, col=1, secondary_y=True)
            # Line = % rate on primary Y (left), with text values - thicker line, darker text
            fig_sf_contact.add_trace(go.Scatter(x=group_data['week_label'], y=group_data['sf_contact_day7_rate'],
                mode='lines+markers+text', line=dict(color=COLORS['chart_yellow'], width=4),
                marker=dict(size=8, color=COLORS['chart_yellow']), text=group_data['sf_contact_day7_rate'].apply(lambda x: f"{x:.1f}%"),
                textposition='top center', textfont=dict(size=10, color='#FFFFFF', family='Arial Black'), showlegend=False),
                row=i+1, col=1, secondary_y=False)
            
            y_max_count = group_data['sf_contact_day7_count'].max() * 1.3
            y_max_rate = group_data['sf_contact_day7_rate'].max() * 1.3
            group_label = f'<b>{group}</b>'
            # Primary Y (left) = % rate
            fig_sf_contact.update_yaxes(title_text='', range=[0, max(y_max_rate, 100)], tickfont=dict(size=9, color=COLORS['chart_yellow']),
                showgrid=True, gridcolor=COLORS['border'], ticksuffix='%', row=i+1, col=1, secondary_y=False)
            # Secondary Y (right) = count
            fig_sf_contact.update_yaxes(range=[0, y_max_count], tickfont=dict(size=9, color=COLORS['text']),
                showgrid=False, row=i+1, col=1, secondary_y=True)
            yref_str_sf = 'y domain' if i == 0 else f'y{i*2+1} domain'
            fig_sf_contact.add_annotation(x=-0.03, y=0.5, xref='paper', yref=yref_str_sf, text=group_label,
                showarrow=False, font=dict(size=12, color=COLORS['chart_yellow'], family='Arial Black'),
                xanchor='right', yanchor='middle', textangle=0)
            fig_sf_contact.add_annotation(x=1.005, y=0.5, xref='paper', yref=yref_str_sf,
                text=f"<b>Avg:</b><br><span style='color:#f0b429'>{avg_rate:.1f}%</span><br><span style='color:#e74c3c'>{avg_count:,.0f}</span>",
                showarrow=False, font=dict(size=18, color=COLORS['text']), xanchor='left', yanchor='middle', align='left')
        
        for i in range(1, n_groups + 1):
            fig_sf_contact.update_xaxes(showgrid=False, showticklabels=(i == n_groups), tickangle=0,
                tickfont=dict(size=10, color=COLORS['text']), row=i, col=1)
        fig_sf_contact.update_layout(title=dict(text='<b>SF Contact % Day 7 by LVC Group (6 Weeks Prior)</b>', font=dict(color=COLORS['text'], size=14), x=0, xanchor='left'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=160, r=100, t=50, b=50), height=550, showlegend=False, bargap=0.3)
        
        # SF Contact Insights box
        insights_items_sf = [html.H5("📊 Statistical Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}),
            html.P("6-Week WoW Z-Score Analysis", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})]
        for item in insights_data_sf:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            insights_items_sf.append(html.Div([
                html.Div([html.Span(f"{item['group']}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '12px'}),
                    html.Span(f" {item['latest_rate']:.1f}%", style={'color': COLORS['primary'], 'fontSize': '12px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div([html.Span(f"{change_symbol} {abs(item['change']):.2f}pp", style={'color': change_color, 'fontSize': '11px'}),
                    html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '10px'}) if item['significance'] else html.Div("No significant change", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '10px'}),
            ], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '8px', 'marginBottom': '8px'}))
        insights_box_sf = html.Div(insights_items_sf, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '20px',
            'border': f"1px solid {COLORS['border']}", 'height': '590px', 'overflowY': 'auto', 'marginBottom': '16px', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # 3. Sent to Sales Trending Chart
        sts_agg = df_vintage.groupby(['vintage_week', 'lvc_group']).agg({
            'sts_eligible': 'sum'
        }).reset_index()
        sts_trending = sts_agg[sts_agg['lvc_group'].isin(lvc_groups)].copy()
        sts_trending = sts_trending[sts_trending['vintage_week'].isin(lvc_trending['vintage_week'].unique())]
        sts_trending['week_label'] = sts_trending['vintage_week'].dt.strftime('%b %-d, %y')
        
        week_totals_sts = sts_trending.groupby('vintage_week')['sts_eligible'].sum().reset_index()
        week_totals_sts.columns = ['vintage_week', 'week_total']
        sts_trending = sts_trending.merge(week_totals_sts, on='vintage_week')
        sts_trending['pct_of_total'] = (sts_trending['sts_eligible'] / sts_trending['week_total'] * 100).round(1)
        sts_trending['wow_change'] = sts_trending.groupby('lvc_group')['pct_of_total'].diff()
        
        fig_sts = make_subplots(rows=n_groups, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[1] * n_groups, specs=[[{"secondary_y": True}] for _ in range(n_groups)])
        
        insights_data_sts = []
        for i, group in enumerate(lvc_groups):
            group_data = sts_trending[sts_trending['lvc_group'] == group].sort_values('vintage_week')
            if group_data.empty:
                continue
            avg_val = group_data['sts_eligible'].mean()
            avg_pct = group_data['pct_of_total'].mean()
            last_6 = group_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change = wow_changes.mean()
                std_change = wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                significance = ""
                if abs(z_score) >= 2.58: significance = "⚠️ Highly Significant (99%)"
                elif abs(z_score) >= 1.96: significance = "📊 Significant (95%)"
                elif abs(z_score) >= 1.645: significance = "📈 Marginally Significant (90%)"
                elif abs(z_score) >= 1.495: significance = "📉 Notable movement"
                insights_data_sts.append({'group': group, 'latest_val': group_data['sts_eligible'].iloc[-1],
                    'latest_pct': group_data['pct_of_total'].iloc[-1], 'change': latest_change,
                    'z_score': z_score, 'significance': significance, 'avg_val': avg_val, 'avg_pct': avg_pct})
            
            fig_sts.add_trace(go.Bar(x=group_data['week_label'], y=group_data['sts_eligible'],
                marker_color='#9b59b6', text=group_data['sts_eligible'].apply(lambda x: f"{x:,.0f}"),
                textposition='outside', textfont=dict(size=9, color=COLORS['text']), showlegend=False),
                row=i+1, col=1, secondary_y=False)
            fig_sts.add_trace(go.Scatter(x=group_data['week_label'], y=group_data['pct_of_total'],
                mode='lines+markers', line=dict(color=COLORS['chart_yellow'], width=2),
                marker=dict(size=5), showlegend=False), row=i+1, col=1, secondary_y=True)
            
            y_max = group_data['sts_eligible'].max() * 1.3
            yref_str_sts = 'y domain' if i == 0 else f'y{i*2+1} domain'
            fig_sts.update_yaxes(title_text='', range=[0, y_max], tickfont=dict(size=9, color=COLORS['text']),
                showgrid=True, gridcolor=COLORS['border'], row=i+1, col=1, secondary_y=False)
            fig_sts.update_yaxes(range=[0, 50], tickfont=dict(size=9, color=COLORS['chart_yellow']),
                showgrid=False, ticksuffix='%', row=i+1, col=1, secondary_y=True)
            fig_sts.add_annotation(x=-0.03, y=0.5, xref='paper', yref=yref_str_sts, text=f'<b>{group}</b>',
                showarrow=False, font=dict(size=12, color=COLORS['chart_yellow'], family='Arial Black'),
                xanchor='right', yanchor='middle', textangle=0)
            fig_sts.add_annotation(x=1.005, y=0.5, xref='paper', yref=yref_str_sts,
                text=f"<b>Avg:</b><br><span style='color:#f0b429'>{avg_pct:.1f}%</span><br><span style='color:#9b59b6'>{avg_val:,.0f}</span>",
                showarrow=False, font=dict(size=18, color=COLORS['text']), xanchor='left', yanchor='middle', align='left')
        
        for i in range(1, n_groups + 1):
            fig_sts.update_xaxes(showgrid=False, showticklabels=(i == n_groups), tickangle=0,
                tickfont=dict(size=10, color=COLORS['text']), row=i, col=1)
        fig_sts.update_layout(title=dict(text='<b>Sent to Sales by LVC Group (6 Weeks Prior)</b>', font=dict(color=COLORS['text'], size=14), x=0, xanchor='left'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=160, r=100, t=50, b=50), height=550, showlegend=False, bargap=0.3)
        
        insights_items_sts = [html.H5("📊 Statistical Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}),
            html.P("6-Week WoW Z-Score Analysis", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})]
        for item in insights_data_sts:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            insights_items_sts.append(html.Div([
                html.Div([html.Span(f"{item['group']}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '12px'}),
                    html.Span(f" {item['latest_val']:,.0f}", style={'color': COLORS['primary'], 'fontSize': '12px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div([html.Span(f"{change_symbol} {abs(item['change']):.2f}pp", style={'color': change_color, 'fontSize': '11px'}),
                    html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '10px'}) if item['significance'] else html.Div("No significant change", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '10px'}),
            ], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '8px', 'marginBottom': '8px'}))
        insights_box_sts = html.Div(insights_items_sts, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '20px',
            'border': f"1px solid {COLORS['border']}", 'height': '590px', 'overflowY': 'auto', 'marginBottom': '16px', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # 4. FAS Day 7 % Trending Chart
        fas_pct_agg = df_vintage.groupby(['vintage_week', 'lvc_group']).agg({
            'fas_day7_count': 'sum',
            'day7_eligible_sts': 'sum'
        }).reset_index()
        fas_pct_trending = fas_pct_agg[fas_pct_agg['lvc_group'].isin(lvc_groups)].copy()
        fas_pct_trending = fas_pct_trending[fas_pct_trending['vintage_week'].isin(lvc_trending['vintage_week'].unique())]
        fas_pct_trending['week_label'] = fas_pct_trending['vintage_week'].dt.strftime('%b %-d, %y')
        fas_pct_trending['fas_day7_rate'] = (fas_pct_trending['fas_day7_count'] / fas_pct_trending['day7_eligible_sts'].replace(0, 1) * 100).round(2)
        
        week_totals_fas = fas_pct_trending.groupby('vintage_week')['fas_day7_count'].sum().reset_index()
        week_totals_fas.columns = ['vintage_week', 'week_total']
        fas_pct_trending = fas_pct_trending.merge(week_totals_fas, on='vintage_week')
        fas_pct_trending['pct_of_total'] = (fas_pct_trending['fas_day7_count'] / fas_pct_trending['week_total'].replace(0, 1) * 100).round(1)
        fas_pct_trending['wow_change'] = fas_pct_trending.groupby('lvc_group')['fas_day7_rate'].diff()
        
        fig_fas_pct = make_subplots(rows=n_groups, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[1] * n_groups, specs=[[{"secondary_y": True}] for _ in range(n_groups)])
        
        insights_data_fas_pct = []
        for i, group in enumerate(lvc_groups):
            group_data = fas_pct_trending[fas_pct_trending['lvc_group'] == group].sort_values('vintage_week')
            if group_data.empty:
                continue
            avg_rate = group_data['fas_day7_rate'].mean()
            avg_count = group_data['fas_day7_count'].mean()
            last_6 = group_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change = wow_changes.mean()
                std_change = wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                significance = ""
                if abs(z_score) >= 2.58: significance = "⚠️ Highly Significant (99%)"
                elif abs(z_score) >= 1.96: significance = "📊 Significant (95%)"
                elif abs(z_score) >= 1.645: significance = "📈 Marginally Significant (90%)"
                elif abs(z_score) >= 1.495: significance = "📉 Notable movement"
                insights_data_fas_pct.append({'group': group, 'latest_rate': group_data['fas_day7_rate'].iloc[-1],
                    'latest_count': group_data['fas_day7_count'].iloc[-1], 'change': latest_change,
                    'z_score': z_score, 'significance': significance, 'avg_rate': avg_rate, 'avg_count': avg_count})
            
            # Bar = count on secondary Y (right), no text - added first so line is on top
            fig_fas_pct.add_trace(go.Bar(x=group_data['week_label'], y=group_data['fas_day7_count'],
                marker_color='#1abc9c', marker_opacity=0.25, showlegend=False),
                row=i+1, col=1, secondary_y=True)
            # Line = % rate on primary Y (left), with text values - thicker line, darker text
            fig_fas_pct.add_trace(go.Scatter(x=group_data['week_label'], y=group_data['fas_day7_rate'],
                mode='lines+markers+text', line=dict(color=COLORS['chart_yellow'], width=4),
                marker=dict(size=8, color=COLORS['chart_yellow']), text=group_data['fas_day7_rate'].apply(lambda x: f"{x:.1f}%"),
                textposition='top center', textfont=dict(size=10, color='#FFFFFF', family='Arial Black'), showlegend=False),
                row=i+1, col=1, secondary_y=False)
            
            y_max_count = group_data['fas_day7_count'].max() * 1.3
            y_max_rate = group_data['fas_day7_rate'].max() * 1.3
            yref_str_fas = 'y domain' if i == 0 else f'y{i*2+1} domain'
            # Primary Y (left) = % rate
            fig_fas_pct.update_yaxes(title_text='', range=[0, max(y_max_rate, 50)], tickfont=dict(size=9, color=COLORS['chart_yellow']),
                showgrid=True, gridcolor=COLORS['border'], ticksuffix='%', row=i+1, col=1, secondary_y=False)
            # Secondary Y (right) = count
            fig_fas_pct.update_yaxes(range=[0, y_max_count], tickfont=dict(size=9, color=COLORS['text']),
                showgrid=False, row=i+1, col=1, secondary_y=True)
            fig_fas_pct.add_annotation(x=-0.03, y=0.5, xref='paper', yref=yref_str_fas, text=f'<b>{group}</b>',
                showarrow=False, font=dict(size=12, color=COLORS['chart_yellow'], family='Arial Black'),
                xanchor='right', yanchor='middle', textangle=0)
            fig_fas_pct.add_annotation(x=1.005, y=0.5, xref='paper', yref=yref_str_fas,
                text=f"<b>Avg:</b><br><span style='color:#f0b429'>{avg_rate:.1f}%</span><br><span style='color:#1abc9c'>{avg_count:,.0f}</span>",
                showarrow=False, font=dict(size=18, color=COLORS['text']), xanchor='left', yanchor='middle', align='left')
        
        for i in range(1, n_groups + 1):
            fig_fas_pct.update_xaxes(showgrid=False, showticklabels=(i == n_groups), tickangle=0,
                tickfont=dict(size=10, color=COLORS['text']), row=i, col=1)
        fig_fas_pct.update_layout(title=dict(text='<b>FAS Day 7 % by LVC Group (6 Weeks Prior)</b>', font=dict(color=COLORS['text'], size=14), x=0, xanchor='left'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=160, r=100, t=50, b=50), height=550, showlegend=False, bargap=0.3)
        
        insights_items_fas_pct = [html.H5("📊 Statistical Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}),
            html.P("6-Week WoW Z-Score Analysis", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})]
        for item in insights_data_fas_pct:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            insights_items_fas_pct.append(html.Div([
                html.Div([html.Span(f"{item['group']}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '12px'}),
                    html.Span(f" {item['latest_rate']:.1f}%", style={'color': COLORS['primary'], 'fontSize': '12px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div([html.Span(f"{change_symbol} {abs(item['change']):.2f}pp", style={'color': change_color, 'fontSize': '11px'}),
                    html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '10px'}) if item['significance'] else html.Div("No significant change", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '10px'}),
            ], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '8px', 'marginBottom': '8px'}))
        insights_box_fas_pct = html.Div(insights_items_fas_pct, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '20px',
            'border': f"1px solid {COLORS['border']}", 'height': '590px', 'overflowY': 'auto', 'marginBottom': '16px', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # 5. FAS Day 7 $ Trending Chart (bars = total $, line = Avg $)
        fas_avg_agg = df_vintage.groupby(['vintage_week', 'lvc_group']).agg({
            'fas_day7_dollars': 'sum',
            'fas_day7_count': 'sum'
        }).reset_index()
        fas_avg_trending = fas_avg_agg[fas_avg_agg['lvc_group'].isin(lvc_groups)].copy()
        fas_avg_trending = fas_avg_trending[fas_avg_trending['vintage_week'].isin(lvc_trending['vintage_week'].unique())]
        fas_avg_trending['week_label'] = fas_avg_trending['vintage_week'].dt.strftime('%b %-d, %y')
        fas_avg_trending['avg_fas_dollars'] = (fas_avg_trending['fas_day7_dollars'] / fas_avg_trending['fas_day7_count'].replace(0, 1)).round(0)
        
        fas_avg_trending['wow_change'] = fas_avg_trending.groupby('lvc_group')['fas_day7_dollars'].diff()
        
        fig_fas_avg = make_subplots(rows=n_groups, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[1] * n_groups, specs=[[{"secondary_y": True}] for _ in range(n_groups)])
        
        insights_data_fas_avg = []
        for i, group in enumerate(lvc_groups):
            group_data = fas_avg_trending[fas_avg_trending['lvc_group'] == group].sort_values('vintage_week')
            if group_data.empty:
                continue
            avg_total = group_data['fas_day7_dollars'].mean()
            avg_per_fas = group_data['avg_fas_dollars'].mean()
            last_6 = group_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change = wow_changes.mean()
                std_change = wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                significance = ""
                if abs(z_score) >= 2.58: significance = "⚠️ Highly Significant (99%)"
                elif abs(z_score) >= 1.96: significance = "📊 Significant (95%)"
                elif abs(z_score) >= 1.645: significance = "📈 Marginally Significant (90%)"
                elif abs(z_score) >= 1.495: significance = "📉 Notable movement"
                insights_data_fas_avg.append({'group': group, 'latest_total': group_data['fas_day7_dollars'].iloc[-1],
                    'latest_avg': group_data['avg_fas_dollars'].iloc[-1], 'change': latest_change,
                    'z_score': z_score, 'significance': significance, 'avg_total': avg_total, 'avg_per_fas': avg_per_fas})
            
            # Bars = Total FAS Day 7 $ (blue like LP2C)
            fig_fas_avg.add_trace(go.Bar(x=group_data['week_label'], y=group_data['fas_day7_dollars'],
                marker_color='#3498db', text=group_data['fas_day7_dollars'].apply(lambda x: f"${x/1000:,.0f}K"),
                textposition='outside', textfont=dict(size=9, color=COLORS['text']), showlegend=False),
                row=i+1, col=1, secondary_y=False)
            # Line = Avg $ per FAS
            fig_fas_avg.add_trace(go.Scatter(x=group_data['week_label'], y=group_data['avg_fas_dollars'],
                mode='lines+markers', line=dict(color=COLORS['chart_yellow'], width=2),
                marker=dict(size=5), showlegend=False), row=i+1, col=1, secondary_y=True)
            
            y_max = group_data['fas_day7_dollars'].max() * 1.3
            y2_max = group_data['avg_fas_dollars'].max() * 1.3
            yref_str_avg = 'y domain' if i == 0 else f'y{i*2+1} domain'
            fig_fas_avg.update_yaxes(title_text='', range=[0, y_max], tickfont=dict(size=9, color=COLORS['text']),
                showgrid=True, gridcolor=COLORS['border'], row=i+1, col=1, secondary_y=False)
            fig_fas_avg.update_yaxes(range=[0, y2_max], tickfont=dict(size=9, color=COLORS['chart_yellow']),
                showgrid=False, tickprefix='$', row=i+1, col=1, secondary_y=True)
            fig_fas_avg.add_annotation(x=-0.03, y=0.5, xref='paper', yref=yref_str_avg, text=f'<b>{group}</b>',
                showarrow=False, font=dict(size=12, color=COLORS['chart_yellow'], family='Arial Black'),
                xanchor='right', yanchor='middle', textangle=0)
            fig_fas_avg.add_annotation(x=1.005, y=0.5, xref='paper', yref=yref_str_avg,
                text=f"<b>Avg:</b><br><span style='color:#f0b429'>${avg_per_fas:,.0f}</span><br><span style='color:#3498db'>${avg_total/1000:,.0f}K</span>",
                showarrow=False, font=dict(size=18, color=COLORS['text']), xanchor='left', yanchor='middle', align='left')
        
        for i in range(1, n_groups + 1):
            fig_fas_avg.update_xaxes(showgrid=False, showticklabels=(i == n_groups), tickangle=0,
                tickfont=dict(size=10, color=COLORS['text']), row=i, col=1)
        fig_fas_avg.update_layout(title=dict(text='<b>FAS Day 7 $ by LVC Group (6 Weeks Prior)</b>', font=dict(color=COLORS['text'], size=14), x=0, xanchor='left'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']),
            margin=dict(l=160, r=100, t=50, b=50), height=550, showlegend=False, bargap=0.3)
        
        insights_items_fas_avg = [html.H5("📊 Statistical Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}),
            html.P("6-Week WoW Z-Score Analysis", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})]
        for item in insights_data_fas_avg:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            insights_items_fas_avg.append(html.Div([
                html.Div([html.Span(f"{item['group']}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '12px'}),
                    html.Span(f" ${item['latest_total']/1000:,.0f}K", style={'color': COLORS['primary'], 'fontSize': '12px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div([html.Span(f"{change_symbol} ${abs(item['change'])/1000:,.0f}K", style={'color': change_color, 'fontSize': '11px'}),
                    html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '10px'}) if item['significance'] else html.Div("No significant change", style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '10px'}),
            ], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '8px', 'marginBottom': '8px'}))
        insights_box_fas_avg = html.Div(insights_items_fas_avg, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '20px',
            'border': f"1px solid {COLORS['border']}", 'height': '590px', 'overflowY': 'auto', 'marginBottom': '16px', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # Channel Mix Analysis v2 Row (separate section)
        # Make the channel charts taller to fit all channels properly
        fig_channel_gross_leads.update_layout(height=150 * n_channel_groups)
        
        scrollable_chart_style = {**CARD_STYLE, 'height': '600px', 'overflowY': 'auto'}
        
        channel_v2_row = html.Div([
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_channel_count)], style=CARD_STYLE), width=6),
                dbc.Col(html.Div([dcc.Graph(figure=fig_channel_dollars)], style=CARD_STYLE), width=6),
            ]),
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_channel_fico)], style=CARD_STYLE), width=6),
                dbc.Col(html.Div([dcc.Graph(figure=fig_channel_lvc)], style=CARD_STYLE), width=6),
            ]),
            # Gross Lead Qty
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_channel_gross_leads)], style=scrollable_chart_style), width=10),
                dbc.Col(insights_box_ch, width=2),
            ]),
            # Sent to Sales
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_ch_sts)], style=scrollable_chart_style), width=10),
                dbc.Col(insights_box_ch_sts, width=2),
            ]),
            # Avg LP2C
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_ch_lp2c)], style=scrollable_chart_style), width=10),
                dbc.Col(insights_box_ch_lp2c, width=2),
            ]),
            # SF Contact % Day 7
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_ch_sf)], style=scrollable_chart_style), width=10),
                dbc.Col(insights_box_ch_sf, width=2),
            ]),
            # FAS Day 7 %
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_ch_fas_pct)], style=scrollable_chart_style), width=10),
                dbc.Col(insights_box_ch_fas_pct, width=2),
            ]),
            # FAS Day 7 $
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_ch_fas_dollar)], style=scrollable_chart_style), width=10),
                dbc.Col(insights_box_ch_fas_dollar, width=2),
            ]),
        ])
        
        # Add channel comparison tables to channel_v2_row (merged from Channel Mix Analysis)
        channel_target = df_vintage[df_vintage['vintage_week'] == target_date].groupby('channel').agg({
            'sts_eligible': 'sum',
            'day7_eligible_sts': 'sum',
            'fas_day7_count': 'sum',
            'fas_day7_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        channel_target = channel_target.rename(columns={'avg_lp2c': 'lp2c'})
        
        channel_prev = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                                  (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))].groupby('channel').agg({
            'sts_eligible': 'sum',
            'day7_eligible_sts': 'sum',
            'fas_day7_count': 'sum',
            'fas_day7_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        channel_prev['sts_eligible'] = channel_prev['sts_eligible'] / 4
        channel_prev['day7_eligible_sts'] = channel_prev['day7_eligible_sts'] / 4
        channel_prev['fas_day7_count'] = channel_prev['fas_day7_count'] / 4
        channel_prev['fas_day7_dollars'] = channel_prev['fas_day7_dollars'] / 4
        channel_prev = channel_prev.rename(columns={'avg_lp2c': 'lp2c_prev'})
        
        channel_merged = channel_target.merge(channel_prev, on='channel', suffixes=('', '_prev'), how='outer').fillna(0)
        channel_merged['fas_day7_rate'] = (channel_merged['fas_day7_count'] / channel_merged['day7_eligible_sts'].replace(0, 1) * 100).round(1)
        channel_merged['fas_delta'] = channel_merged['fas_day7_count'] - channel_merged['fas_day7_count_prev']
        channel_merged['dollar_delta'] = channel_merged['fas_day7_dollars'] - channel_merged['fas_day7_dollars_prev']
        channel_merged['lp2c_delta'] = channel_merged['lp2c'] - channel_merged['lp2c_prev']
        channel_merged = channel_merged.sort_values('fas_day7_count', ascending=False)
        channel_worst = channel_merged.sort_values('fas_delta').head(10)
        
        channel_comparison_row = dbc.Row([
            dbc.Col(html.Div([
                html.H5("Channel Comparison (This Week vs 4-Wk Avg)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=channel_merged[['channel', 'sts_eligible', 'fas_day7_count', 'fas_day7_rate', 'fas_day7_dollars', 'fas_delta', 'lp2c', 'lp2c_delta']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'Channel', 'id': 'channel'},
                        {'name': 'StS #', 'id': 'sts_eligible', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS Day 7 #', 'id': 'fas_day7_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS Day 7 %', 'id': 'fas_day7_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': '$FAS Day 7', 'id': 'fas_day7_dollars', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
                        {'name': 'Δ FAS', 'id': 'fas_delta', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                        {'name': 'LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ LP2C pp', 'id': 'lp2c_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg'], 'overflowX': 'auto'},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '6px', 'fontSize': '11px', 'minWidth': '50px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{fas_delta} > 0', 'column_id': 'fas_delta'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{fas_delta} < 0', 'column_id': 'fas_delta'}, 'color': COLORS['danger']},
                        {'if': {'filter_query': '{lp2c_delta} > 0', 'column_id': 'lp2c_delta'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{lp2c_delta} < 0', 'column_id': 'lp2c_delta'}, 'color': COLORS['danger']},
                    ],
                    page_size=15
                )
            ], style=CARD_STYLE), width=8),
            dbc.Col(html.Div([
                html.H5("⚠️ Worst Performing Channels (vs 4-Wk Avg)", style={'color': COLORS['danger'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=channel_worst[['channel', 'fas_day7_count', 'fas_day7_count_prev', 'fas_delta', 'lp2c', 'lp2c_delta']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'Channel', 'id': 'channel'},
                        {'name': 'FAS Day 7', 'id': 'fas_day7_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Prev', 'id': 'fas_day7_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
                        {'name': 'Δ', 'id': 'fas_delta', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                        {'name': 'LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ LP2C pp', 'id': 'lp2c_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg']},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '6px', 'fontSize': '11px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{fas_delta} < 0', 'column_id': 'fas_delta'}, 'color': COLORS['danger']},
                        {'if': {'filter_query': '{lp2c_delta} < 0', 'column_id': 'lp2c_delta'}, 'color': COLORS['danger']},
                    ],
                    page_size=10
                )
            ], style=CARD_STYLE), width=4),
        ], style={'marginTop': '16px'})
        
        # === Channel Performance Table (similar to MA table) ===
        channel_perf_target = df_vintage[df_vintage['vintage_week'] == target_date].groupby('channel').agg({
            'lead_count': 'sum', 'avg_lp2c': 'mean', 'avg_lvc': 'mean',
            'sf_contact_day7_count': 'sum', 'day7_eligible_sts': 'sum', 'fas_day7_count': 'sum', 
            'fas_day7_dollars': 'sum', 'funded_count': 'sum'
        }).reset_index()
        
        channel_perf_6wk = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                                  (df_vintage['vintage_week'] >= target_date - timedelta(weeks=6))].groupby('channel').agg({
            'lead_count': 'sum', 'avg_lp2c': 'mean', 'avg_lvc': 'mean', 
            'sf_contact_day7_count': 'sum', 'day7_eligible_sts': 'sum', 'fas_day7_count': 'sum',
            'fas_day7_dollars': 'sum', 'funded_count': 'sum'
        }).reset_index()
        
        for col in ['lead_count', 'sf_contact_day7_count', 'day7_eligible_sts', 'fas_day7_count', 'fas_day7_dollars', 'funded_count']:
            channel_perf_6wk[col] = channel_perf_6wk[col] / 6
            
        channel_perf_merged = channel_perf_target.merge(channel_perf_6wk, on='channel', suffixes=('', '_6wk'), how='outer').fillna(0)
        channel_perf_merged = channel_perf_merged.rename(columns={'avg_lp2c': 'lp2c', 'avg_lvc': 'lvc_avg', 'lead_count': 'leads_assigned', 'lead_count_6wk': 'leads_assigned_6wk'})
        
        channel_perf_merged['fas_day7_pct'] = (channel_perf_merged['fas_day7_count'] / channel_perf_merged['day7_eligible_sts'].replace(0, 1) * 100).round(1)
        channel_perf_merged['close_rate'] = (channel_perf_merged['funded_count'] / channel_perf_merged['leads_assigned'].replace(0, 1) * 100).round(1)
        channel_perf_merged['fas_div_lp2c'] = (channel_perf_merged['fas_day7_pct'] / channel_perf_merged['lp2c'].replace(0, 1)).round(2)
        
        channel_perf_merged['fas_day7_pct_6wk'] = (channel_perf_merged['fas_day7_count_6wk'] / channel_perf_merged['day7_eligible_sts_6wk'].replace(0, 1) * 100).round(1)
        channel_perf_merged['close_rate_6wk'] = (channel_perf_merged['funded_count_6wk'] / channel_perf_merged['leads_assigned_6wk'].replace(0, 1) * 100).round(1)
        channel_perf_merged['lp2c_6wk'] = channel_perf_merged['avg_lp2c_6wk'].round(2)
        channel_perf_merged['fas_div_lp2c_6wk'] = (channel_perf_merged['fas_day7_pct_6wk'] / channel_perf_merged['lp2c_6wk'].replace(0, 1)).round(2)
        
        channel_perf_merged['leads_delta'] = ((channel_perf_merged['leads_assigned'] - channel_perf_merged['leads_assigned_6wk']) / channel_perf_merged['leads_assigned_6wk'].replace(0, 1) * 100).round(1)
        channel_perf_merged['fas_dollars_delta'] = ((channel_perf_merged['fas_day7_dollars'] - channel_perf_merged['fas_day7_dollars_6wk']) / channel_perf_merged['fas_day7_dollars_6wk'].replace(0, 1) * 100).round(1)
        channel_perf_merged['fas_pct_delta'] = (channel_perf_merged['fas_day7_pct'] - channel_perf_merged['fas_day7_pct_6wk']).round(1)
        channel_perf_merged['close_rate_delta'] = (channel_perf_merged['close_rate'] - channel_perf_merged['close_rate_6wk']).round(1)
        channel_perf_merged['fas_div_lp2c_delta'] = (channel_perf_merged['fas_div_lp2c'] - channel_perf_merged['fas_div_lp2c_6wk']).round(2)
        channel_perf_merged['lp2c_delta'] = (channel_perf_merged['lp2c'] - channel_perf_merged['lp2c_6wk']).round(1)
        channel_perf_merged['lvc_6wk'] = channel_perf_merged['avg_lvc_6wk'].round(1)
        channel_perf_merged['lvc_delta'] = (channel_perf_merged['lvc_avg'] - channel_perf_merged['lvc_6wk']).round(1)
        
        channel_perf_merged = channel_perf_merged.sort_values('fas_day7_dollars', ascending=False)
        
        table_columns_channel = [
            {'name': 'Leads', 'id': 'leads_assigned', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
            {'name': 'Leads 6wk', 'id': 'leads_assigned_6wk', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
            {'name': 'Leads Δ%', 'id': 'leads_delta', 'type': 'numeric', 'format': {'specifier': '+.0f'}},
            {'name': 'Avg. LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'LP2C 6wk', 'id': 'lp2c_6wk', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'LP2C Δ', 'id': 'lp2c_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
            {'name': 'LVC', 'id': 'lvc_avg', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'LVC 6wk', 'id': 'lvc_6wk', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'LVC Δ', 'id': 'lvc_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
            {'name': 'FAS Day 7 %', 'id': 'fas_day7_pct', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'FAS D7 % 6wk', 'id': 'fas_day7_pct_6wk', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'FAS D7 % Δ', 'id': 'fas_pct_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
            {'name': 'FAS D7/LP2C', 'id': 'fas_div_lp2c', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'F/L 6wk', 'id': 'fas_div_lp2c_6wk', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'F/L Δ', 'id': 'fas_div_lp2c_delta', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
            {'name': 'FAS Day 7 $', 'id': 'fas_day7_dollars', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
            {'name': 'FAS D7 $ 6wk', 'id': 'fas_day7_dollars_6wk', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
            {'name': 'FAS D7 $ Δ%', 'id': 'fas_dollars_delta', 'type': 'numeric', 'format': {'specifier': '+.0f'}},
            {'name': 'Close %', 'id': 'close_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'Close % 6wk', 'id': 'close_rate_6wk', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'Close % Δ', 'id': 'close_rate_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
        ]
        
        channel_perf_cols = ['channel', 'leads_assigned', 'leads_assigned_6wk', 'leads_delta', 'lp2c', 'lp2c_6wk', 'lp2c_delta', 'lvc_avg', 'lvc_6wk', 'lvc_delta',
                        'fas_day7_pct', 'fas_day7_pct_6wk', 'fas_pct_delta', 'fas_div_lp2c', 'fas_div_lp2c_6wk', 'fas_div_lp2c_delta',
                        'fas_day7_dollars', 'fas_day7_dollars_6wk', 'fas_dollars_delta', 'close_rate', 'close_rate_6wk', 'close_rate_delta']
        
        channel_perf_row = dbc.Row([
            dbc.Col(html.Div([
                html.H5("Performance by Channel (Top 20 by FAS Day 7 $)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                html.Div(style={'overflowX': 'auto'}, children=[
                    dash_table.DataTable(
                        data=channel_perf_merged[[c for c in channel_perf_cols if c in channel_perf_merged.columns]].head(20).round(1).to_dict('records'),
                        columns=[{'name': 'Channel', 'id': 'channel'}] + table_columns_channel,
                        style_table={'backgroundColor': COLORS['card_bg'], 'minWidth': '100%'},
                        style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '6px', 'fontSize': '11px'},
                        style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px', 'textAlign': 'center'},
                        style_data_conditional=[
                            {'if': {'column_id': 'leads_delta', 'filter_query': '{leads_delta} > 0'}, 'color': COLORS['success']},
                            {'if': {'column_id': 'leads_delta', 'filter_query': '{leads_delta} < 0'}, 'color': COLORS['danger']},
                            {'if': {'column_id': 'fas_pct_delta', 'filter_query': '{fas_pct_delta} > 0'}, 'color': COLORS['success']},
                            {'if': {'column_id': 'fas_pct_delta', 'filter_query': '{fas_pct_delta} < 0'}, 'color': COLORS['danger']},
                            {'if': {'column_id': 'fas_div_lp2c_delta', 'filter_query': '{fas_div_lp2c_delta} > 0'}, 'color': COLORS['success']},
                            {'if': {'column_id': 'fas_div_lp2c_delta', 'filter_query': '{fas_div_lp2c_delta} < 0'}, 'color': COLORS['danger']},
                            {'if': {'column_id': 'fas_dollars_delta', 'filter_query': '{fas_dollars_delta} > 0'}, 'color': COLORS['success']},
                            {'if': {'column_id': 'fas_dollars_delta', 'filter_query': '{fas_dollars_delta} < 0'}, 'color': COLORS['danger']},
                            {'if': {'column_id': 'total_fas_dollars_delta', 'filter_query': '{total_fas_dollars_delta} > 0'}, 'color': COLORS['success']},
                            {'if': {'column_id': 'total_fas_dollars_delta', 'filter_query': '{total_fas_dollars_delta} < 0'}, 'color': COLORS['danger']},
                            {'if': {'column_id': 'close_rate_delta', 'filter_query': '{close_rate_delta} > 0'}, 'color': COLORS['success']},
                            {'if': {'column_id': 'close_rate_delta', 'filter_query': '{close_rate_delta} < 0'}, 'color': COLORS['danger']},
                            {'if': {'column_id': 'lp2c_delta', 'filter_query': '{lp2c_delta} > 0'}, 'color': COLORS['success']},
                            {'if': {'column_id': 'lp2c_delta', 'filter_query': '{lp2c_delta} < 0'}, 'color': COLORS['danger']},
                            {'if': {'column_id': 'lvc_delta', 'filter_query': '{lvc_delta} > 0'}, 'color': COLORS['success']},
                            {'if': {'column_id': 'lvc_delta', 'filter_query': '{lvc_delta} < 0'}, 'color': COLORS['danger']},
                        ],
                        page_size=20,
                        sort_action='native',
                        sort_mode='single'
                    )
                ])
            ], style=CARD_STYLE), width=12),
        ], style={'marginTop': '16px'})

        # Add WoW comparison tables
        channel_wow_tables = create_wow_comparison_tables(df_vintage, 'channel', 'Channel', is_channel=True)
        
        # Add LT PL Pricing WoW comparison tables
        df_ltpl_pricing = get_ltpl_pricing_data(target_week)
        ltpl_pricing_tables = create_ltpl_pricing_wow_table(df_ltpl_pricing)

        # Append the channel comparison to channel_v2_row
        channel_v2_row = html.Div([channel_v2_row, channel_comparison_row, channel_perf_row, channel_wow_tables, ltpl_pricing_tables])
        
        # Combine all into lvc_row
        lvc_row = html.Div([
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_lvc_count)], style=CARD_STYLE), width=6),
                dbc.Col(html.Div([dcc.Graph(figure=fig_lvc_dollars)], style=CARD_STYLE), width=6),
            ]),
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_gross_leads_pct)], style=CARD_STYLE), width=10),
                dbc.Col(insights_box, width=2),
            ]),
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_sts)], style=CARD_STYLE), width=10),
                dbc.Col(insights_box_sts, width=2),
            ]),
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_lp2c)], style=CARD_STYLE), width=10),
                dbc.Col(insights_box_lp2c, width=2),
            ]),
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_sf_contact)], style=CARD_STYLE), width=10),
                dbc.Col(insights_box_sf, width=2),
            ]),
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_fas_pct)], style=CARD_STYLE), width=10),
                dbc.Col(insights_box_fas_pct, width=2),
            ]),
            dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_fas_avg)], style=CARD_STYLE), width=10),
                dbc.Col(insights_box_fas_avg, width=2),
            ]),
        ])
    
    # === MA Analysis ===
    ma_row = html.Div("Loading MA analysis...")
    df_ma = get_ma_performance_data(target_week)
    if not df_ma.empty:
        df_ma['vintage_week'] = pd.to_datetime(df_ma['vintage_week'])
        
        # Get target week data for tables
        df_ma_target = df_ma[df_ma['vintage_week'] == target_date]
        
        # Add funded_count if missing
        if 'funded_count' not in df_ma.columns:
            df_ma['funded_count'] = 0
        if 'avg_fas_day7_dollars' not in df_ma.columns:
            df_ma['avg_fas_day7_dollars'] = 0
        
        if 'total_fas_dollars' not in df_ma.columns:
            df_ma['total_fas_dollars'] = 0
            
        # Get 6-week average data (excluding target week)
        df_ma_prev = df_ma[(df_ma['vintage_week'] < target_date) & 
                          (df_ma['vintage_week'] >= target_date - timedelta(weeks=6))]
        
        # Table 1: By Starring Group with 6wk avg and delta
        starring_target = df_ma_target.groupby('starring_group').agg({
            'leads_assigned': 'sum', 'avg_lp2c': 'mean', 'avg_lvc': 'mean', 'call_attempts': 'sum',
            'sf_contact_day7_count': 'sum', 'day7_eligible_sts': 'sum', 'fas_day7_count': 'sum', 
            'fas_day7_dollars': 'sum', 'total_fas_dollars': 'sum', 'funded_count': 'sum'
        }).reset_index()
        
        starring_6wk = df_ma_prev.groupby('starring_group').agg({
            'leads_assigned': 'sum', 'avg_lp2c': 'mean', 'avg_lvc': 'mean', 'call_attempts': 'sum',
            'sf_contact_day7_count': 'sum', 'day7_eligible_sts': 'sum', 'fas_day7_count': 'sum',
            'fas_day7_dollars': 'sum', 'total_fas_dollars': 'sum', 'funded_count': 'sum'
        }).reset_index()
        # Average over 6 weeks
        for col in ['leads_assigned', 'call_attempts', 'sf_contact_day7_count', 'day7_eligible_sts', 'fas_day7_count', 'fas_day7_dollars', 'total_fas_dollars', 'funded_count']:
            starring_6wk[col] = starring_6wk[col] / 6
        
        starring_merged = starring_target.merge(starring_6wk, on='starring_group', suffixes=('', '_6wk'), how='outer').fillna(0)
        starring_merged = starring_merged.rename(columns={'avg_lp2c': 'lp2c', 'avg_lvc': 'lvc_avg'})
        starring_merged['contacts_day7_pct'] = (starring_merged['sf_contact_day7_count'] / starring_merged['day7_eligible_sts'].replace(0, 1) * 100).round(1)
        starring_merged['fas_day7_pct'] = (starring_merged['fas_day7_count'] / starring_merged['day7_eligible_sts'].replace(0, 1) * 100).round(1)
        starring_merged['close_rate'] = (starring_merged['funded_count'] / starring_merged['leads_assigned'].replace(0, 1) * 100).round(1)
        starring_merged['fas_div_lp2c'] = (starring_merged['fas_day7_pct'] / starring_merged['lp2c'].replace(0, 1)).round(2)
        # 6wk avg rates
        starring_merged['contacts_day7_pct_6wk'] = (starring_merged['sf_contact_day7_count_6wk'] / starring_merged['day7_eligible_sts_6wk'].replace(0, 1) * 100).round(1)
        starring_merged['fas_day7_pct_6wk'] = (starring_merged['fas_day7_count_6wk'] / starring_merged['day7_eligible_sts_6wk'].replace(0, 1) * 100).round(1)
        starring_merged['close_rate_6wk'] = (starring_merged['funded_count_6wk'] / starring_merged['leads_assigned_6wk'].replace(0, 1) * 100).round(1)
        starring_merged['lp2c_6wk'] = starring_merged['avg_lp2c_6wk'].round(2)
        starring_merged['fas_div_lp2c_6wk'] = (starring_merged['fas_day7_pct_6wk'] / starring_merged['lp2c_6wk'].replace(0, 1)).round(2)
        # Deltas
        starring_merged['leads_delta'] = ((starring_merged['leads_assigned'] - starring_merged['leads_assigned_6wk']) / starring_merged['leads_assigned_6wk'].replace(0, 1) * 100).round(1)
        starring_merged['fas_dollars_delta'] = ((starring_merged['fas_day7_dollars'] - starring_merged['fas_day7_dollars_6wk']) / starring_merged['fas_day7_dollars_6wk'].replace(0, 1) * 100).round(1)
        starring_merged['total_fas_dollars_delta'] = ((starring_merged['total_fas_dollars'] - starring_merged['total_fas_dollars_6wk']) / starring_merged['total_fas_dollars_6wk'].replace(0, 1) * 100).round(1)
        starring_merged['fas_pct_delta'] = (starring_merged['fas_day7_pct'] - starring_merged['fas_day7_pct_6wk']).round(1)
        starring_merged['close_rate_delta'] = (starring_merged['close_rate'] - starring_merged['close_rate_6wk']).round(1)
        starring_merged['fas_div_lp2c_delta'] = (starring_merged['fas_div_lp2c'] - starring_merged['fas_div_lp2c_6wk']).round(2)
        starring_merged['lp2c_delta'] = (starring_merged['lp2c'] - starring_merged['lp2c_6wk']).round(1)
        starring_merged['lvc_6wk'] = starring_merged['avg_lvc_6wk'].round(1)
        starring_merged['lvc_delta'] = (starring_merged['lvc_avg'] - starring_merged['lvc_6wk']).round(1)
        starring_merged = starring_merged.sort_values('fas_day7_dollars', ascending=False)
        
        # Table 2: By Mortgage Advisor with 6wk avg and delta
        ma_target = df_ma_target.groupby('mortgage_advisor').agg({
            'leads_assigned': 'sum', 'avg_lp2c': 'mean', 'avg_lvc': 'mean', 'call_attempts': 'sum',
            'sf_contact_day7_count': 'sum', 'day7_eligible_sts': 'sum', 'fas_day7_count': 'sum',
            'fas_day7_dollars': 'sum', 'total_fas_dollars': 'sum', 'funded_count': 'sum'
        }).reset_index()
        
        ma_6wk = df_ma_prev.groupby('mortgage_advisor').agg({
            'leads_assigned': 'sum', 'avg_lp2c': 'mean', 'avg_lvc': 'mean', 'call_attempts': 'sum',
            'sf_contact_day7_count': 'sum', 'day7_eligible_sts': 'sum', 'fas_day7_count': 'sum',
            'fas_day7_dollars': 'sum', 'total_fas_dollars': 'sum', 'funded_count': 'sum'
        }).reset_index()
        for col in ['leads_assigned', 'call_attempts', 'sf_contact_day7_count', 'day7_eligible_sts', 'fas_day7_count', 'fas_day7_dollars', 'total_fas_dollars', 'funded_count']:
            ma_6wk[col] = ma_6wk[col] / 6
        
        ma_merged = ma_target.merge(ma_6wk, on='mortgage_advisor', suffixes=('', '_6wk'), how='outer').fillna(0)
        ma_merged = ma_merged.rename(columns={'avg_lp2c': 'lp2c', 'avg_lvc': 'lvc_avg'})
        ma_merged['contacts_day7_pct'] = (ma_merged['sf_contact_day7_count'] / ma_merged['day7_eligible_sts'].replace(0, 1) * 100).round(1)
        ma_merged['fas_day7_pct'] = (ma_merged['fas_day7_count'] / ma_merged['day7_eligible_sts'].replace(0, 1) * 100).round(1)
        ma_merged['close_rate'] = (ma_merged['funded_count'] / ma_merged['leads_assigned'].replace(0, 1) * 100).round(1)
        ma_merged['fas_div_lp2c'] = (ma_merged['fas_day7_pct'] / ma_merged['lp2c'].replace(0, 1)).round(2)
        # 6wk avg rates
        ma_merged['contacts_day7_pct_6wk'] = (ma_merged['sf_contact_day7_count_6wk'] / ma_merged['day7_eligible_sts_6wk'].replace(0, 1) * 100).round(1)
        ma_merged['fas_day7_pct_6wk'] = (ma_merged['fas_day7_count_6wk'] / ma_merged['day7_eligible_sts_6wk'].replace(0, 1) * 100).round(1)
        ma_merged['close_rate_6wk'] = (ma_merged['funded_count_6wk'] / ma_merged['leads_assigned_6wk'].replace(0, 1) * 100).round(1)
        ma_merged['lp2c_6wk'] = ma_merged['avg_lp2c_6wk'].round(2)
        ma_merged['fas_div_lp2c_6wk'] = (ma_merged['fas_day7_pct_6wk'] / ma_merged['lp2c_6wk'].replace(0, 1)).round(2)
        # Deltas
        ma_merged['leads_delta'] = ((ma_merged['leads_assigned'] - ma_merged['leads_assigned_6wk']) / ma_merged['leads_assigned_6wk'].replace(0, 1) * 100).round(1)
        ma_merged['fas_dollars_delta'] = ((ma_merged['fas_day7_dollars'] - ma_merged['fas_day7_dollars_6wk']) / ma_merged['fas_day7_dollars_6wk'].replace(0, 1) * 100).round(1)
        ma_merged['total_fas_dollars_delta'] = ((ma_merged['total_fas_dollars'] - ma_merged['total_fas_dollars_6wk']) / ma_merged['total_fas_dollars_6wk'].replace(0, 1) * 100).round(1)
        ma_merged['fas_pct_delta'] = (ma_merged['fas_day7_pct'] - ma_merged['fas_day7_pct_6wk']).round(1)
        ma_merged['close_rate_delta'] = (ma_merged['close_rate'] - ma_merged['close_rate_6wk']).round(1)
        ma_merged['fas_div_lp2c_delta'] = (ma_merged['fas_div_lp2c'] - ma_merged['fas_div_lp2c_6wk']).round(2)
        ma_merged['lp2c_delta'] = (ma_merged['lp2c'] - ma_merged['lp2c_6wk']).round(1)
        ma_merged['lvc_6wk'] = ma_merged['avg_lvc_6wk'].round(1)
        ma_merged['lvc_delta'] = (ma_merged['lvc_avg'] - ma_merged['lvc_6wk']).round(1)
        ma_merged = ma_merged.sort_values('fas_day7_dollars', ascending=False).head(20)
        
        # Keep old names for compatibility
        starring_target = starring_merged
        ma_target = ma_merged
        
        # === NEW: 6-Week Tables by MA and by Starring (FAS%, LP2C, Avg LP2C, ratios, deltas) ===
        # Normalize vintage_week to date for reliable comparison
        df_ma['_vintage_date'] = pd.to_datetime(df_ma['vintage_week']).dt.normalize()
        weeks_6 = sorted(df_ma['_vintage_date'].unique())[-6:]
        week_strs = [pd.to_datetime(w).strftime('%Y-%m-%d') for w in weeks_6]
        
        def build_6week_ratio_table(grp_col, label_col):
            """grp_col = 'mortgage_advisor' or 'starring_group', label_col same for display."""
            sub = df_ma[df_ma['_vintage_date'].isin(weeks_6)].copy()
            if sub.empty:
                return pd.DataFrame()
            by_week = sub.groupby([grp_col, '_vintage_date']).agg({
                'fas_day7_count': 'sum', 'day7_eligible_sts': 'sum', 'avg_lp2c': 'mean', 'leads_assigned': 'sum'
            }).reset_index()
            by_week['fas_day7_pct'] = (by_week['fas_day7_count'] / by_week['day7_eligible_sts'].replace(0, 1) * 100).round(2)
            by_week['lp2c'] = by_week['avg_lp2c'].round(2)
            # Week-level avg LP2C (weighted by leads)
            week_agg = sub.groupby('_vintage_date').apply(
                lambda x: (x['avg_lp2c'] * x['leads_assigned']).sum() / x['leads_assigned'].sum() if x['leads_assigned'].sum() > 0 else 0,
                include_groups=False
            ).reset_index(name='avg_lp2c_week')
            by_week = by_week.merge(week_agg, on='_vintage_date')
            by_week['fas_pct_div_lp2c'] = (by_week['fas_day7_pct'] / by_week['lp2c'].replace(0, 1)).round(2)
            by_week['fas_pct_div_avg_lp2c'] = (by_week['fas_day7_pct'] / by_week['avg_lp2c_week'].replace(0, 1)).round(2)
            
            records = []
            for g in by_week[grp_col].unique():
                r = {label_col: g}
                gdf = by_week[by_week[grp_col] == g]
                for i, w in enumerate(weeks_6):
                    ws = week_strs[i]
                    gw = gdf[gdf['_vintage_date'] == w]
                    if not gw.empty:
                        row = gw.iloc[0]
                        r[f'{ws}_FAS%'] = row['fas_day7_pct']
                        r[f'{ws}_LP2C'] = row['lp2c']
                        r[f'{ws}_AvgLP2C'] = row['avg_lp2c_week']
                        r[f'{ws}_FAS/LP2C'] = row['fas_pct_div_lp2c']
                        r[f'{ws}_FAS/AvgLP2C'] = row['fas_pct_div_avg_lp2c']
                    else:
                        r[f'{ws}_FAS%'] = None
                        r[f'{ws}_LP2C'] = None
                        r[f'{ws}_AvgLP2C'] = None
                        r[f'{ws}_FAS/LP2C'] = None
                        r[f'{ws}_FAS/AvgLP2C'] = None
                target_ws = week_strs[-1]
                for metric, suffix in [('FAS%', '_FAS%'), ('LP2C', '_LP2C'), ('AvgLP2C', '_AvgLP2C'), ('FAS/LP2C', '_FAS/LP2C'), ('FAS/AvgLP2C', '_FAS/AvgLP2C')]:
                    vals = [r.get(f'{ws}{suffix}') for ws in week_strs if r.get(f'{ws}{suffix}') is not None and not pd.isna(r.get(f'{ws}{suffix}'))]
                    avg_val = sum(vals) / len(vals) if vals else None
                    tgt_val = r.get(f'{target_ws}{suffix}')
                    r[f'6wk_avg_{metric}'] = round(float(avg_val), 2) if avg_val is not None else None
                    if tgt_val is not None and not pd.isna(tgt_val) and avg_val is not None:
                        r[f'Delta_{metric}'] = round(float(tgt_val) - float(avg_val), 2)
                    else:
                        r[f'Delta_{metric}'] = None
                records.append(r)
            
            df_records = pd.DataFrame(records)
            # Sort by target week FAS/LP2C descending
            target_metric_col = f'{week_strs[-1]}_FAS/LP2C'
            if target_metric_col in df_records.columns:
                df_records = df_records.sort_values(target_metric_col, ascending=False)
            return df_records
        
        try:
            table_6w_starring = build_6week_ratio_table('starring_group', 'Starring')
            table_6w_ma = build_6week_ratio_table('mortgage_advisor', 'MA')
        except Exception as e:
            import traceback
            traceback.print_exc()
            table_6w_starring = pd.DataFrame()
            table_6w_ma = pd.DataFrame()
        
        # Build column defs and data for 6-week tables (columns = vintage week labels + deltas)
        def make_6week_table_df(tab_df, label_col):
            if tab_df.empty:
                return [], []
            
            # The weeks in descending order (latest week first)
            weeks_6_desc = weeks_6[::-1]
            week_strs_desc = week_strs[::-1]
            week_labels_desc = [pd.to_datetime(w).strftime('%-m/%d') for w in weeks_6_desc]
            
            cols = [{'name': label_col, 'id': label_col}]
            cols += [
                {'name': '6wk Avg', 'id': '6wk_avg_FAS/LP2C'},
                {'name': 'Δ to Target', 'id': 'Delta_FAS/LP2C'},
            ]
            
            for i, ws in enumerate(week_strs_desc):
                wl = week_labels_desc[i]
                cols += [
                    {'name': f'{wl}', 'id': f'{ws}_FAS/LP2C'},
                ]
                
            data = tab_df.to_dict('records')
            for row in data:
                for k in list(row.keys()):
                    if pd.isna(row.get(k)):
                        row[k] = None
                    elif isinstance(row.get(k), float):
                        row[k] = round(float(row[k]), 2)
            return cols, data
        
        cols_starring_6w, data_starring_6w = make_6week_table_df(table_6w_starring, 'Starring')
        cols_ma_6w, data_ma_6w = make_6week_table_df(table_6w_ma, 'MA')
        if not cols_starring_6w:
            cols_starring_6w = [{'name': 'Starring', 'id': 'Starring'}]
        if not cols_ma_6w:
            cols_ma_6w = [{'name': 'MA', 'id': 'MA'}]
        
        # Stat insights for 6-week ratio tables (biggest movers vs 6wk avg)
        def insights_6w_ratio(tab_df, label_col, title):
            if tab_df.empty or 'Delta_FAS/LP2C' not in tab_df.columns:
                return html.Div([html.H5(title, style={'color': COLORS['primary'], 'fontSize': '14px'}), html.P("No data", style={'color': COLORS['text_muted'], 'fontSize': '11px'})],
                    style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '15px', 'border': f"1px solid {COLORS['border']}", 'height': '320px', 'overflowY': 'auto'})
            tab_df = tab_df.copy()
            tab_df['_abs_delta'] = tab_df['Delta_FAS/LP2C'].fillna(0).abs()
            tab_df = tab_df.sort_values('_abs_delta', ascending=False)
            items = [html.H5(title, style={'color': COLORS['primary'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                     html.P("Δ = Target week vs 6wk avg", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '10px'})]
            for _, row in tab_df.head(10).iterrows():
                d = row.get('Delta_FAS/LP2C') or 0
                color = COLORS['success'] if d >= 0 else COLORS['danger']
                items.append(html.Div([
                    html.Span(str(row[label_col])[:20], style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '11px'}),
                    html.Span(f" Δ {d:+.2f}", style={'color': color, 'fontSize': '11px', 'marginLeft': '6px'}),
                ], style={'marginBottom': '6px'}))
            return html.Div(items, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '15px',
                'border': f"1px solid {COLORS['border']}", 'height': '320px', 'overflowY': 'auto', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        insights_6w_starring = insights_6w_ratio(table_6w_starring, 'Starring', '📊 Starring: Δ FAS D7/LP2C vs 6wk avg')
        insights_6w_ma = insights_6w_ratio(table_6w_ma, 'MA', '📊 MA: Δ FAS D7/LP2C vs 6wk avg')
        
        # === Statistical Insights for Starring Groups (6-week trend) ===
        starring_weekly = df_ma.groupby(['vintage_week', 'starring_group']).agg({
            'fas_day7_dollars': 'sum',
            'fas_day7_count': 'sum',
            'day7_eligible_sts': 'sum'
        }).reset_index()
        starring_weekly['fas_day7_pct'] = (starring_weekly['fas_day7_count'] / starring_weekly['day7_eligible_sts'].replace(0, 1) * 100).round(2)
        starring_weekly['wow_change'] = starring_weekly.groupby('starring_group')['fas_day7_dollars'].diff()
        
        insights_data_starring = []
        for group in starring_target['starring_group'].unique():
            group_data = starring_weekly[starring_weekly['starring_group'] == group].sort_values('vintage_week')
            if len(group_data) < 2:
                continue
            last_6 = group_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change, std_change = wow_changes.mean(), wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                latest_val = last_6['fas_day7_dollars'].iloc[-1] if len(last_6) > 0 else 0
                significance = ""
                if abs(z_score) >= 2:
                    significance = "⚠️ Significant change!" if z_score > 0 else "⚠️ Significant decline!"
                elif abs(z_score) >= 1.495:
                    significance = "Notable movement"
                insights_data_starring.append({'group': str(group), 'latest_val': latest_val, 'change': latest_change, 'z_score': z_score, 'significance': significance})
        
        insights_data_starring.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        insights_items_starring = [
            html.H5("📊 Starring Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}),
            html.P("6-Week FAS Day 7 $ Z-Score", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})
        ]
        for item in insights_data_starring[:10]:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            display_name = item['group'][:15] + '...' if len(item['group']) > 15 else item['group']
            insights_items_starring.append(html.Div([
                html.Div([html.Span(f"Star {display_name}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '11px'}),
                    html.Span(f" ${item['latest_val']/1000:,.0f}K", style={'color': COLORS['primary'], 'fontSize': '11px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div([html.Span(f"{change_symbol} ${abs(item['change'])/1000:,.0f}K", style={'color': change_color, 'fontSize': '10px'}),
                    html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'}) if item['significance'] else html.Div("Normal variation", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'}),
            ], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '6px', 'marginBottom': '6px'}))
        
        insights_box_starring = html.Div(insights_items_starring, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '15px',
            'border': f"1px solid {COLORS['border']}", 'height': '350px', 'overflowY': 'auto', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        # === Statistical Insights for MAs (6-week trend) ===
        # Get top 20 MAs by total FAS $ across all weeks
        top_mas = df_ma.groupby('mortgage_advisor')['fas_day7_dollars'].sum().nlargest(20).index.tolist()
        ma_weekly = df_ma[df_ma['mortgage_advisor'].isin(top_mas)].groupby(['vintage_week', 'mortgage_advisor']).agg({
            'fas_day7_dollars': 'sum',
            'fas_day7_count': 'sum',
            'day7_eligible_sts': 'sum'
        }).reset_index()
        ma_weekly['fas_day7_pct'] = (ma_weekly['fas_day7_count'] / ma_weekly['day7_eligible_sts'].replace(0, 1) * 100).round(2)
        ma_weekly['wow_change'] = ma_weekly.groupby('mortgage_advisor')['fas_day7_dollars'].diff()
        
        insights_data_ma = []
        for ma in top_mas:
            ma_data = ma_weekly[ma_weekly['mortgage_advisor'] == ma].sort_values('vintage_week')
            if len(ma_data) < 2:
                continue
            last_6 = ma_data.tail(6)
            wow_changes = last_6['wow_change'].dropna()
            if len(wow_changes) > 1:
                mean_change, std_change = wow_changes.mean(), wow_changes.std()
                latest_change = wow_changes.iloc[-1] if len(wow_changes) > 0 else 0
                z_score = (latest_change - mean_change) / std_change if std_change > 0 else 0
                latest_val = last_6['fas_day7_dollars'].iloc[-1] if len(last_6) > 0 else 0
                significance = ""
                if abs(z_score) >= 2:
                    significance = "⚠️ Significant change!" if z_score > 0 else "⚠️ Significant decline!"
                elif abs(z_score) >= 1.495:
                    significance = "Notable movement"
                insights_data_ma.append({'group': str(ma), 'latest_val': latest_val, 'change': latest_change, 'z_score': z_score, 'significance': significance})
        
        insights_data_ma.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        insights_items_ma = [
            html.H5("📊 MA Insights", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontSize': '14px', 'fontWeight': '600'}),
            html.P("6-Week FAS Day 7 $ Z-Score", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '15px'})
        ]
        for item in insights_data_ma[:15]:
            change_color = COLORS['success'] if item['change'] >= 0 else COLORS['danger']
            change_symbol = "↑" if item['change'] >= 0 else "↓"
            display_name = item['group'][:15] + '...' if len(item['group']) > 15 else item['group']
            insights_items_ma.append(html.Div([
                html.Div([html.Span(f"{display_name}", style={'fontWeight': '600', 'color': COLORS['text'], 'fontSize': '11px'}),
                    html.Span(f" ${item['latest_val']/1000:,.0f}K", style={'color': COLORS['primary'], 'fontSize': '11px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div([html.Span(f"{change_symbol} ${abs(item['change'])/1000:,.0f}K", style={'color': change_color, 'fontSize': '10px'}),
                    html.Span(f" (Z: {item['z_score']:.2f})", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginLeft': '5px'})], style={'marginBottom': '3px'}),
                html.Div(item['significance'], style={'color': COLORS['chart_yellow'] if item['significance'] else COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'}) if item['significance'] else html.Div("Normal variation", style={'color': COLORS['text_muted'], 'fontSize': '9px', 'marginBottom': '8px'}),
            ], style={'borderBottom': f"1px solid {COLORS['border']}", 'paddingBottom': '6px', 'marginBottom': '6px'}))
        
        insights_box_ma = html.Div(insights_items_ma, style={'backgroundColor': COLORS['card_bg'], 'borderRadius': '12px', 'padding': '15px',
            'border': f"1px solid {COLORS['border']}", 'height': '450px', 'overflowY': 'auto', 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'})
        
        table_columns_starring = [
            {'name': 'Leads', 'id': 'leads_assigned', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
            {'name': 'Leads 6wk', 'id': 'leads_assigned_6wk', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
            {'name': 'Leads Δ%', 'id': 'leads_delta', 'type': 'numeric', 'format': {'specifier': '+.0f'}},
            {'name': 'Avg. LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'LP2C 6wk', 'id': 'lp2c_6wk', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'LP2C Δ', 'id': 'lp2c_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
            {'name': 'LVC', 'id': 'lvc_avg', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'LVC 6wk', 'id': 'lvc_6wk', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'LVC Δ', 'id': 'lvc_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
            {'name': 'FAS Day 7 %', 'id': 'fas_day7_pct', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'FAS D7 % 6wk', 'id': 'fas_day7_pct_6wk', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'FAS D7 % Δ', 'id': 'fas_pct_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
            {'name': 'FAS D7/LP2C', 'id': 'fas_div_lp2c', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'F/L 6wk', 'id': 'fas_div_lp2c_6wk', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'F/L Δ', 'id': 'fas_div_lp2c_delta', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
            {'name': 'FAS Day 7 $', 'id': 'fas_day7_dollars', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
            {'name': 'FAS D7 $ 6wk', 'id': 'fas_day7_dollars_6wk', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
            {'name': 'FAS D7 $ Δ%', 'id': 'fas_dollars_delta', 'type': 'numeric', 'format': {'specifier': '+.0f'}},
            {'name': 'Total FAS $', 'id': 'total_fas_dollars', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
            {'name': 'Total FAS $ 6wk', 'id': 'total_fas_dollars_6wk', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
            {'name': 'Total FAS $ Δ%', 'id': 'total_fas_dollars_delta', 'type': 'numeric', 'format': {'specifier': '+.0f'}},
            {'name': 'Close %', 'id': 'close_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'Close % 6wk', 'id': 'close_rate_6wk', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'Close % Δ', 'id': 'close_rate_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
        ]
        
        table_style_cell = {'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '6px', 'fontSize': '11px'}
        table_style_header = {'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px', 'textAlign': 'center'}
        
        # Prepare data for starring table
        starring_cols = ['starring_group', 'leads_assigned', 'leads_assigned_6wk', 'leads_delta', 'lp2c', 'lp2c_6wk', 'lp2c_delta', 'lvc_avg', 'lvc_6wk', 'lvc_delta',
                        'fas_day7_pct', 'fas_day7_pct_6wk', 'fas_pct_delta', 'fas_div_lp2c', 'fas_div_lp2c_6wk', 'fas_div_lp2c_delta',
                        'fas_day7_dollars', 'fas_day7_dollars_6wk', 'fas_dollars_delta', 'total_fas_dollars', 'total_fas_dollars_6wk', 'total_fas_dollars_delta', 'close_rate', 'close_rate_6wk', 'close_rate_delta']
        starring_data = starring_target[[c for c in starring_cols if c in starring_target.columns]].round(1).to_dict('records')
        
        ma_row = html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("Performance by Starring Group", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                    html.Div(style={'overflowX': 'auto'}, children=[
                        dash_table.DataTable(
                            data=starring_data,
                            columns=[{'name': 'Star', 'id': 'starring_group'}] + table_columns_starring,
                            style_table={'backgroundColor': COLORS['card_bg'], 'minWidth': '100%'},
                            style_cell=table_style_cell,
                            style_header=table_style_header,
                            style_data_conditional=[
                                {'if': {'column_id': 'leads_delta', 'filter_query': '{leads_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'leads_delta', 'filter_query': '{leads_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'fas_pct_delta', 'filter_query': '{fas_pct_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'fas_pct_delta', 'filter_query': '{fas_pct_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'fas_div_lp2c_delta', 'filter_query': '{fas_div_lp2c_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'fas_div_lp2c_delta', 'filter_query': '{fas_div_lp2c_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'fas_dollars_delta', 'filter_query': '{fas_dollars_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'fas_dollars_delta', 'filter_query': '{fas_dollars_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'total_fas_dollars_delta', 'filter_query': '{total_fas_dollars_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'total_fas_dollars_delta', 'filter_query': '{total_fas_dollars_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'close_rate_delta', 'filter_query': '{close_rate_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'close_rate_delta', 'filter_query': '{close_rate_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'lp2c_delta', 'filter_query': '{lp2c_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'lp2c_delta', 'filter_query': '{lp2c_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'lvc_delta', 'filter_query': '{lvc_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'lvc_delta', 'filter_query': '{lvc_delta} < 0'}, 'color': COLORS['danger']},
                            ],
                            page_size=15,
                            sort_action='native',
                            sort_mode='single'
                        )
                    ])
                ], style=CARD_STYLE), width=9),
                dbc.Col(insights_box_starring, width=3),
            ]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("Performance by Mortgage Advisor (Top 20 by FAS Day 7 $)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                    html.Div(style={'overflowX': 'auto'}, children=[
                        dash_table.DataTable(
                            data=ma_target[[c for c in starring_cols if c in ma_target.columns and c != 'starring_group'] + ['mortgage_advisor']].round(1).to_dict('records'),
                            columns=[{'name': 'MA', 'id': 'mortgage_advisor'}] + table_columns_starring,
                            style_table={'backgroundColor': COLORS['card_bg'], 'minWidth': '100%'},
                            style_cell=table_style_cell,
                            style_header=table_style_header,
                            style_data_conditional=[
                                {'if': {'column_id': 'leads_delta', 'filter_query': '{leads_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'leads_delta', 'filter_query': '{leads_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'fas_pct_delta', 'filter_query': '{fas_pct_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'fas_pct_delta', 'filter_query': '{fas_pct_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'fas_div_lp2c_delta', 'filter_query': '{fas_div_lp2c_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'fas_div_lp2c_delta', 'filter_query': '{fas_div_lp2c_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'fas_dollars_delta', 'filter_query': '{fas_dollars_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'fas_dollars_delta', 'filter_query': '{fas_dollars_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'total_fas_dollars_delta', 'filter_query': '{total_fas_dollars_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'total_fas_dollars_delta', 'filter_query': '{total_fas_dollars_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'close_rate_delta', 'filter_query': '{close_rate_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'close_rate_delta', 'filter_query': '{close_rate_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'lp2c_delta', 'filter_query': '{lp2c_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'lp2c_delta', 'filter_query': '{lp2c_delta} < 0'}, 'color': COLORS['danger']},
                                {'if': {'column_id': 'lvc_delta', 'filter_query': '{lvc_delta} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'lvc_delta', 'filter_query': '{lvc_delta} < 0'}, 'color': COLORS['danger']},
                            ],
                            page_size=20,
                            sort_action='native',
                            sort_mode='single'
                        )
                    ])
                ], style=CARD_STYLE), width=9),
                dbc.Col(insights_box_ma, width=3),
            ]),
            # New 6-week tables: by Starring and by MA (FAS D7 / LP2C)
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("6-Week by Starring: FAS D7 / LP2C & Δ vs 6wk avg", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                    html.Div(style={'overflowX': 'auto'}, children=[
                        dash_table.DataTable(
                            data=data_starring_6w,
                            columns=cols_starring_6w,
                            style_table={'backgroundColor': COLORS['card_bg'], 'minWidth': '100%'},
                            style_cell=table_style_cell,
                            style_header=table_style_header,
                            style_data_conditional=[
                                {'if': {'column_id': 'Delta_FAS/LP2C', 'filter_query': '{Delta_FAS/LP2C} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'Delta_FAS/LP2C', 'filter_query': '{Delta_FAS/LP2C} < 0'}, 'color': COLORS['danger']},
                            ],
                            page_size=15,
                            sort_action='native',
                            sort_mode='single'
                        )
                    ])
                ], style=CARD_STYLE), width=9),
                dbc.Col(insights_6w_starring, width=3),
            ], style={'marginTop': '20px'}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("6-Week by MA: FAS D7 / LP2C & Δ vs 6wk avg", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                    html.Div(style={'overflowX': 'auto'}, children=[
                        dash_table.DataTable(
                            data=data_ma_6w,
                            columns=cols_ma_6w,
                            style_table={'backgroundColor': COLORS['card_bg'], 'minWidth': '100%'},
                            style_cell=table_style_cell,
                            style_header=table_style_header,
                            style_data_conditional=[
                                {'if': {'column_id': 'Delta_FAS/LP2C', 'filter_query': '{Delta_FAS/LP2C} > 0'}, 'color': COLORS['success']},
                                {'if': {'column_id': 'Delta_FAS/LP2C', 'filter_query': '{Delta_FAS/LP2C} < 0'}, 'color': COLORS['danger']},
                            ],
                            page_size=20,
                            sort_action='native',
                            sort_mode='single'
                        )
                    ])
                ], style=CARD_STYLE), width=9),
                dbc.Col(insights_6w_ma, width=3),
            ], style={'marginTop': '12px'}),
        ])
    
        vintage_ma_export_data = [
            {"name": "Performance by Starring Group", "columns": [{'name': 'Star', 'id': 'starring_group'}] + table_columns_starring, "rows": starring_data},
            {"name": "Performance by MA", "columns": [{'name': 'MA', 'id': 'mortgage_advisor'}] + table_columns_starring, "rows": ma_target[[c for c in starring_cols if c in ma_target.columns and c != 'starring_group'] + ['mortgage_advisor']].round(1).to_dict('records')},
            {"name": "6-Week by Starring", "columns": cols_starring_6w, "rows": data_starring_6w},
            {"name": "6-Week by MA", "columns": cols_ma_6w, "rows": data_ma_6w}
        ]
        
        # Add WoW comparison tables
        ma_wow_tables = create_wow_comparison_tables(df_ma, 'mortgage_advisor', 'MA', is_channel=False)
        
        # Add the export button to the layout
        ma_row = html.Div([
            ma_row,
            ma_wow_tables,
            dbc.Row([
                dbc.Col(html.Div([
                    dbc.Button(
                        "Export MA Data to Excel",
                        id="btn-export-ma-data",
                        color="success",
                        className="me-1",
                        style={'marginTop': '20px'}
                    ),
                ], style={'textAlign': 'right'}), width=12)
            ])
        ])
    
    return kpi_row, lvc_row, channel_v2_row, ma_row, vintage_ma_export_data

# === In-Period Deep Dive Callback ===
@callback(
    [Output('inperiod-kpi-row', 'children'),
     Output('inperiod-lvc-row', 'children'),
     Output('inperiod-channel-row', 'children'),
     Output('inperiod-ma-row', 'children')],
    Input('inperiod-week-selector', 'value')
)
def update_inperiod_deepdive_tab(target_week):
    """Update In-Period Deep Dive tab - reuses vintage data, simpler version"""
    if not target_week:
        return [html.Div("Select a week", style={'color': COLORS['text']})] * 4
    
    try:
        # Reuse the vintage deep dive data (already optimized)
        df_vintage = get_vintage_deep_dive_data(target_week)
        
        target_date = pd.to_datetime(target_week)
        
        # Initialize default rows
        kpi_row = html.Div("No data available", style={'color': COLORS['text']})
        lvc_row = html.Div("No data available", style={'color': COLORS['text']})
        channel_row = html.Div("No data available", style={'color': COLORS['text']})
        ma_row = html.Div("No data available", style={'color': COLORS['text']})
        
        if df_vintage.empty:
            return [kpi_row, lvc_row, channel_row, ma_row]
        
        df_vintage['vintage_week'] = pd.to_datetime(df_vintage['vintage_week'])
        
        # === KPI SECTION ===
        # Aggregate all metrics for target week vs previous 4 weeks
        tgt_data = df_vintage[df_vintage['vintage_week'] == target_date]
        prev_data = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                               (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))]
        
        tgt_leads = tgt_data['lead_count'].sum()
        tgt_sts = tgt_data['sts_eligible'].sum()
        tgt_contact = tgt_data['sf_contact_day7_count'].sum()
        tgt_fas = tgt_data['fas_day7_count'].sum()
        tgt_fas_dollars = tgt_data['fas_day7_dollars'].sum()
        
        prev_leads = prev_data.groupby('vintage_week')['lead_count'].sum().mean() if not prev_data.empty else 0
        prev_sts = prev_data.groupby('vintage_week')['sts_eligible'].sum().mean() if not prev_data.empty else 0
        prev_contact = prev_data.groupby('vintage_week')['sf_contact_day7_count'].sum().mean() if not prev_data.empty else 0
        prev_fas = prev_data.groupby('vintage_week')['fas_day7_count'].sum().mean() if not prev_data.empty else 0
        prev_fas_dollars = prev_data.groupby('vintage_week')['fas_day7_dollars'].sum().mean() if not prev_data.empty else 0
        
        # Calculate deltas
        leads_delta = ((tgt_leads - prev_leads) / prev_leads * 100) if prev_leads > 0 else 0
        sts_delta = ((tgt_sts - prev_sts) / prev_sts * 100) if prev_sts > 0 else 0
        contact_delta = ((tgt_contact - prev_contact) / prev_contact * 100) if prev_contact > 0 else 0
        fas_delta = ((tgt_fas - prev_fas) / prev_fas * 100) if prev_fas > 0 else 0
        fas_dollars_delta = ((tgt_fas_dollars - prev_fas_dollars) / prev_fas_dollars * 100) if prev_fas_dollars > 0 else 0
        
        # Conversion rates
        sts_rate = (tgt_sts / tgt_leads * 100) if tgt_leads > 0 else 0
        contact_rate = (tgt_contact / tgt_sts * 100) if tgt_sts > 0 else 0
        fas_rate = (tgt_fas / tgt_sts * 100) if tgt_sts > 0 else 0
        
        # Sentiment
        if fas_dollars_delta >= 10:
            week_sentiment = "STRONG WEEK"
            sentiment_color = COLORS['success']
        elif fas_dollars_delta >= -5:
            week_sentiment = "FLAT WEEK"
            sentiment_color = COLORS['chart_yellow']
        else:
            week_sentiment = "SOFT WEEK"
            sentiment_color = COLORS['danger']
        
        # Metric cards row
        kpi_row = html.Div([
            html.Div([
                html.H4(f"{week_sentiment}", style={'color': sentiment_color, 'textAlign': 'center', 'marginBottom': '15px'}),
            ]),
            html.Div([
                html.Div(create_metric_card("Gross Leads", f"{tgt_leads:,.0f}", leads_delta, "% vs 4wk avg"), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
                html.Div(create_metric_card("STS", f"{tgt_sts:,.0f}", sts_delta, "% vs 4wk avg"), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
                html.Div(create_metric_card("SF Contact", f"{tgt_contact:,.0f}", contact_delta, "% vs 4wk avg"), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
                html.Div(create_metric_card("FAS Qty", f"{tgt_fas:,.0f}", fas_delta, "% vs 4wk avg"), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
                html.Div(create_metric_card("FAS $", f"${tgt_fas_dollars/1000:,.0f}K", fas_dollars_delta, "% vs 4wk avg"), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
                html.Div(create_metric_card("FAS Rate", f"{fas_rate:.1f}%", None, ""), style={'flex': '1', 'minWidth': '0', 'padding': '0 4px'}),
            ], style={'display': 'flex', 'gap': '0px', 'width': '100%'}),
        ])
        
        # === LVC ANALYSIS ===
        lvc_target = tgt_data.groupby('lvc_group').agg({
            'lead_count': 'sum', 'sts_eligible': 'sum', 'sf_contact_day7_count': 'sum',
            'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum'
        }).reset_index()
        
        lvc_prev = prev_data.groupby('lvc_group').agg({
            'lead_count': 'sum', 'sts_eligible': 'sum', 'sf_contact_day7_count': 'sum',
            'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum'
        }).reset_index()
        lvc_prev[['lead_count', 'sts_eligible', 'sf_contact_day7_count', 'fas_day7_count', 'fas_day7_dollars']] /= 4
        
        lvc_merged = lvc_target.merge(lvc_prev, on='lvc_group', how='outer', suffixes=('_tgt', '_prev')).fillna(0)
        lvc_merged['fas_delta'] = lvc_merged['fas_day7_dollars_tgt'] - lvc_merged['fas_day7_dollars_prev']
        lvc_merged['fas_rate'] = (lvc_merged['fas_day7_count_tgt'] / lvc_merged['sts_eligible_tgt'].replace(0, 1) * 100)
        
        lvc_table_data = []
        for _, row in lvc_merged.iterrows():
            lvc_table_data.append({
                'LVC Group': row['lvc_group'],
                'Leads': f"{row['lead_count_tgt']:,.0f}",
                'STS': f"{row['sts_eligible_tgt']:,.0f}",
                'Contacts': f"{row['sf_contact_day7_count_tgt']:,.0f}",
                'FAS': f"{row['fas_day7_count_tgt']:,.0f}",
                'FAS $': f"${row['fas_day7_dollars_tgt']/1000:,.0f}K",
                'FAS Rate': f"{row['fas_rate']:.1f}%",
                'Δ FAS $': f"${row['fas_delta']/1000:+,.0f}K",
            })
        
        lvc_row = html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    dash_table.DataTable(
                        data=lvc_table_data,
                        columns=[{'name': c, 'id': c} for c in ['LVC Group', 'Leads', 'STS', 'Contacts', 'FAS', 'FAS $', 'FAS Rate', 'Δ FAS $']],
                        style_header={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'fontWeight': 'bold', 'border': f"1px solid {COLORS['border']}"},
                        style_cell={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'padding': '8px', 'textAlign': 'center'},
                        style_data_conditional=[
                            {'if': {'column_id': 'Δ FAS $', 'filter_query': '{Δ FAS $} contains "+"'}, 'color': COLORS['success']},
                            {'if': {'column_id': 'Δ FAS $', 'filter_query': '{Δ FAS $} contains "-"'}, 'color': COLORS['danger']},
                        ],
                    ),
                ], style=CARD_STYLE), width=12),
            ]),
        ])
        
        # === CHANNEL ANALYSIS ===
        ch_target = tgt_data.groupby('channel').agg({
            'lead_count': 'sum', 'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum', 'avg_lp2c': 'mean'
        }).reset_index()
        
        ch_prev = prev_data.groupby('channel').agg({
            'lead_count': 'sum', 'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum', 'avg_lp2c': 'mean'
        }).reset_index()
        ch_prev[['lead_count', 'fas_day7_count', 'fas_day7_dollars']] /= 4
        
        ch_merged = ch_target.merge(ch_prev, on='channel', how='outer', suffixes=('_tgt', '_prev')).fillna(0)
        ch_merged['fas_delta'] = ch_merged['fas_day7_dollars_tgt'] - ch_merged['fas_day7_dollars_prev']
        
        # Top 10 by FAS dollars
        ch_top10 = ch_merged.nlargest(10, 'fas_day7_dollars_tgt')
        
        ch_table_data = []
        for _, row in ch_top10.iterrows():
            ch_table_data.append({
                'Channel': row['channel'],
                'Leads': f"{row['lead_count_tgt']:,.0f}",
                'FAS': f"{row['fas_day7_count_tgt']:,.0f}",
                'FAS $': f"${row['fas_day7_dollars_tgt']/1000:,.0f}K",
                'Δ FAS $': f"${row['fas_delta']/1000:+,.0f}K",
                'Avg LP2C': f"{row['avg_lp2c_tgt']:.1f}%" if row['avg_lp2c_tgt'] > 0 else "N/A",
            })
        
        channel_row = html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    html.H6("Top 10 Channels by FAS $", style={'color': COLORS['text'], 'marginBottom': '10px'}),
                    dash_table.DataTable(
                        data=ch_table_data,
                        columns=[{'name': c, 'id': c} for c in ['Channel', 'Leads', 'FAS', 'FAS $', 'Δ FAS $', 'Avg LP2C']],
                        style_header={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'fontWeight': 'bold', 'border': f"1px solid {COLORS['border']}"},
                        style_cell={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'padding': '8px', 'textAlign': 'center'},
                        style_data_conditional=[
                            {'if': {'column_id': 'Δ FAS $', 'filter_query': '{Δ FAS $} contains "+"'}, 'color': COLORS['success']},
                            {'if': {'column_id': 'Δ FAS $', 'filter_query': '{Δ FAS $} contains "-"'}, 'color': COLORS['danger']},
                        ],
                        sort_action='native',
                    ),
                ], style=CARD_STYLE), width=12),
            ]),
        ])
        
        # === MA PERFORMANCE (using vintage data grouped by MA) ===
        ma_target = tgt_data.groupby('mortgage_advisor').agg({
            'lead_count': 'sum', 'sts_eligible': 'sum', 'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum'
        }).reset_index()
        
        ma_prev = prev_data.groupby('mortgage_advisor').agg({
            'lead_count': 'sum', 'sts_eligible': 'sum', 'fas_day7_count': 'sum', 'fas_day7_dollars': 'sum'
        }).reset_index()
        ma_prev[['lead_count', 'sts_eligible', 'fas_day7_count', 'fas_day7_dollars']] /= 4
        
        ma_merged = ma_target.merge(ma_prev, on='mortgage_advisor', how='outer', suffixes=('_tgt', '_prev')).fillna(0)
        ma_merged['fas_delta'] = ma_merged['fas_day7_dollars_tgt'] - ma_merged['fas_day7_dollars_prev']
        
        # Top 10 by FAS $
        ma_top10 = ma_merged.nlargest(10, 'fas_day7_dollars_tgt')
        
        ma_table_data = []
        for _, row in ma_top10.iterrows():
            ma_table_data.append({
                'MA': row['mortgage_advisor'] if row['mortgage_advisor'] else 'Unknown',
                'Leads': f"{row['lead_count_tgt']:,.0f}",
                'STS': f"{row['sts_eligible_tgt']:,.0f}",
                'FAS': f"{row['fas_day7_count_tgt']:,.0f}",
                'FAS $': f"${row['fas_day7_dollars_tgt']/1000:,.0f}K",
                'Δ FAS $': f"${row['fas_delta']/1000:+,.0f}K",
            })
        
        ma_row = html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    html.H6("Top 10 MAs by FAS $", style={'color': COLORS['text'], 'marginBottom': '10px'}),
                    dash_table.DataTable(
                        data=ma_table_data,
                        columns=[{'name': c, 'id': c} for c in ['MA', 'Leads', 'STS', 'FAS', 'FAS $', 'Δ FAS $']],
                        style_header={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'fontWeight': 'bold', 'border': f"1px solid {COLORS['border']}"},
                        style_cell={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'padding': '8px', 'textAlign': 'center'},
                        style_data_conditional=[
                            {'if': {'column_id': 'Δ FAS $', 'filter_query': '{Δ FAS $} contains "+"'}, 'color': COLORS['success']},
                            {'if': {'column_id': 'Δ FAS $', 'filter_query': '{Δ FAS $} contains "-"'}, 'color': COLORS['danger']},
                        ],
                        sort_action='native',
                    ),
                ], style=CARD_STYLE), width=12),
            ]),
        ])
        
        return kpi_row, lvc_row, channel_row, ma_row
        
    except Exception as e:
        error_msg = html.Div([
            html.P(f"Error loading data", style={'color': COLORS['danger'], 'fontWeight': 'bold'}),
            html.P(f"Error: {str(e)}", style={'color': COLORS['text_muted'], 'fontSize': '12px'}),
        ])
        return [error_msg] * 4

# === In-Period Button Toggle Callback ===
@callback(
    Output('inperiod-button-collapse', 'is_open'),
    Input('inperiod-button-toggle', 'n_clicks'),
    State('inperiod-button-collapse', 'is_open'),
    prevent_initial_call=True
)
def toggle_inperiod_button(n_clicks, is_open):
    """Toggle the in-period button executive summary"""
    if n_clicks:
        return not is_open
    return is_open

# === Collapse/Expand Callbacks ===
@callback(
    Output({'type': 'collapse-content', 'index': MATCH}, 'is_open'),
    Input({'type': 'collapse-btn', 'index': MATCH}, 'n_clicks'),
    State({'type': 'collapse-content', 'index': MATCH}, 'is_open'),
    prevent_initial_call=True
)
def toggle_collapse(n_clicks, is_open):
    """Toggle individual section collapse"""
    if n_clicks:
        return not is_open
    return is_open

# === THE BUTTON Toggle Callback ===
@callback(
    Output('the-button-collapse', 'is_open'),
    Input('the-button-toggle', 'n_clicks'),
    State('the-button-collapse', 'is_open'),
    prevent_initial_call=True
)
def toggle_the_button(n_clicks, is_open):
    """Toggle THE BUTTON executive summary"""
    if n_clicks:
        return not is_open
    return is_open

# === Statistical Insights Toggle Callback ===
@callback(
    Output('insights-collapse', 'is_open'),
    Input('insights-toggle-btn', 'n_clicks'),
    State('insights-collapse', 'is_open'),
    prevent_initial_call=True
)
def toggle_insights(n_clicks, is_open):
    """Toggle Statistical Insights section"""
    if n_clicks:
        return not is_open
    return is_open

@callback(
    [Output({'type': 'collapse-content', 'index': 'key-metrics'}, 'is_open', allow_duplicate=True),
     Output({'type': 'collapse-content', 'index': 'what-changed'}, 'is_open', allow_duplicate=True),
     Output({'type': 'collapse-content', 'index': 'weekly-trends'}, 'is_open', allow_duplicate=True),
     Output({'type': 'collapse-content', 'index': 'vintage-analysis'}, 'is_open', allow_duplicate=True),
     Output({'type': 'collapse-content', 'index': 'daily-comparison'}, 'is_open', allow_duplicate=True)],
    [Input('expand-all-overview', 'n_clicks'),
     Input('collapse-all-overview', 'n_clicks')],
    prevent_initial_call=True
)
def expand_collapse_all_overview(expand_clicks, collapse_clicks):
    """Expand or collapse all overview sections"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return [True] * 5
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'expand-all-overview':
        return [True] * 5
    else:
        return [False] * 5

# === Custom CSS ===
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600&family=Orbitron:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            * {
                box-sizing: border-box;
            }
            body {
                background-color: #0a1628;
                color: #e0e6ed;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                margin: 0;
                padding: 0;
            }
            .tab-container {
                border-bottom: 1px solid #1a3a5c;
            }
            .custom-tabs .tab {
                border: none !important;
                background: transparent !important;
            }
            .dash-dropdown .Select-control,
            .Select-control {
                background-color: #0d1e36 !important;
                border: 1px solid #1a3a5c !important;
                border-radius: 8px !important;
                color: #e0e6ed !important;
                min-height: 38px !important;
            }
            .dash-dropdown .Select-menu-outer,
            .Select-menu-outer {
                background-color: #0d1e36 !important;
                border: 1px solid #1a3a5c !important;
                border-radius: 8px !important;
                margin-top: 4px !important;
            }
            .dash-dropdown .Select-option,
            .Select-option {
                background-color: #0d1e36 !important;
                color: #e0e6ed !important;
                padding: 10px 12px !important;
            }
            .dash-dropdown .Select-option:hover,
            .Select-option.is-focused {
                background-color: #1a3a5c !important;
                color: #00d4aa !important;
            }
            .dash-dropdown .Select-value-label,
            .Select-value-label {
                color: #1a1a2e !important;
                font-weight: 600 !important;
            }
            .Select-single-value,
            .dash-dropdown .Select-single-value,
            .VirtualizedSelectOption,
            .VirtualizedSelectFocusedOption {
                color: #1a1a2e !important;
                font-weight: 600 !important;
            }
            .Select--single > .Select-control .Select-value {
                color: #1a1a2e !important;
            }
            .dash-dropdown .Select-placeholder,
            .Select-placeholder {
                color: #6b7c93 !important;
            }
            .Select-input input {
                color: #e0e6ed !important;
            }
            .Select-arrow-zone {
                color: #00d4aa !important;
            }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                background-color: #1a3a5c !important;
                color: #00d4aa !important;
                font-weight: 500 !important;
                text-transform: uppercase !important;
                font-size: 11px !important;
                letter-spacing: 0.5px !important;
            }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
                border-color: #1a3a5c !important;
            }
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #0a1628;
            }
            ::-webkit-scrollbar-thumb {
                background: #1a3a5c;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #00d4aa;
            }
            .card-hover:hover {
                border-color: #00d4aa !important;
                transition: border-color 0.3s ease;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

import io
import dash
import base64

# === Export Callback ===
def _export_to_xlsx(export_data, default_filename="data.xlsx"):
    """Build an xlsx file from export_data (list of {name, columns: [{id, name}], rows})."""
    if not export_data:
        return None
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for t in export_data:
            name = (t.get("name") or "Sheet")[:31]
            for c in "\\/*?:[]":
                name = name.replace(c, "")
            cols = t.get("columns") or []
            rows = t.get("rows") or []
            col_names = [c.get("name", c.get("id", "")) for c in cols]
            col_ids = [c.get("id", "") for c in cols]
            data = [[r.get(i, "") for i in col_ids] for r in rows]
            df = pd.DataFrame(data, columns=col_names)
            df.to_excel(writer, sheet_name=name or "Sheet", index=False)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@callback(
    Output("vintage-ma-download", "data"),
    Input("btn-export-ma-data", "n_clicks"),
    State("vintage-ma-export-store", "data"),
    prevent_initial_call=True,
)
def download_ma_data(n_clicks, data):
    if not n_clicks or not data:
        return dash.no_update
    content = _export_to_xlsx(data, "vintage_ma_performance.xlsx")
    if not content:
        return dash.no_update
    return dict(content=content, filename="vintage_ma_performance.xlsx", base64=True)


# === Run Server ===
if __name__ == '__main__':
    app.run(debug=True, port=8051)
