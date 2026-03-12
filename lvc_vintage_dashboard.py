"""
LVC Vintage Analysis Dashboard
Dash application for analyzing FAS performance by lead vintage
Two tabs: Overview & Deeper Dive (selectable week)
Modular layout with collapsible sections
"""

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State, ALL, MATCH, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import base64
import io

# === DARK THEME COLORS (Cyber/Tech Style) ===
COLORS = {
    'background': '#0a1628',      # Dark navy
    'card_bg': '#0d1e36',         # Slightly lighter navy
    'text': '#e0e6ed',            # Light gray text
    'text_muted': '#6b7c93',      # Muted blue-gray
    'border': '#1a3a5c',          # Navy border
    'primary': '#00d4aa',         # Cyan/teal accent
    'success': '#00d4aa',         # Cyan for positive
    'warning': '#f0b429',         # Amber warning
    'danger': '#ff6b6b',          # Coral red
    'accent1': '#00d4aa',         # Main cyan
    'accent2': '#00b894',         # Darker teal
    'accent3': '#74b9ff',         # Light blue
    'chart_cyan': '#00d4aa',      # Cyan/teal (LVC 1-2)
    'chart_green': '#2ecc71',     # Emerald green (LVC 3-8) - very distinct
    'chart_yellow': '#f39c12',    # Orange/amber (LVC 9-10)
    'chart_purple': '#9b59b6',    # Purple (PHX Transfer)
    'chart_gray': '#7f8c8d',      # Gray (Other)
}

# === MODULAR LAYOUT CONFIGURATION ===
# Change the order of sections by reordering this list
# Set 'visible' to False to hide a section, 'collapsed' to True to start collapsed
OVERVIEW_LAYOUT_CONFIG = [
    {'id': 'key-metrics', 'title': 'Key Performance Metrics', 'subtitle': 'In-Period: Based on FAS Submit Date', 'visible': True, 'collapsed': False},
    {'id': 'what-changed', 'title': 'What Changed vs Previous 4 Weeks', 'subtitle': 'In-Period: Based on FAS Submit Date', 'visible': True, 'collapsed': False},
    {'id': 'weekly-trends', 'title': 'Weekly FAS Metrics', 'subtitle': 'Last 12 Weeks (Sun-Sat) | In-Period Analysis', 'visible': True, 'collapsed': False},
    {'id': 'vintage-analysis', 'title': 'Vintage Analysis', 'subtitle': 'Vintage: Based on Lead Created Date - Tracking Lead Cohort Performance', 'visible': True, 'collapsed': False},
    {'id': 'daily-comparison', 'title': 'Current Week by Day vs Previous 4 Weeks', 'subtitle': 'Vintage: Comparing by Lead Created Date', 'visible': True, 'collapsed': False},
]

# In-Period Deeper Dive - based on FAS submit date
INPERIOD_DEEPDIVE_CONFIG = [
    {'id': 'inperiod-kpi', 'title': 'Week Deep Dive', 'subtitle': 'What Drove the Performance?', 'visible': True, 'collapsed': False},
    {'id': 'inperiod-lvc', 'title': 'LVC Group Analysis', 'subtitle': 'Which Lead Segments Drove Growth?', 'visible': True, 'collapsed': False},
    {'id': 'inperiod-channel', 'title': 'Channel Mix Analysis', 'subtitle': 'Based on FAS Submit Date (In-Period)', 'visible': True, 'collapsed': False},
    {'id': 'inperiod-persona', 'title': 'Persona Analysis', 'subtitle': 'Persona Mix & LVC Composition | Based on FAS Submit Date (In-Period)', 'visible': True, 'collapsed': False},
    {'id': 'inperiod-outreach', 'title': 'Outreach Metrics', 'subtitle': 'SMS, Calls & MA Performance | Based on FAS Submit Date (In-Period)', 'visible': True, 'collapsed': False},
    {'id': 'inperiod-ma', 'title': 'MA Performance Analysis', 'subtitle': 'Based on FAS Submit Date (In-Period)', 'visible': True, 'collapsed': False},
]

# Vintage Deeper Dive - based on lead_created_date
VINTAGE_DEEPDIVE_CONFIG = [
    {'id': 'vintage-kpi', 'title': 'Week Deep Dive', 'subtitle': 'What Drove the Performance?', 'visible': True, 'collapsed': False},
    {'id': 'vintage-lvc', 'title': 'LVC Group Analysis', 'subtitle': 'Which Lead Segments Drove Growth?', 'visible': True, 'collapsed': False},
    {'id': 'vintage-channel', 'title': 'Channel Mix Analysis', 'subtitle': 'Based on Lead Created Date (Vintage)', 'visible': True, 'collapsed': False},
    {'id': 'vintage-persona', 'title': 'Persona Analysis', 'subtitle': 'Persona Mix & LVC Composition | Based on Lead Created Date (Vintage)', 'visible': True, 'collapsed': False},
    {'id': 'vintage-outreach', 'title': 'Outreach Metrics', 'subtitle': 'SMS, Calls by Vintage | How Are Leads Being Worked?', 'visible': True, 'collapsed': False},
    {'id': 'vintage-ma', 'title': 'MA Performance Analysis', 'subtitle': 'Based on Lead Created Date (Vintage)', 'visible': True, 'collapsed': False},
]

# Keep old name for backwards compatibility
DEEPDIVE_LAYOUT_CONFIG = INPERIOD_DEEPDIVE_CONFIG

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

# Tab styles
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
    title="LVC Vintage Analysis"
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

# === Data Query Functions ===
def get_available_weeks():
    """Get list of available week vintages for dropdown"""
    query = """
    SELECT DISTINCT 
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as week_start
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '2025-10-01'
    AND DATE(lead_created_date) <= CURRENT_DATE()
    ORDER BY week_start DESC
    LIMIT 20
    """
    df = run_query(query)
    if not df.empty:
        df['week_start'] = pd.to_datetime(df['week_start'])
        return df['week_start'].dt.strftime('%Y-%m-%d').tolist()
    return []

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
    ORDER BY 1
    """
    return run_query(query)

def get_daily_comparison_data(target_week):
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
    ORDER BY 1
    """
    return run_query(query)

def get_lvc_breakdown_data(target_week):
    """Get LVC group breakdown for target week vs prev 4 weeks - IN-PERIOD based"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    
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
    """
    return run_query(query)

def get_lvc_vintage_breakdown_data(target_week):
    """Get LVC group breakdown for target week vs prev 4 weeks - VINTAGE based"""
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
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sts_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    GROUP BY 1, 2
    """
    return run_query(query)

def get_what_changed_data(target_week):
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

def get_deep_dive_data(target_week):
    """Get comprehensive data for deep dive analysis"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    
    query = f"""
    WITH base_data AS (
        SELECT
            DATE(full_app_submit_datetime) as fas_date,
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
            DATE(lead_created_date) as vintage_date,
            DATE_DIFF(DATE(full_app_submit_datetime), DATE(lead_created_date), DAY) as days_to_fas,
            CASE 
                WHEN first_call_attempt_datetime IS NOT NULL AND current_sales_assigned_date IS NOT NULL
                THEN TIMESTAMP_DIFF(first_call_attempt_datetime, TIMESTAMP(current_sales_assigned_date), MINUTE)
                ELSE NULL
            END as speed_to_dial_minutes,
            CASE WHEN express_app_started_at IS NOT NULL THEN 1 ELSE 0 END as is_digital_start,
            call_attempts,
            lendage_guid,
            e_loan_amount,
            initial_lead_score_lp2c
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE full_app_submit_datetime IS NOT NULL
        AND DATE(full_app_submit_datetime) >= '{prev_start.strftime('%Y-%m-%d')}'
        AND DATE(full_app_submit_datetime) <= '{target_end.strftime('%Y-%m-%d')}'
    )
    SELECT
        fas_date,
        fas_week,
        lvc_group,
        persona,
        mortgage_advisor,
        channel,
        vintage_date,
        days_to_fas,
        speed_to_dial_minutes,
        is_digital_start,
        call_attempts,
        COUNT(DISTINCT lendage_guid) as fas_count,
        SUM(e_loan_amount) as fas_dollars,
        AVG(e_loan_amount) as avg_loan,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM base_data
    GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    """
    return run_query(query)

def get_vintage_deep_dive_data(target_week):
    """Get comprehensive data for vintage-based deep dive analysis (based on lead_created_date)"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    
    query = f"""
    WITH base_data AS (
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
            CASE WHEN sent_to_sales_date IS NOT NULL THEN 1 ELSE 0 END as is_sts,
            CASE WHEN full_app_submit_datetime IS NOT NULL THEN 1 ELSE 0 END as is_fas,
            CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END as fas_dollars,
            CASE 
                WHEN first_call_attempt_datetime IS NOT NULL AND current_sales_assigned_date IS NOT NULL
                THEN TIMESTAMP_DIFF(first_call_attempt_datetime, TIMESTAMP(current_sales_assigned_date), MINUTE)
                ELSE NULL
            END as speed_to_dial_minutes,
            call_attempts,
            lendage_guid,
            e_loan_amount,
            initial_lead_score_lp2c
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE lead_created_date IS NOT NULL
        AND DATE(lead_created_date) >= '{prev_start.strftime('%Y-%m-%d')}'
        AND DATE(lead_created_date) <= '{target_end.strftime('%Y-%m-%d')}'
    )
    SELECT
        vintage_date,
        vintage_week,
        lvc_group,
        persona,
        mortgage_advisor,
        channel,
        COUNT(DISTINCT lendage_guid) as lead_count,
        COUNT(DISTINCT CASE WHEN is_sts = 1 THEN lendage_guid END) as sts_count,
        COUNT(DISTINCT CASE WHEN is_fas = 1 THEN lendage_guid END) as fas_count,
        SUM(fas_dollars) as fas_dollars,
        AVG(CASE WHEN is_fas = 1 THEN e_loan_amount END) as avg_loan,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM base_data
    GROUP BY 1, 2, 3, 4, 5, 6
    """
    return run_query(query)

def get_ma_funnel_data(target_week):
    """Get MA performance funnel data for target week vs prev 4 weeks"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=4)
    prev_end = target_date - timedelta(days=1)
    
    query = f"""
    WITH target_week AS (
        SELECT
            mortgage_advisor,
            COUNT(DISTINCT CASE WHEN current_sales_assigned_date IS NOT NULL 
                  AND DATE(current_sales_assigned_date) BETWEEN '{target_date.strftime('%Y-%m-%d')}' AND '{target_end.strftime('%Y-%m-%d')}' 
                  THEN lendage_guid END) as assigned,
            COUNT(DISTINCT CASE WHEN contacted_date IS NOT NULL 
                  AND DATE(contacted_date) BETWEEN '{target_date.strftime('%Y-%m-%d')}' AND '{target_end.strftime('%Y-%m-%d')}' 
                  THEN lendage_guid END) as contacted,
            COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL 
                  AND DATE(full_app_submit_datetime) BETWEEN '{target_date.strftime('%Y-%m-%d')}' AND '{target_end.strftime('%Y-%m-%d')}' 
                  THEN lendage_guid END) as fas_count,
            SUM(CASE WHEN full_app_submit_datetime IS NOT NULL 
                AND DATE(full_app_submit_datetime) BETWEEN '{target_date.strftime('%Y-%m-%d')}' AND '{target_end.strftime('%Y-%m-%d')}' 
                THEN e_loan_amount ELSE 0 END) as fas_dollars,
            COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL 
                  AND DATE(sent_to_sales_date) BETWEEN '{target_date.strftime('%Y-%m-%d')}' AND '{target_end.strftime('%Y-%m-%d')}' 
                  THEN lendage_guid END) as sent_to_sales,
            AVG(CASE WHEN full_app_submit_datetime IS NOT NULL 
                AND DATE(full_app_submit_datetime) BETWEEN '{target_date.strftime('%Y-%m-%d')}' AND '{target_end.strftime('%Y-%m-%d')}' 
                THEN initial_lead_score_lp2c END) * 100 as lp2c
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE mortgage_advisor IS NOT NULL
        GROUP BY 1
    ),
    prev_weeks AS (
        SELECT
            mortgage_advisor,
            COUNT(DISTINCT CASE WHEN current_sales_assigned_date IS NOT NULL 
                  AND DATE(current_sales_assigned_date) BETWEEN '{prev_start.strftime('%Y-%m-%d')}' AND '{prev_end.strftime('%Y-%m-%d')}' 
                  THEN lendage_guid END) / 4.0 as assigned_prev,
            COUNT(DISTINCT CASE WHEN contacted_date IS NOT NULL 
                  AND DATE(contacted_date) BETWEEN '{prev_start.strftime('%Y-%m-%d')}' AND '{prev_end.strftime('%Y-%m-%d')}' 
                  THEN lendage_guid END) / 4.0 as contacted_prev,
            COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL 
                  AND DATE(full_app_submit_datetime) BETWEEN '{prev_start.strftime('%Y-%m-%d')}' AND '{prev_end.strftime('%Y-%m-%d')}' 
                  THEN lendage_guid END) / 4.0 as fas_count_prev,
            SUM(CASE WHEN full_app_submit_datetime IS NOT NULL 
                AND DATE(full_app_submit_datetime) BETWEEN '{prev_start.strftime('%Y-%m-%d')}' AND '{prev_end.strftime('%Y-%m-%d')}' 
                THEN e_loan_amount ELSE 0 END) / 4.0 as fas_dollars_prev,
            AVG(CASE WHEN full_app_submit_datetime IS NOT NULL 
                AND DATE(full_app_submit_datetime) BETWEEN '{prev_start.strftime('%Y-%m-%d')}' AND '{prev_end.strftime('%Y-%m-%d')}' 
                THEN initial_lead_score_lp2c END) * 100 as lp2c_prev
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE mortgage_advisor IS NOT NULL
        GROUP BY 1
    )
    SELECT 
        t.mortgage_advisor,
        COALESCE(t.assigned, 0) as assigned,
        COALESCE(t.contacted, 0) as contacted,
        COALESCE(t.fas_count, 0) as fas_count,
        COALESCE(t.fas_dollars, 0) as fas_dollars,
        COALESCE(t.sent_to_sales, 0) as sent_to_sales,
        COALESCE(t.lp2c, 0) as lp2c,
        COALESCE(p.assigned_prev, 0) as assigned_prev,
        COALESCE(p.contacted_prev, 0) as contacted_prev,
        COALESCE(p.fas_count_prev, 0) as fas_count_prev,
        COALESCE(p.fas_dollars_prev, 0) as fas_dollars_prev,
        COALESCE(p.lp2c_prev, 0) as lp2c_prev
    FROM target_week t
    LEFT JOIN prev_weeks p ON t.mortgage_advisor = p.mortgage_advisor
    WHERE t.fas_count > 0 OR t.assigned > 0
    ORDER BY t.fas_dollars DESC
    """
    return run_query(query)

def get_outreach_metrics(target_week):
    """Get outreach metrics (SMS, Calls) for target week vs previous 6 weeks"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=6)
    prev_end = target_date - timedelta(days=1)
    
    query = f"""
    WITH weekly_data AS (
        SELECT
            DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as week_start,
            mortgage_advisor,
            COUNT(DISTINCT lendage_guid) as total_leads,
            COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales,
            COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
            SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars,
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
    )
    SELECT
        week_start,
        mortgage_advisor,
        total_leads,
        sent_to_sales,
        fas_count,
        fas_dollars,
        total_sms,
        total_sms_before_contact,
        total_call_attempts,
        avg_sms,
        avg_sms_before_contact,
        avg_call_attempts,
        CASE WHEN sent_to_sales > 0 THEN total_sms * 1.0 / sent_to_sales ELSE 0 END as sms_per_lead,
        CASE WHEN sent_to_sales > 0 THEN total_sms_before_contact * 1.0 / sent_to_sales ELSE 0 END as sms_pre_contact_per_lead,
        CASE WHEN sent_to_sales > 0 THEN total_call_attempts * 1.0 / sent_to_sales ELSE 0 END as calls_per_lead,
        CASE WHEN sent_to_sales > 0 THEN fas_count * 100.0 / sent_to_sales ELSE 0 END as fas_rate
    FROM weekly_data
    ORDER BY week_start, fas_count DESC
    """
    return run_query(query)

# Outreach Metrics tab: fixed periods for LVC 1-8 calls comparison
OUTREACH_PERIOD_A_START = '2025-12-19'
OUTREACH_PERIOD_A_END   = '2026-01-26'
OUTREACH_PERIOD_B_START = '2026-01-27'
OUTREACH_PERIOD_B_END   = '2026-03-06'

def get_outreach_calls_data():
    """Get call/outreach metrics for LVC 1-8, Period A (12/19/25-1/26/26) vs Period B (1/27/26-3/6/26).
    Returns DataFrame with period, starring, mortgage_advisor, and speed-to-dial / speed-to-contact metrics.
    """
    query = f"""
    WITH base AS (
        SELECT
            lendage_guid,
            CASE
                WHEN DATE(sent_to_sales_date) BETWEEN '{OUTREACH_PERIOD_A_START}' AND '{OUTREACH_PERIOD_A_END}' THEN 'Period A'
                WHEN DATE(sent_to_sales_date) BETWEEN '{OUTREACH_PERIOD_B_START}' AND '{OUTREACH_PERIOD_B_END}' THEN 'Period B'
            END AS period,
            COALESCE(CAST(starring AS STRING), 'Unknown') AS starring,
            mortgage_advisor,
            CASE
                WHEN first_call_attempt_datetime IS NOT NULL AND current_sales_assigned_date IS NOT NULL
                THEN TIMESTAMP_DIFF(first_call_attempt_datetime, TIMESTAMP(current_sales_assigned_date), MINUTE)
            END AS speed_to_dial_minutes,
            CASE
                WHEN sf__contacted_guid IS NOT NULL AND sf_contacted_date IS NOT NULL AND current_sales_assigned_date IS NOT NULL
                THEN TIMESTAMP_DIFF(TIMESTAMP(sf_contacted_date), TIMESTAMP(current_sales_assigned_date), MINUTE)
            END AS speed_to_contact_minutes,
            CASE WHEN first_call_attempt_datetime IS NOT NULL THEN 1 ELSE 0 END AS had_call,
            CASE WHEN sf__contacted_guid IS NOT NULL THEN 1 ELSE 0 END AS contacted,
            COALESCE(call_attempts, 0) AS call_attempts
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE adjusted_lead_value_cohort IN ('1', '2', '3', '4', '5', '6', '7', '8')
        AND sent_to_sales_date IS NOT NULL
        AND DATE(sent_to_sales_date) BETWEEN '{OUTREACH_PERIOD_A_START}' AND '{OUTREACH_PERIOD_B_END}'
    )
    SELECT
        period,
        starring,
        mortgage_advisor,
        COUNT(*) AS lead_count,
        SUM(had_call) AS leads_with_call,
        SUM(contacted) AS contacted_count,
        SUM(call_attempts) AS total_call_attempts,
        AVG(CAST(call_attempts AS FLOAT64)) AS avg_call_attempts,
        AVG(speed_to_dial_minutes) AS avg_speed_to_dial,
        APPROX_QUANTILES(speed_to_dial_minutes, 100)[OFFSET(50)] AS median_speed_to_dial,
        COUNTIF(speed_to_dial_minutes IS NOT NULL AND speed_to_dial_minutes <= 5) AS dial_within_5min,
        COUNTIF(speed_to_dial_minutes IS NOT NULL AND speed_to_dial_minutes <= 15) AS dial_within_15min,
        COUNTIF(speed_to_dial_minutes IS NOT NULL AND speed_to_dial_minutes <= 60) AS dial_within_60min,
        AVG(speed_to_contact_minutes) AS avg_speed_to_contact,
        APPROX_QUANTILES(speed_to_contact_minutes, 100)[OFFSET(50)] AS median_speed_to_contact,
        COUNTIF(speed_to_contact_minutes IS NOT NULL AND speed_to_contact_minutes <= 5) AS contact_within_5min,
        COUNTIF(speed_to_contact_minutes IS NOT NULL AND speed_to_contact_minutes <= 15) AS contact_within_15min,
        COUNTIF(speed_to_contact_minutes IS NOT NULL AND speed_to_contact_minutes <= 60) AS contact_within_60min
    FROM base
    GROUP BY 1, 2, 3
    ORDER BY period, starring, mortgage_advisor
    """
    return run_query(query)

def get_starring_data(target_week, date_field='full_app_submit_datetime'):
    """Get starring distribution and performance data for target week vs previous 6 weeks"""
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    prev_start = target_date - timedelta(weeks=6)
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE({date_field}), WEEK(SUNDAY)) as week_start,
        CASE
            WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
            WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
            WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
            WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
            ELSE 'Other'
        END as lvc_group,
        starring,
        COUNT(DISTINCT lendage_guid) as lead_count,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sts_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        -- New Metrics for additional tables
        COUNT(DISTINCT CASE 
            WHEN full_app_submit_datetime >= TIMESTAMP(DATE(lead_created_date))
            AND DATE(full_app_submit_datetime) <= DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            THEN lendage_guid 
        END) as fas_day7_qty,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted_qty,
        COUNT(DISTINCT CASE 
            WHEN full_app_submit_datetime IS NOT NULL 
            AND funding_end_datetime IS NOT NULL 
            THEN lendage_guid 
        END) as funded_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars,
        SUM(CASE 
            WHEN full_app_submit_datetime >= TIMESTAMP(DATE(lead_created_date))
            AND DATE(full_app_submit_datetime) <= DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
            THEN e_loan_amount ELSE 0 
        END) as fas_day7_dollars,
        AVG(e_loan_amount) as avg_loan,
        AVG(initial_lead_score_lp2c) * 100 as avg_lp2c
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE DATE({date_field}) >= '{prev_start.strftime('%Y-%m-%d')}'
    AND DATE({date_field}) <= '{target_end.strftime('%Y-%m-%d')}'
    AND {date_field} IS NOT NULL
    GROUP BY 1, 2, 3
    ORDER BY 1, 3
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

def create_section_header(title, subtitle=None):
    """Create a styled section header (non-collapsible)"""
    return html.Div([
        html.H3(title, style={
            'color': COLORS['text'],
            'fontSize': '16px',
            'fontWeight': '600',
            'letterSpacing': '0.5px',
            'marginBottom': '4px' if subtitle else '16px',
            'fontFamily': "'Roboto', 'Segoe UI', sans-serif",
        }),
        html.P(subtitle, style={
            'color': COLORS['text_muted'],
            'fontSize': '12px',
            'marginBottom': '16px',
        }) if subtitle else None,
        html.Div(style={
            'width': '40px',
            'height': '2px',
            'backgroundColor': COLORS['primary'],
            'marginBottom': '20px',
        }),
    ])

def create_metric_card(title, value, delta=None, delta_suffix="", positive_is_good=True):
    """Create a metric card component with cyber/tech styling"""
    delta_color = COLORS['success'] if (delta and delta > 0 and positive_is_good) or (delta and delta < 0 and not positive_is_good) else COLORS['danger'] if delta else COLORS['text_muted']
    # Use more decimal places for small values (like pp changes)
    if delta is not None:
        if abs(delta) < 0.1 and delta != 0:
            delta_text = f"{delta:+.2f}{delta_suffix}"
        else:
            delta_text = f"{delta:+.1f}{delta_suffix}"
    else:
        delta_text = ""
    
    return html.Div([
        # Accent line at top
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
    
    # Create sparkline chart if data provided
    sparkline_chart = html.Div()
    if sparkline_data is not None and len(sparkline_data) > 0:
        fig = go.Figure()
        
        # Add sparkline
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
            xaxis=dict(
                showgrid=False, 
                showticklabels=False, 
                zeroline=False,
                fixedrange=True
            ),
            yaxis=dict(
                showgrid=False, 
                showticklabels=False, 
                zeroline=False,
                fixedrange=True
            ),
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

def create_horizontal_bar_chart(data, x_col, y_col, title, color=None, show_values=True):
    """Create a horizontal bar chart"""
    if color is None:
        color = COLORS['chart_cyan']
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=data[y_col],
        x=data[x_col],
        orientation='h',
        marker_color=color,
        marker_line_color=color,
        marker_line_width=1,
        text=data[x_col].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x) if show_values else None,
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=11)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS['text'], size=13, family="Segoe UI, Roboto, sans-serif")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], family="Segoe UI, Roboto, sans-serif"),
        margin=dict(l=120, r=50, t=50, b=40),
        xaxis=dict(gridcolor=COLORS['border'], showgrid=True, zeroline=False, gridwidth=1),
        yaxis=dict(gridcolor=COLORS['border'], showgrid=False),
        height=300,
    )
    
    return fig

def create_line_chart(data, x_col, y_cols, title, colors=None, y_format=None):
    """Create a line chart with cyber/tech styling"""
    fig = go.Figure()
    
    if colors is None:
        colors = [COLORS['chart_cyan'], COLORS['accent3'], COLORS['chart_yellow'], COLORS['chart_purple']]
    
    for i, y_col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode='lines+markers',
            name=y_col,
            line=dict(color=colors[i % len(colors)], width=2, shape='spline'),
            marker=dict(size=6, color=colors[i % len(colors)], line=dict(color=COLORS['background'], width=1)),
            fill='tozeroy' if len(y_cols) == 1 else None,
            fillcolor=f"rgba(0, 212, 170, 0.1)" if len(y_cols) == 1 else None,
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS['text'], size=13, family="Segoe UI, Roboto, sans-serif")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], family="Segoe UI, Roboto, sans-serif"),
        margin=dict(l=60, r=40, t=60, b=40),
        xaxis=dict(gridcolor=COLORS['border'], showgrid=True, gridwidth=1, zeroline=False),
        yaxis=dict(gridcolor=COLORS['border'], showgrid=True, tickformat=y_format, gridwidth=1, zeroline=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
        height=350,
        hovermode='x unified',
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

def create_bar_chart_with_value(data, x_col, y_col, title, subtitle_value=None, color=None, y_format=None, formula_text=None, value_suffix="", value_prefix=""):
    """Create a bar chart with value displayed in title area"""
    if color is None:
        color = COLORS['chart_cyan']
    
    fig = go.Figure()
    
    # Format text labels based on value type
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
    
    # Build title with formula if provided
    title_text = title
    if formula_text:
        title_text = f"{title} <span style='font-size:11px;color:{COLORS['text_muted']}'>{formula_text}</span>"
    
    # Determine y-axis tick format
    y_tickformat = y_format if y_format else None
    y_ticksuffix = value_suffix if value_suffix and value_suffix != 'M' else ''
    y_tickprefix = value_prefix if value_prefix else ''
    
    # Calculate y-axis range with padding for text labels
    y_max = data[y_col].max()
    y_range_max = y_max * 1.15  # Add 15% padding for text labels
    
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

def create_stacked_area_chart(data, x_col, y_cols, title, colors=None):
    """Create a stacked area chart"""
    fig = go.Figure()
    
    if colors is None:
        colors = [COLORS['chart_cyan'], COLORS['chart_green'], COLORS['chart_yellow'], COLORS['chart_purple'], COLORS['chart_gray']]
    
    for i, y_col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode='lines',
            name=y_col,
            stackgroup='one',
            fillcolor=colors[i % len(colors)],
            line=dict(width=0.5, color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS['text'], size=14)),
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        font=dict(color=COLORS['text']),
        margin=dict(l=60, r=40, t=60, b=40),
        xaxis=dict(gridcolor=COLORS['border'], showgrid=True),
        yaxis=dict(gridcolor=COLORS['border'], showgrid=True, tickformat='.0%'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=350,
    )
    
    return fig

# === Layout Components ===
def create_header():
    """Create dashboard header with Achieve logo"""
    # The user can place the actual logo image in the assets folder
    # For now, using a placeholder that matches the brand colors
    
    return html.Div([
        # Left side - Title and subtitle
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
            html.P("Weekly Performance Dashboard | FAS Performance by Lead Vintage & In-Period",
                   style={
                       'color': COLORS['text_muted'], 
                       'marginBottom': '0',
                       'fontSize': '12px',
                       'letterSpacing': '1px',
                       'textTransform': 'uppercase',
                   }),
        ], style={'flex': '1'}),
        # Right side - Logo (place achieve_logo.png in assets folder)
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

def create_week_selector():
    """Create week selection dropdown"""
    weeks = get_available_weeks()
    default_week = weeks[0] if weeks else '2026-02-08'
    
    return html.Div([
        html.Label("Select Analysis Week:", style={'color': COLORS['text'], 'marginRight': '10px'}),
        dcc.Dropdown(
            id='week-selector',
            options=[{'label': w, 'value': w} for w in weeks],
            value=default_week,
            style={'width': '200px', 'display': 'inline-block'},
            className='dash-dropdown-dark'
        )
    ], style={'marginBottom': '20px'})

# === App Layout ===
app.layout = html.Div([
    # Header
    create_header(),
    
            # Tabs
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
                dcc.Tab(label='Outreach Metrics', value='tab-outreach-metrics',
                        style=TAB_STYLE,
                        selected_style=TAB_SELECTED_STYLE),
            ], style={
                'marginBottom': '24px',
                'borderBottom': f"1px solid {COLORS['border']}",
            }),
    
    # Tab Content
    html.Div(id='tab-content'),
    
    # Store for data
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
    elif tab == 'tab-outreach-metrics':
        return create_outreach_metrics_tab()
    return html.Div()

def create_overview_tab():
    """Create Overview tab content with collapsible sections based on LAYOUT_CONFIG"""
    # Map section IDs to their content div IDs
    section_content_map = {
        'key-metrics': 'overview-metrics-row',
        'what-changed': 'what-changed-row',
        'weekly-trends': 'weekly-trends-row',
        'vintage-analysis': 'vintage-analysis-row',
        'daily-comparison': 'daily-comparison-row',
    }
    
    # Build sections based on config order
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
        # Week selector for overview
        html.Div([
            html.Label("Target Week:", style={'color': COLORS['text'], 'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='overview-week-selector',
                options=[{'label': w, 'value': w} for w in get_available_weeks()],
                value=get_available_weeks()[1] if len(get_available_weeks()) > 1 else (get_available_weeks()[0] if get_available_weeks() else None),
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
        
        # Dynamic sections based on config
        html.Div(sections),
    ])

def create_inperiod_deepdive_tab():
    """Create In-Period Deeper Dive tab content (based on FAS submit date)"""
    section_content_map = {
        'inperiod-kpi': 'inperiod-kpi-row',
        'inperiod-lvc': 'inperiod-lvc-row',
        'inperiod-channel': 'inperiod-channel-row',
        'inperiod-persona': 'inperiod-persona-row',
        'inperiod-outreach': 'inperiod-outreach-row',
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
            html.Span("📅 In-Period Analysis", style={'color': COLORS['primary'], 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Span("Based on FAS Submit Date", style={'color': COLORS['text_muted'], 'fontSize': '12px'}),
        ], style={'marginBottom': '12px'}),
        html.Div([
            html.Label("Select Week:", style={'color': COLORS['text'], 'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='inperiod-week-selector',
                options=[{'label': w, 'value': w} for w in get_available_weeks()],
                value=get_available_weeks()[1] if len(get_available_weeks()) > 1 else (get_available_weeks()[0] if get_available_weeks() else None),
                style={'width': '200px', 'color': '#1a1a2e'},
                className='dash-dropdown-dark'
            ),
            html.Button(
                "Expand All",
                id='expand-all-inperiod',
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
                id='collapse-all-inperiod',
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
        dcc.Store(id='inperiod-starring-export-store', data=None),
        dcc.Download(id='inperiod-starring-download'),
    ])

def create_vintage_deepdive_tab():
    """Create Vintage Deeper Dive tab content (based on lead_created_date)"""
    section_content_map = {
        'vintage-kpi': 'vintage-kpi-row',
        'vintage-lvc': 'vintage-lvc-row',
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
            html.Span("📊 Vintage Analysis", style={'color': COLORS['chart_yellow'], 'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
            html.Span("Based on Lead Created Date", style={'color': COLORS['text_muted'], 'fontSize': '12px'}),
        ], style={'marginBottom': '12px'}),
        html.Div([
            html.Label("Select Week:", style={'color': COLORS['text'], 'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='vintage-week-selector',
                options=[{'label': w, 'value': w} for w in get_available_weeks()],
                value=get_available_weeks()[1] if len(get_available_weeks()) > 1 else (get_available_weeks()[0] if get_available_weeks() else None),
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
        dcc.Store(id='vintage-starring-export-store', data=None),
        dcc.Download(id='vintage-starring-download'),
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
    """Update all overview tab components"""
    if not target_week:
        return [html.Div("Select a week", style={'color': COLORS['text']})] * 5
    
    try:
        # Get data
        df_exec = get_executive_summary_data(target_week)
        df_vintage = get_vintage_summary_data(target_week)
        df_daily = get_daily_comparison_data(target_week)
        df_changes = get_what_changed_data(target_week)
        
        target_date = pd.to_datetime(target_week)
        
        # === Key Metrics (In-Period) ===
        metrics_row = html.Div("Loading metrics...", style={'color': COLORS['text']})
        if not df_exec.empty:
            df_exec['week_start'] = pd.to_datetime(df_exec['week_start'])
            df_exec['fas_rate'] = df_exec['fas_count'] / df_exec['sts_count'].replace(0, 1) * 100
            df_exec['fas_per_sts'] = df_exec['fas_dollars'] / df_exec['sts_count'].replace(0, 1)
            
            target_data = df_exec[df_exec['week_start'] == target_date]
            prev_data = df_exec[(df_exec['week_start'] < target_date) & 
                               (df_exec['week_start'] >= target_date - timedelta(weeks=4))]
            
            if len(target_data) > 0 and len(prev_data) > 0:
                target_row = target_data.iloc[0]
                prev_avg = prev_data[['fas_count', 'fas_dollars', 'avg_loan', 'sts_count', 'fas_rate', 'fas_per_sts']].mean()
                
                fas_delta = ((target_row['fas_count'] - prev_avg['fas_count']) / prev_avg['fas_count'] * 100) if prev_avg['fas_count'] > 0 else 0
                dollar_delta = target_row['fas_dollars'] - prev_avg['fas_dollars']
                rate_delta = target_row['fas_rate'] - prev_avg['fas_rate']
                avg_loan_delta = ((target_row['avg_loan'] - prev_avg['avg_loan']) / prev_avg['avg_loan'] * 100) if prev_avg['avg_loan'] > 0 else 0
                fas_per_sts_delta = ((target_row['fas_per_sts'] - prev_avg['fas_per_sts']) / prev_avg['fas_per_sts'] * 100) if prev_avg['fas_per_sts'] > 0 else 0
                sts_delta = ((target_row['sts_count'] - prev_avg['sts_count']) / prev_avg['sts_count'] * 100) if prev_avg['sts_count'] > 0 else 0
                
                metrics_row = dbc.Row([
                    dbc.Col(create_metric_card("FAS Count", f"{target_row['fas_count']:,.0f}", fas_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("FAS $", f"${target_row['fas_dollars']/1000000:.2f}M", dollar_delta/1000000, "M vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("Avg Loan", f"${target_row['avg_loan']:,.0f}", avg_loan_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("FAS Rate", f"{target_row['fas_rate']:.1f}%", rate_delta, "pp vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("FAS $/StS", f"${target_row['fas_per_sts']:,.0f}", fas_per_sts_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("StS Volume", f"{target_row['sts_count']:,.0f}", sts_delta, "% vs 4wk avg"), width=2),
                ])
        
        # === What Changed Section ===
        what_changed_row = html.Div("Loading what changed...", style={'color': COLORS['text']})
        if not df_changes.empty:
            df_changes['fas_week'] = pd.to_datetime(df_changes['fas_week'])
            df_target = df_changes[df_changes['fas_week'] == target_date]
            df_prev = df_changes[(df_changes['fas_week'] < target_date) & 
                                (df_changes['fas_week'] >= target_date - timedelta(weeks=4))]
            
            # LVC Mix with additional metrics
            lvc_target_agg = df_target.groupby('lvc_group').agg({
                'fas_count': 'sum',
                'fas_dollars': 'sum'
            }).reset_index()
            lvc_prev_agg = df_prev.groupby('lvc_group').agg({
                'fas_count': 'sum',
                'fas_dollars': 'sum'
            }).reset_index()
            lvc_prev_agg['fas_count'] = lvc_prev_agg['fas_count'] / 4
            lvc_prev_agg['fas_dollars'] = lvc_prev_agg['fas_dollars'] / 4
            
            lvc_total_target = lvc_target_agg['fas_count'].sum()
            lvc_total_prev = lvc_prev_agg['fas_count'].sum()
            lvc_total_dollars_target = lvc_target_agg['fas_dollars'].sum()
            lvc_total_dollars_prev = lvc_prev_agg['fas_dollars'].sum()
            
            lvc_data = []
            for lvc in ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']:
                tgt_row = lvc_target_agg[lvc_target_agg['lvc_group'] == lvc]
                prv_row = lvc_prev_agg[lvc_prev_agg['lvc_group'] == lvc]
                
                tgt_fas = tgt_row['fas_count'].values[0] if len(tgt_row) > 0 else 0
                prv_fas = prv_row['fas_count'].values[0] if len(prv_row) > 0 else 0
                tgt_dollars = tgt_row['fas_dollars'].values[0] if len(tgt_row) > 0 else 0
                prv_dollars = prv_row['fas_dollars'].values[0] if len(prv_row) > 0 else 0
                
                tgt_pct = (tgt_fas / lvc_total_target * 100) if lvc_total_target > 0 else 0
                prv_pct = (prv_fas / lvc_total_prev * 100) if lvc_total_prev > 0 else 0
                tgt_avg = tgt_dollars / tgt_fas if tgt_fas > 0 else 0
                prv_avg = prv_dollars / prv_fas if prv_fas > 0 else 0
                
                lvc_data.append({
                    'LVC Group': lvc, 
                    'FAS': tgt_fas,
                    'Mix %': tgt_pct, 
                    '$FAS': tgt_dollars,
                    'Avg $FAS': tgt_avg,
                    'Δ Mix': tgt_pct - prv_pct,
                    'Δ Avg': ((tgt_avg - prv_avg) / prv_avg * 100) if prv_avg > 0 else 0
                })
            
            df_lvc = pd.DataFrame(lvc_data)
            
            # Channel Mix with full metrics
            channel_target_agg = df_target.groupby('channel').agg({
                'fas_count': 'sum',
                'fas_dollars': 'sum'
            }).reset_index()
            channel_target_agg = channel_target_agg.nlargest(5, 'fas_count')
            
            channel_prev_agg = df_prev.groupby('channel').agg({
                'fas_count': 'sum',
                'fas_dollars': 'sum'
            }).reset_index()
            channel_prev_agg['fas_count'] = channel_prev_agg['fas_count'] / 4
            channel_prev_agg['fas_dollars'] = channel_prev_agg['fas_dollars'] / 4
            
            channel_total_target = channel_target_agg['fas_count'].sum()
            channel_total_prev = channel_prev_agg['fas_count'].sum()
            
            channel_data = []
            for _, row in channel_target_agg.iterrows():
                ch = row['channel']
                tgt_fas = row['fas_count']
                tgt_dollars = row['fas_dollars']
                
                prv_row = channel_prev_agg[channel_prev_agg['channel'] == ch]
                prv_fas = prv_row['fas_count'].values[0] if len(prv_row) > 0 else 0
                prv_dollars = prv_row['fas_dollars'].values[0] if len(prv_row) > 0 else 0
                
                tgt_pct = (tgt_fas / channel_total_target * 100) if channel_total_target > 0 else 0
                prv_pct = (prv_fas / channel_total_prev * 100) if channel_total_prev > 0 else 0
                tgt_avg = tgt_dollars / tgt_fas if tgt_fas > 0 else 0
                prv_avg = prv_dollars / prv_fas if prv_fas > 0 else 0
                
                channel_data.append({
                    'Channel': ch,
                    'FAS': tgt_fas,
                    'Mix %': tgt_pct,
                    '$FAS': tgt_dollars,
                    'Avg $FAS': tgt_avg,
                    'Δ Mix': tgt_pct - prv_pct,
                    'Δ Avg': ((tgt_avg - prv_avg) / prv_avg * 100) if prv_avg > 0 else 0
                })
            
            df_channel = pd.DataFrame(channel_data)
            
            # Generate insights
            insights = []
            # Best performing LVC
            best_lvc = df_lvc.loc[df_lvc['FAS'].idxmax()]
            insights.append(f"Top LVC: {best_lvc['LVC Group']} ({best_lvc['FAS']:.0f} FAS, {best_lvc['Mix %']:.1f}% mix)")
            
            # Best channel
            if len(df_channel) > 0:
                best_channel = df_channel.iloc[0]
                insights.append(f"Top Channel: {best_channel['Channel']} ({best_channel['FAS']:.0f} FAS)")
            
            # Biggest mix shift in LVC
            biggest_lvc_shift = df_lvc.loc[df_lvc['Δ Mix'].abs().idxmax()]
            if abs(biggest_lvc_shift['Δ Mix']) >= 1:
                direction = "↑" if biggest_lvc_shift['Δ Mix'] > 0 else "↓"
                insights.append(f"Biggest LVC Shift: {biggest_lvc_shift['LVC Group']} {direction}{abs(biggest_lvc_shift['Δ Mix']):.1f}pp")
            
            # Avg loan trend
            avg_loan_target = lvc_total_dollars_target / lvc_total_target if lvc_total_target > 0 else 0
            avg_loan_prev = lvc_total_dollars_prev / lvc_total_prev if lvc_total_prev > 0 else 0
            avg_loan_change = ((avg_loan_target - avg_loan_prev) / avg_loan_prev * 100) if avg_loan_prev > 0 else 0
            if abs(avg_loan_change) >= 1:
                direction = "up" if avg_loan_change > 0 else "down"
                insights.append(f"Avg Loan {direction} {abs(avg_loan_change):.1f}% (${avg_loan_target:,.0f})")
            
            what_changed_row = html.Div([
                # Summary Section ON TOP - Full Width Centered
                html.Div([
                    html.H5("Week Summary & Insights", style={'color': COLORS['text'], 'marginBottom': '28px', 'fontSize': '24px', 'fontWeight': '600', 'textAlign': 'center'}),
                    dbc.Row([
                        # Target Week Column
                        dbc.Col([
                            html.Div([
                                html.Span("TARGET WEEK", style={'color': '#ffffff', 'fontSize': '14px', 'display': 'block', 'marginBottom': '12px', 'letterSpacing': '1px', 'fontWeight': '500'}),
                                html.Span(target_week, style={'color': COLORS['primary'], 'fontSize': '36px', 'fontWeight': '700'}),
                            ], style={'textAlign': 'center'})
                        ], width=2, className='d-flex align-items-center justify-content-center', style={'borderRight': f"1px solid {COLORS['border']}"}),
                        # Key Metrics Column - horizontal layout
                        dbc.Col([
                            html.Div([
                                html.Span("CURRENT WEEK METRICS", style={'color': '#ffffff', 'fontSize': '14px', 'display': 'block', 'marginBottom': '16px', 'letterSpacing': '1px', 'fontWeight': '500', 'textAlign': 'center'}),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.Span("Total FAS", style={'color': COLORS['text_muted'], 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                                            html.Span(f"{lvc_total_target:,.0f}", style={'color': COLORS['primary'], 'fontSize': '22px', 'fontWeight': '600'}),
                                        ], style={'textAlign': 'center'})
                                    ], width=4),
                                    dbc.Col([
                                        html.Div([
                                            html.Span("Total $FAS", style={'color': COLORS['text_muted'], 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
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
                        ], width=3, style={'borderRight': f"1px solid {COLORS['border']}"}),
                        # Change vs Prev Column - horizontal layout
                        dbc.Col([
                            html.Div([
                                html.Span("VS PREVIOUS 4 WEEKS", style={'color': '#ffffff', 'fontSize': '14px', 'display': 'block', 'marginBottom': '16px', 'letterSpacing': '1px', 'fontWeight': '500', 'textAlign': 'center'}),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.Span("FAS Δ", style={'color': COLORS['text_muted'], 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                                            html.Span(f"{lvc_total_target - lvc_total_prev:+,.0f} ({(lvc_total_target - lvc_total_prev)/lvc_total_prev*100:+.1f}%)" if lvc_total_prev > 0 else "N/A", 
                                                      style={'color': COLORS['success'] if lvc_total_target > lvc_total_prev else COLORS['danger'], 'fontSize': '18px', 'fontWeight': '600'}),
                                        ], style={'textAlign': 'center'})
                                    ], width=4),
                                    dbc.Col([
                                        html.Div([
                                            html.Span("$FAS Δ", style={'color': COLORS['text_muted'], 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                                            html.Span(f"${(lvc_total_dollars_target - lvc_total_dollars_prev)/1000000:+.2f}M ({(lvc_total_dollars_target - lvc_total_dollars_prev)/lvc_total_dollars_prev*100:+.1f}%)" if lvc_total_dollars_prev > 0 else "N/A", 
                                                      style={'color': COLORS['success'] if lvc_total_dollars_target > lvc_total_dollars_prev else COLORS['danger'], 'fontSize': '18px', 'fontWeight': '600'}),
                                        ], style={'textAlign': 'center'})
                                    ], width=4),
                                    dbc.Col([
                                        html.Div([
                                            html.Span("Avg Δ", style={'color': COLORS['text_muted'], 'fontSize': '13px', 'display': 'block', 'marginBottom': '4px'}),
                                            html.Span(f"{avg_loan_change:+.1f}%", 
                                                      style={'color': COLORS['success'] if avg_loan_change > 0 else COLORS['danger'], 'fontSize': '18px', 'fontWeight': '600'}),
                                        ], style={'textAlign': 'center'})
                                    ], width=4),
                                ])
                            ], style={'paddingLeft': '20px', 'paddingRight': '20px'})
                        ], width=3, style={'borderRight': f"1px solid {COLORS['border']}"}),
                        # Key Insights Column
                        dbc.Col([
                            html.Div([
                                html.Span("KEY INSIGHTS", style={'color': '#ffffff', 'fontSize': '14px', 'fontWeight': '500', 'display': 'block', 'marginBottom': '12px', 'letterSpacing': '1px', 'textAlign': 'center'}),
                                html.Ul([
                                    html.Li(insight, style={'color': COLORS['primary'], 'marginBottom': '10px', 'fontSize': '16px', 'lineHeight': '1.5', 'fontWeight': '500'})
                                    for insight in insights
                                ], style={'paddingLeft': '16px', 'marginTop': '0', 'marginBottom': '0'})
                            ], style={'paddingLeft': '20px'})
                        ], width=4),
                    ], className='align-items-center')
                ], style={**CARD_STYLE, 'marginBottom': '20px', 'padding': '28px'}),
                
                # LVC and Channel Tables Below
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H5("LVC Performance", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                            dash_table.DataTable(
                                data=df_lvc.to_dict('records'),
                                columns=[
                                    {'name': 'LVC', 'id': 'LVC Group'},
                                    {'name': 'FAS', 'id': 'FAS', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                                    {'name': 'Mix %', 'id': 'Mix %', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                    {'name': '$FAS', 'id': '$FAS', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
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
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H5("Channel Performance (Top 5)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                            dash_table.DataTable(
                                data=df_channel.to_dict('records'),
                                columns=[
                                    {'name': 'Channel', 'id': 'Channel'},
                                    {'name': 'FAS', 'id': 'FAS', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                                    {'name': 'Mix %', 'id': 'Mix %', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                    {'name': '$FAS', 'id': '$FAS', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
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
                    ], width=6),
                ])
            ])
        
        # === Weekly Trends ===
        weekly_trends_row = html.Div("Loading trends...", style={'color': COLORS['text']})
        if not df_exec.empty:
            df_exec_sorted = df_exec.sort_values('week_start')
            # Format as "MM/DD" showing the Sunday start of each week
            df_exec_sorted['week_label'] = df_exec_sorted['week_start'].dt.strftime('%-m/%d')
            
            # Calculate rolling 6-week averages
            df_exec_sorted['fas_count_rolling'] = df_exec_sorted['fas_count'].rolling(window=6, min_periods=1).mean()
            df_exec_sorted['fas_dollars_rolling'] = df_exec_sorted['fas_dollars'].rolling(window=6, min_periods=1).mean()
            df_exec_sorted['fas_rate_rolling'] = df_exec_sorted['fas_rate'].rolling(window=6, min_periods=1).mean()
            df_exec_sorted['fas_per_sts_rolling'] = df_exec_sorted['fas_per_sts'].rolling(window=6, min_periods=1).mean()
            df_exec_sorted['fas_dollars_m'] = df_exec_sorted['fas_dollars'] / 1000000
            df_exec_sorted['fas_dollars_m_rolling'] = df_exec_sorted['fas_dollars_rolling'] / 1000000
            
            # Helper function to create combo chart (bar + rolling avg line)
            def create_combo_chart(data, x_col, y_col, y_rolling_col, title, bar_color, y_format=',.0f', value_suffix='', value_prefix='', formula_text=None):
                fig = go.Figure()
                
                # Bar chart
                fig.add_trace(go.Bar(
                    x=data[x_col],
                    y=data[y_col],
                    name='Weekly',
                    marker_color=bar_color,
                    text=[f"{value_prefix}{v:{y_format}}{value_suffix}" for v in data[y_col]],
                    textposition='outside',
                    textfont=dict(size=10, color=COLORS['text'])
                ))
                
                # Rolling average line
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_rolling_col],
                    name='6-Week Avg',
                    mode='lines+markers',
                    line=dict(color=COLORS['warning'], width=3, dash='solid'),
                    marker=dict(size=6, color=COLORS['warning']),
                ))
                
                title_text = f"{title}<br><span style='font-size:11px;color:{COLORS['text_muted']}'>{formula_text}</span>" if formula_text else title
                y_max = max(data[y_col].max(), data[y_rolling_col].max()) * 1.15
                
                fig.update_layout(
                    title=dict(text=title_text, font=dict(color=COLORS['text'], size=13)),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor=COLORS['card_bg'],
                    font=dict(color=COLORS['text']),
                    margin=dict(l=50, r=30, t=60, b=40),
                    xaxis=dict(gridcolor=COLORS['border'], showgrid=False, tickangle=-45),
                    yaxis=dict(gridcolor=COLORS['border'], showgrid=True, range=[0, y_max]),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10)),
                    height=320,
                    showlegend=True,
                    bargap=0.3
                )
                return fig
            
            # FAS Count combo chart
            fig_fas_count = create_combo_chart(
                df_exec_sorted, 'week_label', 'fas_count', 'fas_count_rolling',
                'FAS Quantity by Week', COLORS['chart_cyan'], y_format=',.0f'
            )
            
            # FAS $ combo chart
            fig_fas_dollars = create_combo_chart(
                df_exec_sorted, 'week_label', 'fas_dollars_m', 'fas_dollars_m_rolling',
                'FAS $ by Week (Millions)', COLORS['chart_green'], y_format='.2f', value_prefix='$', value_suffix='M'
            )
            
            # FAS Rate combo chart
            fig_fas_rate = create_combo_chart(
                df_exec_sorted, 'week_label', 'fas_rate', 'fas_rate_rolling',
                'FAS Rate by Week', COLORS['chart_yellow'], y_format='.1f', value_suffix='%',
                formula_text='(FAS / StS × 100)'
            )
            
            # FAS $/StS combo chart
            fig_fas_per_sts = create_combo_chart(
                df_exec_sorted, 'week_label', 'fas_per_sts', 'fas_per_sts_rolling',
                'FAS $/StS by Week', COLORS['chart_purple'], y_format=',.0f', value_prefix='$'
            )
            
            weekly_trends_row = dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig_fas_count)], style=CARD_STYLE), width=6),
                dbc.Col(html.Div([dcc.Graph(figure=fig_fas_dollars)], style=CARD_STYLE), width=6),
                dbc.Col(html.Div([dcc.Graph(figure=fig_fas_rate)], style=CARD_STYLE), width=6),
                dbc.Col(html.Div([dcc.Graph(figure=fig_fas_per_sts)], style=CARD_STYLE), width=6),
            ])
        
        # === Vintage Analysis ===
        vintage_analysis_row = html.Div("Loading vintage analysis...", style={'color': COLORS['text']})
        if not df_vintage.empty:
            df_vintage['vintage_week'] = pd.to_datetime(df_vintage['vintage_week'])
            df_vintage['fas_rate'] = df_vintage['fas_count'] / df_vintage['sts_count'].replace(0, 1) * 100
            df_vintage['week_label'] = df_vintage['vintage_week'].dt.strftime('%-m/%d')
            df_vintage_sorted = df_vintage.sort_values('vintage_week')
            
            # Compare target week vintage to prev 4 weeks
            target_vintage = df_vintage[df_vintage['vintage_week'] == target_date]
            prev_vintage = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                                      (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))]
            
            # Get LVC vintage breakdown data
            df_lvc_vintage = get_lvc_vintage_breakdown_data(target_week)
            
            vintage_metrics = html.Div("No vintage data for target week", style={'color': COLORS['text']})
            lvc_analysis_content = html.Div()
            insights_content = html.Div()
            
            if len(target_vintage) > 0 and len(prev_vintage) > 0:
                tgt = target_vintage.iloc[0]
                prev_avg = prev_vintage[['total_leads', 'sts_count', 'fas_count', 'fas_dollars', 'avg_loan', 'fas_rate', 'avg_lp2c']].mean()
                
                # Calculate all deltas
                leads_delta = ((tgt['total_leads'] - prev_avg['total_leads']) / prev_avg['total_leads'] * 100) if prev_avg['total_leads'] > 0 else 0
                sts_delta = ((tgt['sts_count'] - prev_avg['sts_count']) / prev_avg['sts_count'] * 100) if prev_avg['sts_count'] > 0 else 0
                fas_delta = ((tgt['fas_count'] - prev_avg['fas_count']) / prev_avg['fas_count'] * 100) if prev_avg['fas_count'] > 0 else 0
                fas_rate_delta = tgt['fas_rate'] - prev_avg['fas_rate']
                fas_dollar_delta = ((tgt['fas_dollars'] - prev_avg['fas_dollars']) / prev_avg['fas_dollars'] * 100) if prev_avg['fas_dollars'] > 0 else 0
                avg_loan_delta = ((tgt['avg_loan'] - prev_avg['avg_loan']) / prev_avg['avg_loan'] * 100) if prev_avg['avg_loan'] > 0 else 0
                lp2c_delta = tgt['avg_lp2c'] - prev_avg['avg_lp2c']  # pp change since already in %
                
                vintage_metrics = dbc.Row([
                    dbc.Col(create_metric_card("Vintage Leads", f"{tgt['total_leads']:,.0f}", leads_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("Vintage StS", f"{tgt['sts_count']:,.0f}", sts_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("Vintage FAS", f"{tgt['fas_count']:,.0f}", fas_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("Vintage FAS $", f"${tgt['fas_dollars']/1000000:.2f}M", fas_dollar_delta, "% vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("Avg LP2C %", f"{tgt['avg_lp2c']:.1f}%", lp2c_delta, "pp vs 4wk avg"), width=2),
                    dbc.Col(create_metric_card("Avg Loan", f"${tgt['avg_loan']:,.0f}" if pd.notna(tgt['avg_loan']) else "N/A", avg_loan_delta if pd.notna(tgt['avg_loan']) else None, "% vs 4wk avg"), width=2),
                ])
                
                # LVC Group Analysis for Vintage
                if not df_lvc_vintage.empty:
                    df_lvc_vintage['vintage_week'] = pd.to_datetime(df_lvc_vintage['vintage_week'])
                    df_lvc_vintage['fas_rate'] = df_lvc_vintage['fas_count'] / df_lvc_vintage['sts_count'].replace(0, 1) * 100
                    
                    lvc_target = df_lvc_vintage[df_lvc_vintage['vintage_week'] == target_date]
                    lvc_prev = df_lvc_vintage[(df_lvc_vintage['vintage_week'] < target_date) & 
                                              (df_lvc_vintage['vintage_week'] >= target_date - timedelta(weeks=4))]
                    
                    # Aggregate LVC data
                    lvc_target_agg = lvc_target.groupby('lvc_group').agg({
                        'total_leads': 'sum', 'sts_count': 'sum', 'fas_count': 'sum', 'fas_dollars': 'sum'
                    }).reset_index()
                    lvc_target_agg['fas_rate'] = lvc_target_agg['fas_count'] / lvc_target_agg['sts_count'].replace(0, 1) * 100
                    
                    lvc_prev_agg = lvc_prev.groupby('lvc_group').agg({
                        'total_leads': 'sum', 'sts_count': 'sum', 'fas_count': 'sum', 'fas_dollars': 'sum'
                    }).reset_index()
                    lvc_prev_agg['total_leads'] = lvc_prev_agg['total_leads'] / 4
                    lvc_prev_agg['sts_count'] = lvc_prev_agg['sts_count'] / 4
                    lvc_prev_agg['fas_count'] = lvc_prev_agg['fas_count'] / 4
                    lvc_prev_agg['fas_dollars'] = lvc_prev_agg['fas_dollars'] / 4
                    lvc_prev_agg['fas_rate'] = lvc_prev_agg['fas_count'] / lvc_prev_agg['sts_count'].replace(0, 1) * 100
                    
                    # Calculate mix percentages
                    total_leads_target = lvc_target_agg['total_leads'].sum()
                    total_leads_prev = lvc_prev_agg['total_leads'].sum()
                    
                    lvc_comparison = []
                    insights_list = []
                    
                    for lvc in ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']:
                        tgt_row = lvc_target_agg[lvc_target_agg['lvc_group'] == lvc]
                        prev_row = lvc_prev_agg[lvc_prev_agg['lvc_group'] == lvc]
                        
                        tgt_leads = tgt_row['total_leads'].values[0] if len(tgt_row) > 0 else 0
                        tgt_fas = tgt_row['fas_count'].values[0] if len(tgt_row) > 0 else 0
                        tgt_rate = tgt_row['fas_rate'].values[0] if len(tgt_row) > 0 else 0
                        tgt_dollars = tgt_row['fas_dollars'].values[0] if len(tgt_row) > 0 else 0
                        
                        prev_leads = prev_row['total_leads'].values[0] if len(prev_row) > 0 else 0
                        prev_fas = prev_row['fas_count'].values[0] if len(prev_row) > 0 else 0
                        prev_rate = prev_row['fas_rate'].values[0] if len(prev_row) > 0 else 0
                        
                        mix_tgt = (tgt_leads / total_leads_target * 100) if total_leads_target > 0 else 0
                        mix_prev = (prev_leads / total_leads_prev * 100) if total_leads_prev > 0 else 0
                        
                        lvc_comparison.append({
                            'LVC Group': lvc,
                            'Leads': tgt_leads,
                            'Mix %': mix_tgt,
                            'FAS': tgt_fas,
                            'FAS Rate': tgt_rate,
                            'Rate Δ': tgt_rate - prev_rate,
                            'FAS $': tgt_dollars,
                        })
                        
                        # Generate insights for key groups
                        if lvc in ['LVC 1-2', 'LVC 3-8']:
                            rate_change = tgt_rate - prev_rate
                            fas_change = tgt_fas - prev_fas
                            if abs(rate_change) >= 0.5 or abs(fas_change) >= 5:
                                direction = "up" if rate_change > 0 else "down"
                                insights_list.append(f"{lvc}: FAS Rate {direction} {abs(rate_change):.1f}pp to {tgt_rate:.1f}% ({tgt_fas:.0f} FAS vs {prev_fas:.1f} prev avg)")
                    
                    df_lvc_comparison = pd.DataFrame(lvc_comparison)
                    
                    # Key insights
                    best_performer = df_lvc_comparison.loc[df_lvc_comparison['FAS Rate'].idxmax()]
                    biggest_volume = df_lvc_comparison.loc[df_lvc_comparison['FAS'].idxmax()]
                    
                    insights_list.insert(0, f"Highest FAS Rate: {best_performer['LVC Group']} at {best_performer['FAS Rate']:.1f}%")
                    insights_list.insert(1, f"Highest Volume: {biggest_volume['LVC Group']} with {biggest_volume['FAS']:.0f} FAS (${biggest_volume['FAS $']/1000000:.2f}M)")
                    
                    lvc_analysis_content = dbc.Row([
                        dbc.Col(html.Div([
                            html.H5("LVC Group Performance (Vintage)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                            dash_table.DataTable(
                                data=df_lvc_comparison.round(1).to_dict('records'),
                                columns=[
                                    {'name': 'LVC Group', 'id': 'LVC Group'},
                                    {'name': 'Leads', 'id': 'Leads', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                                    {'name': 'Mix %', 'id': 'Mix %', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                    {'name': 'FAS', 'id': 'FAS', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                                    {'name': 'FAS Rate', 'id': 'FAS Rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                    {'name': 'Rate Δ (pp)', 'id': 'Rate Δ', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                                    {'name': 'FAS $', 'id': 'FAS $', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
                                ],
                                style_table={'backgroundColor': COLORS['card_bg']},
                                style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '8px'},
                                style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'color': COLORS['primary']},
                                style_data_conditional=[
                                    {'if': {'filter_query': '{Rate Δ} > 0', 'column_id': 'Rate Δ'}, 'color': COLORS['success']},
                                    {'if': {'filter_query': '{Rate Δ} < 0', 'column_id': 'Rate Δ'}, 'color': COLORS['danger']},
                                    {'if': {'filter_query': '{LVC Group} = "LVC 1-2"'}, 'backgroundColor': 'rgba(0, 212, 170, 0.1)'},
                                    {'if': {'filter_query': '{LVC Group} = "LVC 3-8"'}, 'backgroundColor': 'rgba(0, 212, 170, 0.05)'},
                                ]
                            )
                        ], style=CARD_STYLE), width=7),
                        dbc.Col(html.Div([
                            html.H5("Key Insights", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                            html.Ul([
                                html.Li(insight, style={'color': COLORS['text'], 'marginBottom': '8px', 'fontSize': '13px'})
                                for insight in insights_list[:5]
                            ], style={'paddingLeft': '20px', 'marginTop': '8px'})
                        ], style=CARD_STYLE), width=5),
                    ])
            
            # Create bar chart for FAS Rate trend
            fig_vintage_trend = create_bar_chart_with_value(
                df_vintage_sorted, 
                'week_label', 
                'fas_rate', 
                'Vintage FAS Rate by Week',
                subtitle_value=None,
                color=COLORS['chart_cyan'],
                y_format='.1f',
                formula_text='(FAS Count / StS Count × 100)',
                value_suffix='%'
            )
            
            vintage_analysis_row = html.Div([
                html.H5("Target Week Vintage vs Previous 4 Weeks", style={'color': COLORS['text'], 'marginBottom': '16px', 'fontSize': '16px', 'fontWeight': '600'}),
                vintage_metrics,
                html.Div(style={'marginTop': '24px'}),
                lvc_analysis_content,
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
            df_daily['fas_rate'] = df_daily['fas_count'] / df_daily['sts_count'].replace(0, 1) * 100
            
            # Get target week daily data
            target_daily = df_daily[df_daily['week_start'] == target_date].copy()
            prev_daily = df_daily[(df_daily['week_start'] < target_date) & 
                                 (df_daily['week_start'] >= target_date - timedelta(weeks=4))].copy()
            
            if len(target_daily) > 0 and len(prev_daily) > 0:
                # Aggregate prev daily by day of week
                prev_by_dow = prev_daily.groupby('day_of_week').agg({
                    'total_leads': 'mean',
                    'sts_count': 'mean',
                    'fas_count': 'mean',
                    'fas_dollars': 'mean'
                }).reset_index()
                prev_by_dow['fas_rate'] = prev_by_dow['fas_count'] / prev_by_dow['sts_count'].replace(0, 1) * 100
                
                target_daily['date_label'] = target_daily['lead_date'].dt.strftime('%a %m/%d')
                target_daily = target_daily.sort_values('day_of_week')
                
                # Merge with prev
                comparison_df = target_daily.merge(
                    prev_by_dow[['day_of_week', 'fas_count', 'fas_rate']],
                    on='day_of_week',
                    suffixes=('', '_prev')
                )
                comparison_df['fas_delta'] = comparison_df['fas_count'] - comparison_df['fas_count_prev']
                comparison_df['rate_delta'] = comparison_df['fas_rate'] - comparison_df['fas_rate_prev']
                
                fig_daily_fas = create_grouped_bar_chart(
                    comparison_df, 
                    'date_label', 
                    ['fas_count', 'fas_count_prev'], 
                    '📊 Daily FAS: Target Week vs Prev 4-Wk Avg',
                    colors=[COLORS['chart_cyan'], COLORS['chart_gray']]
                )
                fig_daily_fas.update_traces(name='Target Week', selector=dict(name='fas_count'))
                fig_daily_fas.update_traces(name='Prev 4-Wk Avg', selector=dict(name='fas_count_prev'))
                
                daily_comparison_row = dbc.Row([
                    dbc.Col(html.Div([dcc.Graph(figure=fig_daily_fas)], style=CARD_STYLE), width=8),
                    dbc.Col(html.Div([
                        html.H5("Daily Comparison", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                        dash_table.DataTable(
                            data=comparison_df[['date_label', 'fas_count', 'fas_count_prev', 'fas_delta']].round(1).to_dict('records'),
                            columns=[
                                {'name': 'Day', 'id': 'date_label'},
                                {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                                {'name': 'Prev Avg', 'id': 'fas_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
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

@callback(
    [Output('inperiod-kpi-row', 'children'),
     Output('inperiod-lvc-row', 'children'),
     Output('inperiod-channel-row', 'children'),
     Output('inperiod-persona-row', 'children'),
     Output('inperiod-outreach-row', 'children'),
     Output('inperiod-ma-row', 'children'),
     Output('inperiod-starring-export-store', 'data')],
    Input('inperiod-week-selector', 'value')
)
def update_inperiod_deepdive_tab(target_week):
    """Update all in-period deep dive tab components (based on FAS submit date)"""
    inperiod_export_data = None
    if not target_week:
        return [html.Div("Select a week")] * 6 + [None]
    
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    
    # Get data
    df_exec = get_executive_summary_data(target_week)
    df_deep = get_deep_dive_data(target_week)
    df_lvc = get_lvc_breakdown_data(target_week)
    df_ma = get_ma_funnel_data(target_week)
    
    # === KPI Summary ===
    kpi_row = html.Div("Loading KPIs...")
    if not df_exec.empty:
        df_exec['week_start'] = pd.to_datetime(df_exec['week_start'])
        df_exec['fas_rate'] = df_exec['fas_count'] / df_exec['sts_count'].replace(0, 1) * 100
        df_exec['fas_per_sts'] = df_exec['fas_dollars'] / df_exec['sts_count'].replace(0, 1)
        
        # Get last 12 weeks of data for sparklines (including target week)
        twelve_weeks_ago = target_date - timedelta(weeks=11)
        sparkline_df = df_exec[(df_exec['week_start'] >= twelve_weeks_ago) & 
                               (df_exec['week_start'] <= target_date)].sort_values('week_start')
        sparkline_df['week_label'] = sparkline_df['week_start'].dt.strftime('%-m/%d')
        
        target_data = df_exec[df_exec['week_start'] == target_date]
        prev_data = df_exec[(df_exec['week_start'] < target_date) & 
                           (df_exec['week_start'] >= target_date - timedelta(weeks=4))]
        
        if len(target_data) > 0 and len(prev_data) > 0:
            target_row = target_data.iloc[0]
            prev_avg = prev_data[['fas_count', 'fas_dollars', 'avg_loan', 'sts_count', 'fas_rate', 'fas_per_sts', 'avg_lp2c']].mean()
            
            fas_delta_pct = ((target_row['fas_count'] - prev_avg['fas_count']) / prev_avg['fas_count'] * 100) if prev_avg['fas_count'] > 0 else 0
            dollars_delta_pct = ((target_row['fas_dollars'] - prev_avg['fas_dollars']) / prev_avg['fas_dollars'] * 100) if prev_avg['fas_dollars'] > 0 else 0
            rate_delta = target_row['fas_rate'] - prev_avg['fas_rate']
            loan_delta_pct = ((target_row['avg_loan'] - prev_avg['avg_loan']) / prev_avg['avg_loan'] * 100) if prev_avg['avg_loan'] > 0 else 0
            sts_delta_pct = ((target_row['sts_count'] - prev_avg['sts_count']) / prev_avg['sts_count'] * 100) if prev_avg['sts_count'] > 0 else 0
            lp2c_delta = target_row['avg_lp2c'] - prev_avg['avg_lp2c']  # pp change since already in %
            
            # Get sparkline data lists
            spark_labels = sparkline_df['week_label'].tolist()
            spark_sts = sparkline_df['sts_count'].tolist()
            spark_fas = sparkline_df['fas_count'].tolist()
            spark_dollars = sparkline_df['fas_dollars'].tolist()
            spark_rate = sparkline_df['fas_rate'].tolist()
            spark_avg = sparkline_df['avg_loan'].tolist()
            spark_lp2c = sparkline_df['avg_lp2c'].tolist()
            
            kpi_row = dbc.Row([
                dbc.Col(create_metric_card_with_sparkline("StS Volume", f"{target_row['sts_count']:,.0f}", sts_delta_pct, "% vs 4wk avg", sparkline_data=spark_sts, sparkline_labels=spark_labels, value_format="number"), width=2),
                dbc.Col(create_metric_card_with_sparkline("FAS Count", f"{target_row['fas_count']:,.0f}", fas_delta_pct, "% vs 4wk avg", sparkline_data=spark_fas, sparkline_labels=spark_labels, value_format="number"), width=2),
                dbc.Col(create_metric_card_with_sparkline("FAS $", f"${target_row['fas_dollars']/1000000:.2f}M", dollars_delta_pct, "% vs 4wk avg", sparkline_data=spark_dollars, sparkline_labels=spark_labels, value_format="currency"), width=2),
                dbc.Col(create_metric_card_with_sparkline("FAS Rate", f"{target_row['fas_rate']:.1f}%", rate_delta, "pp vs 4wk avg", sparkline_data=spark_rate, sparkline_labels=spark_labels, value_format="percent"), width=2),
                dbc.Col(create_metric_card_with_sparkline("Avg Loan", f"${target_row['avg_loan']:,.0f}", loan_delta_pct, "% vs 4wk avg", sparkline_data=spark_avg, sparkline_labels=spark_labels, value_format="currency"), width=2),
                dbc.Col(create_metric_card_with_sparkline("LP2C (Propensity to Close)", f"{target_row['avg_lp2c']:.1f}%", lp2c_delta, "pp vs 4wk avg", sparkline_data=spark_lp2c, sparkline_labels=spark_labels, value_format="percent"), width=2),
            ])
    
    # === LVC Analysis ===
    lvc_row = html.Div("Loading LVC analysis...")
    if not df_lvc.empty:
        df_lvc['fas_week'] = pd.to_datetime(df_lvc['fas_week'])
        
        lvc_target = df_lvc[df_lvc['fas_week'] == target_date].groupby('lvc_group').agg({
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        lvc_target['period'] = 'This Week'
        lvc_target = lvc_target.rename(columns={'avg_lp2c': 'lp2c'})
        
        lvc_prev = df_lvc[(df_lvc['fas_week'] < target_date) & 
                        (df_lvc['fas_week'] >= target_date - timedelta(weeks=4))].groupby('lvc_group').agg({
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        lvc_prev['fas_count'] = lvc_prev['fas_count'] / 4
        lvc_prev['fas_dollars'] = lvc_prev['fas_dollars'] / 4
        lvc_prev['period'] = '4-Wk Avg'
        lvc_prev = lvc_prev.rename(columns={'avg_lp2c': 'lp2c_prev'})
        
        lvc_combined = pd.concat([lvc_target, lvc_prev])
        lvc_order = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']
        
        # Create pivot tables for charts
        lvc_pivot_count = lvc_combined.pivot(index='lvc_group', columns='period', values='fas_count').reset_index()
        lvc_pivot_dollars = lvc_combined.pivot(index='lvc_group', columns='period', values='fas_dollars').reset_index()
        
        # Calculate % change for each LVC group
        lvc_pivot_count['pct_change'] = ((lvc_pivot_count['This Week'] - lvc_pivot_count['4-Wk Avg']) / lvc_pivot_count['4-Wk Avg'].replace(0, 1) * 100)
        lvc_pivot_dollars['pct_change'] = ((lvc_pivot_dollars['This Week'] - lvc_pivot_dollars['4-Wk Avg']) / lvc_pivot_dollars['4-Wk Avg'].replace(0, 1) * 100)
        
        # Custom grouped bar chart with % change - FAS Count (Prev first, then This Week)
        fig_lvc_count = go.Figure()
        fig_lvc_count.add_trace(go.Bar(
            x=lvc_pivot_count['lvc_group'],
            y=lvc_pivot_count['4-Wk Avg'],
            name='4-Wk Avg',
            marker_color=COLORS['chart_gray'],
            text=lvc_pivot_count['4-Wk Avg'].apply(lambda x: f"{x:,.0f}"),
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10)
        ))
        fig_lvc_count.add_trace(go.Bar(
            x=lvc_pivot_count['lvc_group'],
            y=lvc_pivot_count['This Week'],
            name='This Week',
            marker_color=COLORS['chart_cyan'],
            text=lvc_pivot_count['This Week'].apply(lambda x: f"{x:,.0f}"),
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10)
        ))
        # Add % change annotations at bottom center
        for i, row in lvc_pivot_count.iterrows():
            pct = row['pct_change']
            color = COLORS['success'] if pct >= 0 else COLORS['danger']
            fig_lvc_count.add_annotation(
                x=row['lvc_group'],
                y=0,
                text=f"{pct:+.1f}%",
                showarrow=False,
                font=dict(color=color, size=11, weight='bold'),
                yshift=-18
            )
        fig_lvc_count.update_layout(
            title=dict(text='📊 FAS Count by LVC Group', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']),
            margin=dict(l=60, r=40, t=60, b=50),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
            barmode='group',
            height=380,
            bargap=0.3,
        )
        
        # Custom grouped bar chart with % change - FAS $ (Prev first, then This Week)
        fig_lvc_dollars = go.Figure()
        fig_lvc_dollars.add_trace(go.Bar(
            x=lvc_pivot_dollars['lvc_group'],
            y=lvc_pivot_dollars['4-Wk Avg'],
            name='4-Wk Avg',
            marker_color=COLORS['chart_gray'],
            text=lvc_pivot_dollars['4-Wk Avg'].apply(lambda x: f"${x/1000000:.1f}M" if x >= 1000000 else f"${x/1000:.0f}K"),
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10)
        ))
        fig_lvc_dollars.add_trace(go.Bar(
            x=lvc_pivot_dollars['lvc_group'],
            y=lvc_pivot_dollars['This Week'],
            name='This Week',
            marker_color=COLORS['chart_green'],
            text=lvc_pivot_dollars['This Week'].apply(lambda x: f"${x/1000000:.1f}M" if x >= 1000000 else f"${x/1000:.0f}K"),
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10)
        ))
        # Add % change annotations at bottom center
        for i, row in lvc_pivot_dollars.iterrows():
            pct = row['pct_change']
            color = COLORS['success'] if pct >= 0 else COLORS['danger']
            fig_lvc_dollars.add_annotation(
                x=row['lvc_group'],
                y=0,
                text=f"{pct:+.1f}%",
                showarrow=False,
                font=dict(color=color, size=11, weight='bold'),
                yshift=-18
            )
        fig_lvc_dollars.update_layout(
            title=dict(text='💰 FAS $ by LVC Group', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']),
            margin=dict(l=60, r=40, t=60, b=50),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
            barmode='group',
            height=380,
            bargap=0.3,
        )
        
        # LVC comparison table
        lvc_table = lvc_target.merge(
            lvc_prev[['lvc_group', 'fas_count', 'fas_dollars', 'lp2c_prev']],
            on='lvc_group',
            suffixes=('', '_prev'),
            how='outer'
        ).fillna(0)
        lvc_table['Count Δ'] = lvc_table['fas_count'] - lvc_table['fas_count_prev']
        lvc_table['$ Δ'] = lvc_table['fas_dollars'] - lvc_table['fas_dollars_prev']
        lvc_table['lp2c_delta'] = lvc_table['lp2c'] - lvc_table['lp2c_prev']
        
        # Calculate % of total
        total_target = lvc_table['fas_count'].sum()
        total_prev = lvc_table['fas_count_prev'].sum()
        lvc_table['% Target'] = lvc_table['fas_count'] / total_target * 100 if total_target > 0 else 0
        lvc_table['% Prev'] = lvc_table['fas_count_prev'] / total_prev * 100 if total_prev > 0 else 0
        lvc_table['Share Δ'] = lvc_table['% Target'] - lvc_table['% Prev']
        
        lvc_row = dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_lvc_count)], style=CARD_STYLE), width=6),
            dbc.Col(html.Div([dcc.Graph(figure=fig_lvc_dollars)], style=CARD_STYLE), width=6),
            dbc.Col(html.Div([
                html.H5("LVC Group Comparison", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=lvc_table[['lvc_group', 'fas_count', 'fas_count_prev', 'Count Δ', 'lp2c', 'lp2c_delta', '% Target', 'Share Δ']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'LVC Group', 'id': 'lvc_group'},
                        {'name': 'This Wk', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': '4-Wk Avg', 'id': 'fas_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
                        {'name': 'Δ FAS', 'id': 'Count Δ', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                        {'name': 'LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ LP2C pp', 'id': 'lp2c_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                        {'name': 'Mix %', 'id': '% Target', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ Mix', 'id': 'Share Δ', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg']},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '8px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{Count Δ} > 0', 'column_id': 'Count Δ'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{Count Δ} < 0', 'column_id': 'Count Δ'}, 'color': COLORS['danger']},
                        {'if': {'filter_query': '{lp2c_delta} > 0', 'column_id': 'lp2c_delta'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{lp2c_delta} < 0', 'column_id': 'lp2c_delta'}, 'color': COLORS['danger']},
                        {'if': {'filter_query': '{Share Δ} > 0', 'column_id': 'Share Δ'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{Share Δ} < 0', 'column_id': 'Share Δ'}, 'color': COLORS['danger']},
                    ]
                )
            ], style=CARD_STYLE), width=12),
        ])
    
    # === Channel Analysis ===
    channel_row = html.Div("Loading channel analysis...")
    if not df_deep.empty:
        df_deep['fas_week'] = pd.to_datetime(df_deep['fas_week'])
        
        # Query to get Sent to Sales by channel (for FAS % calculation)
        target_end = target_date + timedelta(days=6)
        prev_start = target_date - timedelta(weeks=4)
        sts_query = f"""
        SELECT
            DATE_TRUNC(DATE(sent_to_sales_date), WEEK(SUNDAY)) as sts_week,
            COALESCE(sub_group, 'Unknown') as channel,
            COUNT(DISTINCT lendage_guid) as sts_count
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE sent_to_sales_date IS NOT NULL
        AND DATE(sent_to_sales_date) >= '{prev_start.strftime('%Y-%m-%d')}'
        AND DATE(sent_to_sales_date) <= '{target_end.strftime('%Y-%m-%d')}'
        GROUP BY 1, 2
        """
        df_sts_channel = run_query(sts_query)
        
        # Get target week channel data with FAS count, dollars, and LP2C
        channel_target = df_deep[df_deep['fas_week'] == target_date].groupby('channel').agg({
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        channel_target = channel_target.rename(columns={'avg_lp2c': 'lp2c'})
        
        # Get prev 4 weeks channel data
        channel_prev = df_deep[(df_deep['fas_week'] < target_date) & 
                              (df_deep['fas_week'] >= target_date - timedelta(weeks=4))].groupby('channel').agg({
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        channel_prev['fas_count'] = channel_prev['fas_count'] / 4
        channel_prev['fas_dollars'] = channel_prev['fas_dollars'] / 4
        channel_prev = channel_prev.rename(columns={'avg_lp2c': 'lp2c_prev'})
        
        # Get STS data by channel
        if not df_sts_channel.empty:
            df_sts_channel['sts_week'] = pd.to_datetime(df_sts_channel['sts_week'])
            sts_target = df_sts_channel[df_sts_channel['sts_week'] == target_date].groupby('channel')['sts_count'].sum().reset_index()
            sts_prev = df_sts_channel[(df_sts_channel['sts_week'] < target_date) & 
                                      (df_sts_channel['sts_week'] >= target_date - timedelta(weeks=4))].groupby('channel')['sts_count'].sum().reset_index()
            sts_prev['sts_count'] = sts_prev['sts_count'] / 4
        else:
            sts_target = pd.DataFrame(columns=['channel', 'sts_count'])
            sts_prev = pd.DataFrame(columns=['channel', 'sts_count'])
        
        # Merge target and prev FAS data
        channel_merged = channel_target.merge(channel_prev, on='channel', suffixes=('', '_prev'), how='outer').fillna(0)
        
        # Merge STS data
        channel_merged = channel_merged.merge(sts_target, on='channel', how='left').fillna(0)
        channel_merged = channel_merged.rename(columns={'sts_count': 'sts'})
        channel_merged = channel_merged.merge(sts_prev, on='channel', how='left').fillna(0)
        channel_merged = channel_merged.rename(columns={'sts_count': 'sts_prev'})
        
        # Calculate FAS % (FAS # / STS #)
        channel_merged['fas_pct'] = (channel_merged['fas_count'] / channel_merged['sts'].replace(0, 1) * 100).round(1)
        channel_merged['fas_pct_prev'] = (channel_merged['fas_count_prev'] / channel_merged['sts_prev'].replace(0, 1) * 100).round(1)
        channel_merged['fas_pct_delta'] = channel_merged['fas_pct'] - channel_merged['fas_pct_prev']
        
        channel_merged['fas_delta'] = channel_merged['fas_count'] - channel_merged['fas_count_prev']
        channel_merged['dollar_delta'] = channel_merged['fas_dollars'] - channel_merged['fas_dollars_prev']
        channel_merged['sts_delta'] = channel_merged['sts'] - channel_merged['sts_prev']
        channel_merged['lp2c_delta'] = channel_merged['lp2c'] - channel_merged['lp2c_prev']
        
        # Format $FAS for display (400K, 1.5M, etc.)
        def format_dollars(val):
            if val >= 1000000:
                return f"${val/1000000:.1f}M"
            elif val >= 1000:
                return f"${val/1000:.0f}K"
            else:
                return f"${val:.0f}"
        
        channel_merged['fas_dollars_fmt'] = channel_merged['fas_dollars'].apply(format_dollars)
        
        # Keep all data for worst performers, then get top 15
        channel_all = channel_merged.copy()
        channel_merged = channel_merged.sort_values('fas_count', ascending=False).head(15)
        
        # Get worst performers (biggest negative delta, min 5 FAS to be meaningful)
        channel_worst = channel_all[channel_all['fas_count'] >= 5].nsmallest(10, 'fas_delta')
        
        # Create horizontal bar chart with $FAS labels
        fig_channel = go.Figure()
        fig_channel.add_trace(go.Bar(
            y=channel_merged['channel'],
            x=channel_merged['fas_count'],
            orientation='h',
            marker_color=COLORS['chart_cyan'],
            text=[f"{fas:,.0f} ({dollars})" for fas, dollars in zip(channel_merged['fas_count'], channel_merged['fas_dollars_fmt'])],
            textposition='outside',
            textfont=dict(size=10, color=COLORS['text'])
        ))
        fig_channel.update_layout(
            title=dict(text='📢 Top 15 Channels by FAS Count', font=dict(color=COLORS['text'], size=14)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=150, r=100, t=50, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=True, range=[0, channel_merged['fas_count'].max() * 1.25]),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=False, autorange='reversed'),
            height=400,
            showlegend=False
        )
        
        channel_row = dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_channel)], style=CARD_STYLE), width=5),
            dbc.Col(html.Div([
                html.H5("Channel Comparison (This Week vs 4-Wk Avg)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=channel_merged[['channel', 'sts', 'fas_count', 'fas_pct', 'fas_dollars', 'fas_delta', 'lp2c', 'lp2c_delta']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'Channel', 'id': 'channel'},
                        {'name': 'StS #', 'id': 'sts', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS #', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS %', 'id': 'fas_pct', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': '$FAS', 'id': 'fas_dollars', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
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
            ], style=CARD_STYLE), width=4),
            dbc.Col(html.Div([
                html.H5("⚠️ Worst Performing Channels (vs 4-Wk Avg)", style={'color': COLORS['danger'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=channel_worst[['channel', 'fas_count', 'fas_count_prev', 'fas_delta', 'lp2c', 'lp2c_delta']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'Channel', 'id': 'channel'},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Prev', 'id': 'fas_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
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
            ], style=CARD_STYLE), width=3),
        ])
    
    # === Persona Analysis ===
    persona_row = html.Div("Loading persona analysis...")
    if not df_deep.empty:
        # Query to get Sent to Sales by persona (for FAS % calculation)
        sts_persona_query = f"""
        SELECT
            DATE_TRUNC(DATE(sent_to_sales_date), WEEK(SUNDAY)) as sts_week,
            persona,
            COUNT(DISTINCT lendage_guid) as sts_count
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE sent_to_sales_date IS NOT NULL
        AND DATE(sent_to_sales_date) >= '{prev_start.strftime('%Y-%m-%d')}'
        AND DATE(sent_to_sales_date) <= '{target_end.strftime('%Y-%m-%d')}'
        GROUP BY 1, 2
        """
        df_sts_persona = run_query(sts_persona_query)
        
        # Get target week persona data with LP2C
        persona_target = df_deep[df_deep['fas_week'] == target_date].groupby('persona').agg({
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        persona_target = persona_target.rename(columns={'avg_lp2c': 'lp2c'})
        
        # Get StS for target week
        if not df_sts_persona.empty:
            df_sts_persona['sts_week'] = pd.to_datetime(df_sts_persona['sts_week'])
            sts_target_persona = df_sts_persona[df_sts_persona['sts_week'] == target_date].groupby('persona')['sts_count'].sum().reset_index()
            persona_target = persona_target.merge(sts_target_persona, on='persona', how='left').fillna(0)
        else:
            persona_target['sts_count'] = 0
        
        # Get prev 4 weeks persona data with LP2C
        persona_prev = df_deep[(df_deep['fas_week'] < target_date) & 
                              (df_deep['fas_week'] >= target_date - timedelta(weeks=4))].groupby('persona').agg({
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        persona_prev['fas_count'] = persona_prev['fas_count'] / 4
        persona_prev['fas_dollars'] = persona_prev['fas_dollars'] / 4
        persona_prev = persona_prev.rename(columns={'avg_lp2c': 'lp2c_prev'})
        
        # Merge target and prev
        persona_merged = persona_target.merge(persona_prev, on='persona', suffixes=('', '_prev'), how='outer').fillna(0)
        persona_merged['fas_delta'] = persona_merged['fas_count'] - persona_merged['fas_count_prev']
        persona_merged['dollar_delta'] = persona_merged['fas_dollars'] - persona_merged['fas_dollars_prev']
        persona_merged['fas_pct'] = (persona_merged['fas_count'] / persona_merged['sts_count'].replace(0, 1) * 100).round(1)
        persona_merged['lp2c_delta'] = persona_merged['lp2c'] - persona_merged['lp2c_prev']
        
        # Calculate mix %
        total_fas = persona_merged['fas_count'].sum()
        total_fas_prev = persona_merged['fas_count_prev'].sum()
        persona_merged['mix_pct'] = (persona_merged['fas_count'] / total_fas * 100) if total_fas > 0 else 0
        persona_merged['mix_pct_prev'] = (persona_merged['fas_count_prev'] / total_fas_prev * 100) if total_fas_prev > 0 else 0
        persona_merged['mix_delta'] = persona_merged['mix_pct'] - persona_merged['mix_pct_prev']
        
        # Keep all for worst performers
        persona_all = persona_merged.copy()
        persona_merged = persona_merged.sort_values('fas_count', ascending=False)
        
        # Get worst performers (biggest negative delta)
        persona_worst = persona_all[persona_all['fas_count'] >= 3].nsmallest(5, 'fas_delta')
        
        # Format $FAS for display
        def format_dollars_persona(val):
            if val >= 1000000:
                return f"${val/1000000:.1f}M"
            elif val >= 1000:
                return f"${val/1000:.0f}K"
            else:
                return f"${val:.0f}"
        
        persona_merged['fas_dollars_fmt'] = persona_merged['fas_dollars'].apply(format_dollars_persona)
        
        # Create horizontal bar chart
        fig_persona = go.Figure()
        fig_persona.add_trace(go.Bar(
            y=persona_merged['persona'],
            x=persona_merged['fas_count'],
            orientation='h',
            marker_color=COLORS['chart_purple'],
            text=[f"{fas:,.0f} ({dollars})" for fas, dollars in zip(persona_merged['fas_count'], persona_merged['fas_dollars_fmt'])],
            textposition='outside',
            textfont=dict(size=10, color=COLORS['text'])
        ))
        fig_persona.update_layout(
            title=dict(text='👤 Persona by FAS Count (In-Period)', font=dict(color=COLORS['text'], size=14)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=120, r=80, t=50, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=True, range=[0, persona_merged['fas_count'].max() * 1.3]),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=False, autorange='reversed'),
            height=350, showlegend=False
        )
        
        # LVC Group x Persona composition
        lvc_persona = df_deep[df_deep['fas_week'] == target_date].groupby(['lvc_group', 'persona'])['fas_count'].sum().reset_index()
        lvc_persona_pivot = lvc_persona.pivot(index='persona', columns='lvc_group', values='fas_count').fillna(0)
        
        # Create stacked bar chart for LVC x Persona
        fig_lvc_persona = go.Figure()
        lvc_colors = {'LVC 1-2': COLORS['chart_cyan'], 'LVC 3-8': COLORS['chart_green'], 
                      'LVC 9-10': COLORS['chart_yellow'], 'PHX Transfer': COLORS['chart_purple'], 'Other': COLORS['chart_gray']}
        
        for lvc_group in lvc_persona_pivot.columns:
            fig_lvc_persona.add_trace(go.Bar(
                y=lvc_persona_pivot.index,
                x=lvc_persona_pivot[lvc_group],
                name=lvc_group,
                orientation='h',
                marker_color=lvc_colors.get(lvc_group, COLORS['chart_gray'])
            ))
        
        fig_lvc_persona.update_layout(
            title=dict(text='📊 LVC Composition by Persona (In-Period)', font=dict(color=COLORS['text'], size=14), y=0.98),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=120, r=30, t=80, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=False, autorange='reversed'),
            barmode='stack',
            legend=dict(orientation='h', yanchor='top', y=1.15, xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
            height=380
        )
        
        persona_row = dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_persona)], style=CARD_STYLE), width=4),
            dbc.Col(html.Div([
                html.H5("Persona Mix (In-Period - This Week vs 4-Wk Avg)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=persona_merged[['persona', 'fas_count', 'fas_delta', 'lp2c', 'lp2c_delta', 'mix_pct', 'mix_delta']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'Persona', 'id': 'persona'},
                        {'name': 'FAS #', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Δ FAS', 'id': 'fas_delta', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                        {'name': 'LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ LP2C pp', 'id': 'lp2c_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                        {'name': 'Mix %', 'id': 'mix_pct', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ Mix', 'id': 'mix_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg'], 'overflowX': 'auto'},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '6px', 'fontSize': '11px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{fas_delta} > 0', 'column_id': 'fas_delta'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{fas_delta} < 0', 'column_id': 'fas_delta'}, 'color': COLORS['danger']},
                        {'if': {'filter_query': '{lp2c_delta} > 0', 'column_id': 'lp2c_delta'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{lp2c_delta} < 0', 'column_id': 'lp2c_delta'}, 'color': COLORS['danger']},
                        {'if': {'filter_query': '{mix_delta} > 0', 'column_id': 'mix_delta'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{mix_delta} < 0', 'column_id': 'mix_delta'}, 'color': COLORS['danger']},
                    ],
                    page_size=10
                )
            ], style=CARD_STYLE), width=3),
            dbc.Col(html.Div([
                html.H5("⚠️ Worst Performing Personas", style={'color': COLORS['danger'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=persona_worst[['persona', 'fas_count', 'fas_count_prev', 'fas_delta', 'lp2c', 'lp2c_delta']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'Persona', 'id': 'persona'},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Prev', 'id': 'fas_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
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
                    page_size=5
                )
            ], style=CARD_STYLE), width=2),
            dbc.Col(html.Div([dcc.Graph(figure=fig_lvc_persona)], style=CARD_STYLE), width=3),
        ])
    
    # === Outreach Metrics ===
    outreach_row = html.Div("Loading outreach metrics...")
    df_outreach = get_outreach_metrics(target_week)
    if not df_outreach.empty:
        df_outreach['week_start'] = pd.to_datetime(df_outreach['week_start'])
        
        # Weekly trends aggregation
        weekly_outreach = df_outreach.groupby('week_start').agg({
            'total_leads': 'sum',
            'sent_to_sales': 'sum',
            'fas_count': 'sum',
            'total_sms': 'sum',
            'total_sms_before_contact': 'sum',
            'total_call_attempts': 'sum'
        }).reset_index()
        
        weekly_outreach['sms_per_lead'] = weekly_outreach['total_sms'] / weekly_outreach['sent_to_sales'].replace(0, 1)
        weekly_outreach['sms_pre_contact_per_lead'] = weekly_outreach['total_sms_before_contact'] / weekly_outreach['sent_to_sales'].replace(0, 1)
        weekly_outreach['calls_per_lead'] = weekly_outreach['total_call_attempts'] / weekly_outreach['sent_to_sales'].replace(0, 1)
        weekly_outreach['fas_rate'] = weekly_outreach['fas_count'] / weekly_outreach['sent_to_sales'].replace(0, 1) * 100
        weekly_outreach['week_label'] = weekly_outreach['week_start'].dt.strftime('%-m/%d')
        
        # Target week vs prev stats
        target_outreach = weekly_outreach[weekly_outreach['week_start'] == target_date]
        prev_outreach = weekly_outreach[(weekly_outreach['week_start'] < target_date) & 
                                        (weekly_outreach['week_start'] >= target_date - timedelta(weeks=4))]
        
        if len(target_outreach) > 0 and len(prev_outreach) > 0:
            target_stats = target_outreach.iloc[0]
            prev_avg = prev_outreach[['sms_per_lead', 'sms_pre_contact_per_lead', 'calls_per_lead']].mean()
            
            sms_delta = target_stats['sms_per_lead'] - prev_avg['sms_per_lead']
            sms_pre_delta = target_stats['sms_pre_contact_per_lead'] - prev_avg['sms_pre_contact_per_lead']
            calls_delta = target_stats['calls_per_lead'] - prev_avg['calls_per_lead']
        else:
            sms_delta = sms_pre_delta = calls_delta = 0
            target_stats = {'sms_per_lead': 0, 'sms_pre_contact_per_lead': 0, 'calls_per_lead': 0}
        
        # Create trend charts
        fig_sms_trend = go.Figure()
        fig_sms_trend.add_trace(go.Bar(
            x=weekly_outreach['week_label'],
            y=weekly_outreach['sms_per_lead'],
            name='SMS/Lead',
            marker_color=COLORS['chart_cyan'],
            text=weekly_outreach['sms_per_lead'].round(2),
            textposition='outside'
        ))
        fig_sms_trend.update_layout(
            title=dict(text='📱 SMS per Lead by Week', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=50, r=30, t=50, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True, range=[0, weekly_outreach['sms_per_lead'].max() * 1.15]),
            height=280,
            showlegend=False
        )
        
        fig_sms_pre_trend = go.Figure()
        fig_sms_pre_trend.add_trace(go.Bar(
            x=weekly_outreach['week_label'],
            y=weekly_outreach['sms_pre_contact_per_lead'],
            name='SMS Pre-Contact/Lead',
            marker_color=COLORS['chart_green'],
            text=weekly_outreach['sms_pre_contact_per_lead'].round(2),
            textposition='outside'
        ))
        fig_sms_pre_trend.update_layout(
            title=dict(text='📱 SMS Before Contact per Lead', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=50, r=30, t=50, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True, range=[0, weekly_outreach['sms_pre_contact_per_lead'].max() * 1.15] if weekly_outreach['sms_pre_contact_per_lead'].max() > 0 else [0, 1]),
            height=280,
            showlegend=False
        )
        
        fig_calls_trend = go.Figure()
        fig_calls_trend.add_trace(go.Bar(
            x=weekly_outreach['week_label'],
            y=weekly_outreach['calls_per_lead'],
            name='Calls/Lead',
            marker_color=COLORS['chart_yellow'],
            text=weekly_outreach['calls_per_lead'].round(2),
            textposition='outside'
        ))
        fig_calls_trend.update_layout(
            title=dict(text='📞 Call Attempts per Lead', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=50, r=30, t=50, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True, range=[0, weekly_outreach['calls_per_lead'].max() * 1.15]),
            height=280,
            showlegend=False
        )
        
        # Top MAs by outreach metrics
        target_ma_outreach = df_outreach[df_outreach['week_start'] == target_date].copy()
        if not target_ma_outreach.empty:
            # Recalculate per-lead metrics at MA level
            target_ma_outreach['sms_per_lead'] = target_ma_outreach['total_sms'] / target_ma_outreach['sent_to_sales'].replace(0, 1)
            target_ma_outreach['sms_pre_contact_per_lead'] = target_ma_outreach['total_sms_before_contact'] / target_ma_outreach['sent_to_sales'].replace(0, 1)
            target_ma_outreach['calls_per_lead'] = target_ma_outreach['total_call_attempts'] / target_ma_outreach['sent_to_sales'].replace(0, 1)
            target_ma_outreach['fas_rate'] = target_ma_outreach['fas_count'] / target_ma_outreach['sent_to_sales'].replace(0, 1) * 100
            
            # Filter to MAs with meaningful volume
            ma_with_volume = target_ma_outreach[target_ma_outreach['sent_to_sales'] >= 10].copy()
            
            # Top MAs by calls per lead
            top_callers = ma_with_volume.nlargest(10, 'calls_per_lead')[['mortgage_advisor', 'sent_to_sales', 'calls_per_lead', 'sms_per_lead', 'fas_count', 'fas_rate']]
            
            # Top MAs by SMS per lead  
            top_sms = ma_with_volume.nlargest(10, 'sms_per_lead')[['mortgage_advisor', 'sent_to_sales', 'sms_per_lead', 'sms_pre_contact_per_lead', 'fas_count', 'fas_rate']]
        else:
            top_callers = pd.DataFrame()
            top_sms = pd.DataFrame()
        
        outreach_calc = "Calc: SMS/Lead = SUM(total_sms_outbound_count)/sent_to_sales; SMS Pre-Contact/Lead = SUM(total_sms_outbound_before_contact)/sent_to_sales; Calls/Lead = SUM(call_attempts)/sent_to_sales; by week (lead_created_date), by MA."
        outreach_row = dbc.Row([
            dbc.Col(html.Div(outreach_calc, style={'fontSize': '11px', 'color': COLORS['text_muted'], 'marginBottom': '8px'}), width=12),
            # KPI cards
            dbc.Col(create_metric_card("SMS/Lead", f"{target_stats['sms_per_lead']:.2f}", sms_delta, " vs 4wk avg"), width=4),
            dbc.Col(create_metric_card("SMS Pre-Contact/Lead", f"{target_stats['sms_pre_contact_per_lead']:.2f}", sms_pre_delta, " vs 4wk avg"), width=4),
            dbc.Col(create_metric_card("Calls/Lead", f"{target_stats['calls_per_lead']:.2f}", calls_delta, " vs 4wk avg"), width=4),
            
            # Trend charts
            dbc.Col(html.Div([dcc.Graph(figure=fig_sms_trend)], style=CARD_STYLE), width=4),
            dbc.Col(html.Div([dcc.Graph(figure=fig_sms_pre_trend)], style=CARD_STYLE), width=4),
            dbc.Col(html.Div([dcc.Graph(figure=fig_calls_trend)], style=CARD_STYLE), width=4),
            
            # Top MAs tables
            dbc.Col(html.Div([
                html.H5("Top MAs by Calls/Lead (min 10 StS)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=top_callers.round(2).to_dict('records') if not top_callers.empty else [],
                    columns=[
                        {'name': 'MA', 'id': 'mortgage_advisor'},
                        {'name': 'StS', 'id': 'sent_to_sales', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Calls/Lead', 'id': 'calls_per_lead', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'SMS/Lead', 'id': 'sms_per_lead', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS %', 'id': 'fas_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg']},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '6px', 'fontSize': '12px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '11px'},
                    page_size=10
                )
            ], style=CARD_STYLE), width=6),
            dbc.Col(html.Div([
                html.H5("Top MAs by SMS/Lead (min 10 StS)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=top_sms.round(2).to_dict('records') if not top_sms.empty else [],
                    columns=[
                        {'name': 'MA', 'id': 'mortgage_advisor'},
                        {'name': 'StS', 'id': 'sent_to_sales', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'SMS/Lead', 'id': 'sms_per_lead', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'SMS Pre-Cntct', 'id': 'sms_pre_contact_per_lead', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS %', 'id': 'fas_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg']},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '6px', 'fontSize': '12px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '11px'},
                    page_size=10
                )
            ], style=CARD_STYLE), width=6),
        ])
    
    # === MA Analysis ===
    ma_row = html.Div("Loading MA analysis...")
    if not df_ma.empty:
        df_ma['fas_delta'] = df_ma['fas_count'] - df_ma['fas_count_prev']
        df_ma['dollar_delta'] = df_ma['fas_dollars'] - df_ma['fas_dollars_prev']
        df_ma['contact_rate'] = df_ma['contacted'] / df_ma['assigned'].replace(0, 1) * 100
        df_ma['fas_rate'] = df_ma['fas_count'] / df_ma['assigned'].replace(0, 1) * 100
        df_ma['lp2c_delta'] = df_ma['lp2c'] - df_ma['lp2c_prev']
        
        top_mas = df_ma.head(15).copy()
        top_gainers = df_ma.nlargest(10, 'fas_delta')
        top_decliners = df_ma[df_ma['fas_count_prev'] > 0].nsmallest(10, 'fas_delta')
        
        ma_summary_stats = dbc.Row([
            dbc.Col(create_metric_card("Total FAS", f"{df_ma['fas_count'].sum():,.0f}"), width=3),
            dbc.Col(create_metric_card("Total FAS $", f"${df_ma['fas_dollars'].sum()/1000000:.2f}M"), width=3),
            dbc.Col(create_metric_card("MAs with FAS", f"{len(df_ma[df_ma['fas_count'] > 0])}"), width=3),
            dbc.Col(create_metric_card("Top 10 MA %", f"{df_ma.head(10)['fas_dollars'].sum()/df_ma['fas_dollars'].sum()*100:.0f}%" if df_ma['fas_dollars'].sum() > 0 else "0%"), width=3),
        ])
        
        ma_row = dbc.Row([
            dbc.Col(ma_summary_stats, width=12),
            dbc.Col(html.Div([
                html.H5("Top 15 MAs by FAS $", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=top_mas[['mortgage_advisor', 'assigned', 'contacted', 'fas_count', 'fas_rate', 'fas_delta', 'lp2c', 'lp2c_delta', 'fas_dollars']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'MA', 'id': 'mortgage_advisor'},
                        {'name': 'Assigned', 'id': 'assigned', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Contacted', 'id': 'contacted', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS %', 'id': 'fas_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ FAS', 'id': 'fas_delta', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                        {'name': 'LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ LP2C pp', 'id': 'lp2c_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                        {'name': 'FAS $', 'id': 'fas_dollars', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg'], 'overflowX': 'auto'},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '6px 10px', 'fontSize': '12px'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'mortgage_advisor'}, 'textAlign': 'left', 'minWidth': '120px', 'maxWidth': '160px'},
                        {'if': {'column_id': 'assigned'}, 'minWidth': '60px'},
                        {'if': {'column_id': 'contacted'}, 'minWidth': '70px'},
                        {'if': {'column_id': 'fas_count'}, 'minWidth': '45px'},
                        {'if': {'column_id': 'fas_rate'}, 'minWidth': '50px'},
                        {'if': {'column_id': 'fas_delta'}, 'minWidth': '55px'},
                        {'if': {'column_id': 'lp2c'}, 'minWidth': '50px'},
                        {'if': {'column_id': 'lp2c_delta'}, 'minWidth': '60px'},
                        {'if': {'column_id': 'fas_dollars'}, 'minWidth': '80px'},
                    ],
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '11px', 'textAlign': 'center'},
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
                html.H5("📈 Biggest Gainers vs Prev", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=top_gainers[['mortgage_advisor', 'fas_count', 'fas_rate', 'fas_count_prev', 'fas_delta', 'lp2c']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'MA', 'id': 'mortgage_advisor'},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS %', 'id': 'fas_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Prev', 'id': 'fas_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
                        {'name': 'Δ', 'id': 'fas_delta', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                        {'name': 'LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg']},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '4px 6px', 'fontSize': '11px'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'mortgage_advisor'}, 'textAlign': 'left', 'minWidth': '100px'},
                    ],
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px', 'textAlign': 'center'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{fas_delta} > 0', 'column_id': 'fas_delta'}, 'color': COLORS['success']},
                    ],
                    page_size=10
                )
            ], style=CARD_STYLE), width=2),
            dbc.Col(html.Div([
                html.H5("📉 Biggest Decliners vs Prev", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=top_decliners[['mortgage_advisor', 'fas_count', 'fas_rate', 'fas_count_prev', 'fas_delta', 'lp2c']].round(1).to_dict('records') if not top_decliners.empty else [],
                    columns=[
                        {'name': 'MA', 'id': 'mortgage_advisor'},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS %', 'id': 'fas_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Prev', 'id': 'fas_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
                        {'name': 'Δ', 'id': 'fas_delta', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                        {'name': 'LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg']},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '4px 6px', 'fontSize': '11px'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'mortgage_advisor'}, 'textAlign': 'left', 'minWidth': '100px'},
                    ],
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px', 'textAlign': 'center'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{fas_delta} < 0', 'column_id': 'fas_delta'}, 'color': COLORS['danger']},
                    ],
                    page_size=10
                )
            ], style=CARD_STYLE), width=2),
        ])
    
    # === Starring Analysis (In-Period) ===
    starring_row = html.Div([
        html.H5("⭐ Starring Distribution (In-Period)", style={'color': COLORS['text'], 'marginBottom': '12px', 'marginTop': '20px', 'fontSize': '16px', 'fontWeight': '600'}),
        html.P("Loading starring data...", style={'color': COLORS['text_muted'], 'fontSize': '11px'})
    ])
    try:
        df_starring = get_starring_data(target_week, 'full_app_submit_datetime')
        print(f"[DEBUG] In-Period Starring data rows: {len(df_starring) if not df_starring.empty else 0}")
    except Exception as e:
        print(f"[DEBUG] Starring query error: {e}")
        df_starring = pd.DataFrame()
    
    if not df_starring.empty:
        try:
            print(f"[DEBUG] Starring columns: {df_starring.columns.tolist()}")
            print(f"[DEBUG] Starring sample: {df_starring.head(3).to_dict('records')}")
            df_starring['week_start'] = pd.to_datetime(df_starring['week_start'])
            df_starring['starring'] = df_starring['starring'].astype(str).replace('nan', 'Unknown').replace('<NA>', 'Unknown')
            # Group starring 0 and Unknown into "0"
            df_starring['starring'] = df_starring['starring'].replace(['Unknown', '0.0'], '0')
            
            # Aggregate by starring for target and prev
            starring_target = df_starring[df_starring['week_start'] == target_date].groupby('starring').agg({
                'sts_count': 'sum', 'fas_count': 'sum', 'fas_dollars': 'sum', 'avg_lp2c': 'mean'
            }).reset_index()
            
            starring_prev = df_starring[(df_starring['week_start'] < target_date) & 
                                        (df_starring['week_start'] >= target_date - timedelta(weeks=4))].groupby('starring').agg({
                'sts_count': 'sum', 'fas_count': 'sum', 'fas_dollars': 'sum', 'avg_lp2c': 'mean'
            }).reset_index()
            starring_prev[['sts_count', 'fas_count', 'fas_dollars']] /= 4
            
            # Merge and calculate
            starring_merged = starring_target.merge(starring_prev, on='starring', suffixes=('', '_prev'), how='outer').fillna(0)
            starring_merged['fas_rate'] = starring_merged['fas_count'] / starring_merged['sts_count'].replace(0, 1) * 100
            starring_merged['fas_rate_prev'] = starring_merged['fas_count_prev'] / starring_merged['sts_count_prev'].replace(0, 1) * 100
            starring_merged['fas_delta'] = starring_merged['fas_count'] - starring_merged['fas_count_prev']
            starring_merged['rate_delta'] = starring_merged['fas_rate'] - starring_merged['fas_rate_prev']
            starring_merged['lp2c_delta'] = starring_merged['avg_lp2c'] - starring_merged['avg_lp2c_prev']
            
            # Sort by starring (5, 3, 1, 0)
            star_order = {'1': 1, '3': 2, '5': 3, '0': 4, '1.0': 1, '3.0': 2, '5.0': 3, '0.0': 4}
            starring_merged['sort_order'] = starring_merged['starring'].astype(str).map(lambda x: star_order.get(x, 5))
            starring_merged = starring_merged.sort_values('sort_order')
            
            # LVC x Starring breakdown for target week
            lvc_starring = df_starring[df_starring['week_start'] == target_date].groupby(['lvc_group', 'starring']).agg({
                'fas_count': 'sum', 'sts_count': 'sum', 'fas_dollars': 'sum'
            }).reset_index()
            lvc_starring['fas_rate'] = lvc_starring['fas_count'] / lvc_starring['sts_count'].replace(0, 1) * 100
            
            # Create starring distribution chart
            fig_starring = go.Figure()
            fig_starring.add_trace(go.Bar(
                x=starring_merged['starring'].astype(str),
                y=starring_merged['fas_count'],
                name='FAS Count',
                marker_color=COLORS['chart_cyan'],
                text=starring_merged['fas_count'].round(0).astype(int),
                textposition='outside'
            ))
            fig_starring.update_layout(
                title=dict(text='⭐ FAS by Starring', font=dict(color=COLORS['text'], size=14)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor=COLORS['card_bg'],
                font=dict(color=COLORS['text']),
                xaxis=dict(title='Star Rating', gridcolor=COLORS['border']),
                yaxis=dict(title='FAS Count', gridcolor=COLORS['border'], range=[0, starring_merged['fas_count'].max() * 1.15] if starring_merged['fas_count'].max() > 0 else [0, 1]),
                height=280,
                margin=dict(l=50, r=30, t=50, b=40),
                showlegend=False
            )
            
            # Create LVC x Starring stacked bar
            lvc_order = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']
            fig_lvc_starring = go.Figure()
            for lvc in lvc_order:
                lvc_data = lvc_starring[lvc_starring['lvc_group'] == lvc]
                if not lvc_data.empty:
                    fig_lvc_starring.add_trace(go.Bar(
                        x=lvc_data['starring'].astype(str),
                        y=lvc_data['fas_count'],
                        name=lvc,
                        text=lvc_data['fas_count'].round(0).astype(int),
                        textposition='inside'
                    ))
            fig_lvc_starring.update_layout(
                title=dict(text='⭐ LVC Group × Starring Distribution', font=dict(color=COLORS['text'], size=14)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor=COLORS['card_bg'],
                font=dict(color=COLORS['text']),
                xaxis=dict(title='Star Rating', gridcolor=COLORS['border']),
                yaxis=dict(title='FAS Count', gridcolor=COLORS['border']),
                barmode='stack',
                height=280,
                margin=dict(l=50, r=30, t=50, b=40),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=10))
            )
            
            # Create LVC Group × Starring weekly distribution tables
            all_weeks = sorted(df_starring['week_start'].unique())
            target_week_dt = all_weeks[-1] if all_weeks else None
            prev_6_weeks = all_weeks[:-1][-6:] if len(all_weeks) > 1 else []
            weeks_for_table = sorted(all_weeks[-6:], reverse=True)
            week_labels = [w.strftime('%-m/%d') for w in weeks_for_table]
            
            lvc_order = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']
            star_order = ['5', '3', '1', '0']
            
            dist_rows = []
            fas_day7_rows = []
            contact_rows = []
            fas_dollars_rows = []
            
            for lvc in lvc_order:
                lvc_data = df_starring[df_starring['lvc_group'] == lvc]
                if lvc_data.empty:
                    continue
                lvc_stars = [s for s in star_order if s in lvc_data['starring'].unique()]
                
                for star in lvc_stars:
                    d_row = {'lvc_group': lvc, 'starring': star}
                    f7_row = {'lvc_group': lvc, 'starring': star}
                    c_row = {'lvc_group': lvc, 'starring': star}
                    fas_d_row = {'lvc_group': lvc, 'starring': star}
                    
                    # Calculate 6-week averages and deltas
                    prev_data = lvc_data[(lvc_data['starring'] == star) & (lvc_data['week_start'].isin(prev_6_weeks))]
                    target_data = lvc_data[(lvc_data['starring'] == star) & (lvc_data['week_start'] == target_week_dt)]
                    
                    # Assigned Leads
                    avg_leads = prev_data['lead_count'].sum() / len(prev_6_weeks) if prev_6_weeks else 0
                    t_leads = target_data['lead_count'].sum() if not target_data.empty else 0
                    d_row['prev_avg'] = f"{avg_leads:,.0f}"
                    d_row['delta'] = f"{t_leads - avg_leads:+,.0f}"
                    
                    # FAS Day 7 %
                    p_f7 = prev_data['fas_day7_qty'].sum()
                    p_sts = prev_data['sts_count'].sum()
                    avg_f7_pct = (p_f7 / p_sts * 100) if p_sts > 0 else 0
                    t_f7 = target_data['fas_day7_qty'].sum() if not target_data.empty else 0
                    t_sts = target_data['sts_count'].sum() if not target_data.empty else 0
                    t_f7_pct = (t_f7 / t_sts * 100) if t_sts > 0 else 0
                    f7_row['prev_avg'] = f"{avg_f7_pct:.1f}%"
                    f7_row['delta'] = f"{t_f7_pct - avg_f7_pct:+.1f}pp"
                    
                    # SF Contact %
                    p_contacted = prev_data['contacted_qty'].sum()
                    p_assigned = prev_data['lead_count'].sum()
                    avg_c_pct = (p_contacted / p_assigned * 100) if p_assigned > 0 else 0
                    t_contacted = target_data['contacted_qty'].sum() if not target_data.empty else 0
                    t_assigned = target_data['lead_count'].sum() if not target_data.empty else 0
                    t_c_pct = (t_contacted / t_assigned * 100) if t_assigned > 0 else 0
                    c_row['prev_avg'] = f"{avg_c_pct:.1f}%"
                    c_row['delta'] = f"{t_c_pct - avg_c_pct:+.1f}pp"
                    
                    # FAS $
                    p_fas_d = prev_data['fas_dollars'].sum()
                    avg_fas_d = p_fas_d / len(prev_6_weeks) if prev_6_weeks else 0
                    t_fas_d = target_data['fas_dollars'].sum() if not target_data.empty else 0
                    fas_d_row['prev_avg'] = f"${avg_fas_d:,.0f}"
                    fas_d_row['delta'] = f"{t_fas_d - avg_fas_d:+,.0f}" if t_fas_d - avg_fas_d < 0 else f"+{t_fas_d - avg_fas_d:+,.0f}"
                    # Quick fix to make sure + shows up cleanly:
                    fas_d_delta = t_fas_d - avg_fas_d
                    fas_d_row['delta'] = f"+${fas_d_delta:,.0f}" if fas_d_delta > 0 else (f"-${abs(fas_d_delta):,.0f}" if fas_d_delta < 0 else "$0")
                    
                    for i, week in enumerate(weeks_for_table):
                        week_data = lvc_data[(lvc_data['starring'] == star) & (lvc_data['week_start'] == week)]
                        
                        # Assigned Leads
                        count = int(week_data['lead_count'].sum()) if not week_data.empty else 0
                        lvc_week_total = lvc_data[lvc_data['week_start'] == week]['lead_count'].sum()
                        pct = (count / lvc_week_total * 100) if lvc_week_total > 0 else 0
                        d_row[f'wk_{i}'] = f"{count:,} ({pct:.0f}%)"
                        
                        # FAS Day 7 % (FAS Day 7 / Sent to Sales)
                        f7 = int(week_data['fas_day7_qty'].sum()) if not week_data.empty else 0
                        sts = int(week_data['sts_count'].sum()) if not week_data.empty else 0
                        f7_pct = (f7 / sts * 100) if sts > 0 else 0
                        f7_row[f'wk_{i}'] = f"{f7_pct:.1f}%"
                        
                        # SF Contact % (Contacted / Assigned)
                        contacted = int(week_data['contacted_qty'].sum()) if not week_data.empty else 0
                        c_pct = (contacted / count * 100) if count > 0 else 0
                        c_row[f'wk_{i}'] = f"{c_pct:.1f}%"
                        
                        # FAS $
                        fas_d = week_data['fas_dollars'].sum() if not week_data.empty else 0
                        fas_d_row[f'wk_{i}'] = f"${fas_d:,.0f}"
                        
                    dist_rows.append(d_row)
                    fas_day7_rows.append(f7_row)
                    contact_rows.append(c_row)
                    fas_dollars_rows.append(fas_d_row)
            
            dist_columns = [
                {'name': 'LVC', 'id': 'lvc_group'},
                {'name': '⭐', 'id': 'starring'},
                {'name': '6Wk Avg', 'id': 'prev_avg'},
                {'name': 'Δ vs Avg', 'id': 'delta'},
            ] + [{'name': label, 'id': f'wk_{i}'} for i, label in enumerate(week_labels)]
            
            # Star-only (no LVC) table: FAS % Day 7 by starring over weeks vs 6wk avg
            df_by_star = df_starring.groupby(['week_start', 'starring']).agg({
                'sts_count': 'sum', 'fas_day7_qty': 'sum'
            }).reset_index()
            df_by_star['fas_rate_d7'] = df_by_star['fas_day7_qty'] / df_by_star['sts_count'].replace(0, 1) * 100
            star_overall_rows = []
            for star in star_order:
                star_data = df_by_star[df_by_star['starring'] == star]
                prev_data = star_data[star_data['week_start'].isin(prev_6_weeks)]
                target_data = star_data[star_data['week_start'] == target_week_dt]
                avg_rate = prev_data['fas_rate_d7'].mean() if not prev_data.empty and len(prev_6_weeks) else 0
                t_rate = target_data['fas_rate_d7'].iloc[0] if not target_data.empty else 0
                row = {'starring': star, 'prev_avg': f"{avg_rate:.1f}%", 'delta': f"{t_rate - avg_rate:+.1f}pp"}
                for i, week in enumerate(weeks_for_table):
                    wd = star_data[star_data['week_start'] == week]
                    row[f'wk_{i}'] = f"{wd['fas_rate_d7'].iloc[0]:.1f}%" if not wd.empty and len(wd) > 0 else "—"
                star_overall_rows.append(row)
            # Star-only: Avg FAS $ (overall, not day 7) by starring over weeks vs 6wk avg
            df_by_star_avg = df_starring.groupby(['week_start', 'starring']).agg({
                'fas_dollars': 'sum', 'fas_count': 'sum'
            }).reset_index()
            df_by_star_avg['avg_fas'] = df_by_star_avg['fas_dollars'] / df_by_star_avg['fas_count'].replace(0, 1)
            star_avg_fas_rows = []
            for star in star_order:
                star_data = df_by_star_avg[df_by_star_avg['starring'] == star]
                prev_data = star_data[star_data['week_start'].isin(prev_6_weeks)]
                target_data = star_data[star_data['week_start'] == target_week_dt]
                avg_val = prev_data['avg_fas'].mean() if not prev_data.empty and len(prev_6_weeks) else 0
                t_val = target_data['avg_fas'].iloc[0] if not target_data.empty else 0
                d = t_val - avg_val
                row = {'starring': star, 'prev_avg': f"${avg_val:,.0f}", 'delta': f"+${d:,.0f}" if d >= 0 else f"-${abs(d):,.0f}"}
                for i, week in enumerate(weeks_for_table):
                    wd = star_data[star_data['week_start'] == week]
                    row[f'wk_{i}'] = f"${wd['avg_fas'].iloc[0]:,.0f}" if not wd.empty and len(wd) > 0 else "—"
                star_avg_fas_rows.append(row)
            star_overall_columns = [
                {'name': '⭐', 'id': 'starring'},
                {'name': '6Wk Avg', 'id': 'prev_avg'},
                {'name': 'Δ vs Avg', 'id': 'delta'},
            ] + [{'name': label, 'id': f'wk_{i}'} for i, label in enumerate(week_labels)]
            
            def create_compact_table(data_rows, title, subtitle):
                return html.Div([
                    html.H6(title, style={'color': COLORS['text'], 'marginTop': '16px', 'marginBottom': '4px', 'fontSize': '13px', 'fontWeight': '600'}),
                    html.P(subtitle, style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '8px', 'fontStyle': 'italic'}),
                    dash_table.DataTable(
                        data=data_rows,
                        columns=dist_columns,
                        style_table={'backgroundColor': COLORS['card_bg'], 'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'},
                        style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '3px 6px', 'fontSize': '10px', 'minWidth': '40px', 'maxWidth': '70px'},
                        style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px', 'textAlign': 'center'},
                        style_cell_conditional=[
                            {'if': {'column_id': 'lvc_group'}, 'textAlign': 'center', 'fontWeight': '600', 'minWidth': '50px', 'maxWidth': '60px'},
                            {'if': {'column_id': 'starring'}, 'textAlign': 'center', 'minWidth': '25px', 'maxWidth': '35px'},
                            {'if': {'column_id': 'prev_avg'}, 'backgroundColor': '#122640', 'fontWeight': '500'},
                            {'if': {'column_id': 'delta'}, 'backgroundColor': '#122640', 'fontWeight': '500'},
                        ],
                        style_data_conditional=[
                            {'if': {'filter_query': '{delta} contains "+"', 'column_id': 'delta'}, 'color': COLORS['success']},
                            {'if': {'filter_query': '{delta} contains "-"', 'column_id': 'delta'}, 'color': COLORS['danger']},
                        ],
                    )
                ])
            
            def create_star_overall_table(data_rows, title, subtitle):
                return html.Div([
                    html.H6(title, style={'color': COLORS['text'], 'marginTop': '0', 'marginBottom': '4px', 'fontSize': '13px', 'fontWeight': '600'}),
                    html.P(subtitle, style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '8px', 'fontStyle': 'italic'}),
                    dash_table.DataTable(
                        data=data_rows,
                        columns=star_overall_columns,
                        style_table={'backgroundColor': COLORS['card_bg'], 'overflowX': 'auto', 'maxHeight': '320px', 'overflowY': 'auto'},
                        style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '3px 6px', 'fontSize': '10px', 'minWidth': '40px', 'maxWidth': '70px'},
                        style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px', 'textAlign': 'center'},
                        style_cell_conditional=[
                            {'if': {'column_id': 'starring'}, 'textAlign': 'center', 'fontWeight': '600', 'minWidth': '45px', 'maxWidth': '55px'},
                            {'if': {'column_id': 'prev_avg'}, 'backgroundColor': '#122640', 'fontWeight': '500'},
                            {'if': {'column_id': 'delta'}, 'backgroundColor': '#122640', 'fontWeight': '500'},
                        ],
                        style_data_conditional=[
                            {'if': {'filter_query': '{delta} contains "+"', 'column_id': 'delta'}, 'color': COLORS['success']},
                            {'if': {'filter_query': '{delta} contains "-"', 'column_id': 'delta'}, 'color': COLORS['danger']},
                        ],
                    )
                ])
            
            # Export payload for download (xlsx): list of {name, columns: [{id, name}], rows}
            inperiod_export_data = [
                {"name": "FAS % Day 7 by Starring", "columns": star_overall_columns, "rows": star_overall_rows},
                {"name": "Avg FAS $ by Starring", "columns": star_overall_columns, "rows": star_avg_fas_rows},
                {"name": "Lead Distribution LVC x Starring", "columns": dist_columns, "rows": dist_rows},
                {"name": "FAS % Day 7 LVC x Starring", "columns": dist_columns, "rows": fas_day7_rows},
                {"name": "SF Contact % LVC x Starring", "columns": dist_columns, "rows": contact_rows},
                {"name": "FAS $ Day 7 LVC x Starring", "columns": dist_columns, "rows": fas_dollars_rows},
            ]
            download_btn = dbc.Button(
                "Download Starring Tables (xlsx)",
                id="inperiod-download-starring-btn",
                size="sm",
                style={
                    "backgroundColor": COLORS["primary"],
                    "border": "none",
                    "color": "#fff",
                    "marginTop": "8px",
                },
            )
            starring_row = html.Div([
                html.H5("⭐ Starring Distribution (In-Period)", style={'color': COLORS['text'], 'marginBottom': '12px', 'marginTop': '20px', 'fontSize': '16px', 'fontWeight': '600'}),
                html.P("5 Star = Best lead quality | Distribution based on FAS Submit Date", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '12px'}),
                dbc.Row([
                    dbc.Col(html.Div(create_star_overall_table(star_overall_rows, "📈 FAS % Day 7 by Starring (Last 6 Wks)", "FAS Day 7 / StS by star | vs 6wk avg"), style=CARD_STYLE), width=4),
                    dbc.Col(html.Div(create_star_overall_table(star_avg_fas_rows, "💰 Avg FAS $ by Starring (Last 6 Wks)", "Avg loan by star | vs 6wk avg"), style=CARD_STYLE), width=4),
                    dbc.Col(html.Div([dcc.Graph(figure=fig_starring)], style=CARD_STYLE), width=2),
                    dbc.Col(html.Div([dcc.Graph(figure=fig_lvc_starring)], style=CARD_STYLE), width=2),
                ]),
                dbc.Row([
                    dbc.Col(create_compact_table(dist_rows, "📊 Lead Distribution by LVC × Starring (Last 6 Wks)", "Based on Leads Assigned Count"), width=6),
                    dbc.Col(create_compact_table(fas_day7_rows, "🚀 FAS % Day 7 by LVC × Starring (Last 6 Wks)", "Based on FAS Day 7 / Sent to Sales"), width=6),
                ], style={'marginTop': '16px'}),
                dbc.Row([
                    dbc.Col(create_compact_table(contact_rows, "📞 SF Contact % by LVC × Starring (Last 6 Wks)", "Based on Contacted / Assigned Leads"), width=6),
                    dbc.Col(create_compact_table(fas_dollars_rows, "💰 Total FAS $ Day 7 by LVC × Starring (Last 6 Wks)", "Based on FAS Day 7 e_loan_amount"), width=6),
                ], style={'marginTop': '16px'}),
                dbc.Row([dbc.Col(download_btn, width=12)], style={'marginTop': '8px'}),
            ])
        except Exception as e:
            print(f"[DEBUG] Starring processing error: {e}")
            inperiod_export_data = None
            starring_row = html.Div([
                html.H5("⭐ Starring Distribution (In-Period)", style={'color': COLORS['text'], 'marginBottom': '12px', 'marginTop': '20px', 'fontSize': '16px', 'fontWeight': '600'}),
                html.P(f"Error processing starring data: {str(e)}", style={'color': COLORS['danger'], 'fontSize': '12px'})
            ])
        
    # Add starring to ma_row
    ma_row = html.Div([ma_row, starring_row])
    
    # === Vintage Deep Analysis ===
    vintage_deep_row = html.Div("Loading vintage analysis...")
    if not df_deep.empty:
        df_deep['fas_date'] = pd.to_datetime(df_deep['fas_date'])
        df_deep['vintage_date'] = pd.to_datetime(df_deep['vintage_date'])
        df_deep['fas_week'] = pd.to_datetime(df_deep['fas_week'])
        
        df_target = df_deep[df_deep['fas_week'] == target_date]
        df_prev = df_deep[(df_deep['fas_week'] < target_date) & 
                         (df_deep['fas_week'] >= target_date - timedelta(weeks=4))]
        
        # Calculate avg days to FAS
        avg_days_target = (df_target['days_to_fas'] * df_target['fas_count']).sum() / df_target['fas_count'].sum() if df_target['fas_count'].sum() > 0 else 0
        avg_days_prev = (df_prev['days_to_fas'] * df_prev['fas_count']).sum() / df_prev['fas_count'].sum() if df_prev['fas_count'].sum() > 0 else 0
        
        # Digital app start %
        digital_target = df_target[df_target['is_digital_start'] == 1]['fas_count'].sum()
        total_target = df_target['fas_count'].sum()
        digital_pct_target = digital_target / total_target * 100 if total_target > 0 else 0
        
        digital_prev = df_prev[df_prev['is_digital_start'] == 1]['fas_count'].sum()
        total_prev = df_prev['fas_count'].sum()
        digital_pct_prev = digital_prev / total_prev * 100 if total_prev > 0 else 0
        
        vintage_deep_row = dbc.Row([
            dbc.Col(create_metric_card("Avg Days to FAS (Target)", f"{avg_days_target:.1f} days", avg_days_target - avg_days_prev, " vs 4wk avg", positive_is_good=False), width=3),
            dbc.Col(create_metric_card("Avg Days to FAS (Prev)", f"{avg_days_prev:.1f} days"), width=3),
            dbc.Col(create_metric_card("Digital Start % (Target)", f"{digital_pct_target:.1f}%", digital_pct_target - digital_pct_prev, "pp vs 4wk avg"), width=3),
            dbc.Col(create_metric_card("Digital Start % (Prev)", f"{digital_pct_prev:.1f}%"), width=3),
        ])
    
    # === Daily Breakout ===
    daily_breakout_row = html.Div("Loading daily breakout...")
    if not df_deep.empty:
        daily_by_lvc = df_deep[df_deep['fas_week'] == target_date].groupby(['fas_date', 'lvc_group'])['fas_count'].sum().reset_index()
        daily_by_lvc['fas_date'] = pd.to_datetime(daily_by_lvc['fas_date'])
        daily_by_lvc['date_label'] = daily_by_lvc['fas_date'].dt.strftime('%a %m/%d')
        
        # Pivot for stacked chart
        daily_pivot = daily_by_lvc.pivot(index='date_label', columns='lvc_group', values='fas_count').fillna(0).reset_index()
        
        # Calculate % of daily total
        lvc_cols = [c for c in daily_pivot.columns if c != 'date_label']
        daily_pivot['total'] = daily_pivot[lvc_cols].sum(axis=1)
        for col in lvc_cols:
            daily_pivot[f'{col}_pct'] = daily_pivot[col] / daily_pivot['total'] * 100
        
        # Create stacked bar chart
        fig_daily_lvc = go.Figure()
        colors_lvc = [COLORS['chart_cyan'], COLORS['chart_green'], COLORS['chart_yellow'], COLORS['chart_purple'], COLORS['chart_gray']]
        
        for i, lvc in enumerate(['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']):
            if lvc in daily_pivot.columns:
                fig_daily_lvc.add_trace(go.Bar(
                    x=daily_pivot['date_label'],
                    y=daily_pivot[lvc],
                    name=lvc,
                    marker_color=colors_lvc[i]
                ))
        
        fig_daily_lvc.update_layout(
            title=dict(text='Daily FAS by LVC Group', font=dict(color=COLORS['text'], size=13, family="Segoe UI, Roboto, sans-serif")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=60, r=40, t=60, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            barmode='stack',
            height=350,
        )
        
        daily_breakout_row = dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_daily_lvc)], style=CARD_STYLE), width=8),
            dbc.Col(html.Div([
                html.H5("Daily Totals", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=daily_pivot[['date_label', 'total']].to_dict('records'),
                    columns=[
                        {'name': 'Day', 'id': 'date_label'},
                        {'name': 'Total FAS', 'id': 'total', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg']},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '8px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold'},
                )
            ], style=CARD_STYLE), width=4),
        ])
    
    return kpi_row, lvc_row, channel_row, persona_row, outreach_row, ma_row, inperiod_export_data


def _starring_export_to_xlsx(export_data, default_filename="starring_tables.xlsx"):
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
    return base64.b64encode(buffer.getvalue()).decode()


@callback(
    Output("inperiod-starring-download", "data"),
    Input("inperiod-download-starring-btn", "n_clicks"),
    State("inperiod-starring-export-store", "data"),
    prevent_initial_call=True,
)
def _download_inperiod_starring(n_clicks, data):
    if not n_clicks or not data:
        return dash.no_update
    content = _starring_export_to_xlsx(data, "inperiod_starring_tables.xlsx")
    if not content:
        return dash.no_update
    return dict(content=content, filename="inperiod_starring_tables.xlsx", base64=True)


@callback(
    Output("vintage-starring-download", "data"),
    Input("vintage-download-starring-btn", "n_clicks"),
    State("vintage-starring-export-store", "data"),
    prevent_initial_call=True,
)
def _download_vintage_starring(n_clicks, data):
    if not n_clicks or not data:
        return dash.no_update
    content = _starring_export_to_xlsx(data, "vintage_starring_tables.xlsx")
    if not content:
        return dash.no_update
    return dict(content=content, filename="vintage_starring_tables.xlsx", base64=True)


def create_outreach_metrics_tab():
    """Outreach Metrics tab: Calls for LVC 1-8, Period A (12/19/25-1/26/26) vs Period B (1/27/26-3/6/26)."""
    df = get_outreach_calls_data()
    if df is None or df.empty:
        return html.Div([
            html.H5("Outreach Metrics (Calls)", style={'color': COLORS['text']}),
            html.P("No data for LVC 1-8 in the selected periods.", style={'color': COLORS['text_muted']}),
        ], style=CARD_STYLE)

    df = df.fillna(0)
    # Weighted averages for speed metrics (need to compute per period)
    df['_dial_w'] = df['avg_speed_to_dial'] * df['leads_with_call']
    df['_contact_w'] = df['avg_speed_to_contact'] * df['contacted_count']

    # --- Aggregate view ---
    agg = df.groupby('period').agg({
        'lead_count': 'sum',
        'leads_with_call': 'sum',
        'contacted_count': 'sum',
        'total_call_attempts': 'sum',
        'dial_within_5min': 'sum',
        'dial_within_15min': 'sum',
        'dial_within_60min': 'sum',
        'contact_within_5min': 'sum',
        'contact_within_15min': 'sum',
        'contact_within_60min': 'sum',
        '_dial_w': 'sum',
        '_contact_w': 'sum',
    }).reset_index()
    agg['avg_call_attempts'] = agg['total_call_attempts'] / agg['lead_count'].replace(0, 1)
    agg['pct_called'] = (agg['leads_with_call'] / agg['lead_count'].replace(0, 1) * 100)
    agg['pct_contacted'] = (agg['contacted_count'] / agg['lead_count'].replace(0, 1) * 100)
    agg['avg_speed_to_dial'] = agg['_dial_w'] / agg['leads_with_call'].replace(0, 1)
    agg['avg_speed_to_contact'] = agg['_contact_w'] / agg['contacted_count'].replace(0, 1)
    agg['pct_dial_5min'] = (agg['dial_within_5min'] / agg['leads_with_call'].replace(0, 1) * 100)
    agg['pct_dial_15min'] = (agg['dial_within_15min'] / agg['leads_with_call'].replace(0, 1) * 100)
    agg['pct_dial_60min'] = (agg['dial_within_60min'] / agg['leads_with_call'].replace(0, 1) * 100)
    agg['pct_contact_5min'] = (agg['contact_within_5min'] / agg['contacted_count'].replace(0, 1) * 100)
    agg['pct_contact_15min'] = (agg['contact_within_15min'] / agg['contacted_count'].replace(0, 1) * 100)
    agg['pct_contact_60min'] = (agg['contact_within_60min'] / agg['contacted_count'].replace(0, 1) * 100)

    period_a = agg[agg['period'] == 'Period A'].iloc[0] if 'Period A' in agg['period'].values else None
    period_b = agg[agg['period'] == 'Period B'].iloc[0] if 'Period B' in agg['period'].values else None

    def kpi(val_a, val_b, fmt='number'):
        if val_a is None or val_b is None:
            return html.Div("—", style=METRIC_CARD_STYLE)
        delta = (val_b - val_a) if val_a else val_b
        if fmt == 'pct':
            d_s = f"{delta:+.1f}pp"
        elif fmt == 'int':
            d_s = f"{delta:+,.0f}"
        else:
            d_s = f"{delta:+.2f}"
        return create_metric_card("", f"{val_b:.1f}" if fmt == 'pct' else f"{val_b:,.2f}" if fmt == 'number' else f"{val_b:,.0f}", delta, f" vs Period A", positive_is_good=(delta >= 0 if fmt in ('pct','number') else delta >= 0))

    # Aggregate KPI cards
    cards_agg = []
    if period_a is not None and period_b is not None:
        cards_agg = [
            dbc.Col(create_metric_card("Leads (StS)", f"{period_b['lead_count']:,.0f}", period_b['lead_count'] - period_a['lead_count'], " vs Period A", positive_is_good=True), width=2),
            dbc.Col(create_metric_card("% Called", f"{period_b['pct_called']:.1f}%", (period_b['pct_called'] - period_a['pct_called']), "pp vs Period A", positive_is_good=True), width=2),
            dbc.Col(create_metric_card("% Contacted", f"{period_b['pct_contacted']:.1f}%", (period_b['pct_contacted'] - period_a['pct_contacted']), "pp vs Period A", positive_is_good=True), width=2),
            dbc.Col(create_metric_card("Avg Call Attemps/Lead", f"{period_b['avg_call_attempts']:.2f}", period_b['avg_call_attempts'] - period_a['avg_call_attempts'], " vs Period A", positive_is_good=True), width=2),
            dbc.Col(create_metric_card("Speed to Dial (avg min)", f"{period_b['avg_speed_to_dial']:.0f}", period_b['avg_speed_to_dial'] - period_a['avg_speed_to_dial'], " vs Period A", positive_is_good=False), width=2),
            dbc.Col(create_metric_card("Speed to Contact (avg min)", f"{period_b['avg_speed_to_contact']:.0f}", period_b['avg_speed_to_contact'] - period_a['avg_speed_to_contact'], " vs Period A", positive_is_good=False), width=2),
        ]

    # Speed breakdown cards (Period B % within 5/15/60)
    if period_b is not None:
        cards_speed = [
            dbc.Col(html.Div([
                html.Div("Dial within 5/15/60 min", style={'fontSize': '11px', 'color': COLORS['text_muted']}),
                html.Div(f"{period_b['pct_dial_5min']:.0f}% / {period_b['pct_dial_15min']:.0f}% / {period_b['pct_dial_60min']:.0f}%", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': COLORS['primary']}),
                html.Div("(Period B)", style={'fontSize': '10px', 'color': COLORS['text_muted']}),
            ], style=METRIC_CARD_STYLE), width=3),
            dbc.Col(html.Div([
                html.Div("Contact within 5/15/60 min", style={'fontSize': '11px', 'color': COLORS['text_muted']}),
                html.Div(f"{period_b['pct_contact_5min']:.0f}% / {period_b['pct_contact_15min']:.0f}% / {period_b['pct_contact_60min']:.0f}%", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': COLORS['primary']}),
                html.Div("(Period B)", style={'fontSize': '10px', 'color': COLORS['text_muted']}),
            ], style=METRIC_CARD_STYLE), width=3),
        ]
    else:
        cards_speed = []

    # --- Starring view ---
    by_star = df.groupby(['period', 'starring']).agg({
        'lead_count': 'sum',
        'leads_with_call': 'sum',
        'contacted_count': 'sum',
        'total_call_attempts': 'sum',
        'dial_within_5min': 'sum',
        'dial_within_15min': 'sum',
        'dial_within_60min': 'sum',
        'contact_within_5min': 'sum',
        'contact_within_15min': 'sum',
        'contact_within_60min': 'sum',
        '_dial_w': 'sum',
        '_contact_w': 'sum',
    }).reset_index()
    by_star['avg_speed_to_dial'] = by_star['_dial_w'] / by_star['leads_with_call'].replace(0, 1)
    by_star['avg_speed_to_contact'] = by_star['_contact_w'] / by_star['contacted_count'].replace(0, 1)
    by_star['pct_called'] = (by_star['leads_with_call'] / by_star['lead_count'].replace(0, 1) * 100).round(1)
    by_star['pct_contacted'] = (by_star['contacted_count'] / by_star['lead_count'].replace(0, 1) * 100).round(1)
    by_star['pct_dial_5min'] = (by_star['dial_within_5min'] / by_star['leads_with_call'].replace(0, 1) * 100).round(1)
    by_star['pct_contact_5min'] = (by_star['contact_within_5min'] / by_star['contacted_count'].replace(0, 1) * 100).round(1)
    by_star['avg_call_attempts'] = (by_star['total_call_attempts'] / by_star['lead_count'].replace(0, 1)).round(2)
    star_order = ['5', '3', '1', 'Unknown', '5.0', '3.0', '1.0']
    by_star['starring'] = pd.Categorical(by_star['starring'].astype(str), categories=star_order, ordered=True)
    by_star = by_star.sort_values(['period', 'starring']).drop(columns=['_dial_w', '_contact_w'], errors='ignore')

    starring_table_data = by_star[['period', 'starring', 'lead_count', 'leads_with_call', 'pct_called', 'contacted_count', 'pct_contacted', 'avg_call_attempts', 'avg_speed_to_dial', 'avg_speed_to_contact', 'pct_dial_5min', 'pct_contact_5min']].copy()
    starring_table_data['avg_speed_to_dial'] = starring_table_data['avg_speed_to_dial'].round(0)
    starring_table_data['avg_speed_to_contact'] = starring_table_data['avg_speed_to_contact'].round(0)
    starring_table_data = starring_table_data.rename(columns={
        'pct_called': '% Called', 'pct_contacted': '% Contacted', 'pct_dial_5min': '% Dial ≤5m', 'pct_contact_5min': '% Contact ≤5m',
        'avg_speed_to_dial': 'Avg Dial (m)', 'avg_speed_to_contact': 'Avg Contact (m)', 'avg_call_attempts': 'Avg Calls/Lead'
    })

    # --- MA view ---
    by_ma = df.groupby(['period', 'mortgage_advisor']).agg({
        'lead_count': 'sum',
        'leads_with_call': 'sum',
        'contacted_count': 'sum',
        'total_call_attempts': 'sum',
        'dial_within_5min': 'sum',
        'dial_within_15min': 'sum',
        'dial_within_60min': 'sum',
        'contact_within_5min': 'sum',
        'contact_within_15min': 'sum',
        'contact_within_60min': 'sum',
        '_dial_w': 'sum',
        '_contact_w': 'sum',
    }).reset_index()
    by_ma['avg_speed_to_dial'] = by_ma['_dial_w'] / by_ma['leads_with_call'].replace(0, 1)
    by_ma['avg_speed_to_contact'] = by_ma['_contact_w'] / by_ma['contacted_count'].replace(0, 1)
    by_ma['pct_called'] = (by_ma['leads_with_call'] / by_ma['lead_count'].replace(0, 1) * 100).round(1)
    by_ma['pct_contacted'] = (by_ma['contacted_count'] / by_ma['lead_count'].replace(0, 1) * 100).round(1)
    by_ma['pct_dial_5min'] = (by_ma['dial_within_5min'] / by_ma['leads_with_call'].replace(0, 1) * 100).round(1)
    by_ma['pct_contact_5min'] = (by_ma['contact_within_5min'] / by_ma['contacted_count'].replace(0, 1) * 100).round(1)
    by_ma['avg_call_attempts'] = (by_ma['total_call_attempts'] / by_ma['lead_count'].replace(0, 1)).round(2)
    by_ma = by_ma.drop(columns=['_dial_w', '_contact_w'], errors='ignore')
    by_ma = by_ma[by_ma['lead_count'] >= 5]  # min volume
    by_ma = by_ma.sort_values(['period', 'lead_count'], ascending=[True, False])
    ma_table_data = by_ma[['period', 'mortgage_advisor', 'lead_count', 'leads_with_call', 'pct_called', 'contacted_count', 'pct_contacted', 'avg_call_attempts', 'avg_speed_to_dial', 'avg_speed_to_contact', 'pct_dial_5min', 'pct_contact_5min']].copy()
    ma_table_data['avg_speed_to_dial'] = ma_table_data['avg_speed_to_dial'].round(0)
    ma_table_data['avg_speed_to_contact'] = ma_table_data['avg_speed_to_contact'].round(0)
    ma_table_data = ma_table_data.rename(columns={
        'mortgage_advisor': 'MA', 'pct_called': '% Called', 'pct_contacted': '% Contacted', 'pct_dial_5min': '% Dial ≤5m', 'pct_contact_5min': '% Contact ≤5m',
        'avg_speed_to_dial': 'Avg Dial (m)', 'avg_speed_to_contact': 'Avg Contact (m)', 'avg_call_attempts': 'Avg Calls/Lead', 'leads_with_call': 'Called', 'contacted_count': 'Contacted'
    })

    calc_agg = "Calc: Lead count = COUNT(*); % Called = leads_with_call/lead_count (first_call_attempt_datetime NOT NULL); % Contacted = contacted_count/lead_count (sf__contacted_guid NOT NULL); Speed to dial = AVG(TIMESTAMP_DIFF(first_call_attempt_datetime, current_sales_assigned_date, MIN)); Speed to contact = AVG(TIMESTAMP_DIFF(sf_contacted_date, current_sales_assigned_date, MIN)) when sf__contacted_guid NOT NULL."
    calc_speed = "Calc: Dial/Contact within 5/15/60 min = COUNT(speed_to_dial_minutes ≤ 5 etc.) / leads_with_call or contacted_count."
    return html.Div([
        create_section_header(
            "Outreach Metrics (Calls)",
            f"LVC 1-8 only | Period A: {OUTREACH_PERIOD_A_START} – {OUTREACH_PERIOD_A_END}  vs  Period B: {OUTREACH_PERIOD_B_START} – {OUTREACH_PERIOD_B_END}"
        ),
        html.Hr(style={'borderColor': COLORS['border'], 'margin': '16px 0'}),
        html.H5("Aggregate (Period A vs Period B)", style={'color': COLORS['text'], 'marginBottom': '4px', 'fontSize': '14px', 'fontWeight': '600'}),
        html.Div(calc_agg, style={'fontSize': '11px', 'color': COLORS['text_muted'], 'marginBottom': '12px'}),
        dbc.Row(cards_agg),
        html.Div(calc_speed, style={'fontSize': '11px', 'color': COLORS['text_muted'], 'marginTop': '8px', 'marginBottom': '8px'}) if cards_speed else html.Div(),
        dbc.Row(cards_speed) if cards_speed else html.Div(),
        html.Hr(style={'borderColor': COLORS['border'], 'margin': '24px 0'}),
        html.H5("By Starring", style={'color': COLORS['text'], 'marginBottom': '4px', 'fontSize': '14px', 'fontWeight': '600'}),
        html.Div("Calc: Same metrics as aggregate, grouped by period and starring.", style={'fontSize': '11px', 'color': COLORS['text_muted'], 'marginBottom': '12px'}),
        html.Div([
            dash_table.DataTable(
                data=starring_table_data.to_dict('records'),
                columns=[{'name': c, 'id': c, 'type': 'numeric', 'format': {'specifier': ',.1f'} if c in ['% Called', '% Contacted', '% Dial ≤5m', '% Contact ≤5m'] else {'specifier': ',.0f'} if c in ['Avg Dial (m)', 'Avg Contact (m)', 'lead_count', 'leads_with_call', 'contacted_count'] else None} for c in starring_table_data.columns],
                style_table={'backgroundColor': COLORS['card_bg']},
                style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'padding': '8px'},
                style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold'},
            )
        ], style=CARD_STYLE),
        html.Hr(style={'borderColor': COLORS['border'], 'margin': '24px 0'}),
        html.H5("By MA (min 5 leads)", style={'color': COLORS['text'], 'marginBottom': '4px', 'fontSize': '14px', 'fontWeight': '600'}),
        html.Div("Calc: Same metrics as aggregate, grouped by period and mortgage_advisor; rows with lead_count ≥ 5.", style={'fontSize': '11px', 'color': COLORS['text_muted'], 'marginBottom': '12px'}),
        html.Div([
            dash_table.DataTable(
                data=ma_table_data.to_dict('records'),
                columns=[{'name': c, 'id': c, 'type': 'numeric', 'format': {'specifier': ',.1f'} if c in ['% Called', '% Contacted', '% Dial ≤5m', '% Contact ≤5m'] else {'specifier': ',.0f'} if c in ['Avg Dial (m)', 'Avg Contact (m)', 'lead_count', 'Called', 'Contacted'] else {'specifier': ',.2f'} if c == 'Avg Calls/Lead' else None} for c in ma_table_data.columns],
                style_table={'backgroundColor': COLORS['card_bg']},
                style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'padding': '8px'},
                style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold'},
                page_size=20,
            )
        ], style=CARD_STYLE),
    ], style={'marginTop': '16px'})

# === Vintage Deep Dive Callback ===
@callback(
    [Output('vintage-kpi-row', 'children'),
     Output('vintage-lvc-row', 'children'),
     Output('vintage-channel-row', 'children'),
     Output('vintage-persona-row', 'children'),
     Output('vintage-outreach-row', 'children'),
     Output('vintage-ma-row', 'children'),
     Output('vintage-starring-export-store', 'data')],
    Input('vintage-week-selector', 'value')
)
def update_vintage_deepdive_tab(target_week):
    """Update all vintage deep dive tab components (based on lead_created_date)"""
    vintage_export_data = None
    if not target_week:
        return [html.Div("Select a week")] * 6 + [None]
    
    target_date = pd.to_datetime(target_week)
    target_end = target_date + timedelta(days=6)
    
    # Get vintage-based data
    df_vintage = get_vintage_deep_dive_data(target_week)
    
    # === KPI Summary ===
    kpi_row = html.Div("Loading KPIs...")
    if not df_vintage.empty:
        df_vintage['vintage_week'] = pd.to_datetime(df_vintage['vintage_week'])
        df_vintage['fas_rate'] = df_vintage['fas_count'] / df_vintage['sts_count'].replace(0, 1) * 100
        
        # Get last 12 weeks of data for sparklines (including target week)
        twelve_weeks_ago = target_date - timedelta(weeks=11)
        sparkline_data = df_vintage[(df_vintage['vintage_week'] >= twelve_weeks_ago) & 
                                    (df_vintage['vintage_week'] <= target_date)]
        
        # Aggregate by week for sparklines
        weekly_sparkline = sparkline_data.groupby('vintage_week').agg({
            'lead_count': 'sum',
            'sts_count': 'sum',
            'fas_count': 'sum',
            'fas_dollars': 'sum'
        }).reset_index().sort_values('vintage_week')
        weekly_sparkline['fas_rate'] = weekly_sparkline['fas_count'] / weekly_sparkline['sts_count'].replace(0, 1) * 100
        weekly_sparkline['avg_loan'] = weekly_sparkline['fas_dollars'] / weekly_sparkline['fas_count'].replace(0, 1)
        weekly_sparkline['week_label'] = weekly_sparkline['vintage_week'].dt.strftime('%-m/%d')
        
        # Target week data
        target_data = df_vintage[df_vintage['vintage_week'] == target_date]
        prev_data = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                               (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))]
        
        # Aggregate metrics
        tgt_leads = target_data['lead_count'].sum()
        tgt_sts = target_data['sts_count'].sum()
        tgt_fas = target_data['fas_count'].sum()
        tgt_dollars = target_data['fas_dollars'].sum()
        tgt_fas_rate = (tgt_fas / tgt_sts * 100) if tgt_sts > 0 else 0
        tgt_avg_loan = (tgt_dollars / tgt_fas) if tgt_fas > 0 else 0
        
        prev_leads = prev_data['lead_count'].sum() / 4
        prev_sts = prev_data['sts_count'].sum() / 4
        prev_fas = prev_data['fas_count'].sum() / 4
        prev_dollars = prev_data['fas_dollars'].sum() / 4
        prev_fas_rate = (prev_fas / prev_sts * 100) if prev_sts > 0 else 0
        prev_avg_loan = (prev_dollars / prev_fas) if prev_fas > 0 else 0
        
        # Calculate deltas
        fas_delta = ((tgt_fas - prev_fas) / prev_fas * 100) if prev_fas > 0 else 0
        dollars_delta = ((tgt_dollars - prev_dollars) / prev_dollars * 100) if prev_dollars > 0 else 0
        rate_delta = tgt_fas_rate - prev_fas_rate
        avg_loan_delta = ((tgt_avg_loan - prev_avg_loan) / prev_avg_loan * 100) if prev_avg_loan > 0 else 0
        leads_delta = ((tgt_leads - prev_leads) / prev_leads * 100) if prev_leads > 0 else 0
        sts_delta = ((tgt_sts - prev_sts) / prev_sts * 100) if prev_sts > 0 else 0
        
        # Get sparkline data lists
        spark_labels = weekly_sparkline['week_label'].tolist()
        spark_leads = weekly_sparkline['lead_count'].tolist()
        spark_sts = weekly_sparkline['sts_count'].tolist()
        spark_fas = weekly_sparkline['fas_count'].tolist()
        spark_dollars = weekly_sparkline['fas_dollars'].tolist()
        spark_rate = weekly_sparkline['fas_rate'].tolist()
        spark_avg = weekly_sparkline['avg_loan'].tolist()
        
        kpi_row = dbc.Row([
            dbc.Col(create_metric_card_with_sparkline("Lead Qty", f"{tgt_leads:,.0f}", leads_delta, "% vs 4wk avg", sparkline_data=spark_leads, sparkline_labels=spark_labels, value_format="number"), width=2),
            dbc.Col(create_metric_card_with_sparkline("StS Volume", f"{tgt_sts:,.0f}", sts_delta, "% vs 4wk avg", sparkline_data=spark_sts, sparkline_labels=spark_labels, value_format="number"), width=2),
            dbc.Col(create_metric_card_with_sparkline("FAS Count", f"{tgt_fas:,.0f}", fas_delta, "% vs 4wk avg", sparkline_data=spark_fas, sparkline_labels=spark_labels, value_format="number"), width=2),
            dbc.Col(create_metric_card_with_sparkline("FAS $", f"${tgt_dollars:,.0f}", dollars_delta, "% vs 4wk avg", sparkline_data=spark_dollars, sparkline_labels=spark_labels, value_format="currency"), width=2),
            dbc.Col(create_metric_card_with_sparkline("FAS Rate", f"{tgt_fas_rate:.1f}%", rate_delta, "pp vs 4wk avg", sparkline_data=spark_rate, sparkline_labels=spark_labels, value_format="percent"), width=2),
            dbc.Col(create_metric_card_with_sparkline("Avg Loan", f"${tgt_avg_loan:,.0f}", avg_loan_delta, "% vs 4wk avg", sparkline_data=spark_avg, sparkline_labels=spark_labels, value_format="currency"), width=2),
        ])
    
    # === LVC Analysis ===
    lvc_row = html.Div("Loading LVC analysis...")
    if not df_vintage.empty:
        lvc_target = df_vintage[df_vintage['vintage_week'] == target_date].groupby('lvc_group').agg({
            'lead_count': 'sum',
            'sts_count': 'sum',
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        lvc_target = lvc_target.rename(columns={'avg_lp2c': 'lp2c'})
        
        lvc_prev = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                             (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))].groupby('lvc_group').agg({
            'lead_count': 'sum',
            'sts_count': 'sum',
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        lvc_prev['lead_count'] = lvc_prev['lead_count'] / 4
        lvc_prev['sts_count'] = lvc_prev['sts_count'] / 4
        lvc_prev['fas_count'] = lvc_prev['fas_count'] / 4
        lvc_prev['fas_dollars'] = lvc_prev['fas_dollars'] / 4
        lvc_prev = lvc_prev.rename(columns={'avg_lp2c': 'lp2c_prev'})
        
        # Create pivot tables for charts
        lvc_target['period'] = 'This Week'
        lvc_prev['period'] = '4-Wk Avg'
        lvc_combined = pd.concat([lvc_target, lvc_prev])
        
        lvc_pivot_count = lvc_combined.pivot(index='lvc_group', columns='period', values='fas_count').reset_index().fillna(0)
        lvc_pivot_dollars = lvc_combined.pivot(index='lvc_group', columns='period', values='fas_dollars').reset_index().fillna(0)
        
        # Calculate % change
        if 'This Week' in lvc_pivot_count.columns and '4-Wk Avg' in lvc_pivot_count.columns:
            lvc_pivot_count['pct_change'] = ((lvc_pivot_count['This Week'] - lvc_pivot_count['4-Wk Avg']) / lvc_pivot_count['4-Wk Avg'].replace(0, 1) * 100)
            lvc_pivot_dollars['pct_change'] = ((lvc_pivot_dollars['This Week'] - lvc_pivot_dollars['4-Wk Avg']) / lvc_pivot_dollars['4-Wk Avg'].replace(0, 1) * 100)
        else:
            lvc_pivot_count['pct_change'] = 0
            lvc_pivot_dollars['pct_change'] = 0
        
        # FAS Count chart
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
            title=dict(text='📊 FAS Count by LVC Group (Vintage)', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']), margin=dict(l=60, r=40, t=60, b=50),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
            barmode='group', height=380, bargap=0.3,
        )
        
        # FAS $ chart
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
            title=dict(text='💰 FAS $ by LVC Group (Vintage)', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']), margin=dict(l=60, r=40, t=60, b=50),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
            barmode='group', height=380, bargap=0.3,
        )
        
        # LVC comparison table
        lvc_table = lvc_target.merge(
            lvc_prev[['lvc_group', 'fas_count', 'fas_dollars', 'lp2c_prev']],
            on='lvc_group', suffixes=('', '_prev'), how='outer'
        ).fillna(0)
        lvc_table['Count Δ'] = lvc_table['fas_count'] - lvc_table['fas_count_prev']
        lvc_table['fas_rate'] = lvc_table['fas_count'] / lvc_table['sts_count'].replace(0, 1) * 100
        lvc_table['lp2c_delta'] = lvc_table['lp2c'] - lvc_table['lp2c_prev']
        
        total_target = lvc_table['fas_count'].sum()
        total_prev = lvc_table['fas_count_prev'].sum()
        lvc_table['% Target'] = lvc_table['fas_count'] / total_target * 100 if total_target > 0 else 0
        lvc_table['% Prev'] = lvc_table['fas_count_prev'] / total_prev * 100 if total_prev > 0 else 0
        lvc_table['Share Δ'] = lvc_table['% Target'] - lvc_table['% Prev']
        
        lvc_row = dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_lvc_count)], style=CARD_STYLE), width=6),
            dbc.Col(html.Div([dcc.Graph(figure=fig_lvc_dollars)], style=CARD_STYLE), width=6),
            dbc.Col(html.Div([
                html.H5("LVC Group Comparison (Vintage)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=lvc_table[['lvc_group', 'fas_count', 'fas_count_prev', 'Count Δ', 'lp2c', 'lp2c_delta', '% Target', 'Share Δ']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'LVC Group', 'id': 'lvc_group'},
                        {'name': 'This Wk', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': '4-Wk Avg', 'id': 'fas_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
                        {'name': 'Δ FAS', 'id': 'Count Δ', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                        {'name': 'LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ LP2C pp', 'id': 'lp2c_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                        {'name': 'Mix %', 'id': '% Target', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ Mix', 'id': 'Share Δ', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg']},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '8px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{Count Δ} > 0', 'column_id': 'Count Δ'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{Count Δ} < 0', 'column_id': 'Count Δ'}, 'color': COLORS['danger']},
                        {'if': {'filter_query': '{lp2c_delta} > 0', 'column_id': 'lp2c_delta'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{lp2c_delta} < 0', 'column_id': 'lp2c_delta'}, 'color': COLORS['danger']},
                        {'if': {'filter_query': '{Share Δ} > 0', 'column_id': 'Share Δ'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{Share Δ} < 0', 'column_id': 'Share Δ'}, 'color': COLORS['danger']},
                    ]
                )
            ], style=CARD_STYLE), width=12),
        ])
    
    # === Channel Analysis ===
    channel_row = html.Div("Loading channel analysis...")
    if not df_vintage.empty:
        channel_target = df_vintage[df_vintage['vintage_week'] == target_date].groupby('channel').agg({
            'sts_count': 'sum',
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        channel_target = channel_target.rename(columns={'avg_lp2c': 'lp2c'})
        
        channel_prev = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                                  (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))].groupby('channel').agg({
            'sts_count': 'sum',
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        channel_prev['sts_count'] = channel_prev['sts_count'] / 4
        channel_prev['fas_count'] = channel_prev['fas_count'] / 4
        channel_prev['fas_dollars'] = channel_prev['fas_dollars'] / 4
        channel_prev = channel_prev.rename(columns={'avg_lp2c': 'lp2c_prev'})
        
        channel_merged = channel_target.merge(channel_prev, on='channel', suffixes=('', '_prev'), how='outer').fillna(0)
        channel_merged['fas_pct'] = (channel_merged['fas_count'] / channel_merged['sts_count'].replace(0, 1) * 100).round(1)
        channel_merged['fas_delta'] = channel_merged['fas_count'] - channel_merged['fas_count_prev']
        channel_merged['dollar_delta'] = channel_merged['fas_dollars'] - channel_merged['fas_dollars_prev']
        channel_merged['lp2c_delta'] = channel_merged['lp2c'] - channel_merged['lp2c_prev']
        
        # Keep all for worst performers
        channel_all = channel_merged.copy()
        channel_merged = channel_merged.sort_values('fas_count', ascending=False).head(15)
        
        # Get worst performers
        channel_worst = channel_all[channel_all['fas_count'] >= 5].nsmallest(10, 'fas_delta')
        
        def format_dollars(val):
            if val >= 1000000:
                return f"${val/1000000:.1f}M"
            elif val >= 1000:
                return f"${val/1000:.0f}K"
            else:
                return f"${val:.0f}"
        
        channel_merged['fas_dollars_fmt'] = channel_merged['fas_dollars'].apply(format_dollars)
        
        fig_channel = go.Figure()
        fig_channel.add_trace(go.Bar(
            y=channel_merged['channel'],
            x=channel_merged['fas_count'],
            orientation='h',
            marker_color=COLORS['chart_yellow'],
            text=[f"{fas:,.0f} ({dollars})" for fas, dollars in zip(channel_merged['fas_count'], channel_merged['fas_dollars_fmt'])],
            textposition='outside',
            textfont=dict(size=10, color=COLORS['text'])
        ))
        fig_channel.update_layout(
            title=dict(text='📢 Top 15 Channels by FAS Count (Vintage)', font=dict(color=COLORS['text'], size=14)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=150, r=100, t=50, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=True, range=[0, channel_merged['fas_count'].max() * 1.25]),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=False, autorange='reversed'),
            height=400, showlegend=False
        )
        
        channel_row = dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_channel)], style=CARD_STYLE), width=5),
            dbc.Col(html.Div([
                html.H5("Channel Comparison (Vintage - This Week vs 4-Wk Avg)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=channel_merged[['channel', 'sts_count', 'fas_count', 'fas_pct', 'fas_dollars', 'fas_delta', 'lp2c', 'lp2c_delta']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'Channel', 'id': 'channel'},
                        {'name': 'StS #', 'id': 'sts_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS #', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS %', 'id': 'fas_pct', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': '$FAS', 'id': 'fas_dollars', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
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
            ], style=CARD_STYLE), width=4),
            dbc.Col(html.Div([
                html.H5("⚠️ Worst Performing Channels (vs 4-Wk Avg)", style={'color': COLORS['danger'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=channel_worst[['channel', 'fas_count', 'fas_count_prev', 'fas_delta', 'lp2c', 'lp2c_delta']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'Channel', 'id': 'channel'},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Prev', 'id': 'fas_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
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
            ], style=CARD_STYLE), width=3),
        ])
    
    # === Persona Analysis ===
    persona_row = html.Div("Loading persona analysis...")
    if not df_vintage.empty:
        # Get target week persona data with LP2C
        persona_target = df_vintage[df_vintage['vintage_week'] == target_date].groupby('persona').agg({
            'sts_count': 'sum',
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        persona_target = persona_target.rename(columns={'avg_lp2c': 'lp2c'})
        
        # Get prev 4 weeks persona data with LP2C
        persona_prev = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                              (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))].groupby('persona').agg({
            'sts_count': 'sum',
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        persona_prev['sts_count'] = persona_prev['sts_count'] / 4
        persona_prev['fas_count'] = persona_prev['fas_count'] / 4
        persona_prev['fas_dollars'] = persona_prev['fas_dollars'] / 4
        persona_prev = persona_prev.rename(columns={'avg_lp2c': 'lp2c_prev'})
        
        # Merge target and prev
        persona_merged = persona_target.merge(persona_prev, on='persona', suffixes=('', '_prev'), how='outer').fillna(0)
        persona_merged['fas_delta'] = persona_merged['fas_count'] - persona_merged['fas_count_prev']
        persona_merged['dollar_delta'] = persona_merged['fas_dollars'] - persona_merged['fas_dollars_prev']
        persona_merged['fas_pct'] = (persona_merged['fas_count'] / persona_merged['sts_count'].replace(0, 1) * 100).round(1)
        persona_merged['lp2c_delta'] = persona_merged['lp2c'] - persona_merged['lp2c_prev']
        
        # Calculate mix %
        total_fas = persona_merged['fas_count'].sum()
        total_fas_prev = persona_merged['fas_count_prev'].sum()
        persona_merged['mix_pct'] = (persona_merged['fas_count'] / total_fas * 100) if total_fas > 0 else 0
        persona_merged['mix_pct_prev'] = (persona_merged['fas_count_prev'] / total_fas_prev * 100) if total_fas_prev > 0 else 0
        persona_merged['mix_delta'] = persona_merged['mix_pct'] - persona_merged['mix_pct_prev']
        
        # Keep all for worst performers
        persona_all = persona_merged.copy()
        persona_merged = persona_merged.sort_values('fas_count', ascending=False)
        
        # Get worst performers
        persona_worst = persona_all[persona_all['fas_count'] >= 3].nsmallest(5, 'fas_delta')
        
        # Format $FAS for display
        def format_dollars_persona_vintage(val):
            if val >= 1000000:
                return f"${val/1000000:.1f}M"
            elif val >= 1000:
                return f"${val/1000:.0f}K"
            else:
                return f"${val:.0f}"
        
        persona_merged['fas_dollars_fmt'] = persona_merged['fas_dollars'].apply(format_dollars_persona_vintage)
        
        # Create horizontal bar chart
        fig_persona = go.Figure()
        fig_persona.add_trace(go.Bar(
            y=persona_merged['persona'],
            x=persona_merged['fas_count'],
            orientation='h',
            marker_color=COLORS['chart_purple'],
            text=[f"{fas:,.0f} ({dollars})" for fas, dollars in zip(persona_merged['fas_count'], persona_merged['fas_dollars_fmt'])],
            textposition='outside',
            textfont=dict(size=10, color=COLORS['text'])
        ))
        fig_persona.update_layout(
            title=dict(text='👤 Persona by FAS Count (Vintage)', font=dict(color=COLORS['text'], size=14)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=120, r=80, t=50, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=True, range=[0, persona_merged['fas_count'].max() * 1.3]),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=False, autorange='reversed'),
            height=350, showlegend=False
        )
        
        # LVC Group x Persona composition
        lvc_persona = df_vintage[df_vintage['vintage_week'] == target_date].groupby(['lvc_group', 'persona'])['fas_count'].sum().reset_index()
        lvc_persona_pivot = lvc_persona.pivot(index='persona', columns='lvc_group', values='fas_count').fillna(0)
        
        # Create stacked bar chart for LVC x Persona
        fig_lvc_persona = go.Figure()
        lvc_colors = {'LVC 1-2': COLORS['chart_cyan'], 'LVC 3-8': COLORS['chart_green'], 
                      'LVC 9-10': COLORS['chart_yellow'], 'PHX Transfer': COLORS['chart_purple'], 'Other': COLORS['chart_gray']}
        
        for lvc_group in lvc_persona_pivot.columns:
            fig_lvc_persona.add_trace(go.Bar(
                y=lvc_persona_pivot.index,
                x=lvc_persona_pivot[lvc_group],
                name=lvc_group,
                orientation='h',
                marker_color=lvc_colors.get(lvc_group, COLORS['chart_gray'])
            ))
        
        fig_lvc_persona.update_layout(
            title=dict(text='📊 LVC Composition by Persona (Vintage)', font=dict(color=COLORS['text'], size=14), y=0.98),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=120, r=30, t=80, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=False, autorange='reversed'),
            barmode='stack',
            legend=dict(orientation='h', yanchor='top', y=1.15, xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
            height=380
        )
        
        persona_row = dbc.Row([
            dbc.Col(html.Div([dcc.Graph(figure=fig_persona)], style=CARD_STYLE), width=4),
            dbc.Col(html.Div([
                html.H5("Persona Mix (Vintage - This Week vs 4-Wk Avg)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=persona_merged[['persona', 'fas_count', 'fas_delta', 'lp2c', 'lp2c_delta', 'mix_pct', 'mix_delta']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'Persona', 'id': 'persona'},
                        {'name': 'FAS #', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Δ FAS', 'id': 'fas_delta', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                        {'name': 'LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ LP2C pp', 'id': 'lp2c_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                        {'name': 'Mix %', 'id': 'mix_pct', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ Mix', 'id': 'mix_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg'], 'overflowX': 'auto'},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '6px', 'fontSize': '11px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{fas_delta} > 0', 'column_id': 'fas_delta'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{fas_delta} < 0', 'column_id': 'fas_delta'}, 'color': COLORS['danger']},
                        {'if': {'filter_query': '{lp2c_delta} > 0', 'column_id': 'lp2c_delta'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{lp2c_delta} < 0', 'column_id': 'lp2c_delta'}, 'color': COLORS['danger']},
                        {'if': {'filter_query': '{mix_delta} > 0', 'column_id': 'mix_delta'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{mix_delta} < 0', 'column_id': 'mix_delta'}, 'color': COLORS['danger']},
                    ],
                    page_size=10
                )
            ], style=CARD_STYLE), width=3),
            dbc.Col(html.Div([
                html.H5("⚠️ Worst Performing Personas", style={'color': COLORS['danger'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=persona_worst[['persona', 'fas_count', 'fas_count_prev', 'fas_delta', 'lp2c', 'lp2c_delta']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'Persona', 'id': 'persona'},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Prev', 'id': 'fas_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
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
                    page_size=5
                )
            ], style=CARD_STYLE), width=2),
            dbc.Col(html.Div([dcc.Graph(figure=fig_lvc_persona)], style=CARD_STYLE), width=3),
        ])
    
    # === MA Analysis ===
    ma_row = html.Div("Loading MA analysis...")
    if not df_vintage.empty:
        # Target week MA data
        ma_target = df_vintage[df_vintage['vintage_week'] == target_date].groupby('mortgage_advisor').agg({
            'lead_count': 'sum',
            'sts_count': 'sum',
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        ma_target = ma_target.rename(columns={'avg_lp2c': 'lp2c'})
        
        # Previous 4 weeks MA data (averaged)
        ma_prev = df_vintage[(df_vintage['vintage_week'] < target_date) & 
                            (df_vintage['vintage_week'] >= target_date - timedelta(weeks=4))].groupby('mortgage_advisor').agg({
            'lead_count': 'sum',
            'sts_count': 'sum',
            'fas_count': 'sum',
            'fas_dollars': 'sum',
            'avg_lp2c': 'mean'
        }).reset_index()
        ma_prev['fas_count'] = ma_prev['fas_count'] / 4
        ma_prev['fas_dollars'] = ma_prev['fas_dollars'] / 4
        ma_prev['sts_count'] = ma_prev['sts_count'] / 4
        ma_prev = ma_prev.rename(columns={'avg_lp2c': 'lp2c_prev'})
        
        # Merge target and prev
        ma_merged = ma_target.merge(ma_prev[['mortgage_advisor', 'fas_count', 'fas_dollars', 'lp2c_prev']], 
                                    on='mortgage_advisor', suffixes=('', '_prev'), how='left').fillna(0)
        ma_merged['fas_rate'] = ma_merged['fas_count'] / ma_merged['sts_count'].replace(0, 1) * 100
        ma_merged['fas_delta'] = ma_merged['fas_count'] - ma_merged['fas_count_prev']
        ma_merged['dollar_delta'] = ma_merged['fas_dollars'] - ma_merged['fas_dollars_prev']
        ma_merged['lp2c_delta'] = ma_merged['lp2c'] - ma_merged['lp2c_prev']
        
        # Sort by FAS dollars and get top 15
        top_mas = ma_merged.sort_values('fas_dollars', ascending=False).head(15)
        top_gainers = ma_merged.nlargest(10, 'fas_delta')
        top_decliners = ma_merged[ma_merged['fas_count_prev'] > 0].nsmallest(10, 'fas_delta')
        
        # Summary stats
        total_fas = ma_merged['fas_count'].sum()
        total_dollars = ma_merged['fas_dollars'].sum()
        mas_with_fas = len(ma_merged[ma_merged['fas_count'] > 0])
        top_10_pct = (ma_merged.nlargest(10, 'fas_dollars')['fas_dollars'].sum() / total_dollars * 100) if total_dollars > 0 else 0
        
        ma_summary_stats = dbc.Row([
            dbc.Col(create_metric_card("Total FAS", f"{total_fas:,.0f}"), width=3),
            dbc.Col(create_metric_card("Total FAS $", f"${total_dollars/1000000:.2f}M"), width=3),
            dbc.Col(create_metric_card("MAs with FAS", f"{mas_with_fas}"), width=3),
            dbc.Col(create_metric_card("Top 10 MA %", f"{top_10_pct:.0f}%"), width=3),
        ])
        
        ma_row = dbc.Row([
            dbc.Col(ma_summary_stats, width=12),
            dbc.Col(html.Div([
                html.H5("Top 15 MAs by FAS $", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '16px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=top_mas[['mortgage_advisor', 'sts_count', 'fas_count', 'fas_rate', 'fas_delta', 'lp2c', 'lp2c_delta', 'fas_dollars']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'MA', 'id': 'mortgage_advisor'},
                        {'name': 'StS', 'id': 'sts_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS %', 'id': 'fas_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ FAS', 'id': 'fas_delta', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                        {'name': 'LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Δ LP2C pp', 'id': 'lp2c_delta', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
                        {'name': 'FAS $', 'id': 'fas_dollars', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg'], 'overflowX': 'auto'},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '6px 10px', 'fontSize': '12px'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'mortgage_advisor'}, 'textAlign': 'left', 'minWidth': '120px', 'maxWidth': '160px'},
                        {'if': {'column_id': 'sts_count'}, 'minWidth': '45px'},
                        {'if': {'column_id': 'fas_count'}, 'minWidth': '45px'},
                        {'if': {'column_id': 'fas_rate'}, 'minWidth': '50px'},
                        {'if': {'column_id': 'fas_delta'}, 'minWidth': '55px'},
                        {'if': {'column_id': 'lp2c'}, 'minWidth': '50px'},
                        {'if': {'column_id': 'lp2c_delta'}, 'minWidth': '60px'},
                        {'if': {'column_id': 'fas_dollars'}, 'minWidth': '80px'},
                    ],
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '11px', 'textAlign': 'center'},
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
                html.H5("📈 Biggest Gainers vs Prev", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=top_gainers[['mortgage_advisor', 'fas_count', 'fas_rate', 'fas_count_prev', 'fas_delta', 'lp2c']].round(1).to_dict('records'),
                    columns=[
                        {'name': 'MA', 'id': 'mortgage_advisor'},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS %', 'id': 'fas_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Prev', 'id': 'fas_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
                        {'name': 'Δ', 'id': 'fas_delta', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                        {'name': 'LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg']},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '4px 6px', 'fontSize': '11px'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'mortgage_advisor'}, 'textAlign': 'left', 'minWidth': '100px'},
                    ],
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px', 'textAlign': 'center'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{fas_delta} > 0', 'column_id': 'fas_delta'}, 'color': COLORS['success']},
                    ],
                    page_size=10
                )
            ], style=CARD_STYLE), width=2),
            dbc.Col(html.Div([
                html.H5("📉 Biggest Decliners vs Prev", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=top_decliners[['mortgage_advisor', 'fas_count', 'fas_rate', 'fas_count_prev', 'fas_delta', 'lp2c']].round(1).to_dict('records') if not top_decliners.empty else [],
                    columns=[
                        {'name': 'MA', 'id': 'mortgage_advisor'},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS %', 'id': 'fas_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                        {'name': 'Prev', 'id': 'fas_count_prev', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
                        {'name': 'Δ', 'id': 'fas_delta', 'type': 'numeric', 'format': {'specifier': '+,.1f'}},
                        {'name': 'LP2C %', 'id': 'lp2c', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg']},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '4px 6px', 'fontSize': '11px'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'mortgage_advisor'}, 'textAlign': 'left', 'minWidth': '100px'},
                    ],
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px', 'textAlign': 'center'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{fas_delta} < 0', 'column_id': 'fas_delta'}, 'color': COLORS['danger']},
                    ],
                    page_size=10
                )
            ], style=CARD_STYLE), width=2),
        ])
    
    # === Starring Analysis (Vintage) ===
    starring_row = html.Div([
        html.H5("⭐ Starring Distribution (Vintage)", style={'color': COLORS['text'], 'marginBottom': '12px', 'marginTop': '20px', 'fontSize': '16px', 'fontWeight': '600'}),
        html.P("Loading starring data...", style={'color': COLORS['text_muted'], 'fontSize': '11px'})
    ])
    try:
        df_starring = get_starring_data(target_week, 'lead_created_date')
        print(f"[DEBUG] Vintage Starring data rows: {len(df_starring) if not df_starring.empty else 0}")
    except Exception as e:
        print(f"[DEBUG] Vintage Starring query error: {e}")
        df_starring = pd.DataFrame()
    
    if not df_starring.empty:
        df_starring['week_start'] = pd.to_datetime(df_starring['week_start'])
        df_starring['starring'] = df_starring['starring'].astype(str).replace('nan', 'Unknown').replace('<NA>', 'Unknown')
        # Group starring 0 and Unknown into "0"
        df_starring['starring'] = df_starring['starring'].replace(['Unknown', '0.0'], '0')
        
        # Aggregate by starring for target and prev
        starring_target = df_starring[df_starring['week_start'] == target_date].groupby('starring').agg({
            'sts_count': 'sum', 'fas_count': 'sum', 'fas_dollars': 'sum', 'avg_lp2c': 'mean'
        }).reset_index()
        
        starring_prev = df_starring[(df_starring['week_start'] < target_date) & 
                                    (df_starring['week_start'] >= target_date - timedelta(weeks=4))].groupby('starring').agg({
            'sts_count': 'sum', 'fas_count': 'sum', 'fas_dollars': 'sum', 'avg_lp2c': 'mean'
        }).reset_index()
        starring_prev[['sts_count', 'fas_count', 'fas_dollars']] /= 4
        
        # Merge and calculate
        starring_merged = starring_target.merge(starring_prev, on='starring', suffixes=('', '_prev'), how='outer').fillna(0)
        starring_merged['fas_rate'] = starring_merged['fas_count'] / starring_merged['sts_count'].replace(0, 1) * 100
        starring_merged['fas_rate_prev'] = starring_merged['fas_count_prev'] / starring_merged['sts_count_prev'].replace(0, 1) * 100
        starring_merged['fas_delta'] = starring_merged['fas_count'] - starring_merged['fas_count_prev']
        starring_merged['rate_delta'] = starring_merged['fas_rate'] - starring_merged['fas_rate_prev']
        starring_merged['lp2c_delta'] = starring_merged['avg_lp2c'] - starring_merged['avg_lp2c_prev']
        
        # Sort by starring (5, 3, 1, 0)
        star_order = {'1': 1, '3': 2, '5': 3, '0': 4, '1.0': 1, '3.0': 2, '5.0': 3, '0.0': 4}
        starring_merged['sort_order'] = starring_merged['starring'].astype(str).map(lambda x: star_order.get(x, 5))
        starring_merged = starring_merged.sort_values('sort_order')
        
        # LVC x Starring breakdown for target week
        lvc_starring = df_starring[df_starring['week_start'] == target_date].groupby(['lvc_group', 'starring']).agg({
            'fas_count': 'sum', 'sts_count': 'sum', 'fas_dollars': 'sum'
        }).reset_index()
        lvc_starring['fas_rate'] = lvc_starring['fas_count'] / lvc_starring['sts_count'].replace(0, 1) * 100
        
        # Create starring distribution chart
        fig_starring = go.Figure()
        fig_starring.add_trace(go.Bar(
            x=starring_merged['starring'].astype(str),
            y=starring_merged['fas_count'],
            name='FAS Count',
            marker_color=COLORS['chart_cyan'],
            text=starring_merged['fas_count'].round(0).astype(int),
            textposition='outside'
        ))
        fig_starring.update_layout(
            title=dict(text='⭐ FAS by Starring (Vintage)', font=dict(color=COLORS['text'], size=14)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            xaxis=dict(title='Star Rating', gridcolor=COLORS['border']),
            yaxis=dict(title='FAS Count', gridcolor=COLORS['border'], range=[0, starring_merged['fas_count'].max() * 1.15] if starring_merged['fas_count'].max() > 0 else [0, 1]),
            height=280,
            margin=dict(l=50, r=30, t=50, b=40),
            showlegend=False
        )
        
        # Create LVC x Starring stacked bar
        lvc_order = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']
        fig_lvc_starring = go.Figure()
        for lvc in lvc_order:
            lvc_data = lvc_starring[lvc_starring['lvc_group'] == lvc]
            if not lvc_data.empty:
                fig_lvc_starring.add_trace(go.Bar(
                    x=lvc_data['starring'].astype(str),
                    y=lvc_data['fas_count'],
                    name=lvc,
                    text=lvc_data['fas_count'].round(0).astype(int),
                    textposition='inside'
                ))
        fig_lvc_starring.update_layout(
            title=dict(text='⭐ LVC Group × Starring Distribution (Vintage)', font=dict(color=COLORS['text'], size=14)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            xaxis=dict(title='Star Rating', gridcolor=COLORS['border']),
            yaxis=dict(title='FAS Count', gridcolor=COLORS['border']),
            barmode='stack',
            height=280,
            margin=dict(l=50, r=30, t=50, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=10))
        )
        
        # Create LVC Group × Starring weekly distribution tables
        all_weeks = sorted(df_starring['week_start'].unique())
        target_week_dt = all_weeks[-1] if all_weeks else None
        prev_6_weeks = all_weeks[:-1][-6:] if len(all_weeks) > 1 else []
        weeks_for_table = sorted(all_weeks[-6:], reverse=True)
        week_labels = [w.strftime('%-m/%d') for w in weeks_for_table]
        
        lvc_order = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']
        star_order_list = ['5', '3', '1', '0']
        
        dist_rows = []
        fas_day7_rows = []
        contact_rows = []
        fas_dollars_rows = []
        
        for lvc in lvc_order:
            lvc_data = df_starring[df_starring['lvc_group'] == lvc]
            if lvc_data.empty:
                continue
            lvc_stars = [s for s in star_order_list if s in lvc_data['starring'].unique()]
            
            for star in lvc_stars:
                d_row = {'lvc_group': lvc, 'starring': star}
                f7_row = {'lvc_group': lvc, 'starring': star}
                c_row = {'lvc_group': lvc, 'starring': star}
                fas_d_row = {'lvc_group': lvc, 'starring': star}
                
                # Calculate 6-week averages and deltas
                prev_data = lvc_data[(lvc_data['starring'] == star) & (lvc_data['week_start'].isin(prev_6_weeks))]
                target_data = lvc_data[(lvc_data['starring'] == star) & (lvc_data['week_start'] == target_week_dt)]
                
                # Assigned Leads
                avg_leads = prev_data['lead_count'].sum() / len(prev_6_weeks) if prev_6_weeks else 0
                t_leads = target_data['lead_count'].sum() if not target_data.empty else 0
                d_row['prev_avg'] = f"{avg_leads:,.0f}"
                d_row['delta'] = f"{t_leads - avg_leads:+,.0f}"
                
                # FAS Day 7 %
                p_f7 = prev_data['fas_day7_qty'].sum()
                p_sts = prev_data['sts_count'].sum()
                avg_f7_pct = (p_f7 / p_sts * 100) if p_sts > 0 else 0
                t_f7 = target_data['fas_day7_qty'].sum() if not target_data.empty else 0
                t_sts = target_data['sts_count'].sum() if not target_data.empty else 0
                t_f7_pct = (t_f7 / t_sts * 100) if t_sts > 0 else 0
                f7_row['prev_avg'] = f"{avg_f7_pct:.1f}%"
                f7_row['delta'] = f"{t_f7_pct - avg_f7_pct:+.1f}pp"
                
                # SF Contact %
                p_contacted = prev_data['contacted_qty'].sum()
                p_assigned = prev_data['lead_count'].sum()
                avg_c_pct = (p_contacted / p_assigned * 100) if p_assigned > 0 else 0
                t_contacted = target_data['contacted_qty'].sum() if not target_data.empty else 0
                t_assigned = target_data['lead_count'].sum() if not target_data.empty else 0
                t_c_pct = (t_contacted / t_assigned * 100) if t_assigned > 0 else 0
                c_row['prev_avg'] = f"{avg_c_pct:.1f}%"
                c_row['delta'] = f"{t_c_pct - avg_c_pct:+.1f}pp"
                
                # FAS $ Day 7
                p_fas_d = prev_data['fas_day7_dollars'].sum()
                avg_fas_d = p_fas_d / len(prev_6_weeks) if prev_6_weeks else 0
                t_fas_d = target_data['fas_day7_dollars'].sum() if not target_data.empty else 0
                fas_d_row['prev_avg'] = f"${avg_fas_d:,.0f}"
                fas_d_delta = t_fas_d - avg_fas_d
                fas_d_row['delta'] = f"+${fas_d_delta:,.0f}" if fas_d_delta > 0 else (f"-${abs(fas_d_delta):,.0f}" if fas_d_delta < 0 else "$0")
                
                for i, week in enumerate(weeks_for_table):
                    week_data = lvc_data[(lvc_data['starring'] == star) & (lvc_data['week_start'] == week)]
                    
                    # Assigned Leads
                    count = int(week_data['lead_count'].sum()) if not week_data.empty else 0
                    lvc_week_total = lvc_data[lvc_data['week_start'] == week]['lead_count'].sum()
                    pct = (count / lvc_week_total * 100) if lvc_week_total > 0 else 0
                    d_row[f'wk_{i}'] = f"{count:,} ({pct:.0f}%)"
                    
                    # FAS Day 7 % (FAS Day 7 / Sent to Sales)
                    f7 = int(week_data['fas_day7_qty'].sum()) if not week_data.empty else 0
                    sts = int(week_data['sts_count'].sum()) if not week_data.empty else 0
                    f7_pct = (f7 / sts * 100) if sts > 0 else 0
                    f7_row[f'wk_{i}'] = f"{f7_pct:.1f}%"
                    
                    # SF Contact % (Contacted / Assigned)
                    contacted = int(week_data['contacted_qty'].sum()) if not week_data.empty else 0
                    c_pct = (contacted / count * 100) if count > 0 else 0
                    c_row[f'wk_{i}'] = f"{c_pct:.1f}%"
                    
                    # FAS $ Day 7
                    fas_d = week_data['fas_day7_dollars'].sum() if not week_data.empty else 0
                    fas_d_row[f'wk_{i}'] = f"${fas_d:,.0f}"
                    
                dist_rows.append(d_row)
                fas_day7_rows.append(f7_row)
                contact_rows.append(c_row)
                fas_dollars_rows.append(fas_d_row)
        
        dist_columns = [
            {'name': 'LVC', 'id': 'lvc_group'},
            {'name': '⭐', 'id': 'starring'},
            {'name': '6Wk Avg', 'id': 'prev_avg'},
            {'name': 'Δ vs Avg', 'id': 'delta'},
        ] + [{'name': label, 'id': f'wk_{i}'} for i, label in enumerate(week_labels)]
        
        # Star-only (no LVC) table: FAS % Day 7 by starring over weeks vs 6wk avg
        df_by_star_v = df_starring.groupby(['week_start', 'starring']).agg({
            'sts_count': 'sum', 'fas_day7_qty': 'sum'
        }).reset_index()
        df_by_star_v['fas_rate_d7'] = df_by_star_v['fas_day7_qty'] / df_by_star_v['sts_count'].replace(0, 1) * 100
        star_overall_rows_v = []
        for star in star_order_list:
            star_data = df_by_star_v[df_by_star_v['starring'] == star]
            prev_data = star_data[star_data['week_start'].isin(prev_6_weeks)]
            target_data = star_data[star_data['week_start'] == target_week_dt]
            avg_rate = prev_data['fas_rate_d7'].mean() if not prev_data.empty and len(prev_6_weeks) else 0
            t_rate = target_data['fas_rate_d7'].iloc[0] if not target_data.empty else 0
            row = {'starring': star, 'prev_avg': f"{avg_rate:.1f}%", 'delta': f"{t_rate - avg_rate:+.1f}pp"}
            for i, week in enumerate(weeks_for_table):
                wd = star_data[star_data['week_start'] == week]
                row[f'wk_{i}'] = f"{wd['fas_rate_d7'].iloc[0]:.1f}%" if not wd.empty and len(wd) > 0 else "—"
            star_overall_rows_v.append(row)
        # Star-only: Avg FAS $ (overall) by starring over weeks vs 6wk avg
        df_by_star_avg_v = df_starring.groupby(['week_start', 'starring']).agg({
            'fas_dollars': 'sum', 'fas_count': 'sum'
        }).reset_index()
        df_by_star_avg_v['avg_fas'] = df_by_star_avg_v['fas_dollars'] / df_by_star_avg_v['fas_count'].replace(0, 1)
        star_avg_fas_rows_v = []
        for star in star_order_list:
            star_data = df_by_star_avg_v[df_by_star_avg_v['starring'] == star]
            prev_data = star_data[star_data['week_start'].isin(prev_6_weeks)]
            target_data = star_data[star_data['week_start'] == target_week_dt]
            avg_val = prev_data['avg_fas'].mean() if not prev_data.empty and len(prev_6_weeks) else 0
            t_val = target_data['avg_fas'].iloc[0] if not target_data.empty else 0
            d = t_val - avg_val
            row = {'starring': star, 'prev_avg': f"${avg_val:,.0f}", 'delta': f"+${d:,.0f}" if d >= 0 else f"-${abs(d):,.0f}"}
            for i, week in enumerate(weeks_for_table):
                wd = star_data[star_data['week_start'] == week]
                row[f'wk_{i}'] = f"${wd['avg_fas'].iloc[0]:,.0f}" if not wd.empty and len(wd) > 0 else "—"
            star_avg_fas_rows_v.append(row)
        star_overall_columns_v = [
            {'name': '⭐', 'id': 'starring'},
            {'name': '6Wk Avg', 'id': 'prev_avg'},
            {'name': 'Δ vs Avg', 'id': 'delta'},
        ] + [{'name': label, 'id': f'wk_{i}'} for i, label in enumerate(week_labels)]
        
        def create_star_overall_table_v(data_rows, title, subtitle):
            return html.Div([
                html.H6(title, style={'color': COLORS['text'], 'marginTop': '0', 'marginBottom': '4px', 'fontSize': '13px', 'fontWeight': '600'}),
                html.P(subtitle, style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '8px', 'fontStyle': 'italic'}),
                dash_table.DataTable(
                    data=data_rows,
                    columns=star_overall_columns_v,
                    style_table={'backgroundColor': COLORS['card_bg'], 'overflowX': 'auto', 'maxHeight': '320px', 'overflowY': 'auto'},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '3px 6px', 'fontSize': '10px', 'minWidth': '40px', 'maxWidth': '70px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px', 'textAlign': 'center'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'starring'}, 'textAlign': 'center', 'fontWeight': '600', 'minWidth': '45px', 'maxWidth': '55px'},
                        {'if': {'column_id': 'prev_avg'}, 'backgroundColor': '#122640', 'fontWeight': '500'},
                        {'if': {'column_id': 'delta'}, 'backgroundColor': '#122640', 'fontWeight': '500'},
                    ],
                    style_data_conditional=[
                        {'if': {'filter_query': '{delta} contains "+"', 'column_id': 'delta'}, 'color': COLORS['success']},
                        {'if': {'filter_query': '{delta} contains "-"', 'column_id': 'delta'}, 'color': COLORS['danger']},
                    ],
                )
            ])
        
        def create_compact_table(data_rows, title, subtitle):
            return html.Div([
                html.H6(title, style={'color': COLORS['text'], 'marginTop': '16px', 'marginBottom': '4px', 'fontSize': '13px', 'fontWeight': '600'}),
                html.P(subtitle, style={'color': COLORS['text_muted'], 'fontSize': '10px', 'marginBottom': '8px', 'fontStyle': 'italic'}),
                dash_table.DataTable(
                    data=data_rows,
                    columns=dist_columns,
                    style_table={'backgroundColor': COLORS['card_bg'], 'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'center', 'padding': '3px 6px', 'fontSize': '10px', 'minWidth': '40px', 'maxWidth': '70px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '10px', 'textAlign': 'center'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'lvc_group'}, 'textAlign': 'center', 'fontWeight': '600', 'minWidth': '50px', 'maxWidth': '60px'},
                        {'if': {'column_id': 'starring'}, 'textAlign': 'center', 'minWidth': '25px', 'maxWidth': '35px'},
                        {'if': {'column_id': 'prev_avg'}, 'backgroundColor': '#122640', 'fontWeight': '500'},
                        {'if': {'column_id': 'delta'}, 'backgroundColor': '#122640', 'fontWeight': '500'},
                    ],
                        style_data_conditional=[
                            {'if': {'filter_query': '{delta} contains "+"', 'column_id': 'delta'}, 'color': COLORS['success']},
                            {'if': {'filter_query': '{delta} contains "-"', 'column_id': 'delta'}, 'color': COLORS['danger']},
                        ],
                )
            ])
        
        vintage_export_data = [
            {"name": "FAS % Day 7 by Starring", "columns": star_overall_columns_v, "rows": star_overall_rows_v},
            {"name": "Avg FAS $ by Starring", "columns": star_overall_columns_v, "rows": star_avg_fas_rows_v},
            {"name": "Lead Distribution LVC x Starring", "columns": dist_columns, "rows": dist_rows},
            {"name": "FAS % Day 7 LVC x Starring", "columns": dist_columns, "rows": fas_day7_rows},
            {"name": "SF Contact % LVC x Starring", "columns": dist_columns, "rows": contact_rows},
            {"name": "FAS $ Day 7 LVC x Starring", "columns": dist_columns, "rows": fas_dollars_rows},
        ]
        download_btn_v = dbc.Button(
            "Download Starring Tables (xlsx)",
            id="vintage-download-starring-btn",
            size="sm",
            style={
                "backgroundColor": COLORS["primary"],
                "border": "none",
                "color": "#fff",
                "marginTop": "8px",
            },
        )
        starring_row = html.Div([
            html.H5("⭐ Starring Distribution (Vintage)", style={'color': COLORS['text'], 'marginBottom': '12px', 'marginTop': '20px', 'fontSize': '16px', 'fontWeight': '600'}),
            html.P("5 Star = Best lead quality | Distribution based on Lead Created Date", style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '12px'}),
            dbc.Row([
                dbc.Col(html.Div(create_star_overall_table_v(star_overall_rows_v, "📈 FAS % Day 7 by Starring (Last 6 Wks)", "FAS Day 7 / StS by star | vs 6wk avg"), style=CARD_STYLE), width=4),
                dbc.Col(html.Div(create_star_overall_table_v(star_avg_fas_rows_v, "💰 Avg FAS $ by Starring (Last 6 Wks)", "Avg loan by star | vs 6wk avg"), style=CARD_STYLE), width=4),
                dbc.Col(html.Div([dcc.Graph(figure=fig_starring)], style=CARD_STYLE), width=2),
                dbc.Col(html.Div([dcc.Graph(figure=fig_lvc_starring)], style=CARD_STYLE), width=2),
            ]),
            dbc.Row([
                dbc.Col(create_compact_table(dist_rows, "📊 Lead Distribution by LVC × Starring (Last 6 Wks)", "Based on Leads Assigned Count"), width=6),
                dbc.Col(create_compact_table(fas_day7_rows, "🚀 FAS % Day 7 by LVC × Starring (Last 6 Wks)", "Based on FAS Day 7 / Sent to Sales"), width=6),
            ], style={'marginTop': '16px'}),
            dbc.Row([
                dbc.Col(create_compact_table(contact_rows, "📞 SF Contact % by LVC × Starring (Last 6 Wks)", "Based on Contacted / Assigned Leads"), width=6),
                dbc.Col(create_compact_table(fas_dollars_rows, "💰 Total FAS $ Day 7 by LVC × Starring (Last 6 Wks)", "Based on FAS Day 7 e_loan_amount"), width=6),
            ], style={'marginTop': '16px'}),
            dbc.Row([dbc.Col(download_btn_v, width=12)], style={'marginTop': '8px'}),
        ])
        
    # Add starring to ma_row
    ma_row = html.Div([ma_row, starring_row])
    
    # === Outreach Metrics (Vintage) ===
    outreach_row = html.Div("Loading outreach metrics...")
    df_outreach = get_outreach_metrics(target_week)
    if not df_outreach.empty:
        df_outreach['week_start'] = pd.to_datetime(df_outreach['week_start'])
        
        # Weekly trends aggregation by vintage
        weekly_outreach = df_outreach.groupby('week_start').agg({
            'total_leads': 'sum',
            'sent_to_sales': 'sum',
            'fas_count': 'sum',
            'total_sms': 'sum',
            'total_sms_before_contact': 'sum',
            'total_call_attempts': 'sum'
        }).reset_index()
        
        weekly_outreach['sms_per_lead'] = weekly_outreach['total_sms'] / weekly_outreach['sent_to_sales'].replace(0, 1)
        weekly_outreach['sms_pre_contact_per_lead'] = weekly_outreach['total_sms_before_contact'] / weekly_outreach['sent_to_sales'].replace(0, 1)
        weekly_outreach['calls_per_lead'] = weekly_outreach['total_call_attempts'] / weekly_outreach['sent_to_sales'].replace(0, 1)
        weekly_outreach['fas_rate'] = weekly_outreach['fas_count'] / weekly_outreach['sent_to_sales'].replace(0, 1) * 100
        weekly_outreach['week_label'] = weekly_outreach['week_start'].dt.strftime('%-m/%d')
        
        # Target week vs prev stats
        target_outreach = weekly_outreach[weekly_outreach['week_start'] == target_date]
        prev_outreach = weekly_outreach[(weekly_outreach['week_start'] < target_date) & 
                                        (weekly_outreach['week_start'] >= target_date - timedelta(weeks=4))]
        
        if len(target_outreach) > 0 and len(prev_outreach) > 0:
            target_stats = target_outreach.iloc[0]
            prev_avg = prev_outreach[['sms_per_lead', 'sms_pre_contact_per_lead', 'calls_per_lead']].mean()
            
            sms_delta = target_stats['sms_per_lead'] - prev_avg['sms_per_lead']
            sms_pre_delta = target_stats['sms_pre_contact_per_lead'] - prev_avg['sms_pre_contact_per_lead']
            calls_delta = target_stats['calls_per_lead'] - prev_avg['calls_per_lead']
        else:
            sms_delta = sms_pre_delta = calls_delta = 0
            target_stats = {'sms_per_lead': 0, 'sms_pre_contact_per_lead': 0, 'calls_per_lead': 0}
        
        # Create trend charts
        fig_sms_trend = go.Figure()
        fig_sms_trend.add_trace(go.Bar(
            x=weekly_outreach['week_label'],
            y=weekly_outreach['sms_per_lead'],
            name='SMS/Lead',
            marker_color=COLORS['chart_cyan'],
            text=weekly_outreach['sms_per_lead'].round(2),
            textposition='outside'
        ))
        fig_sms_trend.update_layout(
            title=dict(text='📱 SMS per Lead by Vintage Week', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=50, r=30, t=50, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False, title='Vintage Week'),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True, range=[0, weekly_outreach['sms_per_lead'].max() * 1.15] if weekly_outreach['sms_per_lead'].max() > 0 else [0, 1]),
            height=280,
            showlegend=False
        )
        
        fig_sms_pre_trend = go.Figure()
        fig_sms_pre_trend.add_trace(go.Bar(
            x=weekly_outreach['week_label'],
            y=weekly_outreach['sms_pre_contact_per_lead'],
            name='SMS Pre-Contact/Lead',
            marker_color=COLORS['chart_green'],
            text=weekly_outreach['sms_pre_contact_per_lead'].round(2),
            textposition='outside'
        ))
        fig_sms_pre_trend.update_layout(
            title=dict(text='📱 SMS Before Contact by Vintage', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=50, r=30, t=50, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False, title='Vintage Week'),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True, range=[0, weekly_outreach['sms_pre_contact_per_lead'].max() * 1.15] if weekly_outreach['sms_pre_contact_per_lead'].max() > 0 else [0, 1]),
            height=280,
            showlegend=False
        )
        
        fig_calls_trend = go.Figure()
        fig_calls_trend.add_trace(go.Bar(
            x=weekly_outreach['week_label'],
            y=weekly_outreach['calls_per_lead'],
            name='Calls/Lead',
            marker_color=COLORS['chart_yellow'],
            text=weekly_outreach['calls_per_lead'].round(2),
            textposition='outside'
        ))
        fig_calls_trend.update_layout(
            title=dict(text='📞 Call Attempts by Vintage Week', font=dict(color=COLORS['text'], size=13)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=COLORS['card_bg'],
            font=dict(color=COLORS['text']),
            margin=dict(l=50, r=30, t=50, b=40),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False, title='Vintage Week'),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True, range=[0, weekly_outreach['calls_per_lead'].max() * 1.15] if weekly_outreach['calls_per_lead'].max() > 0 else [0, 1]),
            height=280,
            showlegend=False
        )
        
        # Top MAs by outreach for this vintage
        target_ma_outreach = df_outreach[df_outreach['week_start'] == target_date].copy()
        if not target_ma_outreach.empty:
            target_ma_outreach['sms_per_lead'] = target_ma_outreach['total_sms'] / target_ma_outreach['sent_to_sales'].replace(0, 1)
            target_ma_outreach['sms_pre_contact_per_lead'] = target_ma_outreach['total_sms_before_contact'] / target_ma_outreach['sent_to_sales'].replace(0, 1)
            target_ma_outreach['calls_per_lead'] = target_ma_outreach['total_call_attempts'] / target_ma_outreach['sent_to_sales'].replace(0, 1)
            target_ma_outreach['fas_rate'] = target_ma_outreach['fas_count'] / target_ma_outreach['sent_to_sales'].replace(0, 1) * 100
            
            ma_with_volume = target_ma_outreach[target_ma_outreach['sent_to_sales'] >= 10].copy()
            top_callers = ma_with_volume.nlargest(10, 'calls_per_lead')[['mortgage_advisor', 'sent_to_sales', 'calls_per_lead', 'sms_per_lead', 'fas_count', 'fas_rate']]
            top_sms = ma_with_volume.nlargest(10, 'sms_per_lead')[['mortgage_advisor', 'sent_to_sales', 'sms_per_lead', 'sms_pre_contact_per_lead', 'fas_count', 'fas_rate']]
        else:
            top_callers = pd.DataFrame()
            top_sms = pd.DataFrame()
        
        outreach_calc_vintage = "Calc: SMS/Lead = SUM(total_sms_outbound_count)/sent_to_sales; SMS Pre-Contact/Lead = SUM(total_sms_outbound_before_contact)/sent_to_sales; Calls/Lead = SUM(call_attempts)/sent_to_sales; by week (lead_created_date), by MA."
        outreach_row = dbc.Row([
            dbc.Col(html.Div(outreach_calc_vintage, style={'fontSize': '11px', 'color': COLORS['text_muted'], 'marginBottom': '8px'}), width=12),
            dbc.Col(create_metric_card("SMS/Lead (Vintage)", f"{target_stats['sms_per_lead']:.2f}", sms_delta, " vs 4wk avg"), width=4),
            dbc.Col(create_metric_card("SMS Pre-Contact/Lead", f"{target_stats['sms_pre_contact_per_lead']:.2f}", sms_pre_delta, " vs 4wk avg"), width=4),
            dbc.Col(create_metric_card("Calls/Lead (Vintage)", f"{target_stats['calls_per_lead']:.2f}", calls_delta, " vs 4wk avg"), width=4),
            
            dbc.Col(html.Div([dcc.Graph(figure=fig_sms_trend)], style=CARD_STYLE), width=4),
            dbc.Col(html.Div([dcc.Graph(figure=fig_sms_pre_trend)], style=CARD_STYLE), width=4),
            dbc.Col(html.Div([dcc.Graph(figure=fig_calls_trend)], style=CARD_STYLE), width=4),
            
            dbc.Col(html.Div([
                html.H5("Top MAs by Calls/Lead for This Vintage (min 10 StS)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=top_callers.round(2).to_dict('records') if not top_callers.empty else [],
                    columns=[
                        {'name': 'MA', 'id': 'mortgage_advisor'},
                        {'name': 'StS', 'id': 'sent_to_sales', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'Calls/Lead', 'id': 'calls_per_lead', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'SMS/Lead', 'id': 'sms_per_lead', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS %', 'id': 'fas_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg']},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '6px', 'fontSize': '12px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '11px'},
                    page_size=10
                )
            ], style=CARD_STYLE), width=6),
            
            dbc.Col(html.Div([
                html.H5("Top MAs by SMS/Lead for This Vintage (min 10 StS)", style={'color': COLORS['text'], 'marginBottom': '12px', 'fontSize': '14px', 'fontWeight': '600'}),
                dash_table.DataTable(
                    data=top_sms.round(2).to_dict('records') if not top_sms.empty else [],
                    columns=[
                        {'name': 'MA', 'id': 'mortgage_advisor'},
                        {'name': 'StS', 'id': 'sent_to_sales', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'SMS/Lead', 'id': 'sms_per_lead', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'SMS Pre-Contact', 'id': 'sms_pre_contact_per_lead', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'FAS', 'id': 'fas_count', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                        {'name': 'FAS %', 'id': 'fas_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    ],
                    style_table={'backgroundColor': COLORS['card_bg']},
                    style_cell={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}", 'textAlign': 'left', 'padding': '6px', 'fontSize': '12px'},
                    style_header={'backgroundColor': COLORS['border'], 'fontWeight': 'bold', 'fontSize': '11px'},
                    page_size=10
                )
            ], style=CARD_STYLE), width=6),
        ])
    
    return kpi_row, lvc_row, channel_row, persona_row, outreach_row, ma_row, vintage_export_data

# === Collapse Callbacks for Vintage Deeper Dive ===
@callback(
    Output({'type': 'collapse', 'index': ALL}, 'is_open', allow_duplicate=True),
    [Input('expand-all-vintage', 'n_clicks'),
     Input('collapse-all-vintage', 'n_clicks')],
    [State({'type': 'collapse', 'index': ALL}, 'is_open')],
    prevent_initial_call=True
)
def expand_collapse_all_vintage(expand_clicks, collapse_clicks, current_states):
    ctx = callback_context
    if not ctx.triggered:
        return current_states
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'expand-all-vintage':
        return [True] * len(current_states)
    elif button_id == 'collapse-all-vintage':
        return [False] * len(current_states)
    return current_states

# === Collapse Callbacks for In-Period Deeper Dive ===
@callback(
    Output({'type': 'collapse', 'index': ALL}, 'is_open', allow_duplicate=True),
    [Input('expand-all-inperiod', 'n_clicks'),
     Input('collapse-all-inperiod', 'n_clicks')],
    [State({'type': 'collapse', 'index': ALL}, 'is_open')],
    prevent_initial_call=True
)
def expand_collapse_all_inperiod(expand_clicks, collapse_clicks, current_states):
    ctx = callback_context
    if not ctx.triggered:
        return current_states
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'expand-all-inperiod':
        return [True] * len(current_states)
    elif button_id == 'collapse-all-inperiod':
        return [False] * len(current_states)
    return current_states

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
            /* Tabs styling */
            .tab-container {
                border-bottom: 1px solid #1a3a5c;
            }
            .custom-tabs .tab {
                border: none !important;
                background: transparent !important;
            }
            /* Dropdown styling */
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
            /* Dash core dropdown single value text */
            .Select-single-value,
            .dash-dropdown .Select-single-value,
            .VirtualizedSelectOption,
            .VirtualizedSelectFocusedOption {
                color: #1a1a2e !important;
                font-weight: 600 !important;
            }
            /* New Dash dropdown classes */
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
            /* Table styling */
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
            /* Scrollbar styling */
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
            /* Card hover effect */
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

# Expand/Collapse All for Overview tab
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

# Expand/Collapse All for Deeper Dive tab
@callback(
    [Output({'type': 'collapse-content', 'index': 'deepdive-kpi'}, 'is_open', allow_duplicate=True),
     Output({'type': 'collapse-content', 'index': 'lvc-analysis'}, 'is_open', allow_duplicate=True),
     Output({'type': 'collapse-content', 'index': 'channel-analysis'}, 'is_open', allow_duplicate=True),
     Output({'type': 'collapse-content', 'index': 'ma-analysis'}, 'is_open', allow_duplicate=True),
     Output({'type': 'collapse-content', 'index': 'vintage-deep'}, 'is_open', allow_duplicate=True),
     Output({'type': 'collapse-content', 'index': 'daily-breakout'}, 'is_open', allow_duplicate=True)],
    [Input('expand-all-deepdive', 'n_clicks'),
     Input('collapse-all-deepdive', 'n_clicks')],
    prevent_initial_call=True
)
def expand_collapse_all_deepdive(expand_clicks, collapse_clicks):
    """Expand or collapse all deeper dive sections"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return [True] * 6
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'expand-all-deepdive':
        return [True] * 6
    else:
        return [False] * 6

# === Run Server ===
if __name__ == '__main__':
    app.run(debug=True, port=8050)
