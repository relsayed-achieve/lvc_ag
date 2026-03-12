"""
Sales Overview Dashboard
------------------------
A futuristic dark-themed Dash dashboard for sales funnel analysis.

Funnel Stages:
1. Gross Leads (count distinct lendage_guid)
2. Eligible Leads (loan_derived_status = 'Eligible')
3. Sent to Sales (sent_to_sales_date is not null)
4. Leads Assigned (current_sales_assigned_date is not null)
5. Contacted (sf__contacted_guid is not null)
6. FAS (full_app_submit_datetime is not null)
7. Funded (funding_end_datetime is not null)

Tabs: Overall Metrics, Daily, Weekly, Monthly
Insights by: lvc_group, persona, mortgage_advisors
"""

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
from google.cloud import bigquery
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from functools import lru_cache
import json

# Try to import Vertex AI for Q&A feature
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    vertexai.init(project="ffn-dw-bigquery-prd", location="us-central1")
    gemini_model = GenerativeModel("gemini-1.5-flash")
    AI_AVAILABLE = True
except Exception as e:
    AI_AVAILABLE = False
    gemini_model = None
    print(f"Vertex AI not available: {e}")

# --- Initialize BigQuery Client ---
try:
    project_id = "ffn-dw-bigquery-prd"
    client = bigquery.Client(project=project_id)
    CONNECTION_STATUS = "✅ Connected"
except Exception as e:
    client = None
    CONNECTION_STATUS = f"❌ Not Connected: {e}"

# --- Initialize Dash App ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True
)
app.title = "Sales Overview Dashboard"

# --- Custom CSS for Dark Futuristic Theme ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --bg-primary: #0a0e17;
                --bg-secondary: #0d1421;
                --bg-card: linear-gradient(145deg, #0d1b2a 0%, #162a40 100%);
                --border-glow: rgba(76, 201, 240, 0.3);
                --text-primary: #ffffff;
                --text-secondary: #8892b0;
                --accent-cyan: #4cc9f0;
                --accent-blue: #4361ee;
                --accent-purple: #7209b7;
                --accent-pink: #f72585;
                --accent-green: #00d084;
                --accent-yellow: #f9c74f;
                --accent-orange: #ff7f0e;
            }
            
            body {
                background: var(--bg-primary) !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .dashboard-container {
                background: linear-gradient(135deg, #0a1628 0%, #1a2744 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .metric-card {
                background: var(--bg-card);
                border: 1px solid var(--border-glow);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 15px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
            }
            
            .metric-card:hover {
                border-color: rgba(76, 201, 240, 0.6);
                box-shadow: 0 6px 30px rgba(76, 201, 240, 0.2);
                transform: translateY(-2px);
            }
            
            .card-title {
                color: var(--accent-cyan);
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 10px;
            }
            
            .kpi-value {
                color: var(--text-primary);
                font-size: 32px;
                font-weight: 700;
                margin: 0;
            }
            
            .kpi-value-small {
                color: var(--text-primary);
                font-size: 24px;
                font-weight: 600;
            }
            
            .kpi-label {
                color: var(--text-secondary);
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .badge-up {
                background: linear-gradient(135deg, #00d084 0%, #00b371 100%);
                color: white;
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 11px;
                font-weight: 600;
                margin-left: 8px;
            }
            
            .badge-down {
                background: linear-gradient(135deg, #f72585 0%, #b5179e 100%);
                color: white;
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 11px;
                font-weight: 600;
                margin-left: 8px;
            }
            
            .badge-neutral {
                background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
                color: white;
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 11px;
                font-weight: 600;
                margin-left: 8px;
            }
            
            .funnel-stage {
                background: linear-gradient(90deg, rgba(76, 201, 240, 0.1) 0%, rgba(76, 201, 240, 0.05) 100%);
                border-left: 3px solid var(--accent-cyan);
                padding: 12px 15px;
                margin-bottom: 8px;
                border-radius: 0 8px 8px 0;
            }
            
            .insight-box {
                background: rgba(76, 201, 240, 0.1);
                border: 1px solid rgba(76, 201, 240, 0.2);
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 10px;
            }
            
            .nav-tabs .nav-link {
                color: var(--text-secondary) !important;
                border: none !important;
                background: transparent !important;
                padding: 12px 20px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .nav-tabs .nav-link.active {
                color: var(--accent-cyan) !important;
                border-bottom: 2px solid var(--accent-cyan) !important;
                background: rgba(76, 201, 240, 0.1) !important;
            }
            
            .nav-tabs .nav-link:hover {
                color: var(--text-primary) !important;
            }
            
            .filter-panel {
                background: var(--bg-card);
                border: 1px solid var(--border-glow);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
            }
            
            .section-divider {
                border-top: 1px solid rgba(76, 201, 240, 0.2);
                margin: 25px 0;
            }
            
            .progress-bar-custom {
                height: 8px;
                background: rgba(255,255,255,0.1);
                border-radius: 4px;
                overflow: hidden;
            }
            
            .progress-fill {
                height: 100%;
                border-radius: 4px;
                transition: width 0.5s ease;
            }
            
            /* DataTable styling */
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                background-color: #0d1b2a !important;
                color: #ffffff !important;
                border-color: rgba(76, 201, 240, 0.2) !important;
            }
            
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                background-color: #162a40 !important;
                font-weight: 600 !important;
            }
            
            /* Dropdown styling */
            .Select-control {
                background-color: #162a40 !important;
                border-color: rgba(76, 201, 240, 0.3) !important;
            }
            
            .Select-value-label {
                color: #ffffff !important;
            }
            
            .Select-menu-outer {
                background-color: #162a40 !important;
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

# --- Dark Theme for Plotly Charts ---
DARK_THEME = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': '#ffffff', 'family': 'Segoe UI'},
    'xaxis': {
        'gridcolor': 'rgba(255,255,255,0.05)',
        'tickfont': {'color': '#8892b0'},
        'titlefont': {'color': '#8892b0'}
    },
    'yaxis': {
        'gridcolor': 'rgba(255,255,255,0.05)',
        'tickfont': {'color': '#8892b0'},
        'titlefont': {'color': '#8892b0'}
    },
    'legend': {'font': {'color': '#8892b0'}}
}

# Color palette for charts
COLORS = {
    'cyan': '#4cc9f0',
    'blue': '#4361ee',
    'purple': '#7209b7',
    'pink': '#f72585',
    'green': '#00d084',
    'yellow': '#f9c74f',
    'orange': '#ff7f0e',
    'red': '#d62728'
}

FUNNEL_COLORS = ['#4cc9f0', '#4895ef', '#4361ee', '#7209b7', '#b5179e', '#f72585', '#ff7f0e']

# --- Helper Functions ---
def run_query(sql_query):
    """Execute BigQuery and return DataFrame"""
    if client is None:
        return pd.DataFrame()
    try:
        return client.query(sql_query).to_dataframe()
    except Exception as e:
        print(f"Query error: {e}")
        return pd.DataFrame()

def format_number(num):
    """Format large numbers with K/M suffix"""
    if pd.isna(num):
        return "--"
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:,.0f}"

def format_pct(num):
    """Format percentage"""
    if pd.isna(num):
        return "--"
    return f"{num:.1%}"

def get_change_badge(current, previous):
    """Return a badge component showing period-over-period change"""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return html.Span("--", className="badge-neutral")
    
    pct_change = (current - previous) / previous
    if pct_change > 0:
        return html.Span(f"↑ {pct_change:.1%}", className="badge-up")
    elif pct_change < 0:
        return html.Span(f"↓ {abs(pct_change):.1%}", className="badge-down")
    else:
        return html.Span("→ 0%", className="badge-neutral")

def build_where_clause(start_date, end_date):
    """Build SQL WHERE clause for date filtering"""
    conditions = []
    if start_date and end_date:
        conditions.append(f"DATE(lead_created_date) BETWEEN '{start_date}' AND '{end_date}'")
    if conditions:
        return "WHERE " + " AND ".join(conditions)
    return ""

# --- SQL Query for Funnel Data ---
def get_funnel_query(start_date, end_date, group_by_cols=[]):
    """Generate SQL query for funnel metrics"""
    where_clause = build_where_clause(start_date, end_date)
    
    # Build group by clause
    group_by_sql = ""
    select_cols = ""
    if group_by_cols:
        select_cols = ", ".join(group_by_cols) + ","
        group_by_sql = "GROUP BY " + ", ".join([str(i+1) for i in range(len(group_by_cols))])
    
    query = f"""
    SELECT 
        {select_cols}
        -- Funnel Metrics
        COUNT(DISTINCT lendage_guid) as gross_leads,
        COUNT(DISTINCT CASE WHEN loan_derived_status = 'Eligible' THEN lendage_guid END) as eligible_leads,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales,
        COUNT(DISTINCT CASE WHEN current_sales_assigned_date IS NOT NULL THEN lendage_guid END) as leads_assigned,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas,
        COUNT(DISTINCT CASE WHEN funding_end_datetime IS NOT NULL THEN lendage_guid END) as funded,
        -- Volume Metrics
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_volume,
        SUM(CASE WHEN funding_end_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as funded_volume
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    {where_clause}
    {group_by_sql}
    """
    return query

def get_time_series_query(start_date, end_date, time_grain='day'):
    """Generate SQL query for time series data"""
    where_clause = build_where_clause(start_date, end_date)
    
    if time_grain == 'day':
        date_trunc = "DATE(lead_created_date)"
    elif time_grain == 'week':
        date_trunc = "DATE_TRUNC(DATE(lead_created_date), WEEK(MONDAY))"
    else:  # month
        date_trunc = "DATE_TRUNC(DATE(lead_created_date), MONTH)"
    
    query = f"""
    SELECT 
        {date_trunc} as period_date,
        -- Funnel Metrics
        COUNT(DISTINCT lendage_guid) as gross_leads,
        COUNT(DISTINCT CASE WHEN loan_derived_status = 'Eligible' THEN lendage_guid END) as eligible_leads,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales,
        COUNT(DISTINCT CASE WHEN current_sales_assigned_date IS NOT NULL THEN lendage_guid END) as leads_assigned,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas,
        COUNT(DISTINCT CASE WHEN funding_end_datetime IS NOT NULL THEN lendage_guid END) as funded,
        -- Volume Metrics
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_volume,
        SUM(CASE WHEN funding_end_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as funded_volume
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    {where_clause}
    GROUP BY 1
    ORDER BY 1
    """
    return query

def get_breakdown_query(start_date, end_date, dimension):
    """Generate SQL query for breakdown by dimension - based on Sent to Sales leads"""
    where_clause = build_where_clause(start_date, end_date)
    
    # Add filter for sent_to_sales for LVC and Persona breakdowns
    sts_filter = "AND sent_to_sales_date IS NOT NULL"
    
    # MA exclusion filter (only applied for mortgage_advisors dimension)
    ma_exclusion = ""
    
    if dimension == 'lvc_group':
        dim_sql = """CASE 
            WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
            WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
            WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
            WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
            ELSE 'Other'
        END"""
    elif dimension == 'persona':
        dim_sql = "COALESCE(persona, 'Unknown')"
    else:  # mortgage_advisors
        dim_sql = "COALESCE(mortgage_advisor, 'Unassigned')"
        ma_exclusion = """
        AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%jordan%lee%'
        AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%nigel%'
        AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%childress%'
        AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%jared%'
        AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%jacob%'
        """
    
    query = f"""
    SELECT 
        {dim_sql} as dimension_value,
        COUNT(DISTINCT lendage_guid) as sent_to_sales,
        COUNT(DISTINCT CASE WHEN current_sales_assigned_date IS NOT NULL THEN lendage_guid END) as leads_assigned,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas,
        COUNT(DISTINCT CASE WHEN funding_end_datetime IS NOT NULL THEN lendage_guid END) as funded,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_volume,
        SUM(CASE WHEN funding_end_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as funded_volume
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    {where_clause}
    {sts_filter}
    {ma_exclusion}
    GROUP BY 1
    ORDER BY sent_to_sales DESC
    LIMIT 20
    """
    return query

# MAs to exclude from analysis
EXCLUDED_MAS = ['Jordan Lee', 'Nigel', 'Childress', 'Jared', 'Jacob']
EXCLUDED_MAS_SQL = ", ".join([f"'{ma}'" for ma in EXCLUDED_MAS])

def get_ma_insights_query(start_date, end_date):
    """Generate SQL query for detailed MA insights - based on assigned leads"""
    where_clause = build_where_clause(start_date, end_date)
    
    query = f"""
    SELECT 
        COALESCE(mortgage_advisor, 'Unassigned') as ma_name,
        COUNT(DISTINCT lendage_guid) as assigned_leads,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas,
        COUNT(DISTINCT CASE WHEN funding_end_datetime IS NOT NULL THEN lendage_guid END) as funded,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_volume,
        SUM(CASE WHEN funding_end_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as funded_volume,
        AVG(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount END) as avg_fas_loan,
        AVG(CASE WHEN funding_end_datetime IS NOT NULL THEN e_loan_amount END) as avg_funded_loan
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    {where_clause}
    AND current_sales_assigned_date IS NOT NULL
    AND COALESCE(mortgage_advisor, 'Unassigned') NOT IN ({EXCLUDED_MAS_SQL})
    AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%jordan%lee%'
    AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%nigel%'
    AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%childress%'
    AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%jared%'
    AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%jacob%'
    GROUP BY 1
    HAVING COUNT(DISTINCT lendage_guid) >= 5
    ORDER BY assigned_leads DESC
    """
    return query

def get_ma_time_series_query(start_date, end_date, time_grain='week'):
    """Generate SQL query for MA time series - based on assigned leads"""
    where_clause = build_where_clause(start_date, end_date)
    
    if time_grain == 'day':
        date_trunc = "DATE(lead_created_date)"
    elif time_grain == 'week':
        date_trunc = "DATE_TRUNC(DATE(lead_created_date), WEEK(MONDAY))"
    else:  # month
        date_trunc = "DATE_TRUNC(DATE(lead_created_date), MONTH)"
    
    query = f"""
    SELECT 
        {date_trunc} as period_date,
        COALESCE(mortgage_advisor, 'Unassigned') as ma_name,
        COUNT(DISTINCT lendage_guid) as assigned_leads,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas,
        COUNT(DISTINCT CASE WHEN funding_end_datetime IS NOT NULL THEN lendage_guid END) as funded,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_volume
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    {where_clause}
    AND current_sales_assigned_date IS NOT NULL
    AND COALESCE(mortgage_advisor, 'Unassigned') NOT IN ({EXCLUDED_MAS_SQL})
    AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%jordan%lee%'
    AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%nigel%'
    AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%childress%'
    AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%jared%'
    AND LOWER(COALESCE(mortgage_advisor, '')) NOT LIKE '%jacob%'
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    return query

# --- Component Builders ---
def build_kpi_card(title, value, change_badge=None, sub_text=None):
    """Build a KPI metric card"""
    return html.Div([
        html.Div(title, className="card-title"),
        html.Div([
            html.Span(value, className="kpi-value"),
            change_badge if change_badge else None
        ]),
        html.Div(sub_text, className="kpi-label") if sub_text else None
    ], className="metric-card")

def build_funnel_stage(label, value, pct_of_total, color):
    """Build a funnel stage row"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Span(label, style={'color': '#ffffff', 'fontWeight': '500'}),
            ], width=4),
            dbc.Col([
                html.Span(format_number(value), style={'color': color, 'fontWeight': '600', 'fontSize': '18px'}),
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div(style={
                        'width': f'{min(pct_of_total * 100, 100)}%',
                        'height': '8px',
                        'backgroundColor': color,
                        'borderRadius': '4px'
                    })
                ], style={'backgroundColor': 'rgba(255,255,255,0.1)', 'borderRadius': '4px', 'height': '8px'})
            ], width=4),
            dbc.Col([
                html.Span(format_pct(pct_of_total), style={'color': '#8892b0', 'fontSize': '12px'})
            ], width=1)
        ], align="center")
    ], className="funnel-stage")

def create_funnel_chart(data):
    """Create a horizontal funnel chart"""
    stages = ['Gross Leads', 'Eligible', 'Sent to Sales', 'Assigned', 'Contacted', 'FAS', 'Funded']
    values = [
        data.get('gross_leads', 0),
        data.get('eligible_leads', 0),
        data.get('sent_to_sales', 0),
        data.get('leads_assigned', 0),
        data.get('contacted', 0),
        data.get('fas', 0),
        data.get('funded', 0)
    ]
    
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=FUNNEL_COLORS),
        connector=dict(line=dict(color="rgba(76, 201, 240, 0.3)", width=1))
    ))
    
    fig.update_layout(
        **DARK_THEME,
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )
    
    return fig

def create_trend_chart(df, metric, title, is_rate=False):
    """Create a trend line chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['period_date'],
        y=df[metric],
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(76, 201, 240, 0.2)',
        line=dict(color=COLORS['cyan'], width=2),
        marker=dict(size=6, color=COLORS['cyan']),
        name=title
    ))
    
    fig.update_layout(
        **DARK_THEME,
        title=dict(text=title, font=dict(size=14, color='#4cc9f0')),
        margin=dict(l=40, r=20, t=50, b=40),
        height=250,
        showlegend=False
    )
    
    if is_rate:
        fig.update_yaxes(tickformat='.1%')
    
    return fig

def create_breakdown_bar_chart(df, title, metric='sent_to_sales'):
    """Create a horizontal bar chart for breakdown - based on sent_to_sales"""
    fig = go.Figure()
    
    # Sort by metric descending
    df_sorted = df.sort_values(metric, ascending=True).tail(10)
    
    fig.add_trace(go.Bar(
        y=df_sorted['dimension_value'],
        x=df_sorted[metric],
        orientation='h',
        marker=dict(
            color=df_sorted[metric],
            colorscale=[[0, '#1b3a4b'], [0.5, '#4895ef'], [1, '#4cc9f0']],
            line=dict(width=0)
        ),
        text=df_sorted[metric].apply(lambda x: format_number(x)),
        textposition='inside',
        textfont=dict(color='white')
    ))
    
    fig.update_layout(
        **DARK_THEME,
        title=dict(text=title, font=dict(size=14, color='#4cc9f0')),
        margin=dict(l=120, r=20, t=50, b=40),
        height=300,
        showlegend=False
    )
    
    return fig

def create_conversion_heatmap(df, title):
    """Create a conversion rates heatmap - based on sent_to_sales funnel"""
    # Calculate conversion rates from sent_to_sales base
    df_rates = df.copy()
    df_rates['assign_rate'] = df_rates['leads_assigned'] / df_rates['sent_to_sales'].replace(0, np.nan)
    df_rates['contact_rate'] = df_rates['contacted'] / df_rates['leads_assigned'].replace(0, np.nan)
    df_rates['fas_rate'] = df_rates['fas'] / df_rates['contacted'].replace(0, np.nan)
    df_rates['funded_rate'] = df_rates['funded'] / df_rates['fas'].replace(0, np.nan)
    df_rates['overall_conv'] = df_rates['fas'] / df_rates['sent_to_sales'].replace(0, np.nan)
    
    rate_cols = ['assign_rate', 'contact_rate', 'fas_rate', 'funded_rate', 'overall_conv']
    rate_labels = ['Assign Rate', 'Contact Rate', 'FAS Rate', 'Pull Through', 'Overall Conv']
    
    z_data = df_rates[rate_cols].fillna(0).values
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=rate_labels,
        y=df_rates['dimension_value'].tolist(),
        colorscale=[[0, '#d73027'], [0.25, '#f46d43'], [0.5, '#fee08b'], [0.75, '#a6d96a'], [1, '#1a9850']],
        text=[[f"{v:.1%}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        hovertemplate='%{y}<br>%{x}: %{z:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        **DARK_THEME,
        title=dict(text=title, font=dict(size=14, color='#4cc9f0')),
        margin=dict(l=120, r=20, t=50, b=40),
        height=350
    )
    
    return fig

def create_donut_chart(labels, values, title):
    """Create a donut chart"""
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=[COLORS['cyan'], COLORS['blue'], COLORS['purple'], COLORS['pink'], COLORS['orange']],
                   line=dict(color='#0d1b2a', width=2)),
        textinfo='percent',
        textfont=dict(size=10, color='white'),
        hovertemplate='%{label}<br>%{value:,}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff', 'family': 'Segoe UI'},
        title=dict(text=title, font=dict(size=14, color='#4cc9f0')),
        margin=dict(l=20, r=20, t=50, b=20),
        height=280,
        showlegend=True,
        legend=dict(font=dict(size=9, color='#8892b0'), orientation='h', y=-0.1)
    )
    
    return fig

# --- Layout Components ---

# Filter Panel
filter_panel = html.Div([
    dbc.Row([
        dbc.Col([
            html.Label("Date Range", style={'color': '#8892b0', 'fontSize': '12px', 'marginBottom': '5px'}),
            dcc.DatePickerRange(
                id='date-range-filter',
                min_date_allowed=date(2024, 1, 1),
                max_date_allowed=date.today(),
                start_date=date(2025, 10, 1),
                end_date=date.today(),
                style={'backgroundColor': '#162a40'}
            )
        ], width=4),
        dbc.Col([
            html.Label("Connection Status", style={'color': '#8892b0', 'fontSize': '12px', 'marginBottom': '5px'}),
            html.Div(CONNECTION_STATUS, style={'color': '#4cc9f0', 'fontSize': '14px', 'marginTop': '10px'})
        ], width=3),
        dbc.Col([
            html.Label("Last Updated", style={'color': '#8892b0', 'fontSize': '12px', 'marginBottom': '5px'}),
            html.Div(datetime.now().strftime("%Y-%m-%d %H:%M"), 
                    style={'color': '#ffffff', 'fontSize': '14px', 'marginTop': '10px'})
        ], width=2),
        dbc.Col([
            dbc.Button("Refresh Data", id="refresh-btn", color="info", outline=True, size="sm",
                      style={'marginTop': '20px'})
        ], width=3, style={'textAlign': 'right'})
    ])
], className="filter-panel")

# --- Tab Contents ---

# Tab 1: Overall Metrics
tab_overall = html.Div([
    # KPI Row
    dbc.Row([
        dbc.Col(html.Div(id='kpi-gross-leads'), width=2),
        dbc.Col(html.Div(id='kpi-eligible'), width=2),
        dbc.Col(html.Div(id='kpi-sent-to-sales'), width=2),
        dbc.Col(html.Div(id='kpi-assigned'), width=1),
        dbc.Col(html.Div(id='kpi-contacted'), width=2),
        dbc.Col(html.Div(id='kpi-fas'), width=2),
        dbc.Col(html.Div(id='kpi-funded'), width=1),
    ], className="mb-4"),
    
    # Funnel and Distribution
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Sales Funnel", className="card-title"),
                dcc.Graph(id='funnel-chart', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=5),
        dbc.Col([
            html.Div([
                html.Div("Conversion Rates", className="card-title"),
                html.Div(id='conversion-rates-display')
            ], className="metric-card")
        ], width=3),
        dbc.Col([
            html.Div([
                html.Div("Volume Distribution", className="card-title"),
                dcc.Graph(id='volume-distribution-chart', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=4),
    ], className="mb-4"),
    
    # Breakdowns Row
    html.Div(className="section-divider"),
    html.H5("Performance Breakdown (Based on Sent to Sales)", style={'color': '#4cc9f0', 'marginBottom': '20px'}),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Sent to Sales by LVC Group", className="card-title"),
                dcc.Graph(id='lvc-breakdown-chart', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=4),
        dbc.Col([
            html.Div([
                html.Div("Sent to Sales by Persona", className="card-title"),
                dcc.Graph(id='persona-breakdown-chart', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=4),
        dbc.Col([
            html.Div([
                html.Div("Sent to Sales by Mortgage Advisor", className="card-title"),
                dcc.Graph(id='ma-breakdown-chart', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=4),
    ], className="mb-4"),
    
    # Conversion Heatmap
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Conversion Rates by LVC Group (from Sent to Sales)", className="card-title"),
                dcc.Graph(id='conversion-heatmap', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=12)
    ])
])

# Tab 2: Daily View
tab_daily = html.Div([
    dbc.Row([
        dbc.Col([
            html.H5("Daily Performance", style={'color': '#4cc9f0'}),
            html.P("Comparing to previous day", style={'color': '#8892b0', 'fontSize': '12px'})
        ], width=8),
        dbc.Col([
            html.Div([
                html.Label("Days to Compare", style={'color': '#8892b0', 'fontSize': '11px'}),
                dcc.Dropdown(
                    id='daily-compare-days',
                    options=[
                        {'label': 'Last 7 Days', 'value': 7},
                        {'label': 'Last 14 Days', 'value': 14},
                        {'label': 'Last 30 Days', 'value': 30}
                    ],
                    value=14,
                    style={'backgroundColor': '#162a40'}
                )
            ])
        ], width=4)
    ], className="mb-4"),
    
    # Daily KPIs with comparison
    dbc.Row([
        dbc.Col(html.Div(id='daily-kpi-leads'), width=2),
        dbc.Col(html.Div(id='daily-kpi-eligible'), width=2),
        dbc.Col(html.Div(id='daily-kpi-sts'), width=2),
        dbc.Col(html.Div(id='daily-kpi-contacted'), width=2),
        dbc.Col(html.Div(id='daily-kpi-fas'), width=2),
        dbc.Col(html.Div(id='daily-kpi-funded'), width=2),
    ], className="mb-4"),
    
    # Daily Trends
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Daily Gross Leads Trend", className="card-title"),
                dcc.Graph(id='daily-leads-trend', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=6),
        dbc.Col([
            html.Div([
                html.Div("Daily FAS Trend", className="card-title"),
                dcc.Graph(id='daily-fas-trend', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Daily Conversion Rates", className="card-title"),
                dcc.Graph(id='daily-conversion-trend', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=12)
    ])
])

# Tab 3: Weekly View
tab_weekly = html.Div([
    dbc.Row([
        dbc.Col([
            html.H5("Weekly Performance", style={'color': '#4cc9f0'}),
            html.P("Comparing to previous week", style={'color': '#8892b0', 'fontSize': '12px'})
        ], width=8),
        dbc.Col([
            html.Div([
                html.Label("Weeks to Compare", style={'color': '#8892b0', 'fontSize': '11px'}),
                dcc.Dropdown(
                    id='weekly-compare-weeks',
                    options=[
                        {'label': 'Last 4 Weeks', 'value': 4},
                        {'label': 'Last 8 Weeks', 'value': 8},
                        {'label': 'Last 12 Weeks', 'value': 12}
                    ],
                    value=8,
                    style={'backgroundColor': '#162a40'}
                )
            ])
        ], width=4)
    ], className="mb-4"),
    
    # Weekly KPIs with comparison
    dbc.Row([
        dbc.Col(html.Div(id='weekly-kpi-leads'), width=2),
        dbc.Col(html.Div(id='weekly-kpi-eligible'), width=2),
        dbc.Col(html.Div(id='weekly-kpi-sts'), width=2),
        dbc.Col(html.Div(id='weekly-kpi-contacted'), width=2),
        dbc.Col(html.Div(id='weekly-kpi-fas'), width=2),
        dbc.Col(html.Div(id='weekly-kpi-funded'), width=2),
    ], className="mb-4"),
    
    # Weekly Trends
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Weekly Funnel Volume", className="card-title"),
                dcc.Graph(id='weekly-funnel-trend', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=8),
        dbc.Col([
            html.Div([
                html.Div("Week-over-Week Change", className="card-title"),
                html.Div(id='weekly-wow-change')
            ], className="metric-card")
        ], width=4),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Weekly Breakdown by LVC Group", className="card-title"),
                dcc.Graph(id='weekly-lvc-breakdown', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=12)
    ])
])

# Tab 4: Monthly View
tab_monthly = html.Div([
    dbc.Row([
        dbc.Col([
            html.H5("Monthly Performance", style={'color': '#4cc9f0'}),
            html.P("Comparing to previous month", style={'color': '#8892b0', 'fontSize': '12px'})
        ], width=8),
        dbc.Col([
            html.Div([
                html.Label("Months to Compare", style={'color': '#8892b0', 'fontSize': '11px'}),
                dcc.Dropdown(
                    id='monthly-compare-months',
                    options=[
                        {'label': 'Last 3 Months', 'value': 3},
                        {'label': 'Last 6 Months', 'value': 6},
                        {'label': 'Last 12 Months', 'value': 12}
                    ],
                    value=6,
                    style={'backgroundColor': '#162a40'}
                )
            ])
        ], width=4)
    ], className="mb-4"),
    
    # Monthly KPIs with comparison
    dbc.Row([
        dbc.Col(html.Div(id='monthly-kpi-leads'), width=2),
        dbc.Col(html.Div(id='monthly-kpi-eligible'), width=2),
        dbc.Col(html.Div(id='monthly-kpi-sts'), width=2),
        dbc.Col(html.Div(id='monthly-kpi-contacted'), width=2),
        dbc.Col(html.Div(id='monthly-kpi-fas'), width=2),
        dbc.Col(html.Div(id='monthly-kpi-funded'), width=2),
    ], className="mb-4"),
    
    # Monthly Trends
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Monthly Funnel Volume", className="card-title"),
                dcc.Graph(id='monthly-funnel-trend', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=6),
        dbc.Col([
            html.Div([
                html.Div("Monthly FAS & Funded Volume ($)", className="card-title"),
                dcc.Graph(id='monthly-volume-trend', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Month-over-Month Performance Summary", className="card-title"),
                dash_table.DataTable(
                    id='monthly-summary-table',
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': '#162a40',
                        'color': '#4cc9f0',
                        'fontWeight': 'bold',
                        'border': '1px solid rgba(76, 201, 240, 0.2)'
                    },
                    style_cell={
                        'backgroundColor': '#0d1b2a',
                        'color': 'white',
                        'textAlign': 'left',
                        'border': '1px solid rgba(76, 201, 240, 0.1)',
                        'padding': '10px'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{MoM Change} contains "↑"'},
                            'color': '#00d084'
                        },
                        {
                            'if': {'filter_query': '{MoM Change} contains "↓"'},
                            'color': '#f72585'
                        }
                    ]
                )
            ], className="metric-card")
        ], width=12)
    ])
])

# Tab 5: MA Insights
tab_ma_insights = html.Div([
    dbc.Row([
        dbc.Col([
            html.H5("🏆 Mortgage Advisor Insights", style={'color': '#4cc9f0'}),
            html.P("Performance analysis based on assigned leads", style={'color': '#8892b0', 'fontSize': '12px'})
        ], width=8),
        dbc.Col([
            html.Div([
                html.Label("Time Grain", style={'color': '#8892b0', 'fontSize': '11px'}),
                dcc.Dropdown(
                    id='ma-time-grain',
                    options=[
                        {'label': 'Daily', 'value': 'day'},
                        {'label': 'Weekly', 'value': 'week'},
                        {'label': 'Monthly', 'value': 'month'}
                    ],
                    value='week',
                    style={'backgroundColor': '#162a40'}
                )
            ])
        ], width=4)
    ], className="mb-4"),
    
    # AI Q&A Box
    html.Div([
        html.Div([
            html.Span("🤖 ", style={'fontSize': '18px'}),
            html.Span("Ask AI About Your Data", style={'color': '#4cc9f0', 'fontWeight': 'bold'}),
        ], className="card-title"),
        dbc.Row([
            dbc.Col([
                dbc.Textarea(
                    id='qa-input',
                    placeholder="Ask a question about your MA performance data... e.g., 'Who are the top performers?', 'Which MAs need coaching?', 'What trends do you see?'",
                    style={
                        'backgroundColor': '#0d1b2a',
                        'color': 'white',
                        'border': '1px solid #1b3a4b',
                        'borderRadius': '8px',
                        'minHeight': '60px',
                        'width': '100%'
                    }
                )
            ], width=10),
            dbc.Col([
                dbc.Button(
                    [html.I(className="fas fa-paper-plane"), " Ask"],
                    id='qa-submit-btn',
                    color="info",
                    style={
                        'width': '100%',
                        'height': '60px',
                        'background': 'linear-gradient(135deg, #4895ef 0%, #4cc9f0 100%)',
                        'border': 'none',
                        'borderRadius': '8px',
                        'fontWeight': 'bold'
                    }
                )
            ], width=2)
        ]),
        dcc.Loading(
            id="qa-loading",
            type="circle",
            color="#4cc9f0",
            children=[
                html.Div(
                    id='qa-response',
                    style={
                        'marginTop': '15px',
                        'padding': '15px',
                        'backgroundColor': '#0d1b2a',
                        'borderRadius': '8px',
                        'border': '1px solid #1b3a4b',
                        'minHeight': '80px',
                        'color': '#e6f1ff',
                        'whiteSpace': 'pre-wrap',
                        'display': 'none'
                    }
                )
            ]
        )
    ], className="metric-card mb-4", style={'border': '1px solid #4cc9f0'}),
    
    # Overall MA KPIs
    html.Div([
        html.Div("Overall Team Performance (Assigned Leads)", className="card-title"),
        dbc.Row([
            dbc.Col(html.Div(id='ma-kpi-assigned'), width=2),
            dbc.Col(html.Div(id='ma-kpi-contacted'), width=2),
            dbc.Col(html.Div(id='ma-kpi-fas'), width=2),
            dbc.Col(html.Div(id='ma-kpi-funded'), width=2),
            dbc.Col(html.Div(id='ma-kpi-contact-rate'), width=2),
            dbc.Col(html.Div(id='ma-kpi-fas-rate'), width=2),
        ])
    ], className="metric-card mb-4"),
    
    # Top Performers and Comparison
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Top Performers by FAS Rate", className="card-title"),
                dcc.Graph(id='ma-top-fas-rate-chart', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=4),
        dbc.Col([
            html.Div([
                html.Div("Top Performers by Volume", className="card-title"),
                dcc.Graph(id='ma-top-volume-chart', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=4),
        dbc.Col([
            html.Div([
                html.Div("Top Performers by FAS $", className="card-title"),
                dcc.Graph(id='ma-top-fas-volume-chart', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=4),
    ], className="mb-4"),
    
    # MA Comparison Heatmaps - Separate views for each metric
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Contact Rate by MA", className="card-title"),
                dcc.Graph(id='ma-contact-rate-heatmap', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=4),
        dbc.Col([
            html.Div([
                html.Div("FAS Rate by MA", className="card-title"),
                dcc.Graph(id='ma-fas-rate-heatmap', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=4),
        dbc.Col([
            html.Div([
                html.Div("Pull Through Rate by MA", className="card-title"),
                dcc.Graph(id='ma-pull-through-heatmap', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=4),
    ], className="mb-4"),
    
    # MA Trend Over Time
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Team FAS Trend Over Time", className="card-title"),
                dcc.Graph(id='ma-fas-trend-chart', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=6),
        dbc.Col([
            html.Div([
                html.Div("Contact Rate Trend Over Time", className="card-title"),
                dcc.Graph(id='ma-contact-trend-chart', config={'displayModeBar': False})
            ], className="metric-card")
        ], width=6),
    ], className="mb-4"),
    
    # Detailed MA Table
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Detailed MA Performance Table", className="card-title"),
                dash_table.DataTable(
                    id='ma-detail-table',
                    page_size=15,
                    sort_action='native',
                    filter_action='native',
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': '#162a40',
                        'color': '#4cc9f0',
                        'fontWeight': 'bold',
                        'border': '1px solid rgba(76, 201, 240, 0.2)'
                    },
                    style_cell={
                        'backgroundColor': '#0d1b2a',
                        'color': 'white',
                        'textAlign': 'left',
                        'border': '1px solid rgba(76, 201, 240, 0.1)',
                        'padding': '10px',
                        'minWidth': '80px'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{FAS Rate} > 15%'},
                            'backgroundColor': 'rgba(0, 208, 132, 0.2)'
                        },
                        {
                            'if': {'filter_query': '{FAS Rate} < 5%'},
                            'backgroundColor': 'rgba(247, 37, 133, 0.2)'
                        }
                    ]
                )
            ], className="metric-card")
        ], width=12)
    ]),
    
    # Period Comparison Section
    html.Div(className="section-divider"),
    html.H5("Period-over-Period Comparison", style={'color': '#4cc9f0', 'marginBottom': '20px'}),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("This Period vs Previous Period", className="card-title"),
                html.Div(id='ma-period-comparison')
            ], className="metric-card")
        ], width=6),
        dbc.Col([
            html.Div([
                html.Div("Top Improvers", className="card-title"),
                html.Div(id='ma-top-improvers')
            ], className="metric-card")
        ], width=6),
    ])
])

# --- Main Layout ---
app.layout = html.Div([
    html.Div([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2("📊 Sales Overview Dashboard", 
                       style={'color': '#ffffff', 'fontWeight': '600', 'marginBottom': '5px'}),
                html.P("Full funnel analysis from leads to funded loans", 
                      style={'color': '#8892b0', 'fontSize': '14px'})
            ], width=8),
            dbc.Col([
                html.Div([
                    html.Span("Data Source: ", style={'color': '#8892b0', 'fontSize': '12px'}),
                    html.Span("lendage_lead_vintages_table", 
                             style={'color': '#4cc9f0', 'fontSize': '12px', 'fontWeight': '600'})
                ], style={'textAlign': 'right', 'marginTop': '10px'})
            ], width=4)
        ], className="mb-4"),
        
        # Filter Panel
        filter_panel,
        
        # Tabs
        dbc.Tabs([
            dbc.Tab(tab_overall, label="📈 Overall Metrics", tab_id="tab-overall",
                   tab_style={'marginLeft': '0'}),
            dbc.Tab(tab_daily, label="📅 Daily", tab_id="tab-daily"),
            dbc.Tab(tab_weekly, label="📆 Weekly", tab_id="tab-weekly"),
            dbc.Tab(tab_monthly, label="🗓️ Monthly", tab_id="tab-monthly"),
            dbc.Tab(tab_ma_insights, label="🏆 MA Insights", tab_id="tab-ma-insights"),
        ], id="main-tabs", active_tab="tab-overall", className="mb-4"),
        
        # Hidden store for data
        dcc.Store(id='funnel-data-store'),
        dcc.Store(id='breakdown-data-store'),
        dcc.Store(id='ma-data-store'),
        
    ], className="dashboard-container", style={'padding': '20px'})
])

# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    [Output('funnel-data-store', 'data'),
     Output('breakdown-data-store', 'data')],
    [Input('date-range-filter', 'start_date'),
     Input('date-range-filter', 'end_date'),
     Input('refresh-btn', 'n_clicks')]
)
def load_data(start_date, end_date, n_clicks):
    """Load all data from BigQuery"""
    # Get overall funnel data
    funnel_query = get_funnel_query(start_date, end_date)
    df_funnel = run_query(funnel_query)
    
    # Get breakdowns
    df_lvc = run_query(get_breakdown_query(start_date, end_date, 'lvc_group'))
    df_persona = run_query(get_breakdown_query(start_date, end_date, 'persona'))
    df_ma = run_query(get_breakdown_query(start_date, end_date, 'mortgage_advisors'))
    
    funnel_data = df_funnel.to_dict('records')[0] if not df_funnel.empty else {}
    breakdown_data = {
        'lvc': df_lvc.to_dict('records') if not df_lvc.empty else [],
        'persona': df_persona.to_dict('records') if not df_persona.empty else [],
        'ma': df_ma.to_dict('records') if not df_ma.empty else []
    }
    
    return funnel_data, breakdown_data

# --- Overall Tab Callbacks ---
@callback(
    [Output('kpi-gross-leads', 'children'),
     Output('kpi-eligible', 'children'),
     Output('kpi-sent-to-sales', 'children'),
     Output('kpi-assigned', 'children'),
     Output('kpi-contacted', 'children'),
     Output('kpi-fas', 'children'),
     Output('kpi-funded', 'children'),
     Output('funnel-chart', 'figure'),
     Output('conversion-rates-display', 'children'),
     Output('volume-distribution-chart', 'figure')],
    [Input('funnel-data-store', 'data')]
)
def update_overall_kpis(funnel_data):
    """Update overall KPIs and funnel chart"""
    if not funnel_data:
        empty_fig = go.Figure()
        empty_fig.update_layout(**DARK_THEME)
        empty_card = html.Div("--", className="metric-card")
        return [empty_card]*7 + [empty_fig, html.Div("No data"), empty_fig]
    
    # Build KPI cards
    kpi_leads = build_kpi_card("Gross Leads", format_number(funnel_data.get('gross_leads', 0)),
                               sub_text="Total leads created")
    kpi_eligible = build_kpi_card("Eligible", format_number(funnel_data.get('eligible_leads', 0)),
                                  sub_text=format_pct(funnel_data.get('eligible_leads', 0) / max(funnel_data.get('gross_leads', 1), 1)))
    kpi_sts = build_kpi_card("Sent to Sales", format_number(funnel_data.get('sent_to_sales', 0)),
                             sub_text=format_pct(funnel_data.get('sent_to_sales', 0) / max(funnel_data.get('eligible_leads', 1), 1)))
    kpi_assigned = build_kpi_card("Assigned", format_number(funnel_data.get('leads_assigned', 0)),
                                  sub_text=format_pct(funnel_data.get('leads_assigned', 0) / max(funnel_data.get('sent_to_sales', 1), 1)))
    kpi_contacted = build_kpi_card("Contacted", format_number(funnel_data.get('contacted', 0)),
                                   sub_text=format_pct(funnel_data.get('contacted', 0) / max(funnel_data.get('leads_assigned', 1), 1)))
    kpi_fas = build_kpi_card("FAS", format_number(funnel_data.get('fas', 0)),
                             sub_text=format_pct(funnel_data.get('fas', 0) / max(funnel_data.get('contacted', 1), 1)))
    kpi_funded = build_kpi_card("Funded", format_number(funnel_data.get('funded', 0)),
                                sub_text=format_pct(funnel_data.get('funded', 0) / max(funnel_data.get('fas', 1), 1)))
    
    # Funnel chart
    funnel_fig = create_funnel_chart(funnel_data)
    
    # Conversion rates display
    gross = max(funnel_data.get('gross_leads', 1), 1)
    conversion_rates = html.Div([
        html.Div([
            html.Div("Eligible Rate", className="kpi-label"),
            html.Div(format_pct(funnel_data.get('eligible_leads', 0) / gross), 
                    className="kpi-value-small", style={'color': COLORS['cyan']})
        ], className="mb-3"),
        html.Div([
            html.Div("StS Rate (from Eligible)", className="kpi-label"),
            html.Div(format_pct(funnel_data.get('sent_to_sales', 0) / max(funnel_data.get('eligible_leads', 1), 1)), 
                    className="kpi-value-small", style={'color': COLORS['blue']})
        ], className="mb-3"),
        html.Div([
            html.Div("Contact Rate (from Assigned)", className="kpi-label"),
            html.Div(format_pct(funnel_data.get('contacted', 0) / max(funnel_data.get('leads_assigned', 1), 1)), 
                    className="kpi-value-small", style={'color': COLORS['purple']})
        ], className="mb-3"),
        html.Div([
            html.Div("FAS Rate (from Contacted)", className="kpi-label"),
            html.Div(format_pct(funnel_data.get('fas', 0) / max(funnel_data.get('contacted', 1), 1)), 
                    className="kpi-value-small", style={'color': COLORS['pink']})
        ], className="mb-3"),
        html.Div([
            html.Div("Pull Through (from FAS)", className="kpi-label"),
            html.Div(format_pct(funnel_data.get('funded', 0) / max(funnel_data.get('fas', 1), 1)), 
                    className="kpi-value-small", style={'color': COLORS['green']})
        ])
    ])
    
    # Volume distribution donut
    volume_fig = create_donut_chart(
        ['FAS Volume', 'Funded Volume'],
        [funnel_data.get('fas_volume', 0), funnel_data.get('funded_volume', 0)],
        "Volume Split"
    )
    
    return (kpi_leads, kpi_eligible, kpi_sts, kpi_assigned, kpi_contacted, kpi_fas, kpi_funded,
            funnel_fig, conversion_rates, volume_fig)

@callback(
    [Output('lvc-breakdown-chart', 'figure'),
     Output('persona-breakdown-chart', 'figure'),
     Output('ma-breakdown-chart', 'figure'),
     Output('conversion-heatmap', 'figure')],
    [Input('breakdown-data-store', 'data')]
)
def update_breakdown_charts(breakdown_data):
    """Update breakdown charts"""
    empty_fig = go.Figure()
    empty_fig.update_layout(**DARK_THEME)
    
    if not breakdown_data:
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    # LVC breakdown
    df_lvc = pd.DataFrame(breakdown_data.get('lvc', []))
    lvc_fig = create_breakdown_bar_chart(df_lvc, "Sent to Sales by LVC Group", 'sent_to_sales') if not df_lvc.empty else empty_fig
    
    # Persona breakdown
    df_persona = pd.DataFrame(breakdown_data.get('persona', []))
    persona_fig = create_breakdown_bar_chart(df_persona, "Sent to Sales by Persona", 'sent_to_sales') if not df_persona.empty else empty_fig
    
    # MA breakdown
    df_ma = pd.DataFrame(breakdown_data.get('ma', []))
    ma_fig = create_breakdown_bar_chart(df_ma, "Sent to Sales by MA", 'sent_to_sales') if not df_ma.empty else empty_fig
    
    # Conversion heatmap
    heatmap_fig = create_conversion_heatmap(df_lvc, "Stage Conversion Rates") if not df_lvc.empty else empty_fig
    
    return lvc_fig, persona_fig, ma_fig, heatmap_fig

# --- Daily Tab Callbacks ---
@callback(
    [Output('daily-kpi-leads', 'children'),
     Output('daily-kpi-eligible', 'children'),
     Output('daily-kpi-sts', 'children'),
     Output('daily-kpi-contacted', 'children'),
     Output('daily-kpi-fas', 'children'),
     Output('daily-kpi-funded', 'children'),
     Output('daily-leads-trend', 'figure'),
     Output('daily-fas-trend', 'figure'),
     Output('daily-conversion-trend', 'figure')],
    [Input('date-range-filter', 'start_date'),
     Input('date-range-filter', 'end_date'),
     Input('daily-compare-days', 'value')]
)
def update_daily_tab(start_date, end_date, compare_days):
    """Update daily tab content"""
    empty_fig = go.Figure()
    empty_fig.update_layout(**DARK_THEME)
    empty_card = html.Div("--", className="metric-card")
    
    # Fetch daily data
    query = get_time_series_query(start_date, end_date, 'day')
    df = run_query(query)
    
    if df.empty:
        return [empty_card]*6 + [empty_fig]*3
    
    df['period_date'] = pd.to_datetime(df['period_date'])
    df = df.sort_values('period_date')
    
    # Get today and yesterday
    today_data = df.iloc[-1] if len(df) > 0 else None
    yesterday_data = df.iloc[-2] if len(df) > 1 else None
    
    # Build KPI cards with DoD comparison
    def build_daily_kpi(title, metric):
        current = today_data[metric] if today_data is not None else 0
        previous = yesterday_data[metric] if yesterday_data is not None else 0
        badge = get_change_badge(current, previous)
        return build_kpi_card(f"Today's {title}", format_number(current), badge, "vs Yesterday")
    
    kpi_leads = build_daily_kpi("Leads", 'gross_leads')
    kpi_eligible = build_daily_kpi("Eligible", 'eligible_leads')
    kpi_sts = build_daily_kpi("StS", 'sent_to_sales')
    kpi_contacted = build_daily_kpi("Contacted", 'contacted')
    kpi_fas = build_daily_kpi("FAS", 'fas')
    kpi_funded = build_daily_kpi("Funded", 'funded')
    
    # Filter to compare_days
    df_filtered = df.tail(compare_days)
    
    # Trend charts
    leads_trend = create_trend_chart(df_filtered, 'gross_leads', 'Daily Gross Leads')
    fas_trend = create_trend_chart(df_filtered, 'fas', 'Daily FAS')
    
    # Conversion rate trend
    df_filtered['fas_rate'] = df_filtered['fas'] / df_filtered['sent_to_sales'].replace(0, np.nan)
    df_filtered['contact_rate'] = df_filtered['contacted'] / df_filtered['leads_assigned'].replace(0, np.nan)
    
    conv_fig = go.Figure()
    conv_fig.add_trace(go.Scatter(
        x=df_filtered['period_date'], y=df_filtered['fas_rate'],
        mode='lines+markers', name='FAS Rate',
        line=dict(color=COLORS['cyan'], width=2),
        marker=dict(size=6)
    ))
    conv_fig.add_trace(go.Scatter(
        x=df_filtered['period_date'], y=df_filtered['contact_rate'],
        mode='lines+markers', name='Contact Rate',
        line=dict(color=COLORS['yellow'], width=2),
        marker=dict(size=6)
    ))
    conv_fig.update_layout(**DARK_THEME, height=250)
    conv_fig.update_yaxes(tickformat='.1%')
    
    return (kpi_leads, kpi_eligible, kpi_sts, kpi_contacted, kpi_fas, kpi_funded,
            leads_trend, fas_trend, conv_fig)

# --- Weekly Tab Callbacks ---
@callback(
    [Output('weekly-kpi-leads', 'children'),
     Output('weekly-kpi-eligible', 'children'),
     Output('weekly-kpi-sts', 'children'),
     Output('weekly-kpi-contacted', 'children'),
     Output('weekly-kpi-fas', 'children'),
     Output('weekly-kpi-funded', 'children'),
     Output('weekly-funnel-trend', 'figure'),
     Output('weekly-wow-change', 'children'),
     Output('weekly-lvc-breakdown', 'figure')],
    [Input('date-range-filter', 'start_date'),
     Input('date-range-filter', 'end_date'),
     Input('weekly-compare-weeks', 'value')]
)
def update_weekly_tab(start_date, end_date, compare_weeks):
    """Update weekly tab content"""
    empty_fig = go.Figure()
    empty_fig.update_layout(**DARK_THEME)
    empty_card = html.Div("--", className="metric-card")
    
    # Fetch weekly data
    query = get_time_series_query(start_date, end_date, 'week')
    df = run_query(query)
    
    if df.empty:
        return [empty_card]*6 + [empty_fig, html.Div("No data"), empty_fig]
    
    df['period_date'] = pd.to_datetime(df['period_date'])
    df = df.sort_values('period_date')
    
    # Get this week and last week
    this_week = df.iloc[-1] if len(df) > 0 else None
    last_week = df.iloc[-2] if len(df) > 1 else None
    
    # Build KPI cards with WoW comparison
    def build_weekly_kpi(title, metric):
        current = this_week[metric] if this_week is not None else 0
        previous = last_week[metric] if last_week is not None else 0
        badge = get_change_badge(current, previous)
        return build_kpi_card(f"This Week {title}", format_number(current), badge, "vs Last Week")
    
    kpi_leads = build_weekly_kpi("Leads", 'gross_leads')
    kpi_eligible = build_weekly_kpi("Eligible", 'eligible_leads')
    kpi_sts = build_weekly_kpi("StS", 'sent_to_sales')
    kpi_contacted = build_weekly_kpi("Contacted", 'contacted')
    kpi_fas = build_weekly_kpi("FAS", 'fas')
    kpi_funded = build_weekly_kpi("Funded", 'funded')
    
    # Filter to compare_weeks
    df_filtered = df.tail(compare_weeks)
    
    # Stacked area chart for funnel
    funnel_fig = go.Figure()
    metrics = ['gross_leads', 'sent_to_sales', 'contacted', 'fas', 'funded']
    colors = [COLORS['cyan'], COLORS['blue'], COLORS['purple'], COLORS['pink'], COLORS['green']]
    
    for metric, color in zip(metrics, colors):
        funnel_fig.add_trace(go.Scatter(
            x=df_filtered['period_date'], y=df_filtered[metric],
            mode='lines', name=metric.replace('_', ' ').title(),
            fill='tonexty' if metric != 'gross_leads' else 'tozeroy',
            line=dict(width=0.5, color=color)
        ))
    
    funnel_fig.update_layout(**DARK_THEME, height=350)
    
    # WoW change summary
    wow_items = []
    for metric, label in [('gross_leads', 'Leads'), ('fas', 'FAS'), ('funded', 'Funded')]:
        current = this_week[metric] if this_week is not None else 0
        previous = last_week[metric] if last_week is not None else 0
        badge = get_change_badge(current, previous)
        wow_items.append(html.Div([
            html.Span(f"{label}: ", style={'color': '#8892b0'}),
            html.Span(format_number(current), style={'color': '#ffffff', 'fontWeight': '600'}),
            badge
        ], style={'marginBottom': '15px'}))
    
    wow_content = html.Div(wow_items)
    
    # Weekly LVC breakdown (simplified)
    lvc_fig = go.Figure()
    lvc_fig.add_trace(go.Bar(
        x=df_filtered['period_date'], y=df_filtered['gross_leads'],
        name='Gross Leads', marker_color=COLORS['cyan']
    ))
    lvc_fig.add_trace(go.Bar(
        x=df_filtered['period_date'], y=df_filtered['fas'],
        name='FAS', marker_color=COLORS['pink']
    ))
    lvc_fig.update_layout(**DARK_THEME, barmode='group', height=300)
    
    return (kpi_leads, kpi_eligible, kpi_sts, kpi_contacted, kpi_fas, kpi_funded,
            funnel_fig, wow_content, lvc_fig)

# --- Monthly Tab Callbacks ---
@callback(
    [Output('monthly-kpi-leads', 'children'),
     Output('monthly-kpi-eligible', 'children'),
     Output('monthly-kpi-sts', 'children'),
     Output('monthly-kpi-contacted', 'children'),
     Output('monthly-kpi-fas', 'children'),
     Output('monthly-kpi-funded', 'children'),
     Output('monthly-funnel-trend', 'figure'),
     Output('monthly-volume-trend', 'figure'),
     Output('monthly-summary-table', 'data'),
     Output('monthly-summary-table', 'columns')],
    [Input('date-range-filter', 'start_date'),
     Input('date-range-filter', 'end_date'),
     Input('monthly-compare-months', 'value')]
)
def update_monthly_tab(start_date, end_date, compare_months):
    """Update monthly tab content"""
    empty_fig = go.Figure()
    empty_fig.update_layout(**DARK_THEME)
    empty_card = html.Div("--", className="metric-card")
    
    # Fetch monthly data
    query = get_time_series_query(start_date, end_date, 'month')
    df = run_query(query)
    
    if df.empty:
        return [empty_card]*6 + [empty_fig, empty_fig, [], []]
    
    df['period_date'] = pd.to_datetime(df['period_date'])
    df = df.sort_values('period_date')
    
    # Get this month and last month
    this_month = df.iloc[-1] if len(df) > 0 else None
    last_month = df.iloc[-2] if len(df) > 1 else None
    
    # Build KPI cards with MoM comparison
    def build_monthly_kpi(title, metric):
        current = this_month[metric] if this_month is not None else 0
        previous = last_month[metric] if last_month is not None else 0
        badge = get_change_badge(current, previous)
        return build_kpi_card(f"This Month {title}", format_number(current), badge, "vs Last Month")
    
    kpi_leads = build_monthly_kpi("Leads", 'gross_leads')
    kpi_eligible = build_monthly_kpi("Eligible", 'eligible_leads')
    kpi_sts = build_monthly_kpi("StS", 'sent_to_sales')
    kpi_contacted = build_monthly_kpi("Contacted", 'contacted')
    kpi_fas = build_monthly_kpi("FAS", 'fas')
    kpi_funded = build_monthly_kpi("Funded", 'funded')
    
    # Filter to compare_months
    df_filtered = df.tail(compare_months)
    
    # Monthly funnel trend
    funnel_fig = go.Figure()
    funnel_fig.add_trace(go.Bar(x=df_filtered['period_date'], y=df_filtered['gross_leads'],
                                name='Gross Leads', marker_color=COLORS['cyan']))
    funnel_fig.add_trace(go.Bar(x=df_filtered['period_date'], y=df_filtered['sent_to_sales'],
                                name='Sent to Sales', marker_color=COLORS['blue']))
    funnel_fig.add_trace(go.Bar(x=df_filtered['period_date'], y=df_filtered['fas'],
                                name='FAS', marker_color=COLORS['pink']))
    funnel_fig.update_layout(**DARK_THEME, barmode='group', height=350)
    funnel_fig.update_xaxes(tickformat='%b %Y')
    
    # Volume trend
    volume_fig = go.Figure()
    volume_fig.add_trace(go.Scatter(
        x=df_filtered['period_date'], y=df_filtered['fas_volume'],
        mode='lines+markers', name='FAS Volume',
        fill='tozeroy', fillcolor='rgba(247, 37, 133, 0.2)',
        line=dict(color=COLORS['pink'], width=2)
    ))
    volume_fig.add_trace(go.Scatter(
        x=df_filtered['period_date'], y=df_filtered['funded_volume'],
        mode='lines+markers', name='Funded Volume',
        fill='tozeroy', fillcolor='rgba(0, 208, 132, 0.2)',
        line=dict(color=COLORS['green'], width=2)
    ))
    volume_fig.update_layout(**DARK_THEME, height=350)
    volume_fig.update_xaxes(tickformat='%b %Y')
    volume_fig.update_yaxes(tickformat='$,.0f')
    
    # Summary table
    df_table = df_filtered.copy()
    df_table['month'] = df_table['period_date'].dt.strftime('%b %Y')
    df_table['fas_rate'] = df_table['fas'] / df_table['sent_to_sales'].replace(0, np.nan)
    
    # Calculate MoM change
    df_table['prev_fas'] = df_table['fas'].shift(1)
    df_table['mom_change'] = ((df_table['fas'] - df_table['prev_fas']) / df_table['prev_fas'].replace(0, np.nan)) * 100
    df_table['MoM Change'] = df_table['mom_change'].apply(
        lambda x: f"↑ {x:.1f}%" if pd.notna(x) and x > 0 else (f"↓ {abs(x):.1f}%" if pd.notna(x) and x < 0 else "--")
    )
    
    # Format table
    table_data = df_table[['month', 'gross_leads', 'sent_to_sales', 'fas', 'funded', 'fas_rate', 'MoM Change']].copy()
    table_data.columns = ['Month', 'Gross Leads', 'Sent to Sales', 'FAS', 'Funded', 'FAS Rate', 'MoM Change']
    table_data['Gross Leads'] = table_data['Gross Leads'].apply(lambda x: f"{x:,}")
    table_data['Sent to Sales'] = table_data['Sent to Sales'].apply(lambda x: f"{x:,}")
    table_data['FAS'] = table_data['FAS'].apply(lambda x: f"{x:,}")
    table_data['Funded'] = table_data['Funded'].apply(lambda x: f"{x:,}")
    table_data['FAS Rate'] = table_data['FAS Rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "--")
    
    columns = [{'name': col, 'id': col} for col in table_data.columns]
    
    return (kpi_leads, kpi_eligible, kpi_sts, kpi_contacted, kpi_fas, kpi_funded,
            funnel_fig, volume_fig, table_data.to_dict('records'), columns)

# --- MA Insights Tab Callbacks ---
@callback(
    [Output('ma-kpi-assigned', 'children'),
     Output('ma-kpi-contacted', 'children'),
     Output('ma-kpi-fas', 'children'),
     Output('ma-kpi-funded', 'children'),
     Output('ma-kpi-contact-rate', 'children'),
     Output('ma-kpi-fas-rate', 'children'),
     Output('ma-top-fas-rate-chart', 'figure'),
     Output('ma-top-volume-chart', 'figure'),
     Output('ma-top-fas-volume-chart', 'figure'),
     Output('ma-contact-rate-heatmap', 'figure'),
     Output('ma-fas-rate-heatmap', 'figure'),
     Output('ma-pull-through-heatmap', 'figure'),
     Output('ma-fas-trend-chart', 'figure'),
     Output('ma-contact-trend-chart', 'figure'),
     Output('ma-detail-table', 'data'),
     Output('ma-detail-table', 'columns'),
     Output('ma-period-comparison', 'children'),
     Output('ma-top-improvers', 'children')],
    [Input('date-range-filter', 'start_date'),
     Input('date-range-filter', 'end_date'),
     Input('ma-time-grain', 'value')]
)
def update_ma_insights_tab(start_date, end_date, time_grain):
    """Update MA Insights tab content"""
    empty_fig = go.Figure()
    empty_fig.update_layout(**DARK_THEME)
    empty_card = html.Div("--", className="metric-card")
    empty_columns = []
    
    # Fetch MA data
    ma_query = get_ma_insights_query(start_date, end_date)
    df_ma = run_query(ma_query)
    
    # Fetch time series data
    ts_query = get_ma_time_series_query(start_date, end_date, time_grain)
    df_ts = run_query(ts_query)
    
    if df_ma.empty:
        return ([empty_card]*6 + [empty_fig]*8 + [[], empty_columns, 
                html.Div("No data available"), html.Div("No data available")])
    
    # Calculate rates
    df_ma['contact_rate'] = df_ma['contacted'] / df_ma['assigned_leads'].replace(0, np.nan)
    df_ma['fas_rate'] = df_ma['fas'] / df_ma['assigned_leads'].replace(0, np.nan)
    df_ma['funded_rate'] = df_ma['funded'] / df_ma['fas'].replace(0, np.nan)
    
    # Overall KPIs
    total_assigned = df_ma['assigned_leads'].sum()
    total_contacted = df_ma['contacted'].sum()
    total_fas = df_ma['fas'].sum()
    total_funded = df_ma['funded'].sum()
    overall_contact_rate = total_contacted / total_assigned if total_assigned > 0 else 0
    overall_fas_rate = total_fas / total_assigned if total_assigned > 0 else 0
    
    kpi_assigned = build_kpi_card("Assigned Leads", format_number(total_assigned), sub_text="Total assigned")
    kpi_contacted = build_kpi_card("Contacted", format_number(total_contacted), sub_text=format_pct(overall_contact_rate))
    kpi_fas = build_kpi_card("FAS", format_number(total_fas), sub_text="Applications")
    kpi_funded = build_kpi_card("Funded", format_number(total_funded), sub_text="Loans funded")
    kpi_contact_rate = build_kpi_card("Contact Rate", format_pct(overall_contact_rate), sub_text="Team average")
    kpi_fas_rate = build_kpi_card("FAS Rate", format_pct(overall_fas_rate), sub_text="Team average")
    
    # Top Performers by FAS Rate (min 20 leads for reliability)
    df_reliable = df_ma[df_ma['assigned_leads'] >= 20].copy()
    if not df_reliable.empty:
        df_top_rate = df_reliable.nlargest(10, 'fas_rate')
        fig_fas_rate = go.Figure(go.Bar(
            y=df_top_rate['ma_name'],
            x=df_top_rate['fas_rate'],
            orientation='h',
            marker=dict(color=df_top_rate['fas_rate'], colorscale='Greens'),
            text=df_top_rate['fas_rate'].apply(lambda x: f"{x:.1%}"),
            textposition='inside',
            textfont=dict(color='white')
        ))
        fig_fas_rate.update_layout(**DARK_THEME, height=300, margin=dict(l=150, r=20, t=30, b=30))
        fig_fas_rate.update_xaxes(tickformat='.0%')
    else:
        fig_fas_rate = empty_fig
    
    # Top Performers by Volume
    df_top_vol = df_ma.nlargest(10, 'assigned_leads')
    fig_volume = go.Figure(go.Bar(
        y=df_top_vol['ma_name'],
        x=df_top_vol['assigned_leads'],
        orientation='h',
        marker=dict(color=df_top_vol['assigned_leads'], colorscale=[[0, '#1b3a4b'], [1, '#4cc9f0']]),
        text=df_top_vol['assigned_leads'].apply(lambda x: format_number(x)),
        textposition='inside',
        textfont=dict(color='white')
    ))
    fig_volume.update_layout(**DARK_THEME, height=300, margin=dict(l=150, r=20, t=30, b=30))
    
    # Top Performers by FAS Volume ($)
    df_top_fas_vol = df_ma.nlargest(10, 'fas_volume')
    fig_fas_vol = go.Figure(go.Bar(
        y=df_top_fas_vol['ma_name'],
        x=df_top_fas_vol['fas_volume'],
        orientation='h',
        marker=dict(color=df_top_fas_vol['fas_volume'], colorscale=[[0, '#7209b7'], [1, '#f72585']]),
        text=df_top_fas_vol['fas_volume'].apply(lambda x: f"${x/1000000:.1f}M" if x >= 1000000 else f"${x/1000:.0f}K"),
        textposition='inside',
        textfont=dict(color='white')
    ))
    fig_fas_vol.update_layout(**DARK_THEME, height=300, margin=dict(l=150, r=20, t=30, b=30))
    fig_fas_vol.update_xaxes(tickformat='$,.0f')
    
    # MA Comparison - 3 Separate Heatmap Charts (top 15 by volume)
    df_heatmap = df_ma.nlargest(15, 'assigned_leads').copy()
    
    # Temperature colorscale: Red (low) -> Yellow (mid) -> Green (high)
    temp_colorscale = [[0, '#d73027'], [0.25, '#f46d43'], [0.5, '#fee08b'], [0.75, '#a6d96a'], [1, '#1a9850']]
    
    if not df_heatmap.empty:
        # Sort by the metric for each chart (ascending so highest at top)
        
        # Contact Rate Heatmap
        df_contact = df_heatmap.sort_values('contact_rate', ascending=True).tail(15)
        fig_contact_heatmap = go.Figure(go.Bar(
            y=df_contact['ma_name'],
            x=df_contact['contact_rate'].fillna(0),
            orientation='h',
            marker=dict(
                color=df_contact['contact_rate'].fillna(0),
                colorscale=temp_colorscale,
                cmin=0,
                cmax=df_heatmap['contact_rate'].max() if df_heatmap['contact_rate'].max() > 0 else 1,
                showscale=True,
                colorbar=dict(tickformat='.0%', len=0.9)
            ),
            text=df_contact['contact_rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "--"),
            textposition='inside',
            textfont=dict(color='white', size=10),
            hovertemplate='%{y}<br>Contact Rate: %{x:.1%}<extra></extra>'
        ))
        fig_contact_heatmap.update_layout(**DARK_THEME, height=400, margin=dict(l=120, r=20, t=10, b=30))
        fig_contact_heatmap.update_xaxes(tickformat='.0%', title='')
        fig_contact_heatmap.update_yaxes(title='')
        
        # FAS Rate Heatmap
        df_fas = df_heatmap.sort_values('fas_rate', ascending=True).tail(15)
        fig_fas_heatmap = go.Figure(go.Bar(
            y=df_fas['ma_name'],
            x=df_fas['fas_rate'].fillna(0),
            orientation='h',
            marker=dict(
                color=df_fas['fas_rate'].fillna(0),
                colorscale=temp_colorscale,
                cmin=0,
                cmax=df_heatmap['fas_rate'].max() if df_heatmap['fas_rate'].max() > 0 else 1,
                showscale=True,
                colorbar=dict(tickformat='.0%', len=0.9)
            ),
            text=df_fas['fas_rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "--"),
            textposition='inside',
            textfont=dict(color='white', size=10),
            hovertemplate='%{y}<br>FAS Rate: %{x:.1%}<extra></extra>'
        ))
        fig_fas_heatmap.update_layout(**DARK_THEME, height=400, margin=dict(l=120, r=20, t=10, b=30))
        fig_fas_heatmap.update_xaxes(tickformat='.0%', title='')
        fig_fas_heatmap.update_yaxes(title='')
        
        # Pull Through Rate Heatmap
        df_pt = df_heatmap.sort_values('funded_rate', ascending=True).tail(15)
        fig_pt_heatmap = go.Figure(go.Bar(
            y=df_pt['ma_name'],
            x=df_pt['funded_rate'].fillna(0),
            orientation='h',
            marker=dict(
                color=df_pt['funded_rate'].fillna(0),
                colorscale=temp_colorscale,
                cmin=0,
                cmax=df_heatmap['funded_rate'].max() if df_heatmap['funded_rate'].max() > 0 else 1,
                showscale=True,
                colorbar=dict(tickformat='.0%', len=0.9)
            ),
            text=df_pt['funded_rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "--"),
            textposition='inside',
            textfont=dict(color='white', size=10),
            hovertemplate='%{y}<br>Pull Through: %{x:.1%}<extra></extra>'
        ))
        fig_pt_heatmap.update_layout(**DARK_THEME, height=400, margin=dict(l=120, r=20, t=10, b=30))
        fig_pt_heatmap.update_xaxes(tickformat='.0%', title='')
        fig_pt_heatmap.update_yaxes(title='')
    else:
        fig_contact_heatmap = empty_fig
        fig_fas_heatmap = empty_fig
        fig_pt_heatmap = empty_fig
    
    # Trend Charts
    if not df_ts.empty:
        df_ts['period_date'] = pd.to_datetime(df_ts['period_date'])
        
        # Aggregate for overall team trend
        df_team_trend = df_ts.groupby('period_date').agg({
            'assigned_leads': 'sum',
            'contacted': 'sum',
            'fas': 'sum',
            'funded': 'sum',
            'fas_volume': 'sum'
        }).reset_index()
        df_team_trend['contact_rate'] = df_team_trend['contacted'] / df_team_trend['assigned_leads'].replace(0, np.nan)
        df_team_trend['fas_rate'] = df_team_trend['fas'] / df_team_trend['assigned_leads'].replace(0, np.nan)
        
        # FAS Trend
        fig_fas_trend = go.Figure()
        fig_fas_trend.add_trace(go.Scatter(
            x=df_team_trend['period_date'], y=df_team_trend['fas'],
            mode='lines+markers', name='FAS',
            fill='tozeroy', fillcolor='rgba(247, 37, 133, 0.2)',
            line=dict(color=COLORS['pink'], width=2),
            marker=dict(size=6)
        ))
        fig_fas_trend.add_trace(go.Scatter(
            x=df_team_trend['period_date'], y=df_team_trend['funded'],
            mode='lines+markers', name='Funded',
            fill='tozeroy', fillcolor='rgba(0, 208, 132, 0.2)',
            line=dict(color=COLORS['green'], width=2),
            marker=dict(size=6)
        ))
        fig_fas_trend.update_layout(**DARK_THEME, height=280)
        
        # Contact Rate Trend
        fig_contact_trend = go.Figure()
        fig_contact_trend.add_trace(go.Scatter(
            x=df_team_trend['period_date'], y=df_team_trend['contact_rate'],
            mode='lines+markers', name='Contact Rate',
            line=dict(color=COLORS['yellow'], width=2),
            marker=dict(size=6)
        ))
        fig_contact_trend.add_trace(go.Scatter(
            x=df_team_trend['period_date'], y=df_team_trend['fas_rate'],
            mode='lines+markers', name='FAS Rate',
            line=dict(color=COLORS['cyan'], width=2),
            marker=dict(size=6)
        ))
        fig_contact_trend.update_layout(**DARK_THEME, height=280)
        fig_contact_trend.update_yaxes(tickformat='.1%')
    else:
        fig_fas_trend = empty_fig
        fig_contact_trend = empty_fig
    
    # Detailed MA Table
    table_df = df_ma.copy()
    table_df['Contact Rate'] = table_df['contact_rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "--")
    table_df['FAS Rate'] = table_df['fas_rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "--")
    table_df['Pull Through'] = table_df['funded_rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "--")
    table_df['Avg FAS Loan'] = table_df['avg_fas_loan'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "--")
    table_df['FAS Volume'] = table_df['fas_volume'].apply(lambda x: f"${x:,.0f}")
    table_df['Assigned'] = table_df['assigned_leads'].apply(lambda x: f"{x:,}")
    table_df['Contacted'] = table_df['contacted'].apply(lambda x: f"{x:,}")
    table_df['FAS'] = table_df['fas'].apply(lambda x: f"{x:,}")
    table_df['Funded'] = table_df['funded'].apply(lambda x: f"{x:,}")
    
    display_cols = ['ma_name', 'Assigned', 'Contacted', 'Contact Rate', 'FAS', 'FAS Rate', 
                   'Funded', 'Pull Through', 'FAS Volume', 'Avg FAS Loan']
    table_df = table_df[display_cols]
    table_df.columns = ['MA Name', 'Assigned', 'Contacted', 'Contact Rate', 'FAS', 'FAS Rate',
                       'Funded', 'Pull Through', 'FAS Volume', 'Avg FAS Loan']
    
    columns = [{'name': col, 'id': col, 'type': 'text'} for col in table_df.columns]
    
    # Period Comparison
    if not df_ts.empty:
        df_ts_sorted = df_ts.sort_values('period_date')
        periods = df_ts_sorted['period_date'].unique()
        
        if len(periods) >= 2:
            current_period = periods[-1]
            prev_period = periods[-2]
            
            current_data = df_ts_sorted[df_ts_sorted['period_date'] == current_period]
            prev_data = df_ts_sorted[df_ts_sorted['period_date'] == prev_period]
            
            current_totals = current_data.agg({'assigned_leads': 'sum', 'fas': 'sum', 'contacted': 'sum'})
            prev_totals = prev_data.agg({'assigned_leads': 'sum', 'fas': 'sum', 'contacted': 'sum'})
            
            period_comparison = html.Div([
                html.Div([
                    html.Span("Assigned: ", style={'color': '#8892b0'}),
                    html.Span(format_number(current_totals['assigned_leads']), style={'color': '#ffffff', 'fontWeight': '600'}),
                    get_change_badge(current_totals['assigned_leads'], prev_totals['assigned_leads'])
                ], style={'marginBottom': '12px'}),
                html.Div([
                    html.Span("Contacted: ", style={'color': '#8892b0'}),
                    html.Span(format_number(current_totals['contacted']), style={'color': '#ffffff', 'fontWeight': '600'}),
                    get_change_badge(current_totals['contacted'], prev_totals['contacted'])
                ], style={'marginBottom': '12px'}),
                html.Div([
                    html.Span("FAS: ", style={'color': '#8892b0'}),
                    html.Span(format_number(current_totals['fas']), style={'color': '#ffffff', 'fontWeight': '600'}),
                    get_change_badge(current_totals['fas'], prev_totals['fas'])
                ])
            ])
            
            # Calculate improvers
            current_ma = current_data.set_index('ma_name')
            prev_ma = prev_data.set_index('ma_name')
            
            common_mas = set(current_ma.index) & set(prev_ma.index)
            if common_mas:
                improvements = []
                for ma in common_mas:
                    curr_fas = current_ma.loc[ma, 'fas']
                    prev_fas = prev_ma.loc[ma, 'fas']
                    if prev_fas > 0:
                        change = ((curr_fas - prev_fas) / prev_fas) * 100
                        improvements.append({'ma': ma, 'change': change, 'curr': curr_fas, 'prev': prev_fas})
                
                if improvements:
                    improvements.sort(key=lambda x: x['change'], reverse=True)
                    top_improvers = html.Div([
                        html.Div([
                            html.Span(f"{i+1}. {imp['ma'][:20]}...: " if len(imp['ma']) > 20 else f"{i+1}. {imp['ma']}: ", 
                                     style={'color': '#8892b0'}),
                            html.Span(f"{imp['change']:+.1f}%", 
                                     style={'color': COLORS['green'] if imp['change'] > 0 else COLORS['pink'], 
                                           'fontWeight': '600'}),
                            html.Span(f" ({int(imp['prev'])} → {int(imp['curr'])})", 
                                     style={'color': '#8892b0', 'fontSize': '11px'})
                        ], style={'marginBottom': '8px'})
                        for i, imp in enumerate(improvements[:5])
                    ])
                else:
                    top_improvers = html.Div("No comparison data available")
            else:
                top_improvers = html.Div("No common MAs between periods")
        else:
            period_comparison = html.Div("Need at least 2 periods for comparison")
            top_improvers = html.Div("Need at least 2 periods for comparison")
    else:
        period_comparison = html.Div("No time series data available")
        top_improvers = html.Div("No time series data available")
    
    return (kpi_assigned, kpi_contacted, kpi_fas, kpi_funded, kpi_contact_rate, kpi_fas_rate,
            fig_fas_rate, fig_volume, fig_fas_vol, 
            fig_contact_heatmap, fig_fas_heatmap, fig_pt_heatmap,
            fig_fas_trend, fig_contact_trend,
            table_df.to_dict('records'), columns,
            period_comparison, top_improvers)

# =============================================================================
# AI Q&A CALLBACK
# =============================================================================
@callback(
    [Output('qa-response', 'children'),
     Output('qa-response', 'style')],
    [Input('qa-submit-btn', 'n_clicks')],
    [State('qa-input', 'value'),
     State('date-range-filter', 'start_date'),
     State('date-range-filter', 'end_date')],
    prevent_initial_call=True
)
def process_qa_question(n_clicks, question, start_date, end_date):
    """Process Q&A question and return AI-generated insights"""
    
    base_style = {
        'marginTop': '15px',
        'padding': '15px',
        'backgroundColor': '#0d1b2a',
        'borderRadius': '8px',
        'border': '1px solid #1b3a4b',
        'minHeight': '80px',
        'color': '#e6f1ff',
        'whiteSpace': 'pre-wrap',
        'display': 'block'
    }
    
    if not question or not question.strip():
        return html.Div([
            html.Span("⚠️ ", style={'color': '#ffc107'}),
            "Please enter a question about your MA performance data."
        ]), base_style
    
    try:
        # Fetch current MA data for context
        ma_query = get_ma_insights_query(start_date, end_date)
        df_ma = run_query(ma_query)
        
        if df_ma.empty:
            return html.Div([
                html.Span("⚠️ ", style={'color': '#ffc107'}),
                "No data available for the selected date range. Please adjust your filters."
            ]), base_style
        
        # Calculate key metrics for AI context
        df_ma['contact_rate'] = df_ma['contacted'] / df_ma['assigned_leads'].replace(0, np.nan)
        df_ma['fas_rate'] = df_ma['fas'] / df_ma['assigned_leads'].replace(0, np.nan)
        df_ma['funded_rate'] = df_ma['funded'] / df_ma['fas'].replace(0, np.nan)
        
        # Build data summary for AI
        total_assigned = df_ma['assigned_leads'].sum()
        total_contacted = df_ma['contacted'].sum()
        total_fas = df_ma['fas'].sum()
        total_funded = df_ma['funded'].sum()
        total_fas_volume = df_ma['fas_volume'].sum()
        
        avg_contact_rate = total_contacted / total_assigned if total_assigned > 0 else 0
        avg_fas_rate = total_fas / total_assigned if total_assigned > 0 else 0
        avg_funded_rate = total_funded / total_fas if total_fas > 0 else 0
        
        # Top and bottom performers
        df_reliable = df_ma[df_ma['assigned_leads'] >= 20].copy()
        
        top_fas_rate = df_reliable.nlargest(5, 'fas_rate')[['ma_name', 'assigned_leads', 'fas_rate', 'contact_rate']].to_dict('records') if not df_reliable.empty else []
        bottom_fas_rate = df_reliable.nsmallest(5, 'fas_rate')[['ma_name', 'assigned_leads', 'fas_rate', 'contact_rate']].to_dict('records') if not df_reliable.empty else []
        top_volume = df_ma.nlargest(5, 'assigned_leads')[['ma_name', 'assigned_leads', 'fas', 'fas_volume']].to_dict('records')
        top_fas_volume = df_ma.nlargest(5, 'fas_volume')[['ma_name', 'fas_volume', 'fas_rate', 'assigned_leads']].to_dict('records')
        
        # Build context for AI
        data_context = f"""
MORTGAGE ADVISOR PERFORMANCE DATA SUMMARY
Date Range: {start_date} to {end_date}
Total MAs Analyzed: {len(df_ma)}

OVERALL TEAM METRICS:
- Total Assigned Leads: {total_assigned:,}
- Total Contacted: {total_contacted:,} ({avg_contact_rate:.1%} contact rate)
- Total FAS: {total_fas:,} ({avg_fas_rate:.1%} FAS rate)
- Total Funded: {total_funded:,} ({avg_funded_rate:.1%} pull-through rate)
- Total FAS Volume: ${total_fas_volume:,.0f}

TOP 5 PERFORMERS BY FAS RATE (min 20 leads):
{json.dumps(top_fas_rate, indent=2, default=str)}

BOTTOM 5 PERFORMERS BY FAS RATE (min 20 leads):
{json.dumps(bottom_fas_rate, indent=2, default=str)}

TOP 5 BY VOLUME (Assigned Leads):
{json.dumps(top_volume, indent=2, default=str)}

TOP 5 BY FAS DOLLAR VOLUME:
{json.dumps(top_fas_volume, indent=2, default=str)}

ALL MA DATA (summary):
{df_ma[['ma_name', 'assigned_leads', 'contacted', 'fas', 'funded', 'fas_volume']].head(30).to_string()}
"""
        
        # Check if AI is available
        if AI_AVAILABLE and gemini_model:
            # Use Vertex AI Gemini
            prompt = f"""You are an expert sales performance analyst. Based on the following MA (Mortgage Advisor) performance data, answer this question concisely and insightfully:

QUESTION: {question}

{data_context}

Provide a clear, actionable answer in 3-5 bullet points. Focus on:
1. Direct answer to the question
2. Key insights from the data
3. Actionable recommendations if applicable

Be specific with names and numbers. Keep response under 200 words."""

            response = gemini_model.generate_content(prompt)
            ai_response = response.text
        else:
            # Fallback: Generate insights without AI
            ai_response = generate_rule_based_insights(question, df_ma, data_context, 
                                                       avg_contact_rate, avg_fas_rate, avg_funded_rate,
                                                       top_fas_rate, bottom_fas_rate, top_volume)
        
        return html.Div([
            html.Div([
                html.Span("🤖 ", style={'fontSize': '16px'}),
                html.Span("AI Insights", style={'color': '#4cc9f0', 'fontWeight': 'bold'}),
            ], style={'marginBottom': '10px', 'borderBottom': '1px solid #1b3a4b', 'paddingBottom': '8px'}),
            html.Div(ai_response, style={'lineHeight': '1.6'})
        ]), base_style
        
    except Exception as e:
        return html.Div([
            html.Span("❌ ", style={'color': '#f72585'}),
            f"Error processing question: {str(e)}"
        ]), base_style


def generate_rule_based_insights(question, df_ma, data_context, avg_contact_rate, avg_fas_rate, avg_funded_rate, 
                                  top_fas_rate, bottom_fas_rate, top_volume):
    """Generate insights using rule-based logic when AI is not available"""
    question_lower = question.lower()
    
    # Analyze question type and generate appropriate response
    if any(word in question_lower for word in ['top', 'best', 'performer', 'highest', 'leader']):
        if top_fas_rate:
            top = top_fas_rate[0]
            response = f"""📊 **Top Performers Analysis**

• **Best FAS Rate**: {top['ma_name']} leads with {top['fas_rate']:.1%} FAS rate on {top['assigned_leads']:,} assigned leads
• The top 5 performers average {sum(t['fas_rate'] for t in top_fas_rate)/len(top_fas_rate):.1%} FAS rate
• Team average FAS rate is {avg_fas_rate:.1%}
• Top performers are converting at {(top_fas_rate[0]['fas_rate']/avg_fas_rate - 1)*100:.0f}% above average

💡 **Recommendation**: Study the practices of top performers for team training opportunities."""
        else:
            response = "Not enough data with minimum 20 leads to identify reliable top performers."
            
    elif any(word in question_lower for word in ['bottom', 'worst', 'struggling', 'lowest', 'coaching', 'improve']):
        if bottom_fas_rate:
            bottom = bottom_fas_rate[0]
            response = f"""🎯 **Coaching Opportunities**

• **Needs Attention**: {bottom['ma_name']} has {bottom['fas_rate']:.1%} FAS rate (vs team avg {avg_fas_rate:.1%})
• Contact rate is {bottom['contact_rate']:.1%} - {'below' if bottom['contact_rate'] < avg_contact_rate else 'above'} team average
• {len([b for b in bottom_fas_rate if b['fas_rate'] < avg_fas_rate * 0.7])} MAs are performing >30% below average

💡 **Recommendation**: Focus coaching on contact rate first, as it's often the leading indicator of FAS performance."""
        else:
            response = "Not enough data to identify MAs needing coaching."
            
    elif any(word in question_lower for word in ['trend', 'change', 'improve', 'over time']):
        response = f"""📈 **Performance Overview**

• Current team FAS rate: {avg_fas_rate:.1%}
• Current contact rate: {avg_contact_rate:.1%}
• Pull-through rate: {avg_funded_rate:.1%}
• Total FAS volume: ${df_ma['fas_volume'].sum():,.0f}

💡 **Note**: For trend analysis, please review the trend charts below which show historical performance over time."""
        
    elif any(word in question_lower for word in ['volume', 'leads', 'capacity']):
        if top_volume:
            response = f"""📊 **Volume Analysis**

• **Highest Volume MA**: {top_volume[0]['ma_name']} with {top_volume[0]['assigned_leads']:,} assigned leads
• Top 5 MAs handle {sum(t['assigned_leads'] for t in top_volume):,} leads total
• Average leads per MA: {df_ma['assigned_leads'].mean():.0f}
• Median leads per MA: {df_ma['assigned_leads'].median():.0f}

💡 **Insight**: Volume distribution shows {'concentration' if df_ma['assigned_leads'].std() > df_ma['assigned_leads'].mean() else 'even distribution'} of leads among MAs."""
        else:
            response = "No volume data available."
            
    elif any(word in question_lower for word in ['contact', 'reach', 'call']):
        response = f"""📞 **Contact Rate Analysis**

• Team average contact rate: {avg_contact_rate:.1%}
• Total contacts made: {df_ma['contacted'].sum():,} out of {df_ma['assigned_leads'].sum():,} assigned
• Contact rate directly impacts FAS rate - correlation is typically strong

💡 **Key Insight**: For every 10% improvement in contact rate, expect roughly {avg_fas_rate/avg_contact_rate*0.1:.1%} improvement in FAS rate."""
        
    else:
        # General summary
        response = f"""📊 **MA Performance Summary**

• **Team Size**: {len(df_ma)} active MAs
• **Contact Rate**: {avg_contact_rate:.1%} team average
• **FAS Rate**: {avg_fas_rate:.1%} team average  
• **Pull-Through**: {avg_funded_rate:.1%} of FAS leads funded
• **Total Volume**: ${df_ma['fas_volume'].sum():,.0f} in FAS

🏆 **Top Performer**: {top_fas_rate[0]['ma_name'] if top_fas_rate else 'N/A'} ({top_fas_rate[0]['fas_rate']:.1%} FAS rate)

💡 Ask more specific questions like "Who needs coaching?" or "What are the trends?" for detailed insights."""
    
    return response


# =============================================================================
# RUN APP
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=8051)
