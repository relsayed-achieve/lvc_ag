"""
Sales Disposition Dashboard
Analyzes fallout reasons, call activity, and lead call results by vintage
Source: lendage-data-platform-ops.report_views.v_sales_disposition
"""

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from google.cloud import bigquery
import sys

def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()

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
    'chart_blue': '#3498db',
    'chart_red': '#e74c3c',
    'chart_gray': '#7f8c8d',
}

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

TAB_STYLE = {
    'backgroundColor': 'transparent',
    'color': COLORS['text_muted'],
    'border': 'none',
    'borderBottom': '2px solid transparent',
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
    title="Sales Disposition Analysis"
)

# Custom CSS for dark theme components
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Date Picker Dark Theme */
            .DateInput_input {
                background-color: #0d1e36 !important;
                color: #e0e6ed !important;
                border: 1px solid #1a3a5c !important;
                padding: 8px !important;
                font-size: 13px !important;
            }
            .DateRangePickerInput {
                background-color: #0d1e36 !important;
                border: 1px solid #1a3a5c !important;
                border-radius: 4px !important;
            }
            .DateRangePickerInput_arrow {
                color: #00d4aa !important;
            }
            .DateRangePickerInput_arrow svg {
                fill: #00d4aa !important;
            }
            .CalendarDay__selected {
                background: #00d4aa !important;
                border: 1px solid #00d4aa !important;
            }
            .CalendarDay__selected:hover {
                background: #00b894 !important;
            }
            .DayPickerKeyboardShortcuts_buttonReset {
                display: none;
            }
            
            /* Dropdown Dark Theme */
            .Select-control {
                background-color: #0d1e36 !important;
                border-color: #1a3a5c !important;
            }
            .Select-value-label, .Select-placeholder {
                color: #e0e6ed !important;
            }
            .Select-menu-outer {
                background-color: #0d1e36 !important;
                border-color: #1a3a5c !important;
            }
            .VirtualizedSelectOption {
                background-color: #0d1e36 !important;
                color: #e0e6ed !important;
            }
            .VirtualizedSelectFocusedOption {
                background-color: #1a3a5c !important;
            }
            
            /* Dash Dropdown Specific */
            .dash-dropdown .Select-control {
                background-color: #0d1e36 !important;
                border: 1px solid #1a3a5c !important;
            }
            .dash-dropdown .Select-value-label {
                color: #e0e6ed !important;
            }
            .dash-dropdown .Select-placeholder {
                color: #6b7c93 !important;
            }
            .dash-dropdown .Select-input input {
                color: #e0e6ed !important;
            }
            .dash-dropdown .Select-menu-outer {
                background-color: #0d1e36 !important;
                border: 1px solid #1a3a5c !important;
            }
            .dash-dropdown .Select-option {
                background-color: #0d1e36 !important;
                color: #e0e6ed !important;
            }
            .dash-dropdown .Select-option:hover,
            .dash-dropdown .Select-option.is-focused {
                background-color: #1a3a5c !important;
            }
            .dash-dropdown .Select-option.is-selected {
                background-color: #00d4aa !important;
                color: #0a1628 !important;
            }
            .dash-dropdown .Select-arrow {
                border-color: #6b7c93 transparent transparent !important;
            }
            .dash-dropdown .Select-clear {
                color: #6b7c93 !important;
            }
            .dash-dropdown .Select-multi-value-wrapper .Select-value {
                background-color: #1a3a5c !important;
                border-color: #00d4aa !important;
                color: #e0e6ed !important;
            }
            .dash-dropdown .Select-value-icon {
                border-color: #1a3a5c !important;
            }
            .dash-dropdown .Select-value-icon:hover {
                background-color: #ff6b6b !important;
                color: white !important;
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

# === BigQuery Connection ===
def get_bq_client():
    try:
        project_id = "ffn-dw-bigquery-prd"
        client = bigquery.Client(project=project_id)
        return client
    except Exception as e:
        log(f"BigQuery connection error: {e}")
        return None

def run_query(query):
    client = get_bq_client()
    if client:
        try:
            return client.query(query).to_dataframe()
        except Exception as e:
            log(f"Query error: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# === Data Query Functions ===
def get_filter_options():
    """Get unique values for filter dropdowns"""
    query = """
    SELECT DISTINCT
        lead_source,
        lead_source_group,
        mortgage_advisor,
        call_activity
    FROM `lendage-data-platform-ops.report_views.v_sales_disposition`
    WHERE DATE(lead_created_date) >= '2025-10-01'
    """
    return run_query(query)

def get_main_data(start_date='2025-10-01', end_date=None, contacted_flag=None):
    """Get all data for dashboard"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Build contacted filter
    contacted_filter = ""
    if contacted_flag is not None:
        contacted_filter = f"AND contacted_flag = {contacted_flag}"
    
    query = f"""
    SELECT
        DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as vintage_week,
        lead_source,
        lead_source_group,
        mortgage_advisor,
        call_activity,
        lead_call_result,
        ineligible_reason,
        inactive_reason,
        contacted_flag,
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
        COUNT(DISTINCT lendage_guid) as lead_count
    FROM `lendage-data-platform-ops.report_views.v_sales_disposition`
    WHERE DATE(lead_created_date) >= '{start_date}'
    AND DATE(lead_created_date) <= '{end_date}'
    {contacted_filter}
    GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    """
    return run_query(query)

# === Helper Functions ===
def create_section_header(title, subtitle=None):
    return html.Div([
        html.H5(title, style={
            'color': COLORS['text'],
            'fontSize': '16px',
            'fontWeight': '600',
            'letterSpacing': '0.5px',
            'marginBottom': '4px' if subtitle else '16px',
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
    delta_color = COLORS['success'] if (delta and delta > 0 and positive_is_good) or (delta and delta < 0 and not positive_is_good) else COLORS['danger'] if delta else COLORS['text_muted']
    delta_text = f"{delta:+.1f}{delta_suffix}" if delta is not None else ""
    
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
            'marginBottom': '8px', 
            'textTransform': 'uppercase',
            'letterSpacing': '0.5px',
            'fontWeight': '700',
        }),
        html.H3(value, style={
            'color': COLORS['primary'], 
            'margin': '0', 
            'fontSize': '28px', 
            'fontWeight': '400',
        }),
        html.P(delta_text, style={
            'color': delta_color, 
            'fontSize': '12px', 
            'marginTop': '8px',
            'fontWeight': '500',
        }) if delta_text else html.Div()
    ], style=METRIC_CARD_STYLE)

def create_stacked_bar_chart(df, x_col, y_col, color_col, title, color_map=None):
    """Create stacked bar chart"""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        color_discrete_map=color_map,
        barmode='stack'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        margin=dict(l=60, r=20, t=50, b=60),
        xaxis=dict(gridcolor=COLORS['border'], showgrid=False, tickangle=-45),
        yaxis=dict(gridcolor=COLORS['border'], showgrid=True, gridwidth=1),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        height=400,
    )
    return fig

def create_line_chart(df, x_col, y_col, color_col, title):
    """Create line chart for trends"""
    if color_col and color_col in df.columns:
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            markers=True
        )
    else:
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=title,
            markers=True
        )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        margin=dict(l=60, r=20, t=50, b=60),
        xaxis=dict(gridcolor=COLORS['border'], showgrid=False, tickangle=-45),
        yaxis=dict(gridcolor=COLORS['border'], showgrid=True, gridwidth=1),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        height=400,
    )
    return fig

def create_horizontal_bar_chart(df, x_col, y_col, title, color=None):
    """Create horizontal bar chart"""
    if color is None:
        color = COLORS['chart_cyan']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df[y_col],
        x=df[x_col],
        orientation='h',
        marker_color=color,
        text=df[x_col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else ""),
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=11)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS['text'], size=14)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        margin=dict(l=200, r=60, t=50, b=40),
        xaxis=dict(gridcolor=COLORS['border'], showgrid=True),
        yaxis=dict(gridcolor=COLORS['border'], showgrid=False, autorange='reversed'),
        height=max(300, len(df) * 25),
    )
    return fig

def create_data_table(df, id_name):
    """Create styled data table"""
    return dash_table.DataTable(
        id=id_name,
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': COLORS['card_bg'],
            'color': COLORS['primary'],
            'fontWeight': 'bold',
            'border': f"1px solid {COLORS['border']}",
            'fontSize': '12px',
            'textAlign': 'left',
            'padding': '10px',
        },
        style_cell={
            'backgroundColor': COLORS['background'],
            'color': COLORS['text'],
            'border': f"1px solid {COLORS['border']}",
            'fontSize': '12px',
            'textAlign': 'left',
            'padding': '8px',
            'minWidth': '80px',
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': COLORS['card_bg'],
            }
        ],
        page_size=15,
        sort_action='native',
        filter_action='native',
    )

# === Layout ===
app.layout = html.Div([
    # Header
    html.Div([
        html.H2("Sales Disposition Analysis", style={
            'color': COLORS['text'],
            'marginBottom': '5px',
            'fontWeight': '300',
            'letterSpacing': '1px',
        }),
        html.P("Contacted Leads | By Vintage (Lead Created Date)", style={
            'color': COLORS['text_muted'],
            'fontSize': '14px',
        }),
    ], style={'padding': '20px 30px', 'borderBottom': f"1px solid {COLORS['border']}"}),
    
    # Filters Row
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Date Range", style={'color': COLORS['text'], 'fontSize': '12px', 'marginBottom': '5px'}),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date='2025-10-01',
                    end_date=datetime.now().strftime('%Y-%m-%d'),
                    min_date_allowed='2025-01-01',
                    max_date_allowed=datetime.now().strftime('%Y-%m-%d'),
                    display_format='YYYY-MM-DD',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                    className='dark-date-picker'
                ),
            ], width=3),
            dbc.Col([
                html.Label("Lead Source Group", style={'color': COLORS['text'], 'fontSize': '12px', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='lead-source-group-filter',
                    multi=True,
                    placeholder="All Lead Source Groups",
                    style={'backgroundColor': COLORS['card_bg']}
                ),
            ], width=2),
            dbc.Col([
                html.Label("Lead Source", style={'color': COLORS['text'], 'fontSize': '12px', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='lead-source-filter',
                    multi=True,
                    placeholder="All Lead Sources",
                    style={'backgroundColor': COLORS['card_bg']}
                ),
            ], width=3),
            dbc.Col([
                html.Label("Mortgage Advisor", style={'color': COLORS['text'], 'fontSize': '12px', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='ma-filter',
                    multi=True,
                    placeholder="All MAs",
                    style={'backgroundColor': COLORS['card_bg']}
                ),
            ], width=2),
            dbc.Col([
                html.Label("Contacted Flag", style={'color': COLORS['text'], 'fontSize': '12px', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='contacted-filter',
                    options=[
                        {'label': 'All', 'value': 'all'},
                        {'label': 'Contacted (TRUE)', 'value': 'true'},
                        {'label': 'Not Contacted (FALSE)', 'value': 'false'},
                    ],
                    value='true',
                    clearable=False,
                    style={'backgroundColor': COLORS['card_bg']}
                ),
            ], width=2),
        ]),
    ], style={**CARD_STYLE, 'margin': '20px 30px'}),
    
    # Store for data (optional caching)
    dcc.Store(id='main-data-store'),
    
    # Tabs
    html.Div([
        dcc.Tabs(
            id='main-tabs',
            value='overview',
            children=[
                dcc.Tab(label='Overview', value='overview', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='Call Activity', value='call-activity', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='Lead Call Result', value='call-result', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='Fallout Reasons', value='fallout', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='By Lead Source', value='lead-source', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='By MA', value='by-ma', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            ],
            style={'borderBottom': f"1px solid {COLORS['border']}"}
        ),
    ], style={'margin': '0 30px'}),
    
    # Tab Content with Loading
    dcc.Loading(
        id="loading",
        type="circle",
        color=COLORS['primary'],
        children=[
            html.Div(id='tab-content', style={'padding': '20px 30px', 'minHeight': '400px'}),
        ]
    ),
    
], style={
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'fontFamily': "'Segoe UI', 'Roboto', sans-serif",
})

# === Callbacks ===
@app.callback(
    [Output('lead-source-group-filter', 'options'),
     Output('lead-source-filter', 'options'),
     Output('ma-filter', 'options')],
    Input('date-range', 'start_date')
)
def populate_filters(_):
    try:
        log("Populating filters...")
        df = get_filter_options()
        if df.empty:
            log("Filter options empty")
            return [], [], []
        
        lead_source_groups = [{'label': str(x), 'value': str(x)} for x in df['lead_source_group'].dropna().unique() if x]
        lead_sources = [{'label': str(x), 'value': str(x)} for x in df['lead_source'].dropna().unique() if x]
        mas = [{'label': str(x), 'value': str(x)} for x in df['mortgage_advisor'].dropna().unique() if x]
        
        log(f"Found {len(lead_source_groups)} source groups, {len(lead_sources)} sources, {len(mas)} MAs")
        return sorted(lead_source_groups, key=lambda x: x['label']), sorted(lead_sources, key=lambda x: x['label']), sorted(mas, key=lambda x: x['label'])
    except Exception as e:
        log(f"Error in populate_filters: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []

@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('lead-source-group-filter', 'value'),
     Input('lead-source-filter', 'value'),
     Input('ma-filter', 'value'),
     Input('contacted-filter', 'value')]
)
def render_tab_content(tab, start_date, end_date, lead_source_groups, lead_sources, mas, contacted_flag):
    log(f"=== Callback triggered: tab={tab}, start={start_date}, end={end_date}, contacted={contacted_flag} ===")
    
    # Set defaults if dates are missing
    if not start_date:
        start_date = '2025-10-01'
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Convert contacted filter value
    contacted_bool = None
    if contacted_flag == 'true':
        contacted_bool = True
    elif contacted_flag == 'false':
        contacted_bool = False
    # else 'all' means no filter
    
    log(f"Loading data from {start_date} to {end_date}, contacted={contacted_bool}...")
    
    try:
        df = get_main_data(start_date, end_date, contacted_flag=contacted_bool)
        
        if df.empty:
            log("No data returned from query!")
            return html.Div([
                html.H4("No data found", style={'color': COLORS['warning']}),
                html.P(f"Date range: {start_date} to {end_date}", style={'color': COLORS['text_muted']}),
                html.P("Try adjusting the date range or filters.", style={'color': COLORS['text_muted']}),
            ], style={'padding': '40px', 'textAlign': 'center'})
        
        log(f"Query returned {len(df)} rows")
        
        # Apply filters
        if lead_source_groups:
            df = df[df['lead_source_group'].isin(lead_source_groups)]
            log(f"After lead_source_group filter: {len(df)} rows")
        if lead_sources:
            df = df[df['lead_source'].isin(lead_sources)]
            log(f"After lead_source filter: {len(df)} rows")
        if mas:
            df = df[df['mortgage_advisor'].isin(mas)]
            log(f"After MA filter: {len(df)} rows")
        
        if df.empty:
            return html.Div([
                html.H4("No data matches filters", style={'color': COLORS['warning']}),
                html.P("Try adjusting the filters.", style={'color': COLORS['text_muted']}),
            ], style={'padding': '40px', 'textAlign': 'center'})
        
        df['vintage_week'] = pd.to_datetime(df['vintage_week'])
        log(f"Rendering {tab} with {len(df)} rows...")
        
        if tab == 'overview':
            return render_overview(df)
        elif tab == 'call-activity':
            return render_call_activity(df)
        elif tab == 'call-result':
            return render_call_result(df)
        elif tab == 'fallout':
            return render_fallout(df)
        elif tab == 'lead-source':
            return render_lead_source(df)
        elif tab == 'by-ma':
            return render_by_ma(df)
            
    except Exception as e:
        log(f"Error in callback: {e}")
        import traceback
        traceback.print_exc()
        return html.Div([
            html.H4("Error loading data", style={'color': COLORS['danger']}),
            html.P(str(e), style={'color': COLORS['text_muted']}),
        ], style={'padding': '40px', 'textAlign': 'center'})
    
    return html.Div("Select a tab")

def render_overview(df):
    """Render overview tab"""
    # Weekly totals
    weekly = df.groupby('vintage_week')['lead_count'].sum().reset_index()
    weekly = weekly.sort_values('vintage_week')
    
    # Calculate KPIs
    total_leads = df['lead_count'].sum()
    last_week = weekly.iloc[-1]['lead_count'] if len(weekly) > 0 else 0
    prev_week = weekly.iloc[-2]['lead_count'] if len(weekly) > 1 else 0
    wow_change = ((last_week - prev_week) / prev_week * 100) if prev_week > 0 else 0
    
    # Fallout rate
    fallout_df = df[df['fallout_reason'].notna()]
    fallout_total = fallout_df['lead_count'].sum()
    fallout_rate = (fallout_total / total_leads * 100) if total_leads > 0 else 0
    
    # Call activity breakdown
    call_activity = df.groupby('call_activity')['lead_count'].sum().reset_index()
    call_activity['pct'] = call_activity['lead_count'] / call_activity['lead_count'].sum() * 100
    
    # Lead call result breakdown
    call_result = df.groupby('lead_call_result')['lead_count'].sum().reset_index()
    call_result['pct'] = call_result['lead_count'] / call_result['lead_count'].sum() * 100
    
    return html.Div([
        # KPI Cards
        create_section_header("Key Metrics", "Contacted Leads Overview"),
        dbc.Row([
            dbc.Col(create_metric_card("Total Contacted", f"{total_leads:,}"), width=3),
            dbc.Col(create_metric_card("Last Week", f"{last_week:,}", delta=wow_change, delta_suffix="% WoW"), width=3),
            dbc.Col(create_metric_card("Fallout Rate", f"{fallout_rate:.1f}%", positive_is_good=False), width=3),
            dbc.Col(create_metric_card("Weeks", f"{len(weekly)}"), width=3),
        ], className="mb-4"),
        
        # Weekly Trend
        html.Div([
            create_section_header("Weekly Contacted Leads", "Vintage: Lead Created Date"),
            dcc.Graph(
                figure=create_line_chart(
                    weekly, 'vintage_week', 'lead_count', None,
                    "Contacted Leads by Week"
                ).update_traces(line_color=COLORS['chart_cyan'], marker_color=COLORS['chart_cyan'])
            ),
        ], style=CARD_STYLE),
        
        # Breakdowns
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header("Call Activity", "% of Total"),
                    dcc.Graph(
                        figure=create_horizontal_bar_chart(
                            call_activity.sort_values('lead_count', ascending=True).tail(10),
                            'lead_count', 'call_activity', "Top Call Activities"
                        )
                    ),
                ], style=CARD_STYLE),
            ], width=6),
            dbc.Col([
                html.Div([
                    create_section_header("Lead Call Result", "% of Total"),
                    dcc.Graph(
                        figure=create_horizontal_bar_chart(
                            call_result.sort_values('lead_count', ascending=True),
                            'lead_count', 'lead_call_result', "Lead Call Results"
                        )
                    ),
                ], style=CARD_STYLE),
            ], width=6),
        ]),
    ])

def render_call_activity(df):
    """Render call activity tab"""
    # Weekly breakdown
    weekly = df.groupby(['vintage_week', 'call_activity'])['lead_count'].sum().reset_index()
    weekly_total = df.groupby('vintage_week')['lead_count'].sum().reset_index()
    weekly_total.columns = ['vintage_week', 'total']
    weekly = weekly.merge(weekly_total, on='vintage_week')
    weekly['pct'] = weekly['lead_count'] / weekly['total'] * 100
    
    # Overall breakdown
    overall = df.groupby('call_activity')['lead_count'].sum().reset_index()
    overall['pct'] = overall['lead_count'] / overall['lead_count'].sum() * 100
    overall = overall.sort_values('lead_count', ascending=False)
    
    # Create pivot table for display
    pivot = weekly.pivot(index='vintage_week', columns='call_activity', values='lead_count').fillna(0)
    pivot = pivot.reset_index()
    pivot['vintage_week'] = pivot['vintage_week'].dt.strftime('%Y-%m-%d')
    
    # Percentage pivot
    pivot_pct = weekly.pivot(index='vintage_week', columns='call_activity', values='pct').fillna(0)
    pivot_pct = pivot_pct.reset_index()
    pivot_pct['vintage_week'] = pivot_pct['vintage_week'].dt.strftime('%Y-%m-%d')
    
    return html.Div([
        create_section_header("Call Activity Analysis", "Weekly breakdown by vintage"),
        
        # Stacked bar chart
        html.Div([
            dcc.Graph(
                figure=create_stacked_bar_chart(
                    weekly, 'vintage_week', 'lead_count', 'call_activity',
                    "Call Activity by Week (Count)"
                )
            ),
        ], style=CARD_STYLE),
        
        # Percentage trend lines
        html.Div([
            create_section_header("Call Activity % Trends", "Percentage of weekly total"),
            dcc.Graph(
                figure=create_line_chart(
                    weekly, 'vintage_week', 'pct', 'call_activity',
                    "Call Activity % by Week"
                )
            ),
        ], style=CARD_STYLE),
        
        # Tables
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header("Weekly Counts", "Lead count by call activity"),
                    create_data_table(pivot, 'call-activity-count-table'),
                ], style=CARD_STYLE),
            ], width=12),
        ]),
        
        # Overall summary
        html.Div([
            create_section_header("Overall Summary", "Total period breakdown"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=create_horizontal_bar_chart(
                            overall, 'lead_count', 'call_activity', "Call Activity Distribution"
                        )
                    ),
                ], width=6),
                dbc.Col([
                    create_data_table(
                        overall.round(1).rename(columns={'lead_count': 'Leads', 'pct': '% of Total'}),
                        'call-activity-summary-table'
                    ),
                ], width=6),
            ]),
        ], style=CARD_STYLE),
    ])

def render_call_result(df):
    """Render lead call result tab"""
    # Weekly breakdown
    weekly = df.groupby(['vintage_week', 'lead_call_result'])['lead_count'].sum().reset_index()
    weekly_total = df.groupby('vintage_week')['lead_count'].sum().reset_index()
    weekly_total.columns = ['vintage_week', 'total']
    weekly = weekly.merge(weekly_total, on='vintage_week')
    weekly['pct'] = weekly['lead_count'] / weekly['total'] * 100
    
    # Overall breakdown
    overall = df.groupby('lead_call_result')['lead_count'].sum().reset_index()
    overall['pct'] = overall['lead_count'] / overall['lead_count'].sum() * 100
    overall = overall.sort_values('lead_count', ascending=False)
    
    # Pivot table
    pivot = weekly.pivot(index='vintage_week', columns='lead_call_result', values='lead_count').fillna(0)
    pivot = pivot.reset_index()
    pivot['vintage_week'] = pivot['vintage_week'].dt.strftime('%Y-%m-%d')
    
    return html.Div([
        create_section_header("Lead Call Result Analysis", "Weekly breakdown by vintage"),
        
        # Stacked bar chart
        html.Div([
            dcc.Graph(
                figure=create_stacked_bar_chart(
                    weekly, 'vintage_week', 'lead_count', 'lead_call_result',
                    "Lead Call Result by Week (Count)"
                )
            ),
        ], style=CARD_STYLE),
        
        # Percentage trend lines
        html.Div([
            create_section_header("Lead Call Result % Trends", "Percentage of weekly total"),
            dcc.Graph(
                figure=create_line_chart(
                    weekly, 'vintage_week', 'pct', 'lead_call_result',
                    "Lead Call Result % by Week"
                )
            ),
        ], style=CARD_STYLE),
        
        # Table
        html.Div([
            create_section_header("Weekly Counts", "Lead count by call result"),
            create_data_table(pivot, 'call-result-table'),
        ], style=CARD_STYLE),
        
        # Overall summary
        html.Div([
            create_section_header("Overall Summary", "Total period breakdown"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=create_horizontal_bar_chart(
                            overall, 'lead_count', 'lead_call_result', "Lead Call Result Distribution"
                        )
                    ),
                ], width=6),
                dbc.Col([
                    create_data_table(
                        overall.round(1).rename(columns={'lead_count': 'Leads', 'pct': '% of Total'}),
                        'call-result-summary-table'
                    ),
                ], width=6),
            ]),
        ], style=CARD_STYLE),
    ])

def render_fallout(df):
    """Render fallout reasons tab"""
    # Filter to only fallout records
    fallout_df = df[df['fallout_reason'].notna()]
    
    # Weekly breakdown by category
    weekly_cat = fallout_df.groupby(['vintage_week', 'fallout_category'])['lead_count'].sum().reset_index()
    
    # Weekly breakdown by reason
    weekly = fallout_df.groupby(['vintage_week', 'fallout_reason'])['lead_count'].sum().reset_index()
    weekly_total = fallout_df.groupby('vintage_week')['lead_count'].sum().reset_index()
    weekly_total.columns = ['vintage_week', 'total']
    weekly = weekly.merge(weekly_total, on='vintage_week')
    weekly['pct'] = weekly['lead_count'] / weekly['total'] * 100
    
    # Overall breakdown
    overall = fallout_df.groupby('fallout_reason')['lead_count'].sum().reset_index()
    overall['pct'] = overall['lead_count'] / overall['lead_count'].sum() * 100
    overall = overall.sort_values('lead_count', ascending=False)
    
    # Category breakdown
    category = fallout_df.groupby('fallout_category')['lead_count'].sum().reset_index()
    category['pct'] = category['lead_count'] / category['lead_count'].sum() * 100
    
    return html.Div([
        create_section_header("Fallout Reason Analysis", "Ineligible and Inactive Leads"),
        
        # Category breakdown
        html.Div([
            create_section_header("Fallout Category", "Ineligible vs Inactive"),
            dcc.Graph(
                figure=create_stacked_bar_chart(
                    weekly_cat, 'vintage_week', 'lead_count', 'fallout_category',
                    "Fallout Category by Week",
                    color_map={'Ineligible': COLORS['chart_red'], 'Inactive': COLORS['chart_yellow']}
                )
            ),
        ], style=CARD_STYLE),
        
        # Top fallout reasons bar chart
        html.Div([
            create_section_header("Top Fallout Reasons", "Overall breakdown"),
            dcc.Graph(
                figure=create_horizontal_bar_chart(
                    overall.head(15).sort_values('lead_count', ascending=True),
                    'lead_count', 'fallout_reason', "Top 15 Fallout Reasons"
                )
            ),
        ], style=CARD_STYLE),
        
        # Trends for top reasons
        html.Div([
            create_section_header("Fallout Reason % Trends", "Top 10 reasons over time"),
            dcc.Graph(
                figure=create_line_chart(
                    weekly[weekly['fallout_reason'].isin(overall.head(10)['fallout_reason'])],
                    'vintage_week', 'pct', 'fallout_reason',
                    "Fallout Reason % by Week (Top 10)"
                )
            ),
        ], style=CARD_STYLE),
        
        # Full table
        html.Div([
            create_section_header("All Fallout Reasons", "Complete breakdown"),
            create_data_table(
                overall.round(1).rename(columns={'lead_count': 'Leads', 'pct': '% of Fallout'}),
                'fallout-table'
            ),
        ], style=CARD_STYLE),
    ])

def render_lead_source(df):
    """Render lead source tab"""
    # By lead source group
    source_group = df.groupby('lead_source_group')['lead_count'].sum().reset_index()
    source_group['pct'] = source_group['lead_count'] / source_group['lead_count'].sum() * 100
    source_group = source_group.sort_values('lead_count', ascending=False)
    
    # By lead source
    source = df.groupby('lead_source')['lead_count'].sum().reset_index()
    source['pct'] = source['lead_count'] / source['lead_count'].sum() * 100
    source = source.sort_values('lead_count', ascending=False)
    
    # Weekly by lead source group
    weekly_group = df.groupby(['vintage_week', 'lead_source_group'])['lead_count'].sum().reset_index()
    
    # Fallout by lead source group
    fallout_df = df[df['fallout_reason'].notna()]
    fallout_by_source = fallout_df.groupby(['lead_source_group', 'fallout_category'])['lead_count'].sum().reset_index()
    
    return html.Div([
        create_section_header("Lead Source Analysis", "Breakdown by source"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header("By Lead Source Group", ""),
                    dcc.Graph(
                        figure=create_horizontal_bar_chart(
                            source_group.head(15).sort_values('lead_count', ascending=True),
                            'lead_count', 'lead_source_group', "Lead Source Groups"
                        )
                    ),
                ], style=CARD_STYLE),
            ], width=6),
            dbc.Col([
                html.Div([
                    create_section_header("By Lead Source", ""),
                    dcc.Graph(
                        figure=create_horizontal_bar_chart(
                            source.head(15).sort_values('lead_count', ascending=True),
                            'lead_count', 'lead_source', "Top Lead Sources"
                        )
                    ),
                ], style=CARD_STYLE),
            ], width=6),
        ]),
        
        # Weekly trend by source group
        html.Div([
            create_section_header("Weekly Trend by Lead Source Group", ""),
            dcc.Graph(
                figure=create_stacked_bar_chart(
                    weekly_group, 'vintage_week', 'lead_count', 'lead_source_group',
                    "Contacted Leads by Source Group"
                )
            ),
        ], style=CARD_STYLE),
        
        # Fallout by source
        html.Div([
            create_section_header("Fallout by Lead Source Group", "Ineligible vs Inactive"),
            dcc.Graph(
                figure=create_stacked_bar_chart(
                    fallout_by_source, 'lead_source_group', 'lead_count', 'fallout_category',
                    "Fallout Category by Source Group",
                    color_map={'Ineligible': COLORS['chart_red'], 'Inactive': COLORS['chart_yellow']}
                )
            ),
        ], style=CARD_STYLE),
    ])

def render_by_ma(df):
    """Render by MA tab"""
    # By MA
    ma = df.groupby('mortgage_advisor')['lead_count'].sum().reset_index()
    ma['pct'] = ma['lead_count'] / ma['lead_count'].sum() * 100
    ma = ma.sort_values('lead_count', ascending=False)
    
    # Weekly by top MAs
    top_mas = ma.head(10)['mortgage_advisor'].tolist()
    weekly_ma = df[df['mortgage_advisor'].isin(top_mas)].groupby(['vintage_week', 'mortgage_advisor'])['lead_count'].sum().reset_index()
    
    # Fallout rate by MA
    fallout_df = df[df['fallout_reason'].notna()]
    ma_fallout = fallout_df.groupby('mortgage_advisor')['lead_count'].sum().reset_index()
    ma_fallout.columns = ['mortgage_advisor', 'fallout_count']
    ma_with_fallout = ma.merge(ma_fallout, on='mortgage_advisor', how='left')
    ma_with_fallout['fallout_count'] = ma_with_fallout['fallout_count'].fillna(0)
    ma_with_fallout['fallout_rate'] = ma_with_fallout['fallout_count'] / ma_with_fallout['lead_count'] * 100
    
    # Call result by MA
    ma_result = df.groupby(['mortgage_advisor', 'lead_call_result'])['lead_count'].sum().reset_index()
    ma_result_top = ma_result[ma_result['mortgage_advisor'].isin(top_mas)]
    
    return html.Div([
        create_section_header("Mortgage Advisor Analysis", "Performance by MA"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    create_section_header("Top MAs by Volume", ""),
                    dcc.Graph(
                        figure=create_horizontal_bar_chart(
                            ma.head(20).sort_values('lead_count', ascending=True),
                            'lead_count', 'mortgage_advisor', "Top 20 MAs"
                        )
                    ),
                ], style=CARD_STYLE),
            ], width=6),
            dbc.Col([
                html.Div([
                    create_section_header("Fallout Rate by MA", "Top 20 by volume"),
                    dcc.Graph(
                        figure=create_horizontal_bar_chart(
                            ma_with_fallout.head(20).sort_values('fallout_rate', ascending=True),
                            'fallout_rate', 'mortgage_advisor', "Fallout Rate %"
                        )
                    ),
                ], style=CARD_STYLE),
            ], width=6),
        ]),
        
        # Weekly trend by top MAs
        html.Div([
            create_section_header("Weekly Trend - Top 10 MAs", ""),
            dcc.Graph(
                figure=create_line_chart(
                    weekly_ma, 'vintage_week', 'lead_count', 'mortgage_advisor',
                    "Contacted Leads by MA"
                )
            ),
        ], style=CARD_STYLE),
        
        # Call result by MA
        html.Div([
            create_section_header("Lead Call Result by MA", "Top 10 MAs"),
            dcc.Graph(
                figure=create_stacked_bar_chart(
                    ma_result_top, 'mortgage_advisor', 'lead_count', 'lead_call_result',
                    "Lead Call Result Distribution"
                )
            ),
        ], style=CARD_STYLE),
        
        # Full MA table
        html.Div([
            create_section_header("All MAs", "Complete breakdown"),
            create_data_table(
                ma_with_fallout[['mortgage_advisor', 'lead_count', 'pct', 'fallout_count', 'fallout_rate']].round(1).rename(
                    columns={'lead_count': 'Leads', 'pct': '% of Total', 'fallout_count': 'Fallout', 'fallout_rate': 'Fallout Rate %'}
                ),
                'ma-table'
            ),
        ], style=CARD_STYLE),
    ])

# === Run App ===
if __name__ == '__main__':
    log("Starting Sales Disposition Dashboard...")
    app.run(debug=True, port=8052, host='127.0.0.1')
