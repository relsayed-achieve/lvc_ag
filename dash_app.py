import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import json

# --- Launch Events Configuration ---
LAUNCH_EVENTS = [
    {"name": "LVS at 10%", "date": "2025-07-01"},
    {"name": "LVS at 100%", "date": "2025-08-30"},
    {"name": "New Genesys Queue", "date": "2025-10-14"},
    {"name": "Tiering (2 in bottom, rest T1)", "date": "2025-10-14"},
    {"name": "Routing (waiting in queue)", "date": "2025-10-14"},
    {"name": "Insider Journey (original)", "date": "2025-10-14"},
    {"name": "Feathering removed from MAs", "date": "2025-11-15"},
    {"name": "Delay of cohort 1 and 2 to PHX", "date": "2025-11-26"},
    {"name": "Adding 1, 3, 5 Starring", "date": "2025-12-02"},
]

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
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
app.title = "LVC Report Breakout"

# Custom CSS for Executive Insights tab
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .insights-container {
                background: linear-gradient(135deg, #0a1628 0%, #1a2744 100%);
                min-height: 100vh;
                padding: 20px;
                border-radius: 8px;
            }
            .insight-card {
                background: linear-gradient(145deg, #0d1b2a 0%, #162a40 100%);
                border: 1px solid rgba(76, 201, 240, 0.2);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 15px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
            }
            .insight-card:hover {
                border-color: rgba(76, 201, 240, 0.5);
                box-shadow: 0 6px 25px rgba(76, 201, 240, 0.15);
            }
            .insight-title {
                color: #4cc9f0;
                font-size: 14px;
                font-weight: 600;
                margin-bottom: 15px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .kpi-big {
                color: #ffffff;
                font-size: 36px;
                font-weight: 700;
            }
            .kpi-medium {
                color: #ffffff;
                font-size: 24px;
                font-weight: 600;
            }
            .kpi-label {
                color: #8892b0;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .badge-up {
                background: linear-gradient(135deg, #00d084 0%, #00b371 100%);
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
            }
            .badge-down {
                background: linear-gradient(135deg, #f72585 0%, #b5179e 100%);
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
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

# --- Helper Functions ---
def build_where_clause(start_date, end_date, selected_lvc=None):
    conditions = []
    if start_date and end_date:
        conditions.append(f"DATE(lead_created_date) BETWEEN '{start_date}' AND '{end_date}'")
    if selected_lvc:
        safe_lvc = [f"'{x}'" for x in selected_lvc]
        conditions.append(f"adjusted_lead_value_cohort IN ({', '.join(safe_lvc)})")
    if conditions:
        return "WHERE " + " AND ".join(conditions)
    return ""

def run_query(sql_query):
    """Execute BigQuery and return DataFrame"""
    if client is None:
        return pd.DataFrame()
    try:
        return client.query(sql_query).to_dataframe()
    except Exception as e:
        print(f"Query error: {e}")
        return pd.DataFrame()

def bayesian_adjusted_rate(conversions, volume, overall_rate, prior_strength=50):
    """Bayesian shrinkage for conversion rates"""
    return (conversions + (overall_rate * prior_strength)) / (volume + prior_strength)

def wilson_ci(conversions, volume, confidence=0.95):
    """Wilson score confidence interval"""
    if volume == 0:
        return (0, 0, 0)
    z = stats.norm.ppf((1 + confidence) / 2)
    p = conversions / volume
    denominator = 1 + z**2 / volume
    center = (p + z**2 / (2 * volume)) / denominator
    spread = z * ((p * (1 - p) / volume + z**2 / (4 * volume**2)) ** 0.5) / denominator
    return (max(0, center - spread), min(1, center + spread), spread * 2)

# --- Sidebar Layout ---
sidebar = dbc.Card([
    dbc.CardHeader(html.H5("🔍 Filters", className="mb-0")),
    dbc.CardBody([
        html.Label("Lead Created Date Range"),
        dcc.DatePickerRange(
            id='date-range',
            min_date_allowed=date(2020, 1, 1),
            max_date_allowed=date.today(),
            start_date=date(2025, 10, 1),
            end_date=date.today(),
            className="mb-3"
        ),
        html.Hr(),
        html.Label("Adjusted Lead Value Cohort"),
        dcc.Dropdown(
            id='lvc-filter',
            options=[{'label': str(i), 'value': str(i)} for i in range(1, 11)] + 
                    [{'label': 'X', 'value': 'X'}, {'label': 'Other', 'value': 'Other'}],
            multi=True,
            placeholder="Select LVC values...",
            className="mb-3"
        ),
        html.Hr(),
        html.Div([
            html.Strong("Connection Status: "),
            html.Span(CONNECTION_STATUS, id='connection-status')
        ])
    ])
], className="mb-3", style={"position": "sticky", "top": "10px"})

# --- Tab Content Components ---

# Tab 1: Main Dashboard
tab_main_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("📊 Main Dashboard"),
            html.P("Overview of key LVC metrics", className="text-muted")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Time Grain"),
                dbc.CardBody([
                    dcc.RadioItems(
                        id='main-time-grain',
                        options=[
                            {'label': ' Day', 'value': 'Day'},
                            {'label': ' Week', 'value': 'Week'},
                            {'label': ' Month', 'value': 'Month'}
                        ],
                        value='Week',
                        inline=True
                    )
                ])
            ])
        ], width=4),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Total Leads", className="text-muted"),
                html.H3(id='kpi-total-leads', children="--")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Sent to Sales", className="text-muted"),
                html.H3(id='kpi-sent-to-sales', children="--")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Contacted", className="text-muted"),
                html.H3(id='kpi-contacted', children="--")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("FAS Count", className="text-muted"),
                html.H3(id='kpi-fas-count', children="--")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("FAS Rate", className="text-muted"),
                html.H3(id='kpi-fas-rate', children="--")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Contact Rate", className="text-muted"),
                html.H3(id='kpi-contact-rate', children="--")
            ])
        ]), width=2),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sent to Sales Volume by LVC Group"),
                dbc.CardBody([dcc.Graph(id='main-sts-chart')])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("FAS Rate by LVC Group"),
                dbc.CardBody([dcc.Graph(id='main-fas-rate-chart')])
            ])
        ], width=6),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Contact Rate by LVC Group"),
                dbc.CardBody([dcc.Graph(id='main-contact-rate-chart')])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Avg LP2C Score by LVC Group"),
                dbc.CardBody([dcc.Graph(id='main-lp2c-chart')])
            ])
        ], width=6),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H5("Raw Data"),
            dash_table.DataTable(
                id='main-data-table',
                page_size=20,
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#303030', 'color': 'white', 'fontWeight': 'bold'},
                style_cell={'backgroundColor': '#404040', 'color': 'white', 'textAlign': 'left'},
            )
        ])
    ])
], fluid=True)

# Tab 2: Pre/Post Launch Analysis
tab_launch_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("🚀 Pre/Post Launch Analysis"),
            html.P("Analyze performance before and after product launches", className="text-muted")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Select Launch Event"),
                    dcc.Dropdown(
                        id='launch-event-select',
                        options=[{'label': 'All Launches', 'value': 'all'}] + 
                                [{'label': f"{e['name']} ({e['date']})", 'value': e['date']} for e in LAUNCH_EVENTS],
                        value='all',
                        className="mb-2"
                    ),
                    html.Label("Time Grain"),
                    dcc.RadioItems(
                        id='launch-time-grain',
                        options=[
                            {'label': ' Day', 'value': 'Day'},
                            {'label': ' Week', 'value': 'Week'},
                            {'label': ' Month', 'value': 'Month'}
                        ],
                        value='Week',
                        inline=True,
                        className="mb-2"
                    ),
                    html.Label("Compare Window"),
                    dcc.Dropdown(
                        id='launch-compare-window',
                        options=[
                            {'label': '30 Days', 'value': 30},
                            {'label': '60 Days', 'value': 60},
                            {'label': '90 Days', 'value': 90},
                            {'label': '120 Days', 'value': 120},
                            {'label': 'Back to 1/2024', 'value': 'all'}
                        ],
                        value=90
                    )
                ])
            ])
        ], width=12)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Pre-Launch FAS Rate", className="text-muted"),
                html.H3(id='launch-kpi-pre-fas', children="--"),
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Post-Launch FAS Rate", className="text-muted"),
                html.H3(id='launch-kpi-post-fas', children="--"),
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Pre-Launch Contact Rate", className="text-muted"),
                html.H3(id='launch-kpi-pre-contact', children="--"),
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Post-Launch Contact Rate", className="text-muted"),
                html.H3(id='launch-kpi-post-contact', children="--"),
            ])
        ]), width=3),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("FAS Rate Over Time"),
                dbc.CardBody([dcc.Graph(id='launch-fas-chart')])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Contact Rate Over Time"),
                dbc.CardBody([dcc.Graph(id='launch-contact-chart')])
            ])
        ], width=6),
    ]),
], fluid=True)

# Tab 3: Frank & Felix Growth
tab_frank_felix_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("🎯 Frank & Felix Growth Analysis"),
            html.P("Tracking Frank & Felix persona performance (Oct 2025 - Jan 2026)", className="text-muted")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Time Grain"),
                    dcc.RadioItems(
                        id='ff-time-grain',
                        options=[
                            {'label': ' Day', 'value': 'Day'},
                            {'label': ' Week', 'value': 'Week'},
                            {'label': ' Month', 'value': 'Month'}
                        ],
                        value='Week',
                        inline=True
                    )
                ])
            ])
        ], width=4)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Frank & Felix % of Total", className="text-muted"),
                html.H3(id='ff-kpi-pct', children="--"),
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Frank & Felix FAS Rate", className="text-muted"),
                html.H3(id='ff-kpi-fas-rate', children="--"),
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Others FAS Rate", className="text-muted"),
                html.H3(id='ff-kpi-other-fas-rate', children="--"),
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("F&F Avg Loan Size", className="text-muted"),
                html.H3(id='ff-kpi-avg-loan', children="--"),
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("FAS $ per StS (F&F)", className="text-muted"),
                html.H3(id='ff-kpi-fas-per-sts', children="--"),
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Pull Through (F&F)", className="text-muted"),
                html.H3(id='ff-kpi-pull-through', children="--"),
            ])
        ]), width=2),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Frank & Felix % of Total Sent to Sales"),
                dbc.CardBody([dcc.Graph(id='ff-pct-chart')])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("FAS Rate: Frank & Felix vs Others"),
                dbc.CardBody([dcc.Graph(id='ff-fas-rate-chart')])
            ])
        ], width=6),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("FAS $ per Lead Sent to Sales"),
                dbc.CardBody([dcc.Graph(id='ff-fas-per-sts-chart')])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Lead Value Driven by Loan Size"),
                dbc.CardBody([dcc.Graph(id='ff-value-concentration-chart')])
            ])
        ], width=6),
    ]),
], fluid=True)

# Tab 4: Agent Performance
tab_agent_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("⭐ Agent Performance Analysis"),
            html.P("Normalizing performance metrics across star tiers", className="text-muted")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H5("The Problem", className="alert-heading"),
                html.P([
                    "Agents are grouped into ", html.Strong("5-star, 3-star, and 1-star"), " tiers. ",
                    "Curated leads are routed with priority: 5-star gets first pick → High volume. ",
                    "1-star gets double-filtered leads → Low volume, artificially high conversion rate."
                ]),
                html.P([
                    html.Strong("This creates misleading metrics"), " - a 1-star with 1/3 conversions (33%) ",
                    "looks better than a 5-star with 15/150 (10%), but the 1-star's rate is statistically unreliable."
                ])
            ], color="warning")
        ])
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Bayesian Prior Strength"),
                dbc.CardBody([
                    dcc.Slider(
                        id='agent-prior-strength',
                        min=10, max=100, step=10, value=50,
                        marks={i: str(i) for i in range(10, 101, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Small("Higher = more shrinkage toward average for low-volume agents", className="text-muted")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Volume Threshold for Full Credit"),
                dbc.CardBody([
                    dcc.Slider(
                        id='agent-volume-threshold',
                        min=20, max=100, step=10, value=50,
                        marks={i: str(i) for i in range(20, 101, 20)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Small("Agents below this volume get penalized in score", className="text-muted")
                ])
            ])
        ], width=6),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Solution 1: Raw vs Bayesian Adjusted Rates"),
                dbc.CardBody([dcc.Graph(id='agent-bayesian-chart')])
            ])
        ], width=12)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Solution 2: Confidence Intervals (Wilson Score)"),
                dbc.CardBody([dcc.Graph(id='agent-ci-chart')])
            ])
        ], width=12)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Agent Performance Table"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='agent-table',
                        style_table={'overflowX': 'auto'},
                        style_header={'backgroundColor': '#303030', 'color': 'white', 'fontWeight': 'bold'},
                        style_cell={'backgroundColor': '#404040', 'color': 'white', 'textAlign': 'left'},
                    )
                ])
            ])
        ])
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📋 Promotion Readiness Framework"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='agent-promotion-table',
                        style_table={'overflowX': 'auto'},
                        style_header={'backgroundColor': '#303030', 'color': 'white', 'fontWeight': 'bold'},
                        style_cell={'backgroundColor': '#404040', 'color': 'white', 'textAlign': 'left'},
                        style_data_conditional=[
                            {'if': {'filter_query': '{Status} = "✅ Ready"'}, 'backgroundColor': '#2e7d32'},
                            {'if': {'filter_query': '{Status} = "❌ Not Yet"'}, 'backgroundColor': '#c62828'},
                            {'if': {'filter_query': '{Status} = "⏳ Need More Data"'}, 'backgroundColor': '#f57c00'},
                        ]
                    )
                ])
            ])
        ])
    ]),
], fluid=True)

# Tab 5: LVC to Persona Analysis
tab_persona_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("👤 LVC & Persona Analysis"),
            html.P("Understanding lead flow from LVC groups to personas", className="text-muted")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("LVC to Persona Flow (Sankey)"),
                dbc.CardBody([dcc.Graph(id='persona-sankey-chart')])
            ])
        ], width=12)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("FAS Rate Heatmap: LVC × Persona"),
                dbc.CardBody([dcc.Graph(id='persona-heatmap')])
            ])
        ], width=12)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Persona Summary Table"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='persona-table',
                        page_size=15,
                        style_table={'overflowX': 'auto'},
                        style_header={'backgroundColor': '#303030', 'color': 'white', 'fontWeight': 'bold'},
                        style_cell={'backgroundColor': '#404040', 'color': 'white', 'textAlign': 'left'},
                    )
                ])
            ])
        ])
    ])
], fluid=True)

# --- Custom CSS for Insights Tab ---
INSIGHTS_CARD_STYLE = {
    'backgroundColor': '#0d1b2a',
    'border': '1px solid #1b3a4b',
    'borderRadius': '12px',
    'padding': '15px',
    'marginBottom': '15px'
}

INSIGHTS_HEADER_STYLE = {
    'color': '#4cc9f0',
    'fontSize': '14px',
    'fontWeight': '600',
    'marginBottom': '10px',
    'borderBottom': '1px solid #1b3a4b',
    'paddingBottom': '8px'
}

KPI_NUMBER_STYLE = {
    'color': '#ffffff',
    'fontSize': '28px',
    'fontWeight': '700',
    'margin': '0'
}

KPI_LABEL_STYLE = {
    'color': '#8892b0',
    'fontSize': '12px',
    'textTransform': 'uppercase'
}

# Tab 6: Executive Insights Dashboard
# Custom CSS for insights tab (injected via app's index_string or external)
INSIGHTS_CSS = '''
        .insights-container {
            background: linear-gradient(135deg, #0a1628 0%, #1a2744 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .insight-card {
            background: linear-gradient(145deg, #0d1b2a 0%, #162a40 100%);
            border: 1px solid rgba(76, 201, 240, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        .insight-card:hover {
            border-color: rgba(76, 201, 240, 0.5);
            box-shadow: 0 6px 25px rgba(76, 201, 240, 0.15);
        }
        .insight-title {
            color: #4cc9f0;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .kpi-big {
            color: #ffffff;
            font-size: 36px;
            font-weight: 700;
        }
        .kpi-medium {
            color: #ffffff;
            font-size: 24px;
            font-weight: 600;
        }
        .kpi-label {
            color: #8892b0;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-up {
            background: linear-gradient(135deg, #00d084 0%, #00b371 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-down {
            background: linear-gradient(135deg, #f72585 0%, #b5179e 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .progress-ring {
            transform: rotate(-90deg);
        }
    '''

tab_insights_content = html.Div([
    html.Div([
        # Header Row
        dbc.Row([
            dbc.Col([
                html.H3("📈 Executive Insights Dashboard", 
                       style={'color': '#ffffff', 'fontWeight': '600', 'marginBottom': '5px'}),
                html.P("Real-time performance overview", 
                      style={'color': '#8892b0', 'fontSize': '14px', 'marginBottom': '20px'})
            ], width=8),
            dbc.Col([
                html.Div([
                    html.Span("Last Updated: ", style={'color': '#8892b0', 'fontSize': '12px'}),
                    html.Span(datetime.now().strftime("%Y-%m-%d"), 
                             style={'color': '#4cc9f0', 'fontSize': '14px', 'fontWeight': '600'})
                ], style={'textAlign': 'right'})
            ], width=4)
        ], className="mb-4"),
        
        # Main Content Row
        dbc.Row([
            # Left Column - Distribution Charts
            dbc.Col([
                # LVC Distribution (Donut Chart)
                html.Div([
                    html.Div("LVC Distribution", className="insight-title"),
                    dcc.Graph(id='insights-lvc-donut', config={'displayModeBar': False},
                             style={'height': '220px'})
                ], className="insight-card"),
                
                # Persona Distribution (Radar Chart)
                html.Div([
                    html.Div("Persona Distribution", className="insight-title"),
                    dcc.Graph(id='insights-persona-radar', config={'displayModeBar': False},
                             style={'height': '250px'})
                ], className="insight-card"),
                
                # Weekly Volume (Bar Chart)
                html.Div([
                    html.Div("Weekly Volume", className="insight-title"),
                    dcc.Graph(id='insights-weekly-bars', config={'displayModeBar': False},
                             style={'height': '180px'})
                ], className="insight-card"),
            ], width=3),
            
            # Center Column - Main Metrics & Trend
            dbc.Col([
                # Top KPIs Row
                html.Div([
                    html.Div("Data Exchange", className="insight-title"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span(id='insights-sts-value', className="kpi-big", 
                                         style={'marginRight': '10px'}),
                                html.Span(id='insights-sts-badge', className="badge-up")
                            ]),
                            html.Div("● Sent to Sales", style={'color': '#4cc9f0', 'fontSize': '12px', 'marginTop': '5px'})
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.Span(id='insights-fas-value', className="kpi-big",
                                         style={'marginRight': '10px'}),
                                html.Span(id='insights-fas-badge', className="badge-up")
                            ]),
                            html.Div("● FAS Count", style={'color': '#f72585', 'fontSize': '12px', 'marginTop': '5px'})
                        ], width=6),
                    ]),
                    # Area Chart
                    dcc.Graph(id='insights-trend-area', config={'displayModeBar': False},
                             style={'height': '200px', 'marginTop': '15px'})
                ], className="insight-card"),
                
                # Monthly KPIs
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div("No.1", style={'color': '#8892b0', 'fontSize': '11px'}),
                            html.Div(id='insights-month1-value', className="kpi-medium"),
                            html.Div(id='insights-month1-label', className="kpi-label")
                        ], width=2, style={'textAlign': 'center'}),
                        dbc.Col([
                            html.Div("No.2", style={'color': '#8892b0', 'fontSize': '11px'}),
                            html.Div(id='insights-month2-value', className="kpi-medium"),
                            html.Div(id='insights-month2-label', className="kpi-label")
                        ], width=2, style={'textAlign': 'center'}),
                        dbc.Col([
                            html.Div("No.3", style={'color': '#8892b0', 'fontSize': '11px'}),
                            html.Div(id='insights-month3-value', className="kpi-medium"),
                            html.Div(id='insights-month3-label', className="kpi-label")
                        ], width=2, style={'textAlign': 'center'}),
                        dbc.Col([
                            html.Div("No.4", style={'color': '#8892b0', 'fontSize': '11px'}),
                            html.Div(id='insights-month4-value', className="kpi-medium"),
                            html.Div(id='insights-month4-label', className="kpi-label")
                        ], width=2, style={'textAlign': 'center'}),
                        dbc.Col([
                            html.Div("No.5", style={'color': '#8892b0', 'fontSize': '11px'}),
                            html.Div(id='insights-month5-value', className="kpi-medium"),
                            html.Div(id='insights-month5-label', className="kpi-label")
                        ], width=2, style={'textAlign': 'center'}),
                    ])
                ], className="insight-card"),
                
                # Progress Rings Row
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='insights-ring-fas-rate', config={'displayModeBar': False},
                                     style={'height': '180px'}),
                            html.Div("FAS Rate", style={'color': '#4cc9f0', 'textAlign': 'center', 
                                                        'fontSize': '12px', 'fontWeight': '600'}),
                            html.Div("Conversion Success", style={'color': '#8892b0', 'textAlign': 'center', 
                                                                   'fontSize': '10px'})
                        ], width=4),
                        dbc.Col([
                            dcc.Graph(id='insights-ring-contact-rate', config={'displayModeBar': False},
                                     style={'height': '180px'}),
                            html.Div("Contact Rate", style={'color': '#f9c74f', 'textAlign': 'center',
                                                            'fontSize': '12px', 'fontWeight': '600'}),
                            html.Div("Sales Engagement", style={'color': '#8892b0', 'textAlign': 'center',
                                                                 'fontSize': '10px'})
                        ], width=4),
                        dbc.Col([
                            dcc.Graph(id='insights-ring-pull-through', config={'displayModeBar': False},
                                     style={'height': '180px'}),
                            html.Div("Pull Through", style={'color': '#90be6d', 'textAlign': 'center',
                                                            'fontSize': '12px', 'fontWeight': '600'}),
                            html.Div("Funding Success", style={'color': '#8892b0', 'textAlign': 'center',
                                                                'fontSize': '10px'})
                        ], width=4),
                    ])
                ], className="insight-card"),
            ], width=6),
            
            # Right Column - Status & Rankings
            dbc.Col([
                # Status Card
                html.Div([
                    html.Div("Performance Status", className="insight-title"),
                    html.Div([
                        html.Div([
                            html.Span("Period: ", style={'color': '#8892b0', 'fontSize': '12px'}),
                            html.Span(id='insights-period-label', 
                                     style={'color': '#ffffff', 'fontSize': '14px', 'fontWeight': '600'})
                        ]),
                        html.Div([
                            html.Span("Top Performer: ", style={'color': '#8892b0', 'fontSize': '12px'}),
                            html.Span(id='insights-top-performer',
                                     style={'color': '#4cc9f0', 'fontSize': '14px', 'fontWeight': '600'})
                        ], style={'marginTop': '8px'}),
                    ], style={'marginBottom': '15px'}),
                    dcc.Graph(id='insights-status-donut', config={'displayModeBar': False},
                             style={'height': '180px'})
                ], className="insight-card"),
                
                # LVC Rankings
                html.Div([
                    html.Div("LVC Group Rankings", className="insight-title"),
                    html.Div(id='insights-lvc-rankings')
                ], className="insight-card"),
                
                # Persona Rankings
                html.Div([
                    html.Div("Persona Rankings", className="insight-title"),
                    html.Div(id='insights-persona-rankings')
                ], className="insight-card"),
            ], width=3),
        ])
    ], className="insights-container")
])

# --- Main Layout ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("📊 LVC Report Breakout", className="text-center my-3"),
        ])
    ]),
    dbc.Row([
        # Sidebar
        dbc.Col([sidebar], width=2),
        # Main Content
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(tab_main_content, label="📊 Main Dashboard", tab_id="tab-main"),
                dbc.Tab(tab_insights_content, label="🎨 Executive Insights", tab_id="tab-insights"),
                dbc.Tab(tab_persona_content, label="👤 LVC & Persona", tab_id="tab-persona"),
                dbc.Tab(tab_launch_content, label="🚀 Pre/Post Launch", tab_id="tab-launch"),
                dbc.Tab(tab_frank_felix_content, label="🎯 Frank & Felix", tab_id="tab-frank-felix"),
                dbc.Tab(tab_agent_content, label="⭐ Agent Performance", tab_id="tab-agent"),
            ], id="tabs", active_tab="tab-main")
        ], width=10)
    ])
], fluid=True)

# =============================================================================
# CALLBACKS
# =============================================================================

# --- Main Dashboard Callbacks ---
@callback(
    [Output('kpi-total-leads', 'children'),
     Output('kpi-sent-to-sales', 'children'),
     Output('kpi-contacted', 'children'),
     Output('kpi-fas-count', 'children'),
     Output('kpi-fas-rate', 'children'),
     Output('kpi-contact-rate', 'children'),
     Output('main-sts-chart', 'figure'),
     Output('main-fas-rate-chart', 'figure'),
     Output('main-contact-rate-chart', 'figure'),
     Output('main-lp2c-chart', 'figure'),
     Output('main-data-table', 'data'),
     Output('main-data-table', 'columns')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('lvc-filter', 'value'),
     Input('main-time-grain', 'value')]
)
def update_main_dashboard(start_date, end_date, lvc_filter, time_grain):
    where_clause = build_where_clause(start_date, end_date, lvc_filter)
    
    query = f"""
    SELECT 
        DATE(lead_created_date) as lead_date,
        CASE 
            WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
            WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
            WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
            WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'phx_transfer'
            ELSE 'Other'
        END as lvc_group,
        COUNT(DISTINCT lendage_guid) as lead_qty,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales_qty,
        AVG(CASE WHEN sent_to_sales_date IS NOT NULL THEN initial_lead_score_lp2c END) as avg_lp2c,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted_qty,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_qty,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount END) as fas_volume
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    {where_clause}
    GROUP BY 1, 2
    ORDER BY 1 DESC
    LIMIT 5000
    """
    
    df = run_query(query)
    
    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(template='plotly_dark', annotations=[dict(text="No data", showarrow=False)])
        return "--", "--", "--", "--", "--", "--", empty_fig, empty_fig, empty_fig, empty_fig, [], []
    
    df['lead_date'] = pd.to_datetime(df['lead_date'])
    
    # Aggregate by time grain
    if time_grain == 'Week':
        df['date_col'] = df['lead_date'] - pd.to_timedelta(df['lead_date'].dt.weekday, unit='D')
    elif time_grain == 'Month':
        df['date_col'] = df['lead_date'].dt.to_period('M').dt.to_timestamp()
    else:
        df['date_col'] = df['lead_date']
    
    df_grouped = df.groupby(['date_col', 'lvc_group']).agg({
        'lead_qty': 'sum',
        'sent_to_sales_qty': 'sum',
        'contacted_qty': 'sum',
        'fas_qty': 'sum',
        'fas_volume': 'sum',
        'avg_lp2c': 'mean'
    }).reset_index()
    
    df_grouped['fas_rate'] = df_grouped['fas_qty'] / df_grouped['sent_to_sales_qty'].replace(0, np.nan)
    df_grouped['contact_rate'] = df_grouped['contacted_qty'] / df_grouped['sent_to_sales_qty'].replace(0, np.nan)
    
    # KPIs
    total_leads = df['lead_qty'].sum()
    total_sts = df['sent_to_sales_qty'].sum()
    total_contacted = df['contacted_qty'].sum()
    total_fas = df['fas_qty'].sum()
    fas_rate = total_fas / total_sts if total_sts > 0 else 0
    contact_rate = total_contacted / total_sts if total_sts > 0 else 0
    
    # Charts
    fig_sts = px.line(df_grouped, x='date_col', y='sent_to_sales_qty', color='lvc_group',
                      title='Sent to Sales Volume', template='plotly_dark')
    fig_fas_rate = px.line(df_grouped, x='date_col', y='fas_rate', color='lvc_group',
                           title='FAS Rate', template='plotly_dark')
    fig_fas_rate.update_yaxes(tickformat='.1%')
    fig_contact_rate = px.line(df_grouped, x='date_col', y='contact_rate', color='lvc_group',
                               title='Contact Rate', template='plotly_dark')
    fig_contact_rate.update_yaxes(tickformat='.1%')
    fig_lp2c = px.line(df_grouped, x='date_col', y='avg_lp2c', color='lvc_group',
                       title='Avg LP2C Score', template='plotly_dark')
    
    # Table
    table_data = df_grouped.to_dict('records')
    columns = [{'name': c, 'id': c} for c in df_grouped.columns]
    
    return (f"{total_leads:,}", f"{total_sts:,}", f"{total_contacted:,}", 
            f"{total_fas:,}", f"{fas_rate:.1%}", f"{contact_rate:.1%}",
            fig_sts, fig_fas_rate, fig_contact_rate, fig_lp2c, table_data, columns)

# --- Agent Performance Callbacks ---
@callback(
    [Output('agent-bayesian-chart', 'figure'),
     Output('agent-ci-chart', 'figure'),
     Output('agent-table', 'data'),
     Output('agent-table', 'columns'),
     Output('agent-promotion-table', 'data'),
     Output('agent-promotion-table', 'columns')],
    [Input('agent-prior-strength', 'value'),
     Input('agent-volume-threshold', 'value')]
)
def update_agent_analysis(prior_strength, volume_threshold):
    # Example agent data
    agents = pd.DataFrame({
        'agent_id': ['Agent A', 'Agent B', 'Agent C', 'Agent D', 'Agent E', 'Agent F', 'Agent G', 'Agent H', 'Agent I'],
        'star_tier': ['5-Star', '5-Star', '5-Star', '3-Star', '3-Star', '3-Star', '1-Star', '1-Star', '1-Star'],
        'leads_received': [150, 142, 138, 48, 52, 45, 8, 5, 3],
        'conversions': [15, 18, 12, 7, 5, 6, 2, 1, 1]
    })
    
    agents['raw_rate'] = agents['conversions'] / agents['leads_received']
    overall_rate = agents['conversions'].sum() / agents['leads_received'].sum()
    
    # Bayesian adjustment
    agents['bayesian_rate'] = agents.apply(
        lambda x: bayesian_adjusted_rate(x['conversions'], x['leads_received'], overall_rate, prior_strength), axis=1
    )
    
    # Confidence intervals
    agents['ci_lower'], agents['ci_upper'], agents['ci_width'] = zip(*agents.apply(
        lambda x: wilson_ci(x['conversions'], x['leads_received']), axis=1
    ))
    
    # Volume adjusted score
    agents['volume_factor'] = agents['leads_received'].apply(lambda x: min(1.0, x / volume_threshold))
    agents['volume_adj_score'] = (agents['raw_rate'] * agents['volume_factor']) ** 0.5
    
    # Bayesian chart
    fig_bayes = go.Figure()
    colors = {'5-Star': '#2ca02c', '3-Star': '#ff7f0e', '1-Star': '#d62728'}
    
    for tier in ['5-Star', '3-Star', '1-Star']:
        tier_data = agents[agents['star_tier'] == tier]
        fig_bayes.add_trace(go.Bar(
            name=f'{tier} Raw', x=tier_data['agent_id'], y=tier_data['raw_rate'],
            marker_color=colors[tier], opacity=0.5
        ))
        fig_bayes.add_trace(go.Bar(
            name=f'{tier} Bayesian', x=tier_data['agent_id'], y=tier_data['bayesian_rate'],
            marker_color=colors[tier]
        ))
    
    fig_bayes.update_layout(
        title='Raw vs Bayesian Adjusted Conversion Rates',
        barmode='group', template='plotly_dark',
        yaxis_tickformat='.1%'
    )
    
    # CI chart
    fig_ci = go.Figure()
    for tier in ['5-Star', '3-Star', '1-Star']:
        tier_data = agents[agents['star_tier'] == tier]
        fig_ci.add_trace(go.Scatter(
            x=tier_data['agent_id'], y=tier_data['raw_rate'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=tier_data['ci_upper'] - tier_data['raw_rate'],
                arrayminus=tier_data['raw_rate'] - tier_data['ci_lower']
            ),
            mode='markers', name=tier, marker=dict(size=12, color=colors[tier])
        ))
    
    fig_ci.update_layout(
        title='Conversion Rates with 95% Confidence Intervals',
        template='plotly_dark', yaxis_tickformat='.1%'
    )
    
    # Agent table
    table_df = agents[['agent_id', 'star_tier', 'leads_received', 'conversions', 'raw_rate', 'bayesian_rate', 'volume_adj_score']].copy()
    table_df['raw_rate'] = table_df['raw_rate'].apply(lambda x: f"{x:.1%}")
    table_df['bayesian_rate'] = table_df['bayesian_rate'].apply(lambda x: f"{x:.1%}")
    table_df['volume_adj_score'] = table_df['volume_adj_score'].apply(lambda x: f"{x:.3f}")
    table_df.columns = ['Agent', 'Tier', 'Leads', 'Conversions', 'Raw Rate', 'Bayesian Rate', 'Vol Adj Score']
    
    # Promotion table
    promo_data = pd.DataFrame({
        'Agent': ['Agent G', 'Agent H', 'Agent I', 'Agent D', 'Agent E'],
        'Tier': ['1-Star', '1-Star', '1-Star', '3-Star', '3-Star'],
        'GP Leads': [28, 31, 15, 45, 52],
        'GP Conv': [3, 4, 2, 6, 7],
        'Contact': ['82%', '78%', '75%', '85%', '88%'],
        'Pull Thru': ['65%', '70%', '55%', '72%', '68%'],
    })
    promo_data['Meets Vol'] = promo_data['GP Leads'] >= 30
    promo_data['GP Rate'] = promo_data.apply(lambda x: f"{x['GP Conv']/x['GP Leads']:.1%}", axis=1)
    promo_data['Status'] = promo_data.apply(
        lambda x: '✅ Ready' if x['Meets Vol'] and x['GP Conv']/x['GP Leads'] > 0.12 
        else ('⏳ Need More Data' if not x['Meets Vol'] else '❌ Not Yet'), axis=1
    )
    promo_data['Meets Vol'] = promo_data['Meets Vol'].apply(lambda x: '✅' if x else '❌')
    
    return (fig_bayes, fig_ci, 
            table_df.to_dict('records'), [{'name': c, 'id': c} for c in table_df.columns],
            promo_data.to_dict('records'), [{'name': c, 'id': c} for c in promo_data.columns])

# --- Frank & Felix Callbacks ---
@callback(
    [Output('ff-kpi-pct', 'children'),
     Output('ff-kpi-fas-rate', 'children'),
     Output('ff-kpi-other-fas-rate', 'children'),
     Output('ff-kpi-avg-loan', 'children'),
     Output('ff-kpi-fas-per-sts', 'children'),
     Output('ff-kpi-pull-through', 'children'),
     Output('ff-pct-chart', 'figure'),
     Output('ff-fas-rate-chart', 'figure'),
     Output('ff-fas-per-sts-chart', 'figure'),
     Output('ff-value-concentration-chart', 'figure')],
    [Input('ff-time-grain', 'value')]
)
def update_frank_felix(time_grain):
    query = """
    SELECT 
        DATE(lead_created_date) as lead_date,
        COALESCE(persona, 'Unknown') as persona,
        COUNT(DISTINCT lendage_guid) as lead_count,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales_qty,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL AND funding_end_datetime IS NOT NULL THEN lendage_guid END) as funded_count,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_loan_amount,
        SUM(CASE WHEN sent_to_sales_date IS NOT NULL THEN e_loan_amount ELSE 0 END) as sts_loan_amount,
        AVG(CASE WHEN sent_to_sales_date IS NOT NULL THEN e_loan_amount END) as avg_loan_amount
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE lead_created_date >= '2025-10-01' AND lead_created_date <= '2026-01-30'
    AND persona IS NOT NULL
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    
    df = run_query(query)
    
    empty_fig = go.Figure()
    empty_fig.update_layout(template='plotly_dark', annotations=[dict(text="No data", showarrow=False)])
    
    if df.empty:
        return "--", "--", "--", "--", "--", "--", empty_fig, empty_fig, empty_fig, empty_fig
    
    df['lead_date'] = pd.to_datetime(df['lead_date'])
    df['persona_group'] = df['persona'].apply(lambda x: 'Frank & Felix' if x in ['Frank', 'Felix'] else 'Other Personas')
    
    # Aggregate by time grain
    if time_grain == 'Week':
        df['date_col'] = df['lead_date'] - pd.to_timedelta(df['lead_date'].dt.weekday, unit='D')
    elif time_grain == 'Month':
        df['date_col'] = df['lead_date'].dt.to_period('M').dt.to_timestamp()
    else:
        df['date_col'] = df['lead_date']
    
    df_grouped = df.groupby(['date_col', 'persona_group']).agg({
        'lead_count': 'sum',
        'sent_to_sales_qty': 'sum',
        'fas_count': 'sum',
        'funded_count': 'sum',
        'fas_loan_amount': 'sum',
        'sts_loan_amount': 'sum'
    }).reset_index()
    
    # Calculate metrics
    df_grouped['fas_rate'] = df_grouped['fas_count'] / df_grouped['sent_to_sales_qty'].replace(0, np.nan)
    df_grouped['fas_per_sts'] = df_grouped['fas_loan_amount'] / df_grouped['sent_to_sales_qty'].replace(0, np.nan)
    df_grouped['pull_through'] = df_grouped['funded_count'] / df_grouped['fas_count'].replace(0, np.nan)
    
    # Calculate % of total
    totals_by_date = df_grouped.groupby('date_col')['sent_to_sales_qty'].sum().reset_index()
    totals_by_date.columns = ['date_col', 'total_sts']
    df_grouped = df_grouped.merge(totals_by_date, on='date_col')
    df_grouped['pct_of_total'] = df_grouped['sent_to_sales_qty'] / df_grouped['total_sts']
    
    # Value concentration
    fas_totals = df_grouped.groupby('date_col')['fas_loan_amount'].sum().reset_index()
    fas_totals.columns = ['date_col', 'total_fas_loan']
    df_grouped = df_grouped.merge(fas_totals, on='date_col')
    df_grouped['fas_dollar_share'] = df_grouped['fas_loan_amount'] / df_grouped['total_fas_loan'].replace(0, np.nan)
    df_grouped['value_concentration'] = df_grouped['fas_dollar_share'] / df_grouped['pct_of_total'].replace(0, np.nan)
    
    # KPIs (totals)
    ff_data = df[df['persona_group'] == 'Frank & Felix']
    other_data = df[df['persona_group'] == 'Other Personas']
    
    ff_sts = ff_data['sent_to_sales_qty'].sum()
    other_sts = other_data['sent_to_sales_qty'].sum()
    total_sts = ff_sts + other_sts
    
    ff_pct = ff_sts / total_sts if total_sts > 0 else 0
    ff_fas = ff_data['fas_count'].sum()
    ff_fas_rate = ff_fas / ff_sts if ff_sts > 0 else 0
    other_fas_rate = other_data['fas_count'].sum() / other_sts if other_sts > 0 else 0
    ff_avg_loan = ff_data['fas_loan_amount'].sum() / ff_fas if ff_fas > 0 else 0
    ff_fas_per_sts = ff_data['fas_loan_amount'].sum() / ff_sts if ff_sts > 0 else 0
    ff_funded = ff_data['funded_count'].sum()
    ff_pull_through = ff_funded / ff_fas if ff_fas > 0 else 0
    
    # Charts
    ff_only = df_grouped[df_grouped['persona_group'] == 'Frank & Felix']
    
    fig_pct = px.line(ff_only, x='date_col', y='pct_of_total', 
                      title='Frank & Felix % of Total StS', template='plotly_dark')
    fig_pct.update_yaxes(tickformat='.1%')
    
    fig_fas_rate = px.line(df_grouped, x='date_col', y='fas_rate', color='persona_group',
                           title='FAS Rate Comparison', template='plotly_dark',
                           color_discrete_map={'Frank & Felix': '#2ca02c', 'Other Personas': '#d62728'})
    fig_fas_rate.update_yaxes(tickformat='.1%')
    
    fig_fas_per_sts = px.line(df_grouped, x='date_col', y='fas_per_sts', color='persona_group',
                              title='FAS $ per Lead StS', template='plotly_dark',
                              color_discrete_map={'Frank & Felix': '#2ca02c', 'Other Personas': '#d62728'})
    fig_fas_per_sts.update_yaxes(tickformat='$,.0f')
    
    fig_value = px.line(df_grouped, x='date_col', y='value_concentration', color='persona_group',
                        title='Lead Value Driven by Loan Size', template='plotly_dark',
                        color_discrete_map={'Frank & Felix': '#2ca02c', 'Other Personas': '#d62728'})
    fig_value.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="1.0x Fair Share")
    
    return (f"{ff_pct:.1%}", f"{ff_fas_rate:.1%}", f"{other_fas_rate:.1%}",
            f"${ff_avg_loan:,.0f}", f"${ff_fas_per_sts:,.0f}", f"{ff_pull_through:.1%}",
            fig_pct, fig_fas_rate, fig_fas_per_sts, fig_value)

# --- LVC Persona Callbacks ---
@callback(
    [Output('persona-sankey-chart', 'figure'),
     Output('persona-heatmap', 'figure'),
     Output('persona-table', 'data'),
     Output('persona-table', 'columns')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_persona_analysis(start_date, end_date):
    where_clause = build_where_clause(start_date, end_date)
    
    query = f"""
    SELECT 
        CASE 
            WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
            WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
            WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
            WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'phx_transfer'
            ELSE 'Other'
        END as lvc_group,
        COALESCE(persona, 'Unknown') as persona,
        COUNT(DISTINCT lendage_guid) as lead_count,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales_qty,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    {where_clause}
    GROUP BY 1, 2
    """
    
    df = run_query(query)
    
    empty_fig = go.Figure()
    empty_fig.update_layout(template='plotly_dark', annotations=[dict(text="No data", showarrow=False)])
    
    if df.empty:
        return empty_fig, empty_fig, [], []
    
    df['fas_rate'] = df['fas_count'] / df['sent_to_sales_qty'].replace(0, np.nan)
    
    # Sankey diagram
    lvc_groups = df['lvc_group'].unique().tolist()
    personas = df['persona'].unique().tolist()
    all_nodes = lvc_groups + personas
    
    sources = [all_nodes.index(row['lvc_group']) for _, row in df.iterrows()]
    targets = [all_nodes.index(row['persona']) for _, row in df.iterrows()]
    values = df['lead_count'].tolist()
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, label=all_nodes),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig_sankey.update_layout(title='LVC to Persona Flow', template='plotly_dark')
    
    # Heatmap
    pivot = df.pivot_table(index='persona', columns='lvc_group', values='fas_rate', aggfunc='mean')
    fig_heatmap = px.imshow(pivot, color_continuous_scale='RdYlGn', template='plotly_dark',
                            title='FAS Rate: Persona × LVC Group')
    fig_heatmap.update_xaxes(side='top')
    
    # Table
    table_df = df.groupby('persona').agg({
        'lead_count': 'sum',
        'sent_to_sales_qty': 'sum',
        'fas_count': 'sum'
    }).reset_index()
    table_df['fas_rate'] = table_df['fas_count'] / table_df['sent_to_sales_qty'].replace(0, np.nan)
    table_df['lead_pct'] = table_df['lead_count'] / table_df['lead_count'].sum()
    table_df = table_df.sort_values('lead_count', ascending=False)
    
    table_df['fas_rate'] = table_df['fas_rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    table_df['lead_pct'] = table_df['lead_pct'].apply(lambda x: f"{x:.1%}")
    table_df.columns = ['Persona', 'Leads', 'StS', 'FAS', 'FAS Rate', 'Lead %']
    
    return (fig_sankey, fig_heatmap,
            table_df.to_dict('records'), [{'name': c, 'id': c} for c in table_df.columns])

# --- Pre/Post Launch Callbacks ---
@callback(
    [Output('launch-kpi-pre-fas', 'children'),
     Output('launch-kpi-post-fas', 'children'),
     Output('launch-kpi-pre-contact', 'children'),
     Output('launch-kpi-post-contact', 'children'),
     Output('launch-fas-chart', 'figure'),
     Output('launch-contact-chart', 'figure')],
    [Input('launch-event-select', 'value'),
     Input('launch-time-grain', 'value'),
     Input('launch-compare-window', 'value')]
)
def update_launch_analysis(event_value, time_grain, compare_window):
    # Determine date range
    if compare_window == 'all':
        start_date = '2024-01-01'
    else:
        start_date = (datetime.today() - timedelta(days=int(compare_window) * 2)).strftime('%Y-%m-%d')
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    query = f"""
    SELECT 
        DATE(lead_created_date) as lead_date,
        COUNT(DISTINCT lendage_guid) as lead_count,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales_qty,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted_qty,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    WHERE lead_created_date >= '{start_date}' AND lead_created_date <= '{end_date}'
    GROUP BY 1
    ORDER BY 1
    """
    
    df = run_query(query)
    
    empty_fig = go.Figure()
    empty_fig.update_layout(template='plotly_dark', annotations=[dict(text="No data", showarrow=False)])
    
    if df.empty:
        return "--", "--", "--", "--", empty_fig, empty_fig
    
    df['lead_date'] = pd.to_datetime(df['lead_date'])
    df['fas_rate'] = df['fas_count'] / df['sent_to_sales_qty'].replace(0, np.nan)
    df['contact_rate'] = df['contacted_qty'] / df['sent_to_sales_qty'].replace(0, np.nan)
    
    # Aggregate by time grain
    if time_grain == 'Week':
        df['date_col'] = df['lead_date'] - pd.to_timedelta(df['lead_date'].dt.weekday, unit='D')
    elif time_grain == 'Month':
        df['date_col'] = df['lead_date'].dt.to_period('M').dt.to_timestamp()
    else:
        df['date_col'] = df['lead_date']
    
    df_grouped = df.groupby('date_col').agg({
        'lead_count': 'sum',
        'sent_to_sales_qty': 'sum',
        'contacted_qty': 'sum',
        'fas_count': 'sum'
    }).reset_index()
    
    df_grouped['fas_rate'] = df_grouped['fas_count'] / df_grouped['sent_to_sales_qty'].replace(0, np.nan)
    df_grouped['contact_rate'] = df_grouped['contacted_qty'] / df_grouped['sent_to_sales_qty'].replace(0, np.nan)
    
    # Determine launch date for comparison
    if event_value == 'all':
        launch_date = pd.Timestamp('2025-10-14')  # Use first major launch
    else:
        launch_date = pd.Timestamp(event_value)
    
    pre_data = df_grouped[df_grouped['date_col'] < launch_date]
    post_data = df_grouped[df_grouped['date_col'] >= launch_date]
    
    pre_fas = pre_data['fas_count'].sum() / pre_data['sent_to_sales_qty'].sum() if pre_data['sent_to_sales_qty'].sum() > 0 else 0
    post_fas = post_data['fas_count'].sum() / post_data['sent_to_sales_qty'].sum() if post_data['sent_to_sales_qty'].sum() > 0 else 0
    pre_contact = pre_data['contacted_qty'].sum() / pre_data['sent_to_sales_qty'].sum() if pre_data['sent_to_sales_qty'].sum() > 0 else 0
    post_contact = post_data['contacted_qty'].sum() / post_data['sent_to_sales_qty'].sum() if post_data['sent_to_sales_qty'].sum() > 0 else 0
    
    # Charts with launch markers
    fig_fas = px.line(df_grouped, x='date_col', y='fas_rate', title='FAS Rate Over Time', template='plotly_dark')
    fig_fas.update_yaxes(tickformat='.1%')
    
    fig_contact = px.line(df_grouped, x='date_col', y='contact_rate', title='Contact Rate Over Time', template='plotly_dark')
    fig_contact.update_yaxes(tickformat='.1%')
    
    # Add launch event lines
    for event in LAUNCH_EVENTS:
        event_date = pd.Timestamp(event['date'])
        if event_date >= df_grouped['date_col'].min() and event_date <= df_grouped['date_col'].max():
            fig_fas.add_vline(x=event_date, line_dash="dash", line_color="orange", opacity=0.7)
            fig_contact.add_vline(x=event_date, line_dash="dash", line_color="orange", opacity=0.7)
    
    return (f"{pre_fas:.1%}", f"{post_fas:.1%}", f"{pre_contact:.1%}", f"{post_contact:.1%}",
            fig_fas, fig_contact)

# --- Executive Insights Callbacks ---
@callback(
    [Output('insights-lvc-donut', 'figure'),
     Output('insights-persona-radar', 'figure'),
     Output('insights-weekly-bars', 'figure'),
     Output('insights-sts-value', 'children'),
     Output('insights-sts-badge', 'children'),
     Output('insights-fas-value', 'children'),
     Output('insights-fas-badge', 'children'),
     Output('insights-trend-area', 'figure'),
     Output('insights-month1-value', 'children'),
     Output('insights-month1-label', 'children'),
     Output('insights-month2-value', 'children'),
     Output('insights-month2-label', 'children'),
     Output('insights-month3-value', 'children'),
     Output('insights-month3-label', 'children'),
     Output('insights-month4-value', 'children'),
     Output('insights-month4-label', 'children'),
     Output('insights-month5-value', 'children'),
     Output('insights-month5-label', 'children'),
     Output('insights-ring-fas-rate', 'figure'),
     Output('insights-ring-contact-rate', 'figure'),
     Output('insights-ring-pull-through', 'figure'),
     Output('insights-period-label', 'children'),
     Output('insights-top-performer', 'children'),
     Output('insights-status-donut', 'figure'),
     Output('insights-lvc-rankings', 'children'),
     Output('insights-persona-rankings', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_insights_dashboard(start_date, end_date):
    # Dark theme for all charts
    dark_template = {
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#ffffff'},
            'margin': {'l': 20, 'r': 20, 't': 20, 'b': 20}
        }
    }
    
    where_clause = build_where_clause(start_date, end_date)
    
    # Main query for LVC data
    query = f"""
    SELECT 
        DATE(lead_created_date) as lead_date,
        CASE 
            WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
            WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
            WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
            WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
            ELSE 'Other'
        END as lvc_group,
        COALESCE(persona, 'Unknown') as persona,
        COUNT(DISTINCT lendage_guid) as lead_count,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales_qty,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted_qty,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL AND funding_end_datetime IS NOT NULL THEN lendage_guid END) as funded_count
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    {where_clause}
    GROUP BY 1, 2, 3
    ORDER BY 1
    """
    
    df = run_query(query)
    
    # Empty figure template
    def empty_fig():
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#ffffff'},
            margin={'l': 20, 'r': 20, 't': 20, 'b': 20}
        )
        return fig
    
    if df.empty:
        empty = empty_fig()
        return (empty, empty, empty, "--", "0%", "--", "0%", empty,
                "--", "--", "--", "--", "--", "--", "--", "--", "--", "--",
                empty, empty, empty, "--", "--", empty, [], [])
    
    df['lead_date'] = pd.to_datetime(df['lead_date'])
    
    # --- 1. LVC Distribution Donut Chart ---
    lvc_dist = df.groupby('lvc_group')['sent_to_sales_qty'].sum().reset_index()
    lvc_colors = ['#4cc9f0', '#4895ef', '#4361ee', '#3f37c9', '#7209b7']
    
    fig_lvc_donut = go.Figure(data=[go.Pie(
        labels=lvc_dist['lvc_group'],
        values=lvc_dist['sent_to_sales_qty'],
        hole=0.6,
        marker=dict(colors=lvc_colors, line=dict(color='#0d1b2a', width=2)),
        textinfo='percent',
        textfont=dict(size=10, color='white'),
        hovertemplate='%{label}<br>%{value:,}<br>%{percent}<extra></extra>'
    )])
    fig_lvc_donut.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(font=dict(size=9, color='#8892b0'), orientation='h', y=-0.1),
        margin={'l': 10, 'r': 10, 't': 10, 'b': 30}
    )
    
    # --- 2. Persona Radar Chart ---
    persona_stats = df.groupby('persona').agg({
        'sent_to_sales_qty': 'sum',
        'fas_count': 'sum',
        'contacted_qty': 'sum'
    }).reset_index()
    persona_stats['fas_rate'] = persona_stats['fas_count'] / persona_stats['sent_to_sales_qty'].replace(0, np.nan) * 100
    persona_stats['contact_rate'] = persona_stats['contacted_qty'] / persona_stats['sent_to_sales_qty'].replace(0, np.nan) * 100
    persona_stats = persona_stats.nlargest(6, 'sent_to_sales_qty')
    
    fig_persona_radar = go.Figure()
    fig_persona_radar.add_trace(go.Scatterpolar(
        r=persona_stats['fas_rate'].fillna(0).tolist() + [persona_stats['fas_rate'].fillna(0).iloc[0]],
        theta=persona_stats['persona'].tolist() + [persona_stats['persona'].iloc[0]],
        fill='toself',
        fillcolor='rgba(76, 201, 240, 0.3)',
        line=dict(color='#4cc9f0', width=2),
        name='FAS Rate %'
    ))
    fig_persona_radar.add_trace(go.Scatterpolar(
        r=persona_stats['contact_rate'].fillna(0).tolist() + [persona_stats['contact_rate'].fillna(0).iloc[0]],
        theta=persona_stats['persona'].tolist() + [persona_stats['persona'].iloc[0]],
        fill='toself',
        fillcolor='rgba(144, 190, 109, 0.3)',
        line=dict(color='#90be6d', width=2),
        name='Contact Rate %'
    ))
    fig_persona_radar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=8, color='#8892b0')),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=9, color='#8892b0'))
        ),
        showlegend=True,
        legend=dict(font=dict(size=8, color='#8892b0'), orientation='h', y=-0.15),
        margin={'l': 40, 'r': 40, 't': 20, 'b': 40}
    )
    
    # --- 3. Weekly Volume Bar Chart ---
    df['weekday'] = df['lead_date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_vol = df.groupby('weekday')['sent_to_sales_qty'].sum().reindex(weekday_order).fillna(0)
    
    fig_weekly = go.Figure(data=[go.Bar(
        x=[d[:3] for d in weekday_order],
        y=weekly_vol.values,
        marker=dict(
            color=weekly_vol.values,
            colorscale=[[0, '#1b3a4b'], [0.5, '#4895ef'], [1, '#4cc9f0']],
            line=dict(width=0)
        ),
        hovertemplate='%{x}<br>%{y:,}<extra></extra>'
    )])
    fig_weekly.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(size=9, color='#8892b0'), showgrid=False),
        yaxis=dict(tickfont=dict(size=9, color='#8892b0'), gridcolor='rgba(255,255,255,0.05)'),
        margin={'l': 30, 'r': 10, 't': 10, 'b': 30}
    )
    
    # --- 4. Main KPIs ---
    total_sts = df['sent_to_sales_qty'].sum()
    total_fas = df['fas_count'].sum()
    total_contacted = df['contacted_qty'].sum()
    total_funded = df['funded_count'].sum()
    
    fas_rate = total_fas / total_sts if total_sts > 0 else 0
    contact_rate = total_contacted / total_sts if total_sts > 0 else 0
    pull_through = total_funded / total_fas if total_fas > 0 else 0
    
    # --- 5. Trend Area Chart ---
    df['week'] = df['lead_date'] - pd.to_timedelta(df['lead_date'].dt.weekday, unit='D')
    trend_data = df.groupby('week').agg({
        'sent_to_sales_qty': 'sum',
        'fas_count': 'sum'
    }).reset_index()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_data['week'], y=trend_data['sent_to_sales_qty'],
        fill='tozeroy', fillcolor='rgba(76, 201, 240, 0.3)',
        line=dict(color='#4cc9f0', width=2),
        name='Sent to Sales'
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend_data['week'], y=trend_data['fas_count'],
        fill='tozeroy', fillcolor='rgba(247, 37, 133, 0.3)',
        line=dict(color='#f72585', width=2),
        name='FAS'
    ))
    fig_trend.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(size=9, color='#8892b0'), showgrid=False, tickformat='%b %d'),
        yaxis=dict(tickfont=dict(size=9, color='#8892b0'), gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(font=dict(size=9, color='#8892b0'), orientation='h', y=1.1),
        margin={'l': 40, 'r': 20, 't': 30, 'b': 30}
    )
    
    # --- 6. Monthly KPIs (Top 5 months) ---
    df['month'] = df['lead_date'].dt.to_period('M').dt.to_timestamp()
    monthly_data = df.groupby('month')['sent_to_sales_qty'].sum().nlargest(5).reset_index()
    monthly_data = monthly_data.sort_values('sent_to_sales_qty', ascending=False)
    
    month_values = []
    month_labels = []
    for i in range(5):
        if i < len(monthly_data):
            month_values.append(f"{monthly_data.iloc[i]['sent_to_sales_qty']:,}")
            month_labels.append(monthly_data.iloc[i]['month'].strftime('%B'))
        else:
            month_values.append("--")
            month_labels.append("--")
    
    # --- 7. Progress Ring Charts ---
    def create_ring_chart(value, color, max_val=1):
        fig = go.Figure(data=[go.Pie(
            values=[value * 100, (max_val - value) * 100],
            hole=0.75,
            marker=dict(colors=[color, 'rgba(255,255,255,0.1)'], line=dict(width=0)),
            textinfo='none',
            hoverinfo='skip'
        )])
        fig.add_annotation(
            text=f"{value:.0%}",
            x=0.5, y=0.5, font=dict(size=24, color=color, family='Arial Black'),
            showarrow=False
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin={'l': 10, 'r': 10, 't': 10, 'b': 10}
        )
        return fig
    
    fig_ring_fas = create_ring_chart(fas_rate, '#4cc9f0')
    fig_ring_contact = create_ring_chart(contact_rate, '#f9c74f')
    fig_ring_pull = create_ring_chart(pull_through, '#90be6d')
    
    # --- 8. Status Donut ---
    status_data = {'Converted': total_fas, 'Contacted': total_contacted - total_fas, 
                   'Pending': total_sts - total_contacted}
    fig_status = go.Figure(data=[go.Pie(
        labels=list(status_data.keys()),
        values=list(status_data.values()),
        hole=0.6,
        marker=dict(colors=['#4cc9f0', '#f9c74f', '#8892b0'], line=dict(width=0)),
        textinfo='percent',
        textfont=dict(size=10, color='white')
    )])
    fig_status.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(font=dict(size=9, color='#8892b0'), orientation='h', y=-0.1),
        margin={'l': 10, 'r': 10, 't': 10, 'b': 30}
    )
    
    # --- 9. LVC Rankings ---
    lvc_rank = df.groupby('lvc_group').agg({
        'sent_to_sales_qty': 'sum',
        'fas_count': 'sum'
    }).reset_index()
    lvc_rank['fas_rate'] = lvc_rank['fas_count'] / lvc_rank['sent_to_sales_qty'].replace(0, np.nan)
    lvc_rank = lvc_rank.sort_values('sent_to_sales_qty', ascending=False)
    
    lvc_rankings_html = []
    max_sts = lvc_rank['sent_to_sales_qty'].max()
    for _, row in lvc_rank.iterrows():
        pct = row['sent_to_sales_qty'] / max_sts * 100 if max_sts > 0 else 0
        lvc_rankings_html.append(
            html.Div([
                html.Div([
                    html.Span(row['lvc_group'], style={'color': '#ffffff', 'fontSize': '12px'}),
                    html.Span(f"{row['sent_to_sales_qty']:,}", style={'color': '#8892b0', 'fontSize': '11px', 'float': 'right'})
                ]),
                html.Div([
                    html.Div(style={
                        'width': f'{pct}%',
                        'height': '6px',
                        'backgroundColor': '#4cc9f0',
                        'borderRadius': '3px',
                        'marginTop': '4px'
                    })
                ], style={'backgroundColor': 'rgba(255,255,255,0.1)', 'borderRadius': '3px', 'height': '6px'})
            ], style={'marginBottom': '12px'})
        )
    
    # --- 10. Persona Rankings ---
    persona_rank = df.groupby('persona').agg({
        'sent_to_sales_qty': 'sum',
        'fas_count': 'sum'
    }).reset_index()
    persona_rank['fas_rate'] = persona_rank['fas_count'] / persona_rank['sent_to_sales_qty'].replace(0, np.nan)
    persona_rank = persona_rank.nlargest(5, 'sent_to_sales_qty')
    
    persona_rankings_html = []
    max_persona_sts = persona_rank['sent_to_sales_qty'].max()
    colors = ['#4cc9f0', '#4895ef', '#4361ee', '#7209b7', '#f72585']
    for i, (_, row) in enumerate(persona_rank.iterrows()):
        pct = row['sent_to_sales_qty'] / max_persona_sts * 100 if max_persona_sts > 0 else 0
        persona_rankings_html.append(
            html.Div([
                html.Div([
                    html.Span(row['persona'][:15] + '...' if len(row['persona']) > 15 else row['persona'], 
                             style={'color': '#ffffff', 'fontSize': '11px'}),
                    html.Span(f"{row['sent_to_sales_qty']:,}", style={'color': '#8892b0', 'fontSize': '10px', 'float': 'right'})
                ]),
                html.Div([
                    html.Div(style={
                        'width': f'{pct}%',
                        'height': '5px',
                        'backgroundColor': colors[i % len(colors)],
                        'borderRadius': '2px',
                        'marginTop': '3px'
                    })
                ], style={'backgroundColor': 'rgba(255,255,255,0.1)', 'borderRadius': '2px', 'height': '5px'})
            ], style={'marginBottom': '10px'})
        )
    
    # Period label and top performer
    period_label = f"{start_date} to {end_date}" if start_date and end_date else "All Time"
    top_performer = lvc_rank.iloc[0]['lvc_group'] if len(lvc_rank) > 0 else "N/A"
    
    return (
        fig_lvc_donut,
        fig_persona_radar,
        fig_weekly,
        f"{total_sts:,}",
        f"↑ {fas_rate:.0%}",
        f"{total_fas:,}",
        f"↑ {contact_rate:.0%}",
        fig_trend,
        month_values[0], month_labels[0],
        month_values[1], month_labels[1],
        month_values[2], month_labels[2],
        month_values[3], month_labels[3],
        month_values[4], month_labels[4],
        fig_ring_fas,
        fig_ring_contact,
        fig_ring_pull,
        period_label,
        top_performer,
        fig_status,
        lvc_rankings_html,
        persona_rankings_html
    )

# =============================================================================
# RUN APP
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=8050)
