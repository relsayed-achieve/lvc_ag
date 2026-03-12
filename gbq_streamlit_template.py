"""
GBQ Streamlit Dashboard Template
================================
A standardized template for connecting to Google BigQuery and building 
Streamlit dashboards with HTML export functionality.

Project: ffn-dw-bigquery-prd
Author: Your Name
"""

import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import json
import altair as alt
from datetime import date, datetime, timedelta

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="GBQ Dashboard Template",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM STYLING (Optional)
# =============================================================================
# Add any custom CSS here if needed
# st.markdown("""
#     <style>
#     .main-header { font-size: 2.5rem; font-weight: bold; }
#     .metric-card { padding: 20px; border-radius: 10px; background: #f0f2f6; }
#     </style>
# """, unsafe_allow_html=True)

# =============================================================================
# MAIN TITLE
# =============================================================================
st.title("📊 GBQ Dashboard Template")
st.caption("A standardized template for BigQuery-powered Streamlit dashboards")

# =============================================================================
# SIDEBAR - CONNECTION & FILTERS
# =============================================================================
with st.sidebar:
    st.header("🔍 Filters")
    
    # --- Date Filter Example ---
    default_start = date.today() - timedelta(days=90)
    default_end = date.today()
    
    date_range = st.date_input(
        "Date Range",
        value=(default_start, default_end),
        min_value=date(2020, 1, 1),
        max_value=date.today()
    )
    
    # --- Add your custom filters here ---
    # Example multi-select filter:
    # selected_values = st.multiselect("Filter Field", options=["Option1", "Option2", "Option3"])
    
    st.divider()
    
    # --- GBQ Connection Settings ---
    with st.expander("🔌 Connection Settings", expanded=True):
        auth_mode = st.radio(
            "Authentication Method", 
            ["Local / Browser Auth", "Upload JSON Key", "Paste JSON Content"]
        )
        
        creds = None
        project_id = None
        
        # === Authentication Method 1: Default Credentials (gcloud CLI) ===
        if auth_mode == "Local / Browser Auth":
            try:
                # Attempt to create a client with default credentials and specific project
                project_id = "ffn-dw-bigquery-prd"  # Standard project ID
                client_dummy = bigquery.Client(project=project_id)
                creds = client_dummy._credentials
                st.success(f"✅ Authenticated with project: `{project_id}`")
            except Exception as e:
                st.warning(f"⚠️ Could not connect using default credentials.")
                st.info("To fix this, run: `gcloud auth application-default login` in your terminal.")
                st.code("gcloud auth application-default login", language="bash")
                with st.expander("See error details"):
                    st.error(str(e))
        
        # === Authentication Method 2: Upload Service Account JSON ===
        elif auth_mode == "Upload JSON Key":
            uploaded_file = st.file_uploader("Upload Service Account JSON", type="json")
            if uploaded_file:
                try:
                    json_content = json.load(uploaded_file)
                    creds = service_account.Credentials.from_service_account_info(json_content)
                    project_id = json_content.get("project_id")
                    st.success(f"✅ Loaded credentials for project: `{project_id}`")
                except Exception as e:
                    st.error(f"❌ Error loading JSON: {e}")
        
        # === Authentication Method 3: Paste JSON Content ===
        else:
            json_text = st.text_area("Paste JSON Content Here", height=150)
            if json_text:
                try:
                    json_content = json.loads(json_text)
                    creds = service_account.Credentials.from_service_account_info(json_content)
                    project_id = json_content.get("project_id")
                    st.success(f"✅ Loaded credentials for project: `{project_id}`")
                except Exception as e:
                    st.error(f"❌ Error parsing JSON: {e}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_where_clause(date_range, additional_conditions=None):
    """
    Build a SQL WHERE clause from filter selections.
    
    Args:
        date_range: Tuple of (start_date, end_date) or single date
        additional_conditions: List of additional SQL condition strings
    
    Returns:
        str: SQL WHERE clause or empty string
    """
    conditions = []
    
    # Date condition
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_str = date_range[0].strftime('%Y-%m-%d')
        end_str = date_range[1].strftime('%Y-%m-%d')
        conditions.append(f"DATE(your_date_column) BETWEEN '{start_str}' AND '{end_str}'")
    elif isinstance(date_range, date):
        start_str = date_range.strftime('%Y-%m-%d')
        conditions.append(f"DATE(your_date_column) = '{start_str}'")
    
    # Add any additional conditions
    if additional_conditions:
        conditions.extend(additional_conditions)
    
    if conditions:
        return "WHERE " + " AND ".join(conditions)
    return ""


def run_query(client, sql_query, use_cache=True, cache_ttl=600):
    """
    Execute a BigQuery query and return a DataFrame.
    
    Args:
        client: BigQuery client
        sql_query: SQL query string
        use_cache: Whether to use Streamlit caching
        cache_ttl: Cache time-to-live in seconds
    
    Returns:
        pd.DataFrame: Query results
    """
    @st.cache_data(ttl=cache_ttl)
    def _cached_query(query_str):
        return client.query(query_str).to_dataframe()
    
    if use_cache:
        return _cached_query(sql_query)
    else:
        return client.query(sql_query).to_dataframe()


def create_time_series_chart(data, x_col, y_col, color_col=None, title=""):
    """
    Create a standardized Altair time series chart.
    
    Args:
        data: DataFrame with the data
        x_col: Column name for x-axis (typically date)
        y_col: Column name for y-axis (metric)
        color_col: Optional column for color grouping
        title: Chart title
    
    Returns:
        alt.Chart: Altair chart object
    """
    base = alt.Chart(data).mark_line(point=True)
    
    encoding = {
        'x': alt.X(f'{x_col}:T', title='Date'),
        'y': alt.Y(f'{y_col}:Q', title=title or y_col),
        'tooltip': [alt.Tooltip(f'{x_col}:T', title='Date'), 
                    alt.Tooltip(f'{y_col}:Q', title='Value', format=',.2f')]
    }
    
    if color_col:
        encoding['color'] = alt.Color(f'{color_col}:N', title=color_col)
        encoding['tooltip'].append(color_col)
    
    chart = base.encode(**encoding).interactive().properties(
        title=title,
        height=300
    )
    
    return chart


# =============================================================================
# HTML EXPORT FUNCTIONALITY
# =============================================================================

def generate_html_report(charts_data, report_title="Dashboard Report"):
    """
    Generate a downloadable HTML report with embedded charts.
    
    Args:
        charts_data: List of dicts with 'id', 'title', and 'chart_json' keys
        report_title: Title for the HTML report
    
    Returns:
        str: Complete HTML document
    """
    # Build chart divs
    chart_divs = ""
    embed_scripts = ""
    
    for chart in charts_data:
        chart_divs += f'''
            <div class="chart-box">
                <h3>{chart['title']}</h3>
                <div id="{chart['id']}"></div>
            </div>
        '''
        embed_scripts += f"vegaEmbed('#{chart['id']}', {chart['chart_json']});\n"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report_title}</title>
        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                padding: 20px; 
                background-color: #f4f4f4; 
                margin: 0;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 8px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }}
            h1 {{ color: #1f2937; margin-bottom: 10px; }}
            h2 {{ color: #374151; margin-top: 40px; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }}
            h3 {{ color: #4b5563; margin-bottom: 15px; }}
            .chart-box {{ 
                margin-bottom: 40px; 
                padding: 20px; 
                border: 1px solid #e5e7eb; 
                border-radius: 8px;
                background: #fafafa;
            }}
            .grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
                gap: 20px; 
            }}
            .timestamp {{ color: #6b7280; font-size: 0.9em; }}
            hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 {report_title}</h1>
            <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            <hr>
            
            <div class="grid">
                {chart_divs}
            </div>
        </div>

        <script type="text/javascript">
            {embed_scripts}
        </script>
    </body>
    </html>
    """
    
    return html_content


def add_html_download_button(html_content, filename_prefix="report"):
    """
    Add a download button for HTML content to the sidebar.
    
    Args:
        html_content: The HTML string to download
        filename_prefix: Prefix for the downloaded file name
    """
    st.sidebar.download_button(
        label="⬇️ Download HTML Report",
        data=html_content,
        file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html"
    )


# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================

if creds:
    # Initialize BigQuery client
    client_project = getattr(creds, 'project_id', project_id)
    client = bigquery.Client(credentials=creds, project=client_project)
    
    # Build WHERE clause from filters
    where_clause = build_where_clause(date_range)
    
    # ==========================================================================
    # DASHBOARD TABS
    # ==========================================================================
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analysis", "⚙️ Settings"])
    
    # --- TAB 1: Overview ---
    with tab1:
        st.header("Overview")
        st.info("👋 This is the Overview tab. Add your main metrics and KPIs here.")
        
        # --- Example: Metrics Row ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Records", value="0", delta=None)
        with col2:
            st.metric(label="Metric 2", value="$0", delta=None)
        with col3:
            st.metric(label="Metric 3", value="0%", delta=None)
        with col4:
            st.metric(label="Metric 4", value="0", delta=None)
        
        st.divider()
        
        # --- Example: Query Section ---
        with st.expander("📝 View/Edit SQL Query", expanded=False):
            example_query = f"""
SELECT 
    DATE(your_date_column) as date,
    COUNT(*) as record_count,
    SUM(your_metric) as total_metric
FROM `ffn-dw-bigquery-prd.your_dataset.your_table`
{where_clause}
GROUP BY 1
ORDER BY 1 DESC
LIMIT 1000
            """
            query = st.text_area("SQL Query", value=example_query.strip(), height=200)
        
        # --- Run Query Button ---
        if st.button("▶️ Run Query", type="primary"):
            try:
                with st.spinner("Fetching data..."):
                    df = run_query(client, query, use_cache=False)
                
                if not df.empty:
                    st.success(f"✅ Returned {len(df):,} rows")
                    
                    # Display data
                    st.dataframe(df, use_container_width=True)
                    
                    # Store in session state for other tabs
                    st.session_state['query_result'] = df
                else:
                    st.warning("Query returned no results.")
                    
            except Exception as e:
                st.error(f"❌ Query failed: {e}")
        
        # --- Example Chart Placeholder ---
        st.subheader("📈 Chart Placeholder")
        st.info("Add your visualizations here using Altair, Plotly, or Streamlit's native charts.")
        
        # If we have data, show a sample chart
        if 'query_result' in st.session_state and not st.session_state['query_result'].empty:
            df = st.session_state['query_result']
            st.dataframe(df.head(10))
    
    # --- TAB 2: Analysis ---
    with tab2:
        st.header("Analysis")
        st.info("👋 This is the Analysis tab. Add detailed analysis and breakdowns here.")
        
        # --- Two Column Layout Example ---
        left_col, right_col = st.columns(2)
        
        with left_col:
            st.subheader("Left Section")
            st.write("Add charts or metrics here.")
            # Placeholder for chart
            st.empty()
        
        with right_col:
            st.subheader("Right Section")
            st.write("Add charts or metrics here.")
            # Placeholder for chart
            st.empty()
        
        st.divider()
        
        # --- Expandable Section Example ---
        with st.expander("📊 Detailed Breakdown", expanded=False):
            st.write("Add detailed tables or analysis here.")
    
    # --- TAB 3: Settings ---
    with tab3:
        st.header("Settings & Export")
        st.info("👋 Configure settings and export options here.")
        
        # --- HTML Report Export ---
        st.subheader("📤 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input("Report Title", value="My Dashboard Report")
        
        with col2:
            st.write("")  # Spacing
            st.write("")
            if st.button("📦 Generate HTML Report"):
                # Example: Create charts data for export
                # In real usage, you'd pass your actual chart JSONs
                charts_data = [
                    {
                        'id': 'chart_1',
                        'title': 'Sample Chart 1',
                        'chart_json': alt.Chart(pd.DataFrame({'x': [1,2,3], 'y': [4,5,6]})).mark_line().encode(x='x', y='y').to_json()
                    }
                ]
                
                html_content = generate_html_report(charts_data, report_title)
                add_html_download_button(html_content, filename_prefix="dashboard")
                st.success("✅ Report ready! Click the download button in the sidebar.")
        
        st.divider()
        
        # --- Connection Info ---
        st.subheader("🔌 Connection Info")
        st.code(f"Project: {project_id}")
        st.code(f"Authentication: {auth_mode}")

else:
    # ==========================================================================
    # NO CONNECTION - SHOW INSTRUCTIONS
    # ==========================================================================
    st.warning("⚠️ Please configure your BigQuery connection in the sidebar.")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Option A: Local Authentication (Recommended for Development)**
       - Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
       - Run in terminal:
       ```bash
       gcloud auth application-default login
       ```
       - Refresh this page
    
    2. **Option B: Service Account JSON**
       - Obtain a service account JSON key file from your GCP admin
       - Upload it using the sidebar option
    
    3. **Option C: Paste JSON**
       - Paste the JSON content directly (useful for quick testing)
    
    ---
    
    ### 📁 Standard Project Configuration
    
    | Setting | Value |
    |---------|-------|
    | **Project ID** | `ffn-dw-bigquery-prd` |
    | **Location** | US |
    
    ---
    
    ### 📚 Template Features
    
    - ✅ Multiple authentication methods
    - ✅ Date range filtering
    - ✅ Cached queries for performance
    - ✅ Tab-based layout
    - ✅ HTML report export
    - ✅ Responsive design
    """)


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption("📊 GBQ Streamlit Template | Built with Streamlit & BigQuery")
