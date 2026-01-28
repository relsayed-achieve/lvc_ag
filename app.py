import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import altair as alt

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

st.set_page_config(page_title="GBQ Reporter", layout="wide")

# --- Light Mode CSS ---
st.markdown("""
<style>
    /* Force light mode */
    .stApp {
        background-color: #ffffff;
        color: #1e1e1e;
    }
    
    /* Sidebar light mode */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1e1e1e !important;
    }
    
    /* Text */
    p, span, label, .stMarkdown {
        color: #1e1e1e !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #1e1e1e !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: inherit !important;
    }
    
    /* Tables */
    .stDataFrame {
        background-color: #ffffff;
    }
    
    /* Selectbox and inputs */
    .stSelectbox, .stTextInput {
        color: #1e1e1e;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #1e1e1e !important;
        background-color: #f0f2f6 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #1e1e1e;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 LVC Report Breakout")

# --- Global Sidebar Filters ---
with st.sidebar:
    st.header("🔍 Filters")
    
    # Date Filter
    # Default: 2025-10-01 to Today
    default_start = date(2025, 10, 1)
    default_end = date.today()
    
    date_range = st.date_input(
        "Lead Created Date",
        value=(default_start, default_end),
        min_value=date(2020, 1, 1),
        max_value=date.today()
    )
    
    # LVC Filter
    # We can hardcode common values or query them. Hardcoding for performance.
    lvc_options = [str(i) for i in range(1, 11)] + ['X', 'Other']
    selected_lvc = st.multiselect("Adjusted Lead Value Cohort", options=lvc_options)

    st.divider()
    with st.expander("🔌 Connection Settings", expanded=False):
        auth_mode = st.radio("Authentication Method", ["Local / Browser Auth", "Upload JSON Key", "Paste JSON Content"])
        
        creds = None
        project_id = None
        
        if auth_mode == "Local / Browser Auth":
            try:
                # Attempt to create a client with default credentials and specific project
                project_id = "ffn-dw-bigquery-prd"
                client_dummy = bigquery.Client(project=project_id)
                creds = client_dummy._credentials
                st.success(f"Authenticated with project: `{project_id}`")
            except Exception as e:
                st.warning(f"Could not connect using default credentials. Error: {e}")
                st.info("To fix this, run: `gcloud auth application-default login` in your terminal.")

        elif auth_mode == "Upload JSON Key":
            uploaded_file = st.file_uploader("Upload Service Account JSON", type="json")
            if uploaded_file:
                try:
                    # Read the file content
                    json_content = json.load(uploaded_file)
                    creds = service_account.Credentials.from_service_account_info(json_content)
                    project_id = json_content.get("project_id")
                    st.success(f"Loaded credentials for project: `{project_id}`")
                except Exception as e:
                    st.error(f"Error loading JSON: {e}")
        else:
            json_text = st.text_area("Paste JSON Content Here")
            if json_text:
                try:
                    json_content = json.loads(json_text)
                    creds = service_account.Credentials.from_service_account_info(json_content)
                    project_id =json_content.get("project_id")
                    st.success(f"Loaded credentials for project: `{project_id}`")
                except Exception as e:
                    st.error(f"Error parsing JSON: {e}")

# --- Helper to build WHERE clause ---
def build_where_clause():
    conditions = []
    
    # Date condition
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_str = date_range[0].strftime('%Y-%m-%d')
        end_str = date_range[1].strftime('%Y-%m-%d')
        conditions.append(f"DATE(lead_created_date) BETWEEN '{start_str}' AND '{end_str}'")
    elif isinstance(date_range, date):
        # Single date selected
         start_str = date_range.strftime('%Y-%m-%d')
         conditions.append(f"DATE(lead_created_date) = '{start_str}'")

    # LVC condition
    if selected_lvc:
        # Handle simple values. If they selected 'X', we might match '%X%' or exact 'X'. 
        # Assuming exact match for numbers, LIKE for X if needed, but let's stick to IN for now based on exact values.
        safe_lvc = [f"'{x}'" for x in selected_lvc]
        # Special handling for X logic match if needed, but for now exact match per list
        conditions.append(f"adjusted_lead_value_cohort IN ({', '.join(safe_lvc)})")
        
    if conditions:
        return "WHERE " + " AND ".join(conditions)
    return ""

where_clause = build_where_clause()

# --- Main App Logic ---
if creds:
    # Use the project_id from the credentials or the one we explicitly set
    client_project = getattr(creds, 'project_id', project_id)
    client = bigquery.Client(credentials=creds, project=client_project)
    
    tab_main, tab_launch, tab_insights = st.tabs(["📊 Main Dashboard", "🚀 Pre/Post Launch(s)", "🤖 Comprehensive Insights (Jan 24+)"])
    
    # --- TAB 1: Main Dashboard ---
    with tab_main:
        col1, col2 = st.columns([2, 1])
        
        # Main Dashboard Logic
        st.divider()
        
        # Defined Metric Query with LVC Grouping
        default_query = f"""
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
        COUNT(DISTINCT CASE WHEN current_sales_assigned_date IS NOT NULL THEN lendage_guid END) as assigned_qty,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted_qty,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_qty,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount END) as fas_volume
    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
    {where_clause}
    GROUP BY 1, 2
    ORDER BY 1 DESC
    LIMIT 5000
    """
        
        with st.expander("📝 View/Edit SQL Query", expanded=False):
            query = st.text_area("SQL", value=default_query.strip(), height=300)

        # Auto-run logic (no button)
        try:
            @st.cache_data(ttl=600)
            def run_dashboard_query(sql_query):
                 return client.query(sql_query).to_dataframe()

            with st.spinner("Fetching Dashboard Data..."):
                df = run_dashboard_query(query)
                
            if not df.empty:
                # Ensure proper datetime type for .dt accessor usage
                if 'lead_date' in df.columns:
                     df['lead_date'] = pd.to_datetime(df['lead_date'])
                
                # --- Local Aggregation Helper ---
                def aggregate_data(df_in, grain):
                    df_agg = df_in.copy()
                    if grain == 'Week':
                         df_agg['date_col'] = df_agg['lead_date'] - pd.to_timedelta(df_agg['lead_date'].dt.weekday, unit='D')
                    elif grain == 'Month':
                         df_agg['date_col'] = df_agg['lead_date'].dt.to_period('M').dt.to_timestamp()
                    else: # Day
                         df_agg['date_col'] = df_agg['lead_date']
                    
                    metric_aggs = {
                        'lead_qty': 'sum',
                        'sent_to_sales_qty': 'sum',
                        'assigned_qty': 'sum',
                        'contacted_qty': 'sum',
                        'fas_qty': 'sum',
                        'fas_volume': 'sum',
                        'avg_lp2c': 'mean'
                    }
                    
                    df_grouped = df_agg.groupby(['date_col', 'lvc_group']).agg(metric_aggs).reset_index()
                    
                    # Rate Calcs
                    df_grouped['sales_contact_rate'] = df_grouped.apply(lambda x: x['contacted_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1)
                    df_grouped['fas_rate'] = df_grouped.apply(lambda x: x['fas_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1)
                    
                    return df_grouped

                # --- Visualization Helper ---
                def plot_time_series(df_raw, y_metric, title, is_rate=False, enable_pct_toggle=False, key=None):
                    
                    grain = "Week" # Default
                    show_pct = False
                    display_title = title
                    
                    col_c1, col_c2, col_c3 = st.columns([4, 1, 1.2])
                    
                    with col_c2:
                        grain = st.selectbox("Grain", ["Day", "Week", "Month"], index=1, key=f"grain_{key if key else title}", label_visibility="collapsed")

                    if enable_pct_toggle:
                         with col_c3:
                             # Render toggle first to get state
                             show_pct = st.checkbox("% Total", key=f"pct_{key if key else title}")
                         
                         # Update title if toggled
                         if show_pct:
                             display_title = title.replace("Quantity", "% of Total").replace("Qty", "% of Total")
                             
                    with col_c1:
                         st.subheader(display_title)

                    # Aggregate Data based on Grain
                    data = aggregate_data(df_raw, grain)

                    y_axis = alt.Y(y_metric, title=display_title)
                    tooltip_format = ',d'
                    
                    base = alt.Chart(data)

                    if show_pct:
                        tooltip_format = '.1%'
                        y_axis = alt.Y('pct_share:Q', title='% of Total', axis=alt.Axis(format='%'))
                        
                        base = base.transform_joinaggregate(
                            total_for_period=f'sum({y_metric})',
                            groupby=['date_col']
                        ).transform_calculate(
                            pct_share=f'datum.{y_metric} / datum.total_for_period'
                        )
                        
                        # Use the calculated field
                        plot_y = 'pct_share:Q'
                        tooltip_val = alt.Tooltip('pct_share:Q', format='.1%', title="% Share")
                    else:
                         plot_y = y_axis
                         tooltip_val = alt.Tooltip(y_metric, format=tooltip_format, title="Value")

                    if is_rate or 'rate' in y_metric or 'lp2c' in y_metric:
                        if 'lp2c' not in y_metric:
                             y_axis = alt.Y(y_metric, title=title, axis=alt.Axis(format='%'))
                             tooltip_format = '.1%'
                        else:
                             tooltip_format = '.1f'
                        # Override for Rates (toggle likely false here anyway)
                        plot_y = y_axis 
                        if 'pct_share' not in str(plot_y): # Safety
                             tooltip_val = alt.Tooltip(y_metric, format=tooltip_format, title="Value")


                    chart = base.mark_line(point=True).encode(
                        x=alt.X('date_col:T', title=grain),
                        y=plot_y,
                        color='lvc_group:N',
                        tooltip=[
                            alt.Tooltip('date_col:T', title=grain), 
                            'lvc_group', 
                            tooltip_val
                        ]
                    ).interactive()

                    st.altair_chart(chart, use_container_width=True)

                def display_rate_insights(df_in, num_col, den_col, title):
                    # Totals don't depend on grain, so we can just sum the raw df or any agg level
                    group_stats = df_in.groupby('lvc_group')[[num_col, den_col]].sum().reset_index()
                    group_stats['rate'] = group_stats.apply(lambda x: x[num_col] / x[den_col] if x[den_col] > 0 else 0, axis=1)
                    
                    if not group_stats.empty:
                        # Sort desc
                        group_stats = group_stats.sort_values('rate', ascending=False)
                        best = group_stats.iloc[0]
                        worst = group_stats.iloc[-1]
                        
                        st.caption(f"**{title} Insights:** Best: `{best['lvc_group']}` ({best['rate']:.1%}) | Worst: `{worst['lvc_group']}` ({worst['rate']:.1%})")

                # --- Sequential Charts ---
                plot_time_series(df, 'lead_qty', "Gross Lead Quantity", enable_pct_toggle=True, key="lead")
                plot_time_series(df, 'sent_to_sales_qty', "Sent to Sales Quantity", enable_pct_toggle=True, key="sales")
                plot_time_series(df, 'assigned_qty', "Assigned Quantity", enable_pct_toggle=True, key="assigned")
                plot_time_series(df, 'avg_lp2c', "Average LP2C", key="lp2c")
                
                st.divider()
                st.subheader("Rates")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    plot_time_series(df, 'sales_contact_rate', "Contact Rate", is_rate=True, key="contact")
                    display_rate_insights(df, 'contacted_qty', 'sent_to_sales_qty', "Contact Rate")
                    
                with col_r2:
                    plot_time_series(df, 'fas_rate', "FAS Rate", is_rate=True, key="fas")
                    display_rate_insights(df, 'fas_qty', 'sent_to_sales_qty', "FAS Rate")

                plot_time_series(df, 'fas_volume', "FAS Volume ($)", key="vol")

                with st.expander("Raw Data (Daily)"):
                    st.dataframe(df, use_container_width=True)


            else:
                st.warning("Dashboard Query returned no results.")

        except Exception as e:
            st.error(f"Dashboard Error: {e}")

        # --- 2. Sankey Chart Section ---
        st.divider()
        st.header("🔀 Lead Flow Sankey")
        st.caption("Visualizing migration from Initial LVC to Adjusted LVC to FAS conversion.")
        
        sankey_query = f"""
        WITH flow_data AS (
            SELECT 
                CASE 
                    WHEN CAST(initial_lead_value_cohort AS STRING) IN ('1', '2') THEN '1-2'
                    WHEN CAST(initial_lead_value_cohort AS STRING) IN ('3', '4', '5', '6', '7', '8') THEN '3-8'
                    WHEN CAST(initial_lead_value_cohort AS STRING) IN ('9', '10') THEN '9-10'
                    WHEN CAST(initial_lead_value_cohort AS STRING) LIKE '%X%' THEN 'Phx Transfer'
                    ELSE 'Other'
                END as source_lvc,
                CASE 
                    WHEN CAST(adjusted_lead_value_cohort AS STRING) IN ('1', '2') THEN '1-2'
                    WHEN CAST(adjusted_lead_value_cohort AS STRING) IN ('3', '4', '5', '6', '7', '8') THEN '3-8'
                    WHEN CAST(adjusted_lead_value_cohort AS STRING) IN ('9', '10') THEN '9-10'
                    WHEN CAST(adjusted_lead_value_cohort AS STRING) LIKE '%X%' THEN 'Phx Transfer'
                    ELSE 'Other'
                END as target_lvc,
                COUNT(DISTINCT lendage_guid) as lead_count,
                COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count
            FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
            {where_clause}
            AND initial_lead_value_cohort IS NOT NULL
            GROUP BY 1, 2
        )
        SELECT * FROM flow_data
        """
        
        with st.expander("View Sankey SQL"):
            st.code(sankey_query)

        # Auto-load Sankey (Removed Button)
        try:
             with st.spinner("Fetching Sankey Data..."):
                df_sankey = client.query(sankey_query).to_dataframe()
                
                if not df_sankey.empty:

                        total_leads = df_sankey['lead_count'].sum()
                        
                        initial_nodes = sorted(df_sankey['source_lvc'].unique().tolist())
                        adjusted_nodes = sorted(df_sankey['target_lvc'].unique().tolist())
                        
                        label_list = []
                        
                        # Totals for Label %
                        init_totals = df_sankey.groupby('source_lvc')['lead_count'].sum().to_dict()
                        adj_totals = df_sankey.groupby('target_lvc')['lead_count'].sum().to_dict()
                        
                        initial_labels = [f"Initial: {x} ({init_totals.get(x,0)/total_leads:.1%})" for x in initial_nodes]
                        adjusted_labels = [f"Adj: {x} ({adj_totals.get(x,0)/total_leads:.1%})" for x in adjusted_nodes]
                        
                        fas_total = df_sankey['fas_count'].sum()
                        fas_label = f"FAS ({fas_total/total_leads:.1%})"

                        label_list.extend(initial_labels)
                        label_list.extend(adjusted_labels)
                        label_list.append(fas_label)
                        
                        source_indices = []
                        target_indices = []
                        values = []
                        custom_data = [] # For Tooltip %
                        
                        init_map = {n: i for i, n in enumerate(initial_nodes)}
                        adj_offset = len(initial_nodes)
                        adj_map = {n: i + adj_offset for i, n in enumerate(adjusted_nodes)}
                        fas_idx = len(label_list) - 1
                        
                        # Flow 1: Initial -> Adjusted
                        for _, row in df_sankey.iterrows():
                            if row['source_lvc'] in init_map and row['target_lvc'] in adj_map:
                                src_idx = init_map[row['source_lvc']]
                                val = row['lead_count']
                                source_total = init_totals.get(row['source_lvc'], 1)
                                pct_of_source = val / source_total
                                
                                source_indices.append(src_idx)
                                target_indices.append(adj_map[row['target_lvc']])
                                values.append(val)
                                custom_data.append(f"{pct_of_source:.1%} of {row['source_lvc']} -> {row['target_lvc']}")
                            
                        # Flow 2: Adjusted -> FAS
                        fas_agg = df_sankey.groupby('target_lvc').agg({'fas_count': 'sum', 'lead_count': 'sum'}).reset_index()
                        
                        # Store rates for insights
                        bucket_rates = []
                        
                        for _, row in fas_agg.iterrows():
                            if row['fas_count'] > 0 and row['target_lvc'] in adj_map:
                                src_idx = adj_map[row['target_lvc']]
                                val = row['fas_count']
                                # For Adjusted -> FAS, source total is the total leads in that Adjusted bucket
                                source_total = adj_totals.get(row['target_lvc'], 1) 
                                pct_of_group = val / source_total
                                
                                bucket_rates.append({
                                    'bucket': row['target_lvc'],
                                    'rate': pct_of_group,
                                    'volume': val
                                })
                                
                                source_indices.append(src_idx)
                                target_indices.append(fas_idx)
                                values.append(val)
                                custom_data.append(f"{pct_of_group:.1%} of Adj {row['target_lvc']} -> FAS")

                        # Plot Sankey
                        fig = go.Figure(data=[go.Sankey(
                            node = dict(
                              pad = 15,
                              thickness = 20,
                              line = dict(color = "black", width = 0.5),
                              label = label_list,
                            ),
                            link = dict(
                              source = source_indices,
                              target = target_indices,
                              value = values,
                              customdata = custom_data,
                              hovertemplate='%{value} Leads<br>%{customdata}<extra></extra>'
                          ))])

                        fig.update_layout(title_text=f"Lead Flow (N={total_leads:,})", font_size=10, height=800)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # --- Automated Insights ---
                        if bucket_rates:
                            st.subheader("💡 Key Insights")
                            # Sort by Rate DESC
                            bucket_rates.sort(key=lambda x: x['rate'], reverse=True)
                            
                            # Limit to top 4 to prevent layout issues
                            num_cols = min(len(bucket_rates), 4)
                            insight_cols = st.columns(num_cols)
                            
                            for i in range(num_cols): 
                                item = bucket_rates[i]
                                with insight_cols[i]:
                                    st.metric(
                                        label=f"{item['bucket']} Conv.",
                                        value=f"{item['rate']:.1%}",
                                        help=f"{item['volume']} FAS / {adj_totals.get(item['bucket'])} Leads"
                                    )
                                    
                            # Narrative
                            top_bucket = bucket_rates[0]
                            bottom_bucket = bucket_rates[-1]
                            if top_bucket != bottom_bucket:
                                st.markdown(f"""
                                - **Highest Conversion**: `{top_bucket['bucket']}` leads convert at **{top_bucket['rate']:.1%}**.
                                - **Lowest Conversion**: `{bottom_bucket['bucket']}` leads convert at **{bottom_bucket['rate']:.1%}**.
                                """)
                            else:
                                 st.markdown(f"- **Conversion**: `{top_bucket['bucket']}` leads convert at **{top_bucket['rate']:.1%}**.")
                        else:
                            st.info("No conversion insights available (No FAS volume detected in selection).")
                        
        except Exception as e:
            st.error(f"Sankey Error: {e}")

        # --- 3. LVC to Persona Analysis ---
        st.divider()
        st.header("👤 LVC to Persona Analysis")
        st.caption("Understanding which personas flow into which LVC groups and their FAS performance")
        
        persona_query = f"""
        SELECT 
            CASE 
                WHEN CAST(adjusted_lead_value_cohort AS STRING) IN ('1', '2') THEN 'LVC 1-2'
                WHEN CAST(adjusted_lead_value_cohort AS STRING) IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
                WHEN CAST(adjusted_lead_value_cohort AS STRING) IN ('9', '10') THEN 'LVC 9-10'
                WHEN CAST(adjusted_lead_value_cohort AS STRING) LIKE '%X%' THEN 'PHX Transfer'
                ELSE 'Other'
            END as lvc_group,
            COALESCE(persona, 'Unknown') as persona,
            COUNT(DISTINCT lendage_guid) as lead_count,
            COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales_qty,
            COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        {where_clause}
        AND adjusted_lead_value_cohort IS NOT NULL
        GROUP BY 1, 2
        """
        
        with st.expander("View LVC-Persona SQL"):
            st.code(persona_query)
        
        try:
            with st.spinner("Fetching LVC-Persona Data..."):
                df_persona = client.query(persona_query).to_dataframe()
                
                if not df_persona.empty:
                    # Calculate FAS Rate
                    df_persona['fas_rate'] = df_persona.apply(
                        lambda x: x['fas_count'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                    )
                    
                    # Create two visualizations side by side
                    persona_col1, persona_col2 = st.columns(2)
                    
                    with persona_col1:
                        st.subheader("📊 Lead Distribution: LVC → Persona")
                        
                        # Sankey from LVC to Persona
                        lvc_nodes = sorted(df_persona['lvc_group'].unique().tolist())
                        persona_nodes = sorted(df_persona['persona'].unique().tolist())
                        
                        # Calculate totals for percentages
                        total_leads = df_persona['lead_count'].sum()
                        lvc_totals = df_persona.groupby('lvc_group')['lead_count'].sum().to_dict()
                        persona_totals = df_persona.groupby('persona')['lead_count'].sum().to_dict()
                        
                        # Create node labels with percentages
                        lvc_labels = [f"{node}<br>({lvc_totals.get(node, 0):,} | {lvc_totals.get(node, 0)/total_leads*100:.1f}%)" for node in lvc_nodes]
                        persona_labels = [f"{node}<br>({persona_totals.get(node, 0):,} | {persona_totals.get(node, 0)/total_leads*100:.1f}%)" for node in persona_nodes]
                        all_node_labels = lvc_labels + persona_labels
                        all_nodes = lvc_nodes + persona_nodes
                        
                        # Node colors
                        lvc_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                        persona_colors = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a']
                        node_colors = lvc_colors[:len(lvc_nodes)] + persona_colors[:len(persona_nodes)]
                        
                        sources = []
                        targets = []
                        values = []
                        link_colors = []
                        link_labels = []
                        
                        for _, row in df_persona.iterrows():
                            src_idx = all_nodes.index(row['lvc_group'])
                            tgt_idx = all_nodes.index(row['persona'])
                            sources.append(src_idx)
                            targets.append(tgt_idx)
                            values.append(row['lead_count'])
                            link_colors.append(f"rgba(31, 119, 180, 0.4)")
                            # Calculate % of total and % within LVC
                            pct_total = row['lead_count'] / total_leads * 100
                            pct_lvc = row['lead_count'] / lvc_totals.get(row['lvc_group'], 1) * 100
                            link_labels.append(f"{row['lead_count']:,} leads ({pct_total:.1f}% of total, {pct_lvc:.1f}% of {row['lvc_group']})")
                        
                        fig_persona = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                                label=all_node_labels,
                                color=node_colors
                            ),
                            link=dict(
                                source=sources,
                                target=targets,
                                value=values,
                                color=link_colors,
                                label=link_labels,
                                hovertemplate='%{label}<extra></extra>'
                            ),
                            textfont=dict(color='black', size=12)
                        )])
                        
                        fig_persona.update_layout(
                            font=dict(size=12, color='black'),
                            height=500,
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig_persona, use_container_width=True)
                    
                    with persona_col2:
                        st.subheader("🎯 FAS Rate by LVC & Persona")
                        
                        # Create pivot table for heatmap
                        pivot_fas = df_persona.pivot_table(
                            values='fas_rate', 
                            index='persona', 
                            columns='lvc_group', 
                            aggfunc='mean'
                        ).fillna(0)
                        
                        # Heatmap using Altair
                        df_heatmap = df_persona[['lvc_group', 'persona', 'fas_rate', 'lead_count']].copy()
                        
                        heatmap = alt.Chart(df_heatmap).mark_rect().encode(
                            x=alt.X('lvc_group:N', title='LVC Group', axis=alt.Axis(labelColor='black', labelFontWeight='bold', titleColor='black', titleFontWeight='bold')),
                            y=alt.Y('persona:N', title='Persona', axis=alt.Axis(labelColor='black', labelFontWeight='bold', titleColor='black', titleFontWeight='bold')),
                            color=alt.Color('fas_rate:Q', title='FAS Rate', scale=alt.Scale(scheme='blues')),
                            tooltip=[
                                alt.Tooltip('lvc_group:N', title='LVC'),
                                alt.Tooltip('persona:N', title='Persona'),
                                alt.Tooltip('fas_rate:Q', title='FAS Rate', format='.1%'),
                                alt.Tooltip('lead_count:Q', title='Lead Count', format=',d')
                            ]
                        ).properties(
                            height=400
                        ).configure_view(
                            strokeWidth=0,
                            fill='#ffffff'
                        ).configure(
                            background='#ffffff'
                        )
                        
                        st.altair_chart(heatmap, use_container_width=True)
                    
                    # Summary table
                    st.subheader("📋 Detailed Breakdown")
                    
                    # Aggregate by persona
                    persona_summary = df_persona.groupby('persona').agg({
                        'lead_count': 'sum',
                        'sent_to_sales_qty': 'sum',
                        'fas_count': 'sum'
                    }).reset_index()
                    persona_summary['fas_rate'] = persona_summary.apply(
                        lambda x: x['fas_count'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                    )
                    persona_summary['lead_pct'] = persona_summary['lead_count'] / persona_summary['lead_count'].sum()
                    persona_summary = persona_summary.sort_values('fas_rate', ascending=False)
                    
                    # Format for display
                    display_df = persona_summary.copy()
                    display_df['FAS Rate'] = display_df['fas_rate'].apply(lambda x: f"{x:.1%}")
                    display_df['Lead %'] = display_df['lead_pct'].apply(lambda x: f"{x:.1%}")
                    display_df['Leads'] = display_df['lead_count'].apply(lambda x: f"{x:,}")
                    display_df['FAS Count'] = display_df['fas_count'].apply(lambda x: f"{x:,}")
                    display_df = display_df[['persona', 'Leads', 'Lead %', 'FAS Count', 'FAS Rate']]
                    display_df.columns = ['Persona', 'Leads', 'Lead %', 'FAS Count', 'FAS Rate']
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Insights
                    best_persona = persona_summary.iloc[0]
                    worst_persona = persona_summary.iloc[-1]
                    
                    st.markdown("**🔍 Key Insights:**")
                    st.markdown(f"- **Best Converting Persona**: `{best_persona['persona']}` with **{best_persona['fas_rate']:.1%}** FAS rate")
                    st.markdown(f"- **Lowest Converting Persona**: `{worst_persona['persona']}` with **{worst_persona['fas_rate']:.1%}** FAS rate")
                    
                else:
                    st.warning("No persona data found for the selected filters.")
                    
        except Exception as e:
            st.error(f"LVC-Persona Analysis Error: {e}")
            st.caption("Note: This analysis requires a 'persona' field in the data source.")

        # --- HTML Export Logic ---
        st.divider()
        if 'df' in locals() and not df.empty:
            # Pre-calculate Weekly DF for the report (Standardized View)
            df_export = df.copy()
            if 'lead_date' in df_export.columns:
                 df_export['date_col'] = df_export['lead_date'] - pd.to_timedelta(df_export['lead_date'].dt.weekday, unit='D')
            
            # Reuse aggregation logic briefly
            metric_aggs_exp = {
                'lead_qty': 'sum', 'sent_to_sales_qty': 'sum', 'assigned_qty': 'sum',
                'contacted_qty': 'sum', 'fas_qty': 'sum', 'fas_volume': 'sum', 'avg_lp2c': 'mean'
            }
            df_weekly_export = df_export.groupby(['date_col', 'lvc_group']).agg(metric_aggs_exp).reset_index()
            df_weekly_export.rename(columns={'date_col': 'week_start'}, inplace=True)
            
            # Rates
            df_weekly_export['sales_contact_rate'] = df_weekly_export.apply(lambda x: x['contacted_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1)
            df_weekly_export['fas_rate'] = df_weekly_export.apply(lambda x: x['fas_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1)

            def get_chart_html(data, y_metric, title, is_rate=False):
                y_axis = alt.Y(y_metric, title=title)
                tooltip_format = ',d'
                if is_rate or 'rate' in y_metric or 'lp2c' in y_metric:
                    if 'lp2c' not in y_metric:
                         y_axis = alt.Y(y_metric, title=title, axis=alt.Axis(format='%'))
                         tooltip_format = '.1%'
                    else:
                         tooltip_format = '.1f'

                chart = alt.Chart(data).mark_line(point=True).encode(
                    x=alt.X('week_start:T', title='Week Of'),
                    y=y_axis,
                    color='lvc_group:N',
                    tooltip=[alt.Tooltip('week_start:T', title='Week'), 'lvc_group', alt.Tooltip(y_metric, format=tooltip_format, title="Value")]
                ).properties(title=title)
                return chart.to_json()

            if st.sidebar.button("📦 Prepare HTML Report"):
                with st.spinner("Generating Report..."):
                    # Generate Chart JSONs
                    c_lead = get_chart_html(df_weekly_export, 'lead_qty', "Gross Lead Quantity")
                    c_sent = get_chart_html(df_weekly_export, 'sent_to_sales_qty', "Sent to Sales Quantity")
                    c_assigned = get_chart_html(df_weekly_export, 'assigned_qty', "Assigned Quantity")
                    c_lp2c = get_chart_html(df_weekly_export, 'avg_lp2c', "Average LP2C", is_rate=True)
                    c_contact = get_chart_html(df_weekly_export, 'sales_contact_rate', "Contact Rate", is_rate=True)
                    c_fas_rate = get_chart_html(df_weekly_export, 'fas_rate', "FAS Rate", is_rate=True)
                    c_fas_vol = get_chart_html(df_weekly_export, 'fas_volume', "FAS Volume ($)")
                    
                    # Sankey HTML
                    sankey_html = fig.to_html(full_html=False, include_plotlyjs='cdn') if 'fig' in locals() else "<i>Sankey not loaded</i>"

                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>LVC Report Breakout</title>
                        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
                        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
                        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
                        <style>
                            body {{ font-family: sans-serif; padding: 20px; background-color: #f4f4f4; }}
                            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                            h1, h2 {{ color: #333; }}
                            .chart-box {{ margin-bottom: 40px; padding: 20px; border: 1px solid #eee; border-radius: 5px; }}
                            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>📊 LVC Report Breakout</h1>
                            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                            <hr>
                            
                            <h2>📈 Volume Trends</h2>
                            <div class="grid">
                                <div id="vis_lead" class="chart-box"></div>
                                <div id="vis_lp2c" class="chart-box"></div>
                            </div>
                            <div class="grid">
                                <div id="vis_sent" class="chart-box"></div>
                                <div id="vis_assigned" class="chart-box"></div>
                            </div>
                            
                            <h2>🎯 Efficiency Rates</h2>
                            <div class="grid">
                                <div id="vis_contact" class="chart-box"></div>
                                <div id="vis_fas_rate" class="chart-box"></div>
                            </div>
                            
                            <h2>💰 Volume</h2>
                            <div id="vis_fas_vol" class="chart-box"></div>
                            
                            <h2>🔀 Lead Flow Sankey</h2>
                            <div class="chart-box">
                                {sankey_html}
                            </div>
                        </div>

                        <script type="text/javascript">
                            vegaEmbed('#vis_lead', {c_lead});
                            vegaEmbed('#vis_lp2c', {c_lp2c});
                            vegaEmbed('#vis_sent', {c_sent});
                            vegaEmbed('#vis_assigned', {c_assigned});
                            vegaEmbed('#vis_contact', {c_contact});
                            vegaEmbed('#vis_fas_rate', {c_fas_rate});
                            vegaEmbed('#vis_fas_vol', {c_fas_vol});
                        </script>
                    </body>
                    </html>
                    """
                    
                    st.sidebar.download_button(
                        label="⬇️ Download HTML Report",
                        data=html_content,
                        file_name=f"LVC_Report_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html"
                    )

    # --- TAB 2: Pre/Post Launch Analysis ---
    with tab_launch:
        st.header("🚀 Pre/Post Launch Impact Analysis")
        
        # --- Controls ---
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        
        with ctrl_col1:
            # Add "All Launches" as first option
            event_options = ["📊 All Launches"] + [f"{e['name']} ({e['date']})" for e in LAUNCH_EVENTS]
            selected_event_idx = st.selectbox(
                "Select Launch Event",
                range(len(event_options)),
                format_func=lambda x: event_options[x],
                index=0  # Default to "All Launches"
            )
            
            # Handle "All" vs single event
            show_all_launches = (selected_event_idx == 0)
            if show_all_launches:
                selected_event = None
                launch_date = None
            else:
                selected_event = LAUNCH_EVENTS[selected_event_idx - 1]  # -1 because of "All" option
                launch_date = pd.Timestamp(selected_event['date'])
        
        with ctrl_col2:
            time_grain = st.selectbox("Time Grain", ["Day", "Week", "Month"], index=1)
        
        with ctrl_col3:
            compare_options = ["Back to 1/2024", 30, 60, 90, 120]
            compare_window = st.selectbox(
                "Compare Window", 
                compare_options, 
                index=0,  # Default to "Back to 1/2024"
                format_func=lambda x: "Back to 1/2024" if x == "Back to 1/2024" else f"{x} days"
            )
        
        if show_all_launches:
            st.caption("Analyzing: **All Launch Events** with timeline markers")
        else:
            st.caption(f"Analyzing: **{selected_event['name']}** launched on **{selected_event['date']}**")
        
        # --- Query for Pre/Post Launch Data with FAS Day 7 ---
        # Determine query start date based on compare_window
        query_start_date = '2024-01-01' if compare_window == "Back to 1/2024" else '2025-01-01'
        
        launch_query = f"""
        WITH base_data AS (
            SELECT 
                DATE(lead_created_date) as lead_date,
                CASE 
                    WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
                    WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
                    WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
                    WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
                    ELSE 'Other'
                END as lvc_group,
                lendage_guid,
                sent_to_sales_date,
                sf__contacted_guid,
                full_app_submit_datetime,
                lead_created_date
            FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
            WHERE lead_created_date >= '{query_start_date}'
        )
        SELECT 
            lead_date,
            lvc_group,
            COUNT(DISTINCT lendage_guid) as lead_qty,
            COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales_qty,
            COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted_qty,
            COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_qty,
            -- FAS within 7 days of lead creation (FAS Rate Day 7)
            COUNT(DISTINCT CASE 
                WHEN full_app_submit_datetime IS NOT NULL 
                AND DATE(full_app_submit_datetime) <= DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
                THEN lendage_guid 
            END) as fas_day7_qty
        FROM base_data
        GROUP BY 1, 2
        ORDER BY 1 DESC
        """
        
        try:
            @st.cache_data(ttl=600)
            def run_launch_query(sql_query):
                return client.query(sql_query).to_dataframe()
            
            with st.spinner("Loading launch analysis data..."):
                df_launch = run_launch_query(launch_query)
            
            if not df_launch.empty:
                df_launch['lead_date'] = pd.to_datetime(df_launch['lead_date'])
                
                # Determine date range based on compare_window
                if compare_window == "Back to 1/2024":
                    data_start = pd.Timestamp("2024-01-01")
                    data_end = pd.Timestamp.today()
                else:
                    # Use window around launch date (or first launch if "All")
                    if show_all_launches:
                        # For "All", use earliest launch as reference
                        earliest_launch = pd.Timestamp(min(e['date'] for e in LAUNCH_EVENTS))
                        data_start = earliest_launch - timedelta(days=compare_window)
                        data_end = pd.Timestamp.today()
                    else:
                        data_start = launch_date - timedelta(days=compare_window)
                        data_end = launch_date + timedelta(days=compare_window)
                
                # Filter data to range
                df_window = df_launch[
                    (df_launch['lead_date'] >= data_start) & 
                    (df_launch['lead_date'] <= data_end)
                ]
                
                # For KPI comparison, split pre/post based on selected event or first event if "All"
                if show_all_launches:
                    # Use earliest launch for pre/post split when showing all
                    kpi_launch_date = pd.Timestamp(min(e['date'] for e in LAUNCH_EVENTS))
                else:
                    kpi_launch_date = launch_date
                
                df_pre = df_window[df_window['lead_date'] < kpi_launch_date]
                df_post = df_window[df_window['lead_date'] >= kpi_launch_date]
                
                # --- KPI Cards ---
                st.subheader("📊 Key Metrics Comparison")
                
                def calc_metrics(df):
                    sts = df['sent_to_sales_qty'].sum()
                    return {
                        'lead_qty': df['lead_qty'].sum(),
                        'sent_to_sales_qty': sts,
                        'contacted_qty': df['contacted_qty'].sum(),
                        'fas_qty': df['fas_qty'].sum(),
                        'fas_day7_qty': df['fas_day7_qty'].sum(),
                        'contact_rate': df['contacted_qty'].sum() / sts if sts > 0 else 0,
                        'fas_rate': df['fas_qty'].sum() / sts if sts > 0 else 0,
                        'fas_rate_d7': df['fas_day7_qty'].sum() / sts if sts > 0 else 0,
                    }
                
                pre_metrics = calc_metrics(df_pre)
                post_metrics = calc_metrics(df_post)
                
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                
                with kpi_col1:
                    delta_fas = post_metrics['fas_rate'] - pre_metrics['fas_rate']
                    st.metric(
                        "FAS Rate",
                        f"{post_metrics['fas_rate']:.1%}",
                        f"{delta_fas:+.1%}",
                        delta_color="normal"
                    )
                    st.caption(f"Pre: {pre_metrics['fas_rate']:.1%}")
                    st.caption("📐 `FAS Qty / Sent to Sales Qty`")
                
                with kpi_col2:
                    delta_fas_d7 = post_metrics['fas_rate_d7'] - pre_metrics['fas_rate_d7']
                    st.metric(
                        "FAS Rate Day 7",
                        f"{post_metrics['fas_rate_d7']:.1%}",
                        f"{delta_fas_d7:+.1%}",
                        delta_color="normal"
                    )
                    st.caption(f"Pre: {pre_metrics['fas_rate_d7']:.1%}")
                    st.caption("📐 `FAS within 7 days of lead_created_date / Leads created (StS)`")
                
                with kpi_col3:
                    delta_contact = post_metrics['contact_rate'] - pre_metrics['contact_rate']
                    st.metric(
                        "Contact Rate",
                        f"{post_metrics['contact_rate']:.1%}",
                        f"{delta_contact:+.1%}",
                        delta_color="normal"
                    )
                    st.caption(f"Pre: {pre_metrics['contact_rate']:.1%}")
                    st.caption("📐 `COUNT(DISTINCT sf__contacted_guid) / Sent to Sales Qty`")
                
                st.divider()
                
                # --- Trend Charts ---
                st.subheader("📈 Trend Analysis")
                
                # Aggregate by time grain
                def aggregate_launch_data(df, grain):
                    df_agg = df.copy()
                    if grain == 'Week':
                        df_agg['date_col'] = df_agg['lead_date'] - pd.to_timedelta(df_agg['lead_date'].dt.weekday, unit='D')
                    elif grain == 'Month':
                        df_agg['date_col'] = df_agg['lead_date'].dt.to_period('M').dt.to_timestamp()
                    else:
                        df_agg['date_col'] = df_agg['lead_date']
                    
                    agg_funcs = {
                        'lead_qty': 'sum',
                        'sent_to_sales_qty': 'sum',
                        'contacted_qty': 'sum',
                        'fas_qty': 'sum',
                        'fas_day7_qty': 'sum'
                    }
                    
                    df_grouped = df_agg.groupby(['date_col', 'lvc_group']).agg(agg_funcs).reset_index()
                    df_grouped['fas_rate'] = df_grouped.apply(
                        lambda x: x['fas_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                    )
                    df_grouped['fas_rate_d7'] = df_grouped.apply(
                        lambda x: x['fas_day7_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                    )
                    df_grouped['contact_rate'] = df_grouped.apply(
                        lambda x: x['contacted_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                    )
                    return df_grouped
                
                # Use already filtered df_window for aggregation
                df_agg = aggregate_launch_data(df_window, time_grain)
                
                # Create Overall (Total) aggregation
                df_overall = df_window.copy()
                if time_grain == 'Week':
                    df_overall['date_col'] = df_overall['lead_date'] - pd.to_timedelta(df_overall['lead_date'].dt.weekday, unit='D')
                elif time_grain == 'Month':
                    df_overall['date_col'] = df_overall['lead_date'].dt.to_period('M').dt.to_timestamp()
                else:
                    df_overall['date_col'] = df_overall['lead_date']
                
                df_overall_agg = df_overall.groupby('date_col').agg({
                    'lead_qty': 'sum',
                    'sent_to_sales_qty': 'sum',
                    'contacted_qty': 'sum',
                    'fas_qty': 'sum',
                    'fas_day7_qty': 'sum'
                }).reset_index()
                df_overall_agg['fas_rate'] = df_overall_agg.apply(
                    lambda x: x['fas_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                df_overall_agg['fas_rate_d7'] = df_overall_agg.apply(
                    lambda x: x['fas_day7_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                df_overall_agg['contact_rate'] = df_overall_agg.apply(
                    lambda x: x['contacted_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                
                # Chart with launch date reference line
                def create_launch_chart(data, y_col, title, is_rate=True, by_lvc=False):
                    y_format = '.1%' if is_rate else ',d'
                    y_axis_format = '%' if is_rate else None
                    
                    # X-axis with year shown (multi-line format: Month Day \n Year) - black bold labels
                    x_axis = alt.X(
                        'date_col:T', 
                        title=time_grain,
                        axis=alt.Axis(
                            format='%b %d',
                            labelExpr="datum.label + '\\n' + timeFormat(datum.value, '%Y')",
                            labelAlign='center',
                            labelPadding=5,
                            labelColor='black',
                            labelFontWeight='bold',
                            titleColor='black',
                            titleFontWeight='bold'
                        )
                    )
                    
                    # Y-axis with black bold labels
                    y_axis_config = {
                        'labelColor': 'black',
                        'labelFontWeight': 'bold',
                        'titleColor': 'black',
                        'titleFontWeight': 'bold'
                    }
                    if y_axis_format:
                        y_axis_config['format'] = y_axis_format
                    
                    y_axis = alt.Y(
                        y_col, 
                        title=title, 
                        axis=alt.Axis(**y_axis_config)
                    )
                    
                    if by_lvc:
                        base = alt.Chart(data).mark_line(point=True).encode(
                            x=x_axis,
                            y=y_axis,
                            color='lvc_group:N',
                            tooltip=[
                                alt.Tooltip('date_col:T', title=time_grain, format='%b %d, %Y'),
                                'lvc_group',
                                alt.Tooltip(y_col, format=y_format, title=title)
                            ]
                        )
                    else:
                        base = alt.Chart(data).mark_line(point=True, color='#1f77b4').encode(
                            x=x_axis,
                            y=y_axis,
                            tooltip=[
                                alt.Tooltip('date_col:T', title=time_grain, format='%b %d, %Y'),
                                alt.Tooltip(y_col, format=y_format, title=title)
                            ]
                        )
                    
                    # Build reference lines for all visible launches
                    from collections import defaultdict
                    
                    # Get data date range
                    x_min = data['date_col'].min()
                    x_max = data['date_col'].max()
                    
                    if show_all_launches:
                        # Filter events to those within range
                        visible_events = [e for e in LAUNCH_EVENTS 
                                         if pd.Timestamp(e['date']) >= x_min and pd.Timestamp(e['date']) <= x_max]
                    else:
                        visible_events = [selected_event] if selected_event and launch_date and x_min <= launch_date <= x_max else []
                    
                    if not visible_events:
                        return base.interactive().configure_view(
                            strokeWidth=0,
                            fill='#ffffff'
                        ).configure(
                            background='#ffffff'
                        )
                    
                    # Count same-date events for staggering
                    date_counts = defaultdict(int)
                    for event in visible_events:
                        date_counts[event['date']] += 1
                    
                    date_indices = defaultdict(int)
                    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999', '#66c2a5']
                    
                    layers = [base]
                    
                    for i, event in enumerate(visible_events):
                        event_date = pd.Timestamp(event['date'])
                        event_name = event['name']
                        color = colors[i % len(colors)] if show_all_launches else 'red'
                        
                        same_date_idx = date_indices[event['date']]
                        date_indices[event['date']] += 1
                        num_same_date = date_counts[event['date']]
                        
                        rule_df = pd.DataFrame({'x': [event_date]})
                        
                        # Only one vertical line per date
                        if same_date_idx == 0:
                            rule = alt.Chart(rule_df).mark_rule(
                                color=color, strokeDash=[5, 5], strokeWidth=2
                            ).encode(x='x:T')
                            layers.append(rule)
                        
                        # Stagger labels for same-date events
                        if num_same_date > 1:
                            dy = -15 - (same_date_idx * 14)
                        else:
                            dy = -15 - (i % 3) * 16
                        
                        label_df = pd.DataFrame({'x': [event_date], 'label': [event_name]})
                        label = alt.Chart(label_df).mark_text(
                            align='left', dx=3, dy=dy, color=color, fontSize=11, fontWeight='bold'
                        ).encode(x='x:T', text='label:N')
                        layers.append(label)
                    
                    return alt.layer(*layers).interactive().configure_view(
                        strokeWidth=0,
                        fill='#ffffff'
                    ).configure(
                        background='#ffffff'
                    )
                
                # Overall Charts - Full Width (Presentation Style)
                st.markdown("#### Overall Metrics")
                
                st.markdown("<h4 style='text-align: center;'>FAS Rate</h4>", unsafe_allow_html=True)
                st.altair_chart(
                    create_launch_chart(df_overall_agg, 'fas_rate', 'FAS Rate', is_rate=True),
                    use_container_width=True
                )
                
                st.markdown("<h4 style='text-align: center;'>FAS Rate Day 7</h4>", unsafe_allow_html=True)
                st.altair_chart(
                    create_launch_chart(df_overall_agg, 'fas_rate_d7', 'FAS Rate Day 7', is_rate=True),
                    use_container_width=True
                )
                
                st.markdown("<h4 style='text-align: center;'>Contact Rate</h4>", unsafe_allow_html=True)
                st.altair_chart(
                    create_launch_chart(df_overall_agg, 'contact_rate', 'Contact Rate', is_rate=True),
                    use_container_width=True
                )
                
                st.markdown("<h4 style='text-align: center;'>Sent to Sales Volume</h4>", unsafe_allow_html=True)
                st.altair_chart(
                    create_launch_chart(df_overall_agg, 'sent_to_sales_qty', 'Sent to Sales Volume', is_rate=False),
                    use_container_width=True
                )
                
                # By LVC Group Charts - Full Width
                st.markdown("#### By LVC Group")
                
                st.markdown("<h4 style='text-align: center;'>FAS Rate by LVC Group</h4>", unsafe_allow_html=True)
                st.altair_chart(
                    create_launch_chart(df_agg, 'fas_rate', 'FAS Rate by LVC', is_rate=True, by_lvc=True),
                    use_container_width=True
                )
                
                st.markdown("<h4 style='text-align: center;'>FAS Rate Day 7 by LVC Group</h4>", unsafe_allow_html=True)
                st.altair_chart(
                    create_launch_chart(df_agg, 'fas_rate_d7', 'FAS Rate Day 7 by LVC', is_rate=True, by_lvc=True),
                    use_container_width=True
                )
                
                st.divider()
                
                # --- Seasonality Comparison & Variance Analysis ---
                st.subheader("📊 Seasonality Analysis: Product Launch Impact vs Seasonal Trends")
                st.caption("**Goal:** Determine if post-10/2025 performance decline is due to product launches OR normal seasonality")
                
                # Define comparison ranges
                period_2024_start = pd.Timestamp("2024-10-01")
                period_2024_end = pd.Timestamp("2025-01-31")
                period_2025_start = pd.Timestamp("2025-11-01")
                period_2025_end = pd.Timestamp("2026-01-31")
                
                st.info(f"**Comparing:** Oct 2024 - Jan 2025 vs Nov 2025 - Jan 2026")
                
                # Filter data for each period
                df_period_2024 = df_launch[
                    (df_launch['lead_date'] >= period_2024_start) & 
                    (df_launch['lead_date'] <= period_2024_end)
                ]
                df_period_2025 = df_launch[
                    (df_launch['lead_date'] >= period_2025_start) & 
                    (df_launch['lead_date'] <= min(period_2025_end, pd.Timestamp.today()))
                ]
                
                # Calculate metrics for each period
                def calc_period_metrics(df, period_name):
                    if df.empty:
                        return {
                            'period': period_name,
                            'lead_qty': 0,
                            'sent_to_sales_qty': 0,
                            'contacted_qty': 0,
                            'fas_qty': 0,
                            'fas_day7_qty': 0,
                            'fas_rate': 0,
                            'fas_rate_d7': 0,
                            'contact_rate': 0
                        }
                    sts = df['sent_to_sales_qty'].sum()
                    return {
                        'period': period_name,
                        'lead_qty': df['lead_qty'].sum(),
                        'sent_to_sales_qty': sts,
                        'contacted_qty': df['contacted_qty'].sum(),
                        'fas_qty': df['fas_qty'].sum(),
                        'fas_day7_qty': df['fas_day7_qty'].sum(),
                        'fas_rate': df['fas_qty'].sum() / sts if sts > 0 else 0,
                        'fas_rate_d7': df['fas_day7_qty'].sum() / sts if sts > 0 else 0,
                        'contact_rate': df['contacted_qty'].sum() / sts if sts > 0 else 0
                    }
                
                metrics_2024 = calc_period_metrics(df_period_2024, "Oct'24-Jan'25")
                metrics_2025 = calc_period_metrics(df_period_2025, "Nov'25-Jan'26")
                
                # Display period comparison KPIs
                st.markdown("#### Overall Period Comparison")
                period_col1, period_col2, period_col3 = st.columns(3)
                
                with period_col1:
                    delta_fas_period = metrics_2025['fas_rate'] - metrics_2024['fas_rate']
                    st.metric(
                        "FAS Rate",
                        f"{metrics_2025['fas_rate']:.1%}",
                        f"{delta_fas_period:+.1%} vs prior year",
                        delta_color="normal"
                    )
                    st.caption(f"Oct'24-Jan'25: {metrics_2024['fas_rate']:.1%}")
                
                with period_col2:
                    delta_fas_d7_period = metrics_2025['fas_rate_d7'] - metrics_2024['fas_rate_d7']
                    st.metric(
                        "FAS Rate Day 7",
                        f"{metrics_2025['fas_rate_d7']:.1%}",
                        f"{delta_fas_d7_period:+.1%} vs prior year",
                        delta_color="normal"
                    )
                    st.caption(f"Oct'24-Jan'25: {metrics_2024['fas_rate_d7']:.1%}")
                
                with period_col3:
                    delta_contact_period = metrics_2025['contact_rate'] - metrics_2024['contact_rate']
                    st.metric(
                        "Contact Rate",
                        f"{metrics_2025['contact_rate']:.1%}",
                        f"{delta_contact_period:+.1%} vs prior year",
                        delta_color="normal"
                    )
                    st.caption(f"Oct'24-Jan'25: {metrics_2024['contact_rate']:.1%}")
                
                st.divider()
                
                # --- WoW / MoM Variance Analysis ---
                st.markdown("#### 📈 Week-over-Week (WoW) & Month-over-Month (MoM) Variance")
                st.caption("Analyzing if recent trends are getting worse or stabilizing")
                
                # Aggregate by week for recent data
                df_recent = df_launch[df_launch['lead_date'] >= pd.Timestamp("2025-10-01")].copy()
                df_recent['week'] = df_recent['lead_date'] - pd.to_timedelta(df_recent['lead_date'].dt.weekday, unit='D')
                df_recent['month'] = df_recent['lead_date'].dt.to_period('M').dt.to_timestamp()
                
                # Weekly aggregation
                weekly_agg = df_recent.groupby('week').agg({
                    'lead_qty': 'sum',
                    'sent_to_sales_qty': 'sum',
                    'contacted_qty': 'sum',
                    'fas_qty': 'sum',
                    'fas_day7_qty': 'sum'
                }).reset_index()
                weekly_agg['fas_rate'] = weekly_agg.apply(
                    lambda x: x['fas_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                weekly_agg['fas_rate_d7'] = weekly_agg.apply(
                    lambda x: x['fas_day7_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                weekly_agg['contact_rate'] = weekly_agg.apply(
                    lambda x: x['contacted_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                
                # Calculate WoW changes
                weekly_agg['fas_rate_wow'] = weekly_agg['fas_rate'].diff()
                weekly_agg['fas_rate_d7_wow'] = weekly_agg['fas_rate_d7'].diff()
                weekly_agg['contact_rate_wow'] = weekly_agg['contact_rate'].diff()
                
                # Monthly aggregation
                monthly_agg = df_recent.groupby('month').agg({
                    'lead_qty': 'sum',
                    'sent_to_sales_qty': 'sum',
                    'contacted_qty': 'sum',
                    'fas_qty': 'sum',
                    'fas_day7_qty': 'sum'
                }).reset_index()
                monthly_agg['fas_rate'] = monthly_agg.apply(
                    lambda x: x['fas_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                monthly_agg['fas_rate_d7'] = monthly_agg.apply(
                    lambda x: x['fas_day7_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                monthly_agg['contact_rate'] = monthly_agg.apply(
                    lambda x: x['contacted_qty'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                
                # Calculate MoM changes
                monthly_agg['fas_rate_mom'] = monthly_agg['fas_rate'].diff()
                monthly_agg['fas_rate_d7_mom'] = monthly_agg['fas_rate_d7'].diff()
                monthly_agg['contact_rate_mom'] = monthly_agg['contact_rate'].diff()
                
                # Display WoW table
                var_col1, var_col2 = st.columns(2)
                
                with var_col1:
                    st.markdown("**Weekly Variance (WoW)**")
                    wow_display = weekly_agg[['week', 'fas_rate', 'fas_rate_wow', 'fas_rate_d7', 'fas_rate_d7_wow', 'contact_rate', 'contact_rate_wow']].copy()
                    wow_display['week'] = wow_display['week'].dt.strftime('%Y-%m-%d')
                    wow_display['fas_rate'] = wow_display['fas_rate'].apply(lambda x: f"{x:.1%}")
                    wow_display['fas_rate_wow'] = wow_display['fas_rate_wow'].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
                    wow_display['fas_rate_d7'] = wow_display['fas_rate_d7'].apply(lambda x: f"{x:.1%}")
                    wow_display['fas_rate_d7_wow'] = wow_display['fas_rate_d7_wow'].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
                    wow_display['contact_rate'] = wow_display['contact_rate'].apply(lambda x: f"{x:.1%}")
                    wow_display['contact_rate_wow'] = wow_display['contact_rate_wow'].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
                    wow_display.columns = ['Week', 'FAS Rate', 'FAS WoW', 'FAS D7', 'D7 WoW', 'Contact', 'Cont WoW']
                    st.dataframe(wow_display.tail(8), use_container_width=True, hide_index=True)
                
                with var_col2:
                    st.markdown("**Monthly Variance (MoM)**")
                    mom_display = monthly_agg[['month', 'fas_rate', 'fas_rate_mom', 'fas_rate_d7', 'fas_rate_d7_mom', 'contact_rate', 'contact_rate_mom']].copy()
                    mom_display['month'] = mom_display['month'].dt.strftime('%Y-%m')
                    mom_display['fas_rate'] = mom_display['fas_rate'].apply(lambda x: f"{x:.1%}")
                    mom_display['fas_rate_mom'] = mom_display['fas_rate_mom'].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
                    mom_display['fas_rate_d7'] = mom_display['fas_rate_d7'].apply(lambda x: f"{x:.1%}")
                    mom_display['fas_rate_d7_mom'] = mom_display['fas_rate_d7_mom'].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
                    mom_display['contact_rate'] = mom_display['contact_rate'].apply(lambda x: f"{x:.1%}")
                    mom_display['contact_rate_mom'] = mom_display['contact_rate_mom'].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
                    mom_display.columns = ['Month', 'FAS Rate', 'FAS MoM', 'FAS D7', 'D7 MoM', 'Contact', 'Cont MoM']
                    st.dataframe(mom_display, use_container_width=True, hide_index=True)
                
                st.divider()
                
                # --- Diagnosis: Product Launch vs Seasonality ---
                st.subheader("🔍 Diagnosis: Product Launch Impact vs Seasonality")
                
                # Calculate key diagnostic metrics
                # 1. Did 2024 also see a decline in same period?
                df_2024_oct_dec = df_launch[
                    (df_launch['lead_date'] >= pd.Timestamp("2024-10-01")) & 
                    (df_launch['lead_date'] <= pd.Timestamp("2024-12-31"))
                ]
                df_2024_jul_sep = df_launch[
                    (df_launch['lead_date'] >= pd.Timestamp("2024-07-01")) & 
                    (df_launch['lead_date'] <= pd.Timestamp("2024-09-30"))
                ]
                
                def calc_simple_rate(df):
                    if df.empty:
                        return {'fas_rate': 0, 'contact_rate': 0, 'fas_rate_d7': 0}
                    sts = df['sent_to_sales_qty'].sum()
                    return {
                        'fas_rate': df['fas_qty'].sum() / sts if sts > 0 else 0,
                        'contact_rate': df['contacted_qty'].sum() / sts if sts > 0 else 0,
                        'fas_rate_d7': df['fas_day7_qty'].sum() / sts if sts > 0 else 0
                    }
                
                metrics_2024_q4 = calc_simple_rate(df_2024_oct_dec)
                metrics_2024_q3 = calc_simple_rate(df_2024_jul_sep)
                
                # 2. 2025 comparison
                df_2025_oct_dec = df_launch[
                    (df_launch['lead_date'] >= pd.Timestamp("2025-10-01")) & 
                    (df_launch['lead_date'] <= pd.Timestamp("2025-12-31"))
                ]
                df_2025_jul_sep = df_launch[
                    (df_launch['lead_date'] >= pd.Timestamp("2025-07-01")) & 
                    (df_launch['lead_date'] <= pd.Timestamp("2025-09-30"))
                ]
                
                metrics_2025_q4 = calc_simple_rate(df_2025_oct_dec)
                metrics_2025_q3 = calc_simple_rate(df_2025_jul_sep)
                
                # Calculate seasonal drops
                seasonal_drop_2024 = metrics_2024_q4['fas_rate'] - metrics_2024_q3['fas_rate']
                seasonal_drop_2025 = metrics_2025_q4['fas_rate'] - metrics_2025_q3['fas_rate']
                
                # Display diagnostic table
                st.markdown("#### Seasonal Pattern Comparison (Q3 → Q4)")
                
                diag_df = pd.DataFrame([
                    {
                        'Year': '2024',
                        'Q3 FAS Rate (Jul-Sep)': f"{metrics_2024_q3['fas_rate']:.1%}",
                        'Q4 FAS Rate (Oct-Dec)': f"{metrics_2024_q4['fas_rate']:.1%}",
                        'Seasonal Δ': f"{seasonal_drop_2024:+.1%}",
                        'Q3 Contact Rate': f"{metrics_2024_q3['contact_rate']:.1%}",
                        'Q4 Contact Rate': f"{metrics_2024_q4['contact_rate']:.1%}"
                    },
                    {
                        'Year': '2025',
                        'Q3 FAS Rate (Jul-Sep)': f"{metrics_2025_q3['fas_rate']:.1%}",
                        'Q4 FAS Rate (Oct-Dec)': f"{metrics_2025_q4['fas_rate']:.1%}",
                        'Seasonal Δ': f"{seasonal_drop_2025:+.1%}",
                        'Q3 Contact Rate': f"{metrics_2025_q3['contact_rate']:.1%}",
                        'Q4 Contact Rate': f"{metrics_2025_q4['contact_rate']:.1%}"
                    }
                ])
                st.dataframe(diag_df, use_container_width=True, hide_index=True)
                
                # Generate diagnostic insights
                st.markdown("#### 💡 Diagnostic Insights")
                
                # Calculate extra drop beyond normal seasonality
                extra_drop = seasonal_drop_2025 - seasonal_drop_2024
                
                diag_col1, diag_col2 = st.columns(2)
                
                with diag_col1:
                    st.markdown("**Seasonality Analysis:**")
                    if abs(seasonal_drop_2024) > 0.005:  # If 2024 also had seasonal drop
                        st.info(f"📅 **2024 baseline:** FAS Rate dropped **{seasonal_drop_2024:+.1%}** from Q3→Q4 (normal seasonal pattern)")
                    else:
                        st.info(f"📅 **2024 baseline:** FAS Rate was relatively stable Q3→Q4 ({seasonal_drop_2024:+.1%})")
                    
                    st.info(f"📅 **2025 observed:** FAS Rate dropped **{seasonal_drop_2025:+.1%}** from Q3→Q4")
                
                with diag_col2:
                    st.markdown("**Product Launch Impact:**")
                    if extra_drop < -0.01:  # More than 1% extra drop
                        st.error(f"""
                        ⚠️ **LIKELY PRODUCT IMPACT**
                        
                        2025 Q4 dropped **{abs(extra_drop):.1%}** more than 2024's seasonal pattern.
                        
                        This suggests the Oct 2025 launches (Genesys Queue, Tiering, Routing, Insider Journey) 
                        may have negatively impacted performance beyond normal seasonality.
                        """)
                    elif extra_drop > 0.01:  # Actually better
                        st.success(f"""
                        ✅ **SEASONALITY - NOT PRODUCT LAUNCHES**
                        
                        2025 Q4 actually performed **{extra_drop:+.1%}** better than 2024's seasonal pattern.
                        
                        The decline you're seeing is likely normal seasonality, not product launch issues.
                        """)
                    else:
                        st.warning(f"""
                        ⚖️ **INCONCLUSIVE**
                        
                        2025 vs 2024 seasonal drop difference: **{extra_drop:+.1%}**
                        
                        The difference is within normal variance. Could be a mix of seasonality and minor product impact.
                        Recommend monitoring WoW trends for stabilization.
                        """)
                
                # Recent trend indicator
                st.markdown("#### 📊 Recent Trend Direction")
                
                if len(weekly_agg) >= 3:
                    recent_weeks = weekly_agg.tail(3)
                    recent_wow_changes = recent_weeks['fas_rate_wow'].dropna()
                    
                    if len(recent_wow_changes) >= 2:
                        trend_direction = recent_wow_changes.mean()
                        consecutive_positive = (recent_wow_changes > 0).sum()
                        consecutive_negative = (recent_wow_changes < 0).sum()
                        
                        trend_col1, trend_col2, trend_col3 = st.columns(3)
                        
                        with trend_col1:
                            if trend_direction > 0:
                                st.success(f"📈 **Recovering:** Avg WoW change = **{trend_direction:+.2%}**")
                            else:
                                st.error(f"📉 **Declining:** Avg WoW change = **{trend_direction:+.2%}**")
                        
                        with trend_col2:
                            st.metric("Positive WoW Weeks (last 3)", f"{consecutive_positive}/3")
                        
                        with trend_col3:
                            st.metric("Negative WoW Weeks (last 3)", f"{consecutive_negative}/3")
                        
                        if consecutive_positive >= 2:
                            st.success("✅ **Signal:** Performance appears to be **stabilizing/recovering**. May not need immediate action.")
                        elif consecutive_negative >= 2:
                            st.error("🚨 **Signal:** Performance continues to **decline**. Recommend investigating product launch impacts.")
                
                # Note about data completeness
                days_in_2024 = (period_2024_end - period_2024_start).days + 1
                days_in_2025 = (min(period_2025_end, pd.Timestamp.today()) - period_2025_start).days + 1
                if days_in_2025 < days_in_2024:
                    st.info(f"⚠️ **Note:** Current period has {days_in_2025} days of data vs {days_in_2024} days in prior year. Analysis will improve as more data comes in.")
                
                with st.expander("View Raw Data"):
                    st.dataframe(df_window, use_container_width=True)
            
            else:
                st.warning("No data found for launch analysis.")
        
        except Exception as e:
            st.error(f"Launch Analysis Error: {e}")
            st.exception(e)
    
    # --- TAB 3: Insights ---
    with tab_insights:
        st.header("🔎 Historical & Impact Analysis (Jan 2024+)")
        
        # 1. Fetch Comprehensive Data
        insights_query = """
        SELECT 
            DATE(lead_created_date) as lead_date,
            COUNT(DISTINCT lendage_guid) as lead_qty,
            COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales_qty,
            COUNT(DISTINCT CASE WHEN current_sales_assigned_date IS NOT NULL THEN lendage_guid END) as assigned_qty,
            COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted_qty,
            COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_qty,
            AVG(CASE WHEN sent_to_sales_date IS NOT NULL THEN initial_lead_score_lp2c END) as avg_lp2c
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE lead_created_date >= '2024-01-01'
        GROUP BY 1
        ORDER BY 1
        """
        
        with st.spinner("Analyzing historical patterns..."):
             # We use the same client
             try:
                 df_hist = client.query(insights_query).to_dataframe()
                 
                 if not df_hist.empty:
                     df_hist['lead_date'] = pd.to_datetime(df_hist['lead_date'])
                     
                     # --- A. Pre/Post Analysis (10/15/2025) ---
                     st.subheader("1. Impact Analysis: LVC Change (10/15/2025)")
                     cutoff_date = pd.Timestamp("2025-10-15")
                     
                     df_pre = df_hist[df_hist['lead_date'] < cutoff_date]
                     df_post = df_hist[df_hist['lead_date'] >= cutoff_date]
                     
                     col_i1, col_i2, col_i3, col_i4 = st.columns(4)
                     
                     # Metric Helper
                     def calc_rate(d, n, d_col):
                         return d[n].sum() / d[d_col].sum() if d[d_col].sum() > 0 else 0
                     
                     # Pre Metrics
                     pre_fas_rate = calc_rate(df_pre, 'fas_qty', 'sent_to_sales_qty')
                     pre_lp2c = df_pre['avg_lp2c'].mean()
                     pre_contact = calc_rate(df_pre, 'contacted_qty', 'sent_to_sales_qty')
                     
                     # Post Metrics
                     post_fas_rate = calc_rate(df_post, 'fas_qty', 'sent_to_sales_qty')
                     post_lp2c = df_post['avg_lp2c'].mean()
                     post_contact = calc_rate(df_post, 'contacted_qty', 'sent_to_sales_qty')
                     
                     col_i1.metric("FAS Rate (Post vs Pre)", f"{post_fas_rate:.1%}", f"{(post_fas_rate-pre_fas_rate):.1%}")
                     col_i2.metric("Contact Rate", f"{post_contact:.1%}", f"{(post_contact-pre_contact):.1%}")
                     col_i3.metric("Avg LP2C (Quality)", f"{post_lp2c:.1f}", f"{(post_lp2c-pre_lp2c):.1f}")
                     col_i4.metric("Days Since Change", f"{(datetime.today() - cutoff_date).days} days")
                     
                     st.info(f"Comparing **{len(df_pre)} days** before vs **{len(df_post)} days** after the 10/15/25 cutoff.")
                     
                     # --- B. Seasonality (Monthly) ---
                     st.divider()
                     st.subheader("2. Monthly Seasonality Trends")
                     
                     df_hist['month'] = df_hist['lead_date'].dt.to_period('M').dt.to_timestamp()
                     df_monthly = df_hist.groupby('month').agg({
                         'lead_qty': 'sum',
                         'sent_to_sales_qty': 'sum',
                         'fas_qty': 'sum'
                     }).reset_index()
                     df_monthly['fas_rate'] = df_monthly['fas_qty'] / df_monthly['sent_to_sales_qty']
                     
                     base_m = alt.Chart(df_monthly).encode(x=alt.X('month:T', title='Month'))
                     
                     bar = base_m.mark_bar(color='#aaccff').encode(y=alt.Y('lead_qty', title='Volume'))
                     line = base_m.mark_line(color='red').encode(y=alt.Y('fas_rate', title='FAS Rate', axis=alt.Axis(format='%')))
                     
                     st.altair_chart((bar + line).resolve_scale(y='independent'), use_container_width=True)
                     
                     # --- C. Funnel Analysis ---
                     st.divider()
                     st.subheader("3. Full Funnel Breakdown")
                     
                     funnel_data = pd.DataFrame({
                         'Stage': ['Gross Leads', 'Sent to Sales', 'Assigned', 'Contacted', 'FAS'],
                         'Value': [
                             df_hist['lead_qty'].sum(),
                             df_hist['sent_to_sales_qty'].sum(),
                             df_hist['assigned_qty'].sum(),
                             df_hist['contacted_qty'].sum(),
                             df_hist['fas_qty'].sum()
                         ]
                     })
                     # Calculate conversion from previous step
                     funnel_data['Prev'] = funnel_data['Value'].shift(1).fillna(funnel_data['Value'])
                     funnel_data['Conv %'] = funnel_data['Value'] / funnel_data['Prev']
                     funnel_data.loc[0, 'Conv %'] = 1.0 # 100% for top
                     
                     fig_f = go.Figure(go.Funnel(
                         y = funnel_data['Stage'],
                         x = funnel_data['Value'],
                         textinfo = "value+percent previous"
                     ))
                     st.plotly_chart(fig_f, use_container_width=True)
                 
                 else:
                     st.warning("No historical data found for Insights.")
                     
             except Exception as e:
                 st.error(f"Error fetching insights: {e}")

else:
    st.info("👈 Please authenticate using the sidebar to start running queries.")
