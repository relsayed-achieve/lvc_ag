import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
from scipy import stats
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
    
    # LVC Raw Filter (for query WHERE clause)
    # We can hardcode common values or query them. Hardcoding for performance.
    lvc_options = [str(i) for i in range(1, 11)] + ['X', 'Other']
    selected_lvc = st.multiselect("Adjusted Lead Value Cohort", options=lvc_options, placeholder="All Values")
    
    # LVC Group Filter (for Main Dashboard charts)
    lvc_group_options = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'phx_transfer', 'Other']
    selected_lvc_groups = st.multiselect(
        "LVC Group (Dashboard)",
        options=lvc_group_options,
        default=[],
        placeholder="All LVC Groups",
        key="sidebar_lvc_filter"
    )
    
    st.caption("💡 Filters apply to Main Dashboard")

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
    
    tab_main, tab_persona, tab_launch, tab_frank_felix, tab_digital, tab_call_metrics, tab_outreach, tab_dead_zone, tab_vol_conv, tab_fas_analysis, tab_28_analysis, tab_agent, tab_insights = st.tabs(["📊 Main Dashboard", "👤 LVC & Persona Analysis", "🚀 Pre/Post Launch(s)", "🎯 Frank & Felix Growth", "📱 Digital Intent", "📞 MA Call Metrics", "📞 Outreach (Pre-App)", "💀 Dead Zone", "📈 Volume vs Conversion", "✅ FAS Analysis", "🎯 2/8 Analysis", "⭐ Agent Performance", "🤖 Comprehensive Insights (Jan 24+)"])
    
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
        SUM(CASE WHEN sent_to_sales_date IS NOT NULL THEN e_loan_amount ELSE 0 END) as sts_volume,
        AVG(CASE WHEN sent_to_sales_date IS NOT NULL THEN initial_lead_score_lp2c END) as avg_lp2c,
        COUNT(DISTINCT CASE WHEN current_sales_assigned_date IS NOT NULL THEN lendage_guid END) as assigned_qty,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted_qty,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_qty,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_volume
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
                
                # Apply LVC Group filter from sidebar
                if selected_lvc_groups:
                    df = df[df['lvc_group'].isin(selected_lvc_groups)]
                    st.info(f"🔍 Filtered to LVC Groups: {', '.join(selected_lvc_groups)}")
                
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
                        'sts_volume': 'sum',
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
                    
                    # FAS $ per Sent to Sales Lead
                    df_grouped['fas_per_sts'] = df_grouped.apply(lambda x: x['fas_volume'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1)
                    
                    # Avg FAS Loan Amount
                    df_grouped['avg_fas_loan'] = df_grouped.apply(lambda x: x['fas_volume'] / x['fas_qty'] if x['fas_qty'] > 0 else 0, axis=1)
                    
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
                st.caption("**Field:** `lendage_guid` | **Calc:** `COUNT(DISTINCT lendage_guid)`")
                
                plot_time_series(df, 'sent_to_sales_qty', "Sent to Sales Quantity", enable_pct_toggle=True, key="sales")
                st.caption("**Field:** `sent_to_sales_date` | **Calc:** `COUNT(DISTINCT lendage_guid WHERE sent_to_sales_date IS NOT NULL)`")
                
                plot_time_series(df, 'assigned_qty', "Assigned Quantity", enable_pct_toggle=True, key="assigned")
                st.caption("**Field:** `current_sales_assigned_date` | **Calc:** `COUNT(DISTINCT lendage_guid WHERE current_sales_assigned_date IS NOT NULL)`")
                
                plot_time_series(df, 'avg_lp2c', "Average LP2C", key="lp2c")
                st.caption("**Field:** `initial_lead_score_lp2c` | **Calc:** `AVG(initial_lead_score_lp2c WHERE sent_to_sales_date IS NOT NULL)`")
                
                st.divider()
                st.subheader("Rates")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    plot_time_series(df, 'sales_contact_rate', "Contact Rate", is_rate=True, key="contact")
                    st.caption("**Fields:** `sf__contacted_guid`, `sent_to_sales_date` | **Calc:** `COUNT(contacted) / COUNT(Sent to Sales)`")
                    display_rate_insights(df, 'contacted_qty', 'sent_to_sales_qty', "Contact Rate")
                    
                with col_r2:
                    plot_time_series(df, 'fas_rate', "FAS Rate", is_rate=True, key="fas")
                    st.caption("**Fields:** `full_app_submit_datetime`, `sent_to_sales_date` | **Calc:** `COUNT(FAS) / COUNT(Sent to Sales)`")
                    display_rate_insights(df, 'fas_qty', 'sent_to_sales_qty', "FAS Rate")

                plot_time_series(df, 'fas_volume', "FAS Volume ($)", key="vol")
                st.caption("**Field:** `e_loan_amount` | **Calc:** `SUM(e_loan_amount WHERE full_app_submit_datetime IS NOT NULL)`")
                
                st.divider()
                st.subheader("💰 Revenue & Value Metrics")
                
                # FAS $ per Sent to Sales Lead (by LVC Group)
                plot_time_series(df, 'fas_per_sts', "FAS $ per Sent to Sales Lead (by LVC Group)", key="fas_per_sts")
                st.caption("**Fields:** `e_loan_amount`, `sent_to_sales_date` | **Calc:** `SUM(FAS e_loan_amount) / COUNT(Sent to Sales Leads)`")
                
                # FAS $ per Sent to Sales Lead - AGGREGATE (no LVC breakdown)
                st.subheader("FAS $ per Sent to Sales Lead (Aggregate)")
                
                grain_agg = st.selectbox("Grain", ["Day", "Week", "Month"], index=1, key="grain_fas_per_sts_agg", label_visibility="collapsed")
                
                # Aggregate data without LVC breakdown
                df_agg = df.copy()
                if grain_agg == 'Week':
                    df_agg['date_col'] = df_agg['lead_date'] - pd.to_timedelta(df_agg['lead_date'].dt.weekday, unit='D')
                elif grain_agg == 'Month':
                    df_agg['date_col'] = df_agg['lead_date'].dt.to_period('M').dt.to_timestamp()
                else:
                    df_agg['date_col'] = df_agg['lead_date']
                
                # Group by date only (no LVC)
                df_agg_grouped = df_agg.groupby('date_col').agg({
                    'sent_to_sales_qty': 'sum',
                    'fas_qty': 'sum',
                    'fas_volume': 'sum'
                }).reset_index()
                
                df_agg_grouped['fas_per_sts'] = df_agg_grouped['fas_volume'] / df_agg_grouped['sent_to_sales_qty'].replace(0, 1)
                
                fig_agg = alt.Chart(df_agg_grouped).mark_line(point=True, color='#2ca02c').encode(
                    x=alt.X('date_col:T', title=grain_agg, axis=alt.Axis(format='%b %d')),
                    y=alt.Y('fas_per_sts:Q', title='FAS $ per StS', axis=alt.Axis(format='$,.0f')),
                    tooltip=[
                        alt.Tooltip('date_col:T', title='Date', format='%b %d, %Y'),
                        alt.Tooltip('fas_per_sts:Q', title='FAS $/StS', format='$,.0f'),
                        alt.Tooltip('sent_to_sales_qty:Q', title='StS Volume', format=','),
                        alt.Tooltip('fas_qty:Q', title='FAS Count', format=','),
                        alt.Tooltip('fas_volume:Q', title='FAS $', format='$,.0f')
                    ]
                ).properties(height=300).interactive()
                
                st.altair_chart(fig_agg, use_container_width=True)
                st.caption("**Fields:** `e_loan_amount`, `sent_to_sales_date` | **Calc:** `SUM(FAS e_loan_amount) / COUNT(Sent to Sales Leads)` — **Aggregate across all LVC groups**")
                
                # Lead Value Driven by Loan Size - custom chart
                st.subheader("Lead Value Driven by Loan Size")
                
                grain_lv = st.selectbox("Grain", ["Day", "Week", "Month"], index=1, key="grain_lead_value", label_visibility="collapsed")
                
                # Aggregate data
                df_lv = df.copy()
                if grain_lv == 'Week':
                    df_lv['date_col'] = df_lv['lead_date'] - pd.to_timedelta(df_lv['lead_date'].dt.weekday, unit='D')
                elif grain_lv == 'Month':
                    df_lv['date_col'] = df_lv['lead_date'].dt.to_period('M').dt.to_timestamp()
                else:
                    df_lv['date_col'] = df_lv['lead_date']
                
                # Group by date and LVC
                df_lv_grouped = df_lv.groupby(['date_col', 'lvc_group']).agg({
                    'fas_qty': 'sum',
                    'fas_volume': 'sum'
                }).reset_index()
                
                # Calculate avg FAS loan per LVC group
                df_lv_grouped['avg_fas_loan'] = df_lv_grouped['fas_volume'] / df_lv_grouped['fas_qty'].replace(0, 1)
                
                # Calculate overall avg for each date
                date_totals = df_lv_grouped.groupby('date_col').agg({
                    'fas_qty': 'sum',
                    'fas_volume': 'sum'
                }).reset_index()
                date_totals['overall_avg'] = date_totals['fas_volume'] / date_totals['fas_qty'].replace(0, 1)
                date_totals = date_totals[['date_col', 'overall_avg']]
                
                # Merge to get ratio
                df_lv_grouped = df_lv_grouped.merge(date_totals, on='date_col')
                df_lv_grouped['lead_value_ratio'] = df_lv_grouped['avg_fas_loan'] / df_lv_grouped['overall_avg'].replace(0, 1)
                
                # Chart
                fig_lv = alt.Chart(df_lv_grouped).mark_line(point=True).encode(
                    x=alt.X('date_col:T', title='Date', axis=alt.Axis(format='%b %d', labelColor='white', titleColor='white')),
                    y=alt.Y('lead_value_ratio:Q', title='Lead Value Ratio', axis=alt.Axis(labelColor='white', titleColor='white')),
                    color=alt.Color('lvc_group:N', title='LVC Group'),
                    tooltip=[
                        alt.Tooltip('date_col:T', title='Date', format='%b %d, %Y'),
                        alt.Tooltip('lvc_group:N', title='LVC Group'),
                        alt.Tooltip('lead_value_ratio:Q', title='Lead Value Ratio', format='.2f'),
                        alt.Tooltip('avg_fas_loan:Q', title='Avg FAS Loan', format='$,.0f'),
                        alt.Tooltip('overall_avg:Q', title='Overall Avg', format='$,.0f'),
                        alt.Tooltip('fas_qty:Q', title='FAS Qty', format=',d')
                    ]
                ).properties(
                    height=300
                ).interactive()
                
                # Add reference line at 1.0
                ref_line = alt.Chart(pd.DataFrame({'y': [1.0]})).mark_rule(color='#ff6b6b', strokeDash=[5,5]).encode(
                    y='y:Q'
                )
                
                st.altair_chart(fig_lv + ref_line, use_container_width=True)
                st.caption("**Fields:** `e_loan_amount`, `fas_qty` | **Calc:** `(Avg FAS Loan per LVC) / (Overall Avg FAS Loan)` | **>1 = Higher value leads, <1 = Lower value leads**")

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

    # --- TAB: LVC & Persona Analysis ---
    with tab_persona:
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
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig_persona, use_container_width=True)
                    
                    with persona_col2:
                        st.subheader("🎯 FAS Performance by LVC & Persona")
                        
                        # Heatmap using Altair - based on FAS QTY with red-green gradient
                        df_heatmap = df_persona[['lvc_group', 'persona', 'fas_rate', 'fas_count', 'lead_count', 'sent_to_sales_qty']].copy()
                        
                        # Create heatmap based on FAS count with more color steps
                        # Use quantile scale type for better color distribution
                        fas_min = df_heatmap['fas_count'].min()
                        fas_max = df_heatmap['fas_count'].max()
                        fas_mid = df_heatmap['fas_count'].median()
                        
                        heatmap_base = alt.Chart(df_heatmap).mark_rect().encode(
                            x=alt.X('lvc_group:N', title='LVC Group', axis=alt.Axis(labelColor='white', labelFontWeight='bold', titleColor='white', titleFontWeight='bold')),
                            y=alt.Y('persona:N', title='Persona', axis=alt.Axis(labelColor='white', labelFontWeight='bold', titleColor='white', titleFontWeight='bold')),
                            color=alt.Color('fas_count:Q', title='FAS QTY', 
                                scale=alt.Scale(
                                    domain=[fas_min, fas_min + (fas_mid-fas_min)*0.25, fas_min + (fas_mid-fas_min)*0.5, fas_mid, fas_mid + (fas_max-fas_mid)*0.5, fas_max],
                                    range=['#d73027', '#f46d43', '#fdae61', '#fee08b', '#a6d96a', '#1a9850']
                                )
                            ),
                            tooltip=[
                                alt.Tooltip('lvc_group:N', title='LVC'),
                                alt.Tooltip('persona:N', title='Persona'),
                                alt.Tooltip('fas_count:Q', title='FAS QTY', format=',d'),
                                alt.Tooltip('fas_rate:Q', title='FAS Rate', format='.1%'),
                                alt.Tooltip('lead_count:Q', title='Lead Count', format=',d')
                            ]
                        )
                        
                        # Add text labels showing FAS QTY and FAS Rate
                        heatmap_text = alt.Chart(df_heatmap).mark_text(fontSize=10, color='black', fontWeight='bold').encode(
                            x=alt.X('lvc_group:N'),
                            y=alt.Y('persona:N'),
                            text=alt.Text('label:N')
                        ).transform_calculate(
                            label="datum.fas_count + ' (' + format(datum.fas_rate, '.1%') + ')'"
                        )
                        
                        heatmap = (heatmap_base + heatmap_text).properties(
                            height=400
                        ).configure_view(
                            strokeWidth=0
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

        # --- 4. Finance Group to Persona Analysis ---
        st.divider()
        st.header("💰 Finance Group to Persona Analysis (1/1/2025+)")
        st.caption("Understanding which personas flow into which Finance Groups and their FAS performance (Data from January 1, 2025 onwards)")
        
        finance_persona_query = f"""
        SELECT 
            COALESCE(finance_group, 'Unknown') as finance_group,
            COALESCE(persona, 'Unknown') as persona,
            COUNT(DISTINCT lendage_guid) as lead_count,
            COUNT(DISTINCT lendage_guid) as sent_to_sales_qty,
            COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE lead_created_date >= '2025-01-01'
        AND finance_group IS NOT NULL
        AND sent_to_sales_date IS NOT NULL
        GROUP BY 1, 2
        """
        
        with st.expander("View Finance Group-Persona SQL"):
            st.code(finance_persona_query)
        
        try:
            with st.spinner("Fetching Finance Group-Persona Data..."):
                df_finance_persona = client.query(finance_persona_query).to_dataframe()
                
                if not df_finance_persona.empty:
                    # Calculate FAS Rate
                    df_finance_persona['fas_rate'] = df_finance_persona.apply(
                        lambda x: x['fas_count'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                    )
                    
                    st.subheader("📊 Lead Distribution: Finance Group → Persona")
                    
                    # Calculate totals for percentages
                    total_leads_fg = df_finance_persona['lead_count'].sum()
                    fg_totals = df_finance_persona.groupby('finance_group')['lead_count'].sum().to_dict()
                    persona_totals_fg = df_finance_persona.groupby('persona')['lead_count'].sum().to_dict()
                    
                    # Create node labels with percentages
                    fg_nodes = sorted(df_finance_persona['finance_group'].unique().tolist())
                    persona_nodes_fg = sorted(df_finance_persona['persona'].unique().tolist())
                    
                    fg_labels = [f"{node}<br>({fg_totals.get(node, 0):,} | {fg_totals.get(node, 0)/total_leads_fg*100:.1f}%)" for node in fg_nodes]
                    persona_labels_fg = [f"{node}<br>({persona_totals_fg.get(node, 0):,} | {persona_totals_fg.get(node, 0)/total_leads_fg*100:.1f}%)" for node in persona_nodes_fg]
                    all_node_labels_fg = fg_labels + persona_labels_fg
                    all_nodes_fg = fg_nodes + persona_nodes_fg
                    
                    # Node colors
                    fg_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                    persona_colors_fg = ['#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
                    node_colors_fg = fg_colors[:len(fg_nodes)] + persona_colors_fg[:len(persona_nodes_fg)]
                    
                    sources_fg = []
                    targets_fg = []
                    values_fg = []
                    link_colors_fg = []
                    link_labels_fg = []
                    
                    for _, row in df_finance_persona.iterrows():
                        src_idx = all_nodes_fg.index(row['finance_group'])
                        tgt_idx = all_nodes_fg.index(row['persona'])
                        sources_fg.append(src_idx)
                        targets_fg.append(tgt_idx)
                        values_fg.append(row['lead_count'])
                        link_colors_fg.append(f"rgba(255, 127, 14, 0.4)")
                        # Calculate % of total and % within Finance Group
                        pct_total = row['lead_count'] / total_leads_fg * 100
                        pct_fg = row['lead_count'] / fg_totals.get(row['finance_group'], 1) * 100
                        link_labels_fg.append(f"{row['lead_count']:,} leads ({pct_total:.1f}% of total, {pct_fg:.1f}% of {row['finance_group']})")
                    
                    fig_finance_persona = go.Figure(data=[go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=all_node_labels_fg,
                            color=node_colors_fg
                        ),
                        link=dict(
                            source=sources_fg,
                            target=targets_fg,
                            value=values_fg,
                            color=link_colors_fg,
                            label=link_labels_fg,
                            hovertemplate='%{label}<extra></extra>'
                        ),
                        textfont=dict(color='black', size=12)
                    )])
                    
                    fig_finance_persona.update_layout(
                        font=dict(size=12, color='black'),
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_finance_persona, use_container_width=True)
                    
                    # Summary table
                    st.subheader("📋 Finance Group Performance Summary")
                    
                    fg_summary = df_finance_persona.groupby('finance_group').agg({
                        'lead_count': 'sum',
                        'sent_to_sales_qty': 'sum',
                        'fas_count': 'sum'
                    }).reset_index()
                    fg_summary['fas_rate'] = fg_summary.apply(
                        lambda x: x['fas_count'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                    )
                    fg_summary['lead_pct'] = fg_summary['lead_count'] / fg_summary['lead_count'].sum()
                    fg_summary = fg_summary.sort_values('fas_rate', ascending=False)
                    
                    # Format for display
                    display_fg = fg_summary.copy()
                    display_fg['FAS Rate'] = display_fg['fas_rate'].apply(lambda x: f"{x:.1%}")
                    display_fg['Lead %'] = display_fg['lead_pct'].apply(lambda x: f"{x:.1%}")
                    display_fg['Leads'] = display_fg['lead_count'].apply(lambda x: f"{x:,}")
                    display_fg['FAS Count'] = display_fg['fas_count'].apply(lambda x: f"{x:,}")
                    display_fg = display_fg[['finance_group', 'Leads', 'Lead %', 'FAS Count', 'FAS Rate']]
                    display_fg.columns = ['Finance Group', 'Leads', 'Lead %', 'FAS Count', 'FAS Rate']
                    
                    st.dataframe(display_fg, use_container_width=True, hide_index=True)
                    
                    # Insights
                    best_fg = fg_summary.iloc[0]
                    worst_fg = fg_summary.iloc[-1]
                    
                    st.markdown("**🔍 Key Insights:**")
                    st.markdown(f"- **Best Converting Finance Group**: `{best_fg['finance_group']}` with **{best_fg['fas_rate']:.1%}** FAS rate")
                    st.markdown(f"- **Lowest Converting Finance Group**: `{worst_fg['finance_group']}` with **{worst_fg['fas_rate']:.1%}** FAS rate")
                    
                    # --- What's Driving Each Finance Group ---
                    st.divider()
                    st.subheader("🔎 What's Driving Each Finance Group?")
                    st.caption("Top personas contributing to each finance group by volume and conversion")
                    
                    # Create detailed breakdown by finance group
                    df_drivers = df_finance_persona.copy()
                    
                    # Calculate % within each finance group
                    fg_totals_for_pct = df_drivers.groupby('finance_group')['lead_count'].transform('sum')
                    df_drivers['pct_of_fg'] = df_drivers['lead_count'] / fg_totals_for_pct
                    
                    # Get unique finance groups
                    finance_groups = sorted(df_drivers['finance_group'].unique().tolist())
                    
                    # Create tabs for each finance group
                    fg_tabs = st.tabs(finance_groups)
                    
                    for i, fg in enumerate(finance_groups):
                        with fg_tabs[i]:
                            fg_data = df_drivers[df_drivers['finance_group'] == fg].copy()
                            fg_data = fg_data.sort_values('lead_count', ascending=False)
                            
                            # Summary metrics
                            total_leads_fg = fg_data['lead_count'].sum()
                            total_fas_fg = fg_data['fas_count'].sum()
                            overall_fas_rate = total_fas_fg / total_leads_fg if total_leads_fg > 0 else 0
                            
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("Total Leads (StS)", f"{total_leads_fg:,}")
                            with col_m2:
                                st.metric("Total FAS", f"{total_fas_fg:,}")
                            with col_m3:
                                st.metric("Overall FAS Rate", f"{overall_fas_rate:.1%}")
                            
                            # Persona breakdown table
                            st.markdown("**Persona Breakdown:**")
                            display_drivers = fg_data[['persona', 'lead_count', 'pct_of_fg', 'fas_count', 'fas_rate']].copy()
                            display_drivers['Lead Count'] = display_drivers['lead_count'].apply(lambda x: f"{x:,}")
                            display_drivers['% of Finance Group'] = display_drivers['pct_of_fg'].apply(lambda x: f"{x:.1%}")
                            display_drivers['FAS Count'] = display_drivers['fas_count'].apply(lambda x: f"{x:,}")
                            display_drivers['FAS Rate'] = display_drivers['fas_rate'].apply(lambda x: f"{x:.1%}")
                            display_drivers = display_drivers[['persona', 'Lead Count', '% of Finance Group', 'FAS Count', 'FAS Rate']]
                            display_drivers.columns = ['Persona', 'Leads', '% of Group', 'FAS', 'FAS Rate']
                            
                            st.dataframe(display_drivers, use_container_width=True, hide_index=True)
                            
                            # Top driver insight
                            top_persona = fg_data.iloc[0]
                            best_converting = fg_data.loc[fg_data['fas_rate'].idxmax()]
                            
                            st.markdown(f"""
                            **💡 Insights for {fg}:**
                            - **Highest Volume**: `{top_persona['persona']}` drives **{top_persona['pct_of_fg']:.1%}** of leads ({top_persona['lead_count']:,} leads)
                            - **Best Converter**: `{best_converting['persona']}` has the highest FAS rate at **{best_converting['fas_rate']:.1%}**
                            """)
                    
                    # Deep Dive: All Personas - Monthly Breakdown
                    st.divider()
                    st.subheader("🎯 Deep Dive: Finance Group Stats by Persona Over Time (1/1/2025+)")
                    st.caption("Monthly trends for each persona's finance group performance")
                    
                    # Query for monthly persona breakdown
                    monthly_persona_query = """
                    SELECT 
                        COALESCE(persona, 'Unknown') as persona,
                        COALESCE(finance_group, 'Unknown') as finance_group,
                        FORMAT_DATE('%Y-%m', lead_created_date) as month,
                        COUNT(DISTINCT lendage_guid) as lead_count,
                        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count
                    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                    WHERE lead_created_date >= '2025-01-01'
                    AND finance_group IS NOT NULL
                    AND sent_to_sales_date IS NOT NULL
                    GROUP BY 1, 2, 3
                    ORDER BY 1, 2, 3
                    """
                    
                    df_monthly = client.query(monthly_persona_query).to_dataframe()
                    
                    if not df_monthly.empty:
                        df_monthly['fas_rate'] = df_monthly.apply(
                            lambda x: x['fas_count'] / x['lead_count'] if x['lead_count'] > 0 else 0, axis=1
                        )
                        
                        # Get unique months and personas
                        months = sorted(df_monthly['month'].unique())
                        personas = df_monthly.groupby('persona')['lead_count'].sum().sort_values(ascending=False).index.tolist()
                        finance_groups_list = sorted(df_monthly['finance_group'].unique())
                        
                        # Metric selector
                        deep_dive_metric = st.selectbox(
                            "Select Metric to Display",
                            ['FAS Rate', 'Lead Count', 'FAS Count'],
                            key='deep_dive_metric'
                        )
                        
                        # Create tabs for each persona
                        persona_tabs = st.tabs(personas)
                        
                        for p_idx, persona_name in enumerate(personas):
                            with persona_tabs[p_idx]:
                                persona_monthly = df_monthly[df_monthly['persona'] == persona_name]
                                
                                # Total stats
                                total_leads = persona_monthly['lead_count'].sum()
                                total_fas = persona_monthly['fas_count'].sum()
                                overall_rate = total_fas / total_leads if total_leads > 0 else 0
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Leads (StS)", f"{total_leads:,}")
                                with col2:
                                    st.metric("Total FAS", f"{total_fas:,}")
                                with col3:
                                    st.metric("Overall FAS Rate", f"{overall_rate:.1%}")
                                
                                # Create pivot table: Finance Groups as rows, Months as columns
                                if deep_dive_metric == 'FAS Rate':
                                    pivot_data = persona_monthly.pivot_table(
                                        index='finance_group',
                                        columns='month',
                                        values='fas_rate',
                                        aggfunc='mean'
                                    ).fillna(0)
                                    format_fn = lambda x: f"{x:.1%}" if x > 0 else "-"
                                elif deep_dive_metric == 'Lead Count':
                                    pivot_data = persona_monthly.pivot_table(
                                        index='finance_group',
                                        columns='month',
                                        values='lead_count',
                                        aggfunc='sum'
                                    ).fillna(0)
                                    format_fn = lambda x: f"{int(x):,}" if x > 0 else "-"
                                else:  # FAS Count
                                    pivot_data = persona_monthly.pivot_table(
                                        index='finance_group',
                                        columns='month',
                                        values='fas_count',
                                        aggfunc='sum'
                                    ).fillna(0)
                                    format_fn = lambda x: f"{int(x):,}" if x > 0 else "-"
                                
                                # Add total column
                                if deep_dive_metric == 'FAS Rate':
                                    # For rate, calculate weighted average
                                    fg_totals = persona_monthly.groupby('finance_group').agg({
                                        'lead_count': 'sum',
                                        'fas_count': 'sum'
                                    })
                                    fg_totals['Total'] = fg_totals.apply(
                                        lambda x: x['fas_count'] / x['lead_count'] if x['lead_count'] > 0 else 0, axis=1
                                    )
                                    pivot_data['Total'] = fg_totals['Total']
                                else:
                                    pivot_data['Total'] = pivot_data.sum(axis=1)
                                
                                # Format the pivot table for display
                                display_pivot = pivot_data.applymap(format_fn)
                                display_pivot.index.name = 'Finance Group'
                                
                                st.dataframe(display_pivot, use_container_width=True)
                                
                                # Trend chart for this persona
                                st.markdown(f"**📈 {deep_dive_metric} Trend by Finance Group**")
                                
                                chart_data = persona_monthly.copy()
                                chart_data['month_date'] = pd.to_datetime(chart_data['month'] + '-01')
                                
                                if deep_dive_metric == 'FAS Rate':
                                    y_field = 'fas_rate:Q'
                                    y_title = 'FAS Rate'
                                    y_format = '.0%'
                                elif deep_dive_metric == 'Lead Count':
                                    y_field = 'lead_count:Q'
                                    y_title = 'Lead Count'
                                    y_format = ','
                                else:
                                    y_field = 'fas_count:Q'
                                    y_title = 'FAS Count'
                                    y_format = ','
                                
                                trend_chart = alt.Chart(chart_data).mark_line(point=True).encode(
                                    x=alt.X('month_date:T', title='Month', axis=alt.Axis(
                                        format='%b %Y', labelColor='white', labelFontWeight='bold',
                                        titleColor='white', titleFontWeight='bold'
                                    )),
                                    y=alt.Y(y_field, title=y_title, axis=alt.Axis(
                                        format=y_format, labelColor='white', labelFontWeight='bold',
                                        titleColor='white', titleFontWeight='bold'
                                    )),
                                    color=alt.Color('finance_group:N', title='Finance Group'),
                                    tooltip=[
                                        alt.Tooltip('month:N', title='Month'),
                                        alt.Tooltip('finance_group:N', title='Finance Group'),
                                        alt.Tooltip('lead_count:Q', title='Leads', format=',d'),
                                        alt.Tooltip('fas_count:Q', title='FAS', format=',d'),
                                        alt.Tooltip('fas_rate:Q', title='FAS Rate', format='.1%')
                                    ]
                                ).properties(
                                    height=350
                                ).interactive()
                                
                                st.altair_chart(trend_chart, use_container_width=True)
                                
                                # Best/worst finance groups for this persona
                                fg_perf = persona_monthly.groupby('finance_group').agg({
                                    'lead_count': 'sum',
                                    'fas_count': 'sum'
                                }).reset_index()
                                fg_perf['fas_rate'] = fg_perf.apply(
                                    lambda x: x['fas_count'] / x['lead_count'] if x['lead_count'] > 0 else 0, axis=1
                                )
                                fg_perf = fg_perf.sort_values('fas_rate', ascending=False)
                                
                                if len(fg_perf) > 0:
                                    best = fg_perf.iloc[0]
                                    worst = fg_perf.iloc[-1]
                                    st.markdown(f"""
                                    🏆 **Best Converting:** `{best['finance_group']}` ({best['fas_rate']:.1%} FAS rate, {best['lead_count']:,} leads)
                                    
                                    ⚠️ **Lowest Converting:** `{worst['finance_group']}` ({worst['fas_rate']:.1%} FAS rate, {worst['lead_count']:,} leads)
                                    """)
                    else:
                        st.info("No monthly persona data found.")
                    
                else:
                    st.warning("No finance group data found for the selected filters.")
                    
        except Exception as e:
            st.error(f"Finance Group-Persona Analysis Error: {e}")
            st.caption("Note: This analysis requires a 'finance_group' field in the data source.")

    # Back to Main Dashboard tab for HTML Export
    with tab_main:
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
                    
                    # X-axis with year shown (multi-line format: Month Day \n Year) - white bold labels
                    x_axis = alt.X(
                        'date_col:T', 
                        title=time_grain,
                        axis=alt.Axis(
                            format='%b %d',
                            labelExpr="datum.label + '\\n' + timeFormat(datum.value, '%Y')",
                            labelAlign='center',
                            labelPadding=5,
                            labelColor='white',
                            labelFontWeight='bold',
                            titleColor='white',
                            titleFontWeight='bold'
                        )
                    )
                    
                    # Y-axis with white bold labels
                    y_axis_config = {
                        'labelColor': 'white',
                        'labelFontWeight': 'bold',
                        'titleColor': 'white',
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
                            strokeWidth=0
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
                        strokeWidth=0
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
    
    # --- TAB: Frank & Felix Growth Analysis ---
    with tab_frank_felix:
        st.header("🎯 Frank & Felix Growth Analysis")
        st.caption("Tracking the growth of high-value personas (Frank & Felix) from **Oct 1, 2025 - Jan 30, 2026** and their superior conversion performance")
        
        # Query for persona data with loan amounts over time (fixed date range)
        ff_query = """
        WITH persona_data AS (
            SELECT 
                DATE(lead_created_date) as lead_date,
                COALESCE(persona, 'Unknown') as persona,
                COUNT(DISTINCT lendage_guid) as lead_count,
                COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales_qty,
                COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
                -- Funded count (FAS'd leads that also have funding_end_datetime)
                COUNT(DISTINCT CASE 
                    WHEN full_app_submit_datetime IS NOT NULL 
                    AND funding_end_datetime IS NOT NULL 
                    THEN lendage_guid 
                END) as funded_count,
                SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_loan_amount,
                SUM(CASE WHEN sent_to_sales_date IS NOT NULL THEN e_loan_amount ELSE 0 END) as sts_loan_amount,
                AVG(CASE WHEN sent_to_sales_date IS NOT NULL THEN e_loan_amount END) as avg_loan_amount,
                -- Contribution Margin
                SUM(COALESCE(ue_c_contribution_margin_npv_actual, 0)) as contribution_margin,
                -- FAS loan amounts
                AVG(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount END) as avg_fas_loan,
                -- Funded loan amounts
                SUM(CASE WHEN funding_end_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as funded_loan_amount,
                AVG(CASE WHEN funding_end_datetime IS NOT NULL THEN e_loan_amount END) as avg_funded_loan
            FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
            WHERE lead_created_date >= '2025-10-01'
            AND lead_created_date <= '2026-01-30'
            AND persona IS NOT NULL
            GROUP BY 1, 2
        )
        SELECT * FROM persona_data
        ORDER BY lead_date, persona
        """
        
        try:
            @st.cache_data(ttl=600)
            def run_ff_query(sql_query):
                return client.query(sql_query).to_dataframe()
            
            with st.spinner("Loading Frank & Felix analysis..."):
                df_ff = run_ff_query(ff_query)
            
            if not df_ff.empty:
                df_ff['lead_date'] = pd.to_datetime(df_ff['lead_date'])
                
                # Create persona grouping: Frank, Felix, vs Others
                df_ff['persona_group'] = df_ff['persona'].apply(
                    lambda x: x if x in ['Frank', 'Felix'] else 'Other Personas'
                )
                
                # --- Time Grain Selection ---
                ff_col1, ff_col2 = st.columns([3, 1])
                with ff_col2:
                    ff_grain = st.selectbox("Time Grain", ["Day", "Week", "Month"], index=1, key="ff_grain")
                
                # Aggregate by time grain
                df_ff_agg = df_ff.copy()
                if ff_grain == 'Week':
                    df_ff_agg['date_col'] = df_ff_agg['lead_date'] - pd.to_timedelta(df_ff_agg['lead_date'].dt.weekday, unit='D')
                elif ff_grain == 'Month':
                    df_ff_agg['date_col'] = df_ff_agg['lead_date'].dt.to_period('M').dt.to_timestamp()
                else:
                    df_ff_agg['date_col'] = df_ff_agg['lead_date']
                
                # Aggregate by date and persona group
                df_grouped = df_ff_agg.groupby(['date_col', 'persona_group']).agg({
                    'lead_count': 'sum',
                    'sent_to_sales_qty': 'sum',
                    'fas_count': 'sum',
                    'funded_count': 'sum',
                    'fas_loan_amount': 'sum',
                    'sts_loan_amount': 'sum',
                    'contribution_margin': 'sum'
                }).reset_index()
                
                # Calculate rates
                df_grouped['fas_rate'] = df_grouped.apply(
                    lambda x: x['fas_count'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                
                # Calculate FAS $ per Lead Sent to Sales (Revenue Yield)
                df_grouped['fas_per_sts'] = df_grouped.apply(
                    lambda x: x['fas_loan_amount'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                
                # Calculate Pull Through Rate (Funded / FAS)
                df_grouped['pull_through_rate'] = df_grouped.apply(
                    lambda x: x['funded_count'] / x['fas_count'] if x['fas_count'] > 0 else 0, axis=1
                )
                
                # Calculate Contribution Margin per StS
                df_grouped['cm_per_sts'] = df_grouped.apply(
                    lambda x: x['contribution_margin'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                
                # Calculate Value Concentration Ratio (% of FAS $) / (% of StS) per period
                # First calculate totals per date
                total_fas_per_date = df_grouped.groupby('date_col')['fas_loan_amount'].transform('sum')
                total_sts_per_date = df_grouped.groupby('date_col')['sent_to_sales_qty'].transform('sum')
                df_grouped['fas_dollar_share'] = df_grouped['fas_loan_amount'] / total_fas_per_date
                df_grouped['sts_share'] = df_grouped['sent_to_sales_qty'] / total_sts_per_date
                df_grouped['value_concentration'] = df_grouped.apply(
                    lambda x: x['fas_dollar_share'] / x['sts_share'] if x['sts_share'] > 0 else 0, axis=1
                )
                
                # Calculate % of total Sent to Sales per period (based on StS qty, not leads)
                total_per_date = df_grouped.groupby('date_col')['sent_to_sales_qty'].transform('sum')
                df_grouped['pct_of_total'] = df_grouped['sent_to_sales_qty'] / total_per_date
                
                # Also aggregate by individual persona for detailed view
                df_persona_detail = df_ff_agg.groupby(['date_col', 'persona']).agg({
                    'lead_count': 'sum',
                    'sent_to_sales_qty': 'sum',
                    'fas_count': 'sum',
                    'funded_count': 'sum',
                    'fas_loan_amount': 'sum',
                    'sts_loan_amount': 'sum',
                    'contribution_margin': 'sum'
                }).reset_index()
                df_persona_detail['fas_rate'] = df_persona_detail.apply(
                    lambda x: x['fas_count'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                df_persona_detail['pull_through_rate'] = df_persona_detail.apply(
                    lambda x: x['funded_count'] / x['fas_count'] if x['fas_count'] > 0 else 0, axis=1
                )
                df_persona_detail['cm_per_sts'] = df_persona_detail.apply(
                    lambda x: x['contribution_margin'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                # % of total based on Sent to Sales qty
                total_per_date_detail = df_persona_detail.groupby('date_col')['sent_to_sales_qty'].transform('sum')
                df_persona_detail['pct_of_total'] = df_persona_detail['sent_to_sales_qty'] / total_per_date_detail
                
                # --- Summary KPIs ---
                st.subheader("📊 Key Performance Summary (Oct 1, 2025 - Jan 30, 2026)")
                
                # Calculate overall metrics for Frank+Felix vs Others
                df_overall = df_ff.copy()
                df_overall['persona_group'] = df_overall['persona'].apply(
                    lambda x: 'Frank & Felix' if x in ['Frank', 'Felix'] else 'Other Personas'
                )
                
                overall_summary = df_overall.groupby('persona_group').agg({
                    'lead_count': 'sum',
                    'sent_to_sales_qty': 'sum',
                    'fas_count': 'sum',
                    'funded_count': 'sum',
                    'fas_loan_amount': 'sum',
                    'sts_loan_amount': 'sum',
                    'contribution_margin': 'sum'
                }).reset_index()
                
                total_leads = overall_summary['lead_count'].sum()
                total_sts = overall_summary['sent_to_sales_qty'].sum()
                
                ff_row = overall_summary[overall_summary['persona_group'] == 'Frank & Felix']
                other_row = overall_summary[overall_summary['persona_group'] == 'Other Personas']
                
                if not ff_row.empty and not other_row.empty:
                    ff_leads = ff_row['lead_count'].values[0]
                    ff_sts = ff_row['sent_to_sales_qty'].values[0]
                    ff_fas = ff_row['fas_count'].values[0]
                    ff_funded = ff_row['funded_count'].values[0]
                    ff_loan_amt = ff_row['fas_loan_amount'].values[0]
                    ff_sts_loan = ff_row['sts_loan_amount'].values[0]
                    ff_cm = ff_row['contribution_margin'].values[0]
                    
                    other_leads = other_row['lead_count'].values[0]
                    other_sts = other_row['sent_to_sales_qty'].values[0]
                    other_fas = other_row['fas_count'].values[0]
                    other_funded = other_row['funded_count'].values[0]
                    other_loan_amt = other_row['fas_loan_amount'].values[0]
                    other_sts_loan = other_row['sts_loan_amount'].values[0]
                    other_cm = other_row['contribution_margin'].values[0]
                    
                    ff_fas_rate = ff_fas / ff_sts if ff_sts > 0 else 0
                    other_fas_rate = other_fas / other_sts if other_sts > 0 else 0
                    
                    # % of total based on Sent to Sales qty (not leads)
                    ff_pct = ff_sts / total_sts if total_sts > 0 else 0
                    other_pct = other_sts / total_sts if total_sts > 0 else 0
                    
                    # FAS $ per Lead Sent to Sales (Revenue Yield)
                    ff_fas_per_sts = ff_loan_amt / ff_sts if ff_sts > 0 else 0
                    other_fas_per_sts = other_loan_amt / other_sts if other_sts > 0 else 0
                    
                    ff_avg_loan = ff_loan_amt / ff_fas if ff_fas > 0 else 0
                    other_avg_loan = other_loan_amt / other_fas if other_fas > 0 else 0
                    
                    # Pull Through Rate (Funded / FAS)
                    ff_pull_through = ff_funded / ff_fas if ff_fas > 0 else 0
                    other_pull_through = other_funded / other_fas if other_fas > 0 else 0
                    
                    # Value Concentration Ratio: (% of FAS $) / (% of StS)
                    total_fas_dollar = ff_loan_amt + other_loan_amt
                    ff_fas_dollar_share = ff_loan_amt / total_fas_dollar if total_fas_dollar > 0 else 0
                    other_fas_dollar_share = other_loan_amt / total_fas_dollar if total_fas_dollar > 0 else 0
                    ff_value_concentration = ff_fas_dollar_share / ff_pct if ff_pct > 0 else 0
                    other_value_concentration = other_fas_dollar_share / other_pct if other_pct > 0 else 0
                    
                    # Contribution Margin per StS
                    ff_cm_per_sts = ff_cm / ff_sts if ff_sts > 0 else 0
                    other_cm_per_sts = other_cm / other_sts if other_sts > 0 else 0
                    
                    # Total CM and share
                    total_cm = ff_cm + other_cm
                    ff_cm_share = ff_cm / total_cm if total_cm > 0 else 0
                    other_cm_share = other_cm / total_cm if total_cm > 0 else 0
                    
                    # KPI Cards - Row 1: Volume & Share (based on Sent to Sales)
                    st.markdown("#### 📈 Volume & Market Share (Sent to Sales)")
                    kpi_row1 = st.columns(4)
                    
                    with kpi_row1[0]:
                        st.metric(
                            "Frank & Felix (StS)",
                            f"{ff_sts:,}",
                            help="Total Sent to Sales for Frank & Felix personas"
                        )
                    
                    with kpi_row1[1]:
                        st.metric(
                            "F&F % of Total StS",
                            f"{ff_pct:.1%}",
                            help="Frank & Felix as percentage of all Sent to Sales"
                        )
                    
                    with kpi_row1[2]:
                        st.metric(
                            "Other Personas (StS)",
                            f"{other_sts:,}",
                            help="Total Sent to Sales for all other personas"
                        )
                    
                    with kpi_row1[3]:
                        st.metric(
                            "Others % of Total StS",
                            f"{other_pct:.1%}",
                            help="Other personas as percentage of all Sent to Sales"
                        )
                    
                    # KPI Cards - Row 2: Conversion Comparison
                    st.markdown("#### 🎯 Conversion Performance")
                    kpi_row2 = st.columns(4)
                    
                    with kpi_row2[0]:
                        rate_delta = ff_fas_rate - other_fas_rate
                        st.metric(
                            "F&F FAS Rate",
                            f"{ff_fas_rate:.1%}",
                            f"+{rate_delta:.1%} vs Others" if rate_delta > 0 else f"{rate_delta:.1%} vs Others",
                            delta_color="normal"
                        )
                    
                    with kpi_row2[1]:
                        st.metric(
                            "Others FAS Rate",
                            f"{other_fas_rate:.1%}",
                            help="FAS Rate for non-Frank/Felix personas"
                        )
                    
                    with kpi_row2[2]:
                        fas_per_sts_delta = ff_fas_per_sts - other_fas_per_sts
                        st.metric(
                            "F&F FAS $ per StS",
                            f"${ff_fas_per_sts:,.0f}",
                            f"+${fas_per_sts_delta:,.0f} vs Others" if fas_per_sts_delta > 0 else f"${fas_per_sts_delta:,.0f}",
                            delta_color="normal"
                        )
                        st.caption("FAS Loan $ / Leads Sent to Sales")
                    
                    with kpi_row2[3]:
                        st.metric(
                            "Others FAS $ per StS",
                            f"${other_fas_per_sts:,.0f}",
                            help="FAS Loan $ generated per lead sent to sales"
                        )
                    
                    # KPI Cards - Row 3: Loan Value
                    st.markdown("#### 💰 Loan Value Impact")
                    kpi_row3 = st.columns(4)
                    
                    with kpi_row3[0]:
                        loan_delta = ff_avg_loan - other_avg_loan
                        st.metric(
                            "F&F Avg Loan Size",
                            f"${ff_avg_loan:,.0f}",
                            f"+${loan_delta:,.0f}" if loan_delta > 0 else f"${loan_delta:,.0f}",
                            delta_color="normal"
                        )
                    
                    with kpi_row3[1]:
                        st.metric(
                            "Others Avg Loan Size",
                            f"${other_avg_loan:,.0f}",
                            help="Average loan amount for FAS'd leads"
                        )
                    
                    with kpi_row3[2]:
                        st.metric(
                            "F&F Total FAS $",
                            f"${ff_loan_amt:,.0f}",
                            help="Total loan amount from Frank & Felix FAS"
                        )
                    
                    with kpi_row3[3]:
                        st.metric(
                            "Others Total FAS $",
                            f"${other_loan_amt:,.0f}",
                            help="Total loan amount from other personas FAS"
                        )
                    
                    # KPI Cards - Row 4: Pull Through Rate
                    st.markdown("#### 🎯 Pull Through Rate (Funded / FAS)")
                    kpi_row4 = st.columns(4)
                    
                    with kpi_row4[0]:
                        pull_delta = ff_pull_through - other_pull_through
                        st.metric(
                            "F&F Pull Through",
                            f"{ff_pull_through:.1%}",
                            f"+{pull_delta:.1%} vs Others" if pull_delta > 0 else f"{pull_delta:.1%}",
                            delta_color="normal"
                        )
                        st.caption(f"Funded: {ff_funded:,} / FAS: {ff_fas:,}")
                    
                    with kpi_row4[1]:
                        st.metric(
                            "Others Pull Through",
                            f"{other_pull_through:.1%}",
                            help="% of FAS'd leads that get funded"
                        )
                        st.caption(f"Funded: {other_funded:,} / FAS: {other_fas:,}")
                    
                    with kpi_row4[2]:
                        st.metric(
                            "F&F Funded Count",
                            f"{ff_funded:,}",
                            help="Total funded loans for Frank & Felix"
                        )
                    
                    with kpi_row4[3]:
                        st.metric(
                            "Others Funded Count",
                            f"{other_funded:,}",
                            help="Total funded loans for other personas"
                        )
                    
                    # KPI Cards - Row 5: Lead Value driven by Loan Size
                    st.markdown("#### 📊 Lead Value Driven by Loan Size")
                    st.caption("Ratio > 1.0 means punching above weight (generating more $ than their share of volume)")
                    kpi_row5 = st.columns(4)
                    
                    with kpi_row5[0]:
                        st.metric(
                            "F&F Lead Value Ratio",
                            f"{ff_value_concentration:.2f}x",
                            f"+{ff_value_concentration - 1:.2f}" if ff_value_concentration > 1 else f"{ff_value_concentration - 1:.2f}",
                            delta_color="normal"
                        )
                        st.caption(f"{ff_fas_dollar_share:.1%} of FAS $ ÷ {ff_pct:.1%} of StS")
                    
                    with kpi_row5[1]:
                        st.metric(
                            "Others Lead Value Ratio",
                            f"{other_value_concentration:.2f}x",
                            f"+{other_value_concentration - 1:.2f}" if other_value_concentration > 1 else f"{other_value_concentration - 1:.2f}",
                            delta_color="inverse"
                        )
                        st.caption(f"{other_fas_dollar_share:.1%} of FAS $ ÷ {other_pct:.1%} of StS")
                    
                    with kpi_row5[2]:
                        st.metric(
                            "F&F Share of FAS $",
                            f"{ff_fas_dollar_share:.1%}",
                            help="% of total FAS loan dollars from Frank & Felix"
                        )
                    
                    with kpi_row5[3]:
                        st.metric(
                            "Others Share of FAS $",
                            f"{other_fas_dollar_share:.1%}",
                            help="% of total FAS loan dollars from other personas"
                        )
                    
                    # KPI Cards - Row 6: Contribution Margin
                    st.markdown("#### 💵 Contribution Margin")
                    st.caption("NPV Actual Contribution Margin - the true profitability per lead")
                    kpi_row6 = st.columns(4)
                    
                    with kpi_row6[0]:
                        cm_delta = ff_cm_per_sts - other_cm_per_sts
                        st.metric(
                            "F&F CM per StS",
                            f"${ff_cm_per_sts:,.0f}",
                            f"+${cm_delta:,.0f} vs Others" if cm_delta > 0 else f"${cm_delta:,.0f}",
                            delta_color="normal"
                        )
                        st.caption("Contribution Margin / Leads Sent to Sales")
                    
                    with kpi_row6[1]:
                        st.metric(
                            "Others CM per StS",
                            f"${other_cm_per_sts:,.0f}",
                            help="Contribution Margin per lead sent to sales"
                        )
                    
                    with kpi_row6[2]:
                        st.metric(
                            "F&F Total CM",
                            f"${ff_cm:,.0f}",
                            f"{ff_cm_share:.1%} of total",
                            help="Total Contribution Margin from Frank & Felix"
                        )
                    
                    with kpi_row6[3]:
                        st.metric(
                            "Others Total CM",
                            f"${other_cm:,.0f}",
                            f"{other_cm_share:.1%} of total",
                            help="Total Contribution Margin from other personas"
                        )
                
                st.divider()
                
                # --- Chart 1: % of Total Sent to Sales Over Time ---
                st.subheader("📈 Goal 1: Frank & Felix Growing as % of Total Sent to Sales")
                st.caption("Marketing effort to increase high-value personas (based on Sent to Sales volume)")
                
                # Filter to just Frank and Felix for the share chart
                df_ff_only = df_persona_detail[df_persona_detail['persona'].isin(['Frank', 'Felix'])].copy()
                
                # Also calculate combined Frank+Felix
                df_combined = df_grouped[df_grouped['persona_group'].isin(['Frank', 'Felix'])].groupby('date_col').agg({
                    'sent_to_sales_qty': 'sum',
                    'pct_of_total': 'sum'  # Sum their percentages
                }).reset_index()
                df_combined['persona'] = 'Frank + Felix Combined'
                
                # Calculate the combined percentage properly (based on Sent to Sales)
                total_per_date_all = df_ff_agg.groupby('date_col')['sent_to_sales_qty'].sum().reset_index()
                total_per_date_all.columns = ['date_col', 'total_sts']
                df_combined = df_combined.merge(total_per_date_all, on='date_col')
                df_combined['pct_of_total'] = df_combined['sent_to_sales_qty'] / df_combined['total_sts']
                
                # Create the trend chart
                pct_chart = alt.Chart(df_ff_only).mark_line(point=True).encode(
                    x=alt.X('date_col:T', title=ff_grain, axis=alt.Axis(
                        format='%b %d',
                        labelExpr="datum.label + '\\n' + timeFormat(datum.value, '%Y')",
                        labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    y=alt.Y('pct_of_total:Q', title='% of Total Sent to Sales', axis=alt.Axis(
                        format='%', labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    color=alt.Color('persona:N', scale=alt.Scale(
                        domain=['Frank', 'Felix'],
                        range=['#2ca02c', '#1f77b4']
                    )),
                    tooltip=[
                        alt.Tooltip('date_col:T', title=ff_grain, format='%b %d, %Y'),
                        alt.Tooltip('persona:N', title='Persona'),
                        alt.Tooltip('pct_of_total:Q', title='% of Total StS', format='.1%'),
                        alt.Tooltip('sent_to_sales_qty:Q', title='Sent to Sales', format=',d')
                    ]
                ).properties(
                    height=350
                ).interactive()
                
                st.altair_chart(pct_chart, use_container_width=True)
                
                # Trend analysis for share growth
                first_period = df_combined.head(2)['pct_of_total'].mean()
                last_period = df_combined.tail(2)['pct_of_total'].mean()
                share_change = last_period - first_period
                
                if share_change > 0:
                    st.success(f"✅ **Frank & Felix share of Sent to Sales has grown by {share_change:+.1%}** from {first_period:.1%} to {last_period:.1%}")
                else:
                    st.warning(f"⚠️ Frank & Felix share changed by {share_change:+.1%} from {first_period:.1%} to {last_period:.1%}")
                
                st.divider()
                
                # --- Chart 2: FAS Rate Comparison Over Time ---
                st.subheader("📊 Goal 2: Frank & Felix Convert Better Than Others")
                st.caption("Demonstrating superior FAS performance")
                
                fas_chart = alt.Chart(df_grouped).mark_line(point=True).encode(
                    x=alt.X('date_col:T', title=ff_grain, axis=alt.Axis(
                        format='%b %d',
                        labelExpr="datum.label + '\\n' + timeFormat(datum.value, '%Y')",
                        labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    y=alt.Y('fas_rate:Q', title='FAS Rate', axis=alt.Axis(
                        format='%', labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    color=alt.Color('persona_group:N', scale=alt.Scale(
                        domain=['Frank', 'Felix', 'Other Personas'],
                        range=['#2ca02c', '#1f77b4', '#d62728']
                    )),
                    strokeDash=alt.condition(
                        alt.datum.persona_group == 'Other Personas',
                        alt.value([5, 5]),
                        alt.value([0])
                    ),
                    tooltip=[
                        alt.Tooltip('date_col:T', title=ff_grain, format='%b %d, %Y'),
                        alt.Tooltip('persona_group:N', title='Persona'),
                        alt.Tooltip('fas_rate:Q', title='FAS Rate', format='.1%'),
                        alt.Tooltip('fas_count:Q', title='FAS Count', format=',d'),
                        alt.Tooltip('sent_to_sales_qty:Q', title='Sent to Sales', format=',d')
                    ]
                ).properties(
                    height=350
                ).interactive()
                
                st.altair_chart(fas_chart, use_container_width=True)
                
                st.divider()
                
                # --- Chart: Pull Through Rate Over Time ---
                st.subheader("🎯 Pull Through Rate (Funded / FAS)")
                st.caption("What % of FAS'd leads actually get funded - shows quality of conversions")
                
                pull_through_chart = alt.Chart(df_grouped).mark_line(point=True).encode(
                    x=alt.X('date_col:T', title=ff_grain, axis=alt.Axis(
                        format='%b %d',
                        labelExpr="datum.label + '\\n' + timeFormat(datum.value, '%Y')",
                        labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    y=alt.Y('pull_through_rate:Q', title='Pull Through Rate (Funded/FAS)', axis=alt.Axis(
                        format='%', labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    color=alt.Color('persona_group:N', scale=alt.Scale(
                        domain=['Frank', 'Felix', 'Other Personas'],
                        range=['#2ca02c', '#1f77b4', '#d62728']
                    )),
                    strokeDash=alt.condition(
                        alt.datum.persona_group == 'Other Personas',
                        alt.value([5, 5]),
                        alt.value([0])
                    ),
                    tooltip=[
                        alt.Tooltip('date_col:T', title=ff_grain, format='%b %d, %Y'),
                        alt.Tooltip('persona_group:N', title='Persona'),
                        alt.Tooltip('pull_through_rate:Q', title='Pull Through Rate', format='.1%'),
                        alt.Tooltip('funded_count:Q', title='Funded', format=',d'),
                        alt.Tooltip('fas_count:Q', title='FAS', format=',d')
                    ]
                ).properties(
                    height=350
                ).interactive()
                
                st.altair_chart(pull_through_chart, use_container_width=True)
                
                st.divider()
                
                # --- Chart 3: FAS $ per Lead Sent to Sales (Revenue Yield) ---
                st.subheader("💰 Goal 3: Higher Revenue per Lead (FAS $ per StS)")
                st.caption("Shows dollar value generated per lead sent to sales - even with similar FAS rates, F&F generates more revenue per lead")
                
                revenue_chart = alt.Chart(df_grouped).mark_line(point=True).encode(
                    x=alt.X('date_col:T', title=ff_grain, axis=alt.Axis(
                        format='%b %d',
                        labelExpr="datum.label + '\\n' + timeFormat(datum.value, '%Y')",
                        labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    y=alt.Y('fas_per_sts:Q', title='FAS $ per Lead Sent to Sales', axis=alt.Axis(
                        format='$,.0f', labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    color=alt.Color('persona_group:N', scale=alt.Scale(
                        domain=['Frank', 'Felix', 'Other Personas'],
                        range=['#2ca02c', '#1f77b4', '#d62728']
                    )),
                    strokeDash=alt.condition(
                        alt.datum.persona_group == 'Other Personas',
                        alt.value([5, 5]),
                        alt.value([0])
                    ),
                    tooltip=[
                        alt.Tooltip('date_col:T', title=ff_grain, format='%b %d, %Y'),
                        alt.Tooltip('persona_group:N', title='Persona'),
                        alt.Tooltip('fas_per_sts:Q', title='FAS $ per StS', format='$,.0f'),
                        alt.Tooltip('fas_loan_amount:Q', title='Total FAS Loan $', format='$,.0f'),
                        alt.Tooltip('sent_to_sales_qty:Q', title='Sent to Sales', format=',d')
                    ]
                ).properties(
                    height=350
                ).interactive()
                
                st.altair_chart(revenue_chart, use_container_width=True)
                
                st.divider()
                
                # --- Chart: Contribution Margin per StS Over Time ---
                st.subheader("💵 Contribution Margin per Lead Sent to Sales")
                st.caption("NPV Actual Contribution Margin per lead - the true profitability metric")
                
                cm_chart = alt.Chart(df_grouped).mark_line(point=True).encode(
                    x=alt.X('date_col:T', title=ff_grain, axis=alt.Axis(
                        format='%b %d',
                        labelExpr="datum.label + '\\n' + timeFormat(datum.value, '%Y')",
                        labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    y=alt.Y('cm_per_sts:Q', title='Contribution Margin per StS', axis=alt.Axis(
                        format='$,.0f', labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    color=alt.Color('persona_group:N', scale=alt.Scale(
                        domain=['Frank', 'Felix', 'Other Personas'],
                        range=['#2ca02c', '#1f77b4', '#d62728']
                    )),
                    strokeDash=alt.condition(
                        alt.datum.persona_group == 'Other Personas',
                        alt.value([5, 5]),
                        alt.value([0])
                    ),
                    tooltip=[
                        alt.Tooltip('date_col:T', title=ff_grain, format='%b %d, %Y'),
                        alt.Tooltip('persona_group:N', title='Persona'),
                        alt.Tooltip('cm_per_sts:Q', title='CM per StS', format='$,.0f'),
                        alt.Tooltip('contribution_margin:Q', title='Total CM', format='$,.0f'),
                        alt.Tooltip('sent_to_sales_qty:Q', title='Sent to Sales', format=',d')
                    ]
                ).properties(
                    height=350
                ).interactive()
                
                st.altair_chart(cm_chart, use_container_width=True)
                
                st.divider()
                
                # --- Chart: Lead Value Driven by Loan Size (Value Concentration Ratio) Over Time ---
                st.subheader("📊 Lead Value Driven by Loan Size")
                st.caption("Ratio > 1.0x means generating more FAS $ than their share of Sent to Sales volume (punching above weight)")
                
                # Add a reference line at 1.0
                value_conc_base = alt.Chart(df_grouped).mark_line(point=True).encode(
                    x=alt.X('date_col:T', title=ff_grain, axis=alt.Axis(
                        format='%b %d',
                        labelExpr="datum.label + '\\n' + timeFormat(datum.value, '%Y')",
                        labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    y=alt.Y('value_concentration:Q', title='Value Concentration (FAS $ Share / StS Share)', axis=alt.Axis(
                        format='.2f', labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    color=alt.Color('persona_group:N', scale=alt.Scale(
                        domain=['Frank', 'Felix', 'Other Personas'],
                        range=['#2ca02c', '#1f77b4', '#d62728']
                    )),
                    strokeDash=alt.condition(
                        alt.datum.persona_group == 'Other Personas',
                        alt.value([5, 5]),
                        alt.value([0])
                    ),
                    tooltip=[
                        alt.Tooltip('date_col:T', title=ff_grain, format='%b %d, %Y'),
                        alt.Tooltip('persona_group:N', title='Persona'),
                        alt.Tooltip('value_concentration:Q', title='Value Concentration', format='.2f'),
                        alt.Tooltip('fas_dollar_share:Q', title='FAS $ Share', format='.1%'),
                        alt.Tooltip('sts_share:Q', title='StS Share', format='.1%')
                    ]
                )
                
                # Reference line at 1.0 (fair share)
                reference_line = alt.Chart(pd.DataFrame({'y': [1.0]})).mark_rule(
                    color='white', strokeDash=[3, 3], strokeWidth=1
                ).encode(y='y:Q')
                
                # Reference text
                reference_text = alt.Chart(pd.DataFrame({'y': [1.0], 'text': ['Fair Share (1.0x)']})).mark_text(
                    align='left', dx=5, dy=-5, color='white', fontSize=10
                ).encode(y='y:Q', text='text:N')
                
                value_conc_chart = alt.layer(value_conc_base, reference_line, reference_text).properties(
                    height=350
                ).interactive()
                
                st.altair_chart(value_conc_chart, use_container_width=True)
                
                st.divider()
                
                # --- Chart 4: Other Personas Declining ---
                st.subheader("📉 Goal 4: Other Populations Declining as % of Total Sent to Sales")
                st.caption("As Frank & Felix grow, other personas should decline proportionally")
                
                # Get all personas and their share over time
                decline_chart = alt.Chart(df_persona_detail[~df_persona_detail['persona'].isin(['Frank', 'Felix'])]).mark_area(
                    opacity=0.7
                ).encode(
                    x=alt.X('date_col:T', title=ff_grain, axis=alt.Axis(
                        format='%b %d',
                        labelExpr="datum.label + '\\n' + timeFormat(datum.value, '%Y')",
                        labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    y=alt.Y('pct_of_total:Q', title='% of Total Sent to Sales', stack='normalize', axis=alt.Axis(
                        format='%', labelColor='white', labelFontWeight='bold',
                        titleColor='white', titleFontWeight='bold'
                    )),
                    color=alt.Color('persona:N', title='Persona'),
                    tooltip=[
                        alt.Tooltip('date_col:T', title=ff_grain, format='%b %d, %Y'),
                        alt.Tooltip('persona:N', title='Persona'),
                        alt.Tooltip('pct_of_total:Q', title='% of Total StS', format='.1%'),
                        alt.Tooltip('sent_to_sales_qty:Q', title='Sent to Sales', format=',d')
                    ]
                ).properties(
                    height=350
                ).interactive()
                
                st.altair_chart(decline_chart, use_container_width=True)
                
                # Calculate other personas' decline
                df_others = df_grouped[df_grouped['persona_group'] == 'Other Personas']
                if not df_others.empty:
                    others_first = df_others.head(2)['pct_of_total'].mean()
                    others_last = df_others.tail(2)['pct_of_total'].mean()
                    others_change = others_last - others_first
                    
                    if others_change < 0:
                        st.success(f"✅ **Other personas' share of Sent to Sales has declined by {others_change:.1%}** from {others_first:.1%} to {others_last:.1%}")
                    else:
                        st.info(f"ℹ️ Other personas' share changed by {others_change:+.1%} from {others_first:.1%} to {others_last:.1%}")
                
                st.divider()
                
                # --- Summary Table ---
                st.subheader("📋 Detailed Performance Summary")
                
                # Create comprehensive summary by persona
                summary_by_persona = df_ff.groupby('persona').agg({
                    'lead_count': 'sum',
                    'sent_to_sales_qty': 'sum',
                    'fas_count': 'sum',
                    'funded_count': 'sum',
                    'fas_loan_amount': 'sum',
                    'sts_loan_amount': 'sum',
                    'contribution_margin': 'sum'
                }).reset_index()
                
                summary_by_persona['fas_rate'] = summary_by_persona.apply(
                    lambda x: x['fas_count'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                summary_by_persona['fas_per_sts'] = summary_by_persona.apply(
                    lambda x: x['fas_loan_amount'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                summary_by_persona['pull_through'] = summary_by_persona.apply(
                    lambda x: x['funded_count'] / x['fas_count'] if x['fas_count'] > 0 else 0, axis=1
                )
                summary_by_persona['cm_per_sts'] = summary_by_persona.apply(
                    lambda x: x['contribution_margin'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                summary_by_persona['avg_loan'] = summary_by_persona.apply(
                    lambda x: x['fas_loan_amount'] / x['fas_count'] if x['fas_count'] > 0 else 0, axis=1
                )
                # % of total based on Sent to Sales
                summary_by_persona['pct_of_total'] = summary_by_persona['sent_to_sales_qty'] / summary_by_persona['sent_to_sales_qty'].sum()
                
                summary_by_persona = summary_by_persona.sort_values('fas_rate', ascending=False)
                
                # Format for display
                display_summary = summary_by_persona.copy()
                display_summary['Sent to Sales'] = display_summary['sent_to_sales_qty'].apply(lambda x: f"{x:,}")
                display_summary['% of Total StS'] = display_summary['pct_of_total'].apply(lambda x: f"{x:.1%}")
                display_summary['FAS'] = display_summary['fas_count'].apply(lambda x: f"{x:,}")
                display_summary['FAS Rate'] = display_summary['fas_rate'].apply(lambda x: f"{x:.1%}")
                display_summary['FAS $ per StS'] = display_summary['fas_per_sts'].apply(lambda x: f"${x:,.0f}")
                display_summary['Pull Through'] = display_summary['pull_through'].apply(lambda x: f"{x:.1%}")
                display_summary['CM per StS'] = display_summary['cm_per_sts'].apply(lambda x: f"${x:,.0f}")
                display_summary['Total CM'] = display_summary['contribution_margin'].apply(lambda x: f"${x:,.0f}")
                display_summary['Avg Loan $'] = display_summary['avg_loan'].apply(lambda x: f"${x:,.0f}")
                display_summary['Total FAS $'] = display_summary['fas_loan_amount'].apply(lambda x: f"${x:,.0f}")
                
                display_summary = display_summary[['persona', 'Sent to Sales', '% of Total StS', 'FAS', 'FAS Rate', 'Pull Through', 'FAS $ per StS', 'CM per StS', 'Total CM']]
                display_summary.columns = ['Persona', 'Sent to Sales', '% Total StS', 'FAS', 'FAS Rate', 'Pull Thru', 'FAS $/StS', 'CM/StS', 'Total CM']
                
                # Highlight Frank and Felix rows
                st.dataframe(display_summary, use_container_width=True, hide_index=True)
                
                # Key takeaways
                st.markdown("---")
                st.subheader("💡 Key Takeaways")
                
                ff_total_fas = summary_by_persona[summary_by_persona['persona'].isin(['Frank', 'Felix'])]['fas_loan_amount'].sum()
                total_fas_all = summary_by_persona['fas_loan_amount'].sum()
                ff_dollar_share = ff_total_fas / total_fas_all if total_fas_all > 0 else 0
                
                st.markdown(f"""
                1. **Frank & Felix represent {ff_pct:.1%}** of total Sent to Sales but generate **{ff_dollar_share:.1%}** of total FAS loan volume
                2. **FAS Rate advantage:** Frank & Felix convert at **{ff_fas_rate:.1%}** vs Others at **{other_fas_rate:.1%}** ({ff_fas_rate - other_fas_rate:+.1%} better)
                3. **Loan size advantage:** Average loan for Frank & Felix is **${ff_avg_loan:,.0f}** vs Others at **${other_avg_loan:,.0f}** (${ff_avg_loan - other_avg_loan:+,.0f} higher)
                4. **Marketing ROI:** Increasing Frank & Felix share directly improves both conversion rates AND dollar volume per conversion
                
                *Data period: Oct 1, 2025 - Jan 30, 2026 | % calculations based on Sent to Sales volume*
                """)
                
                st.divider()
                
                # --- SCORECARD: Lead Quality Trends Over Time ---
                st.subheader("📋 Lead Quality Scorecard (Overall Trends)")
                st.caption("Tracking key metrics over time to show improvement in lead quality - All personas combined")
                
                # Aggregate data by time period (overall, not by persona)
                df_scorecard = df_ff.copy()
                if ff_grain == 'Week':
                    df_scorecard['period'] = df_scorecard['lead_date'] - pd.to_timedelta(df_scorecard['lead_date'].dt.weekday, unit='D')
                elif ff_grain == 'Month':
                    df_scorecard['period'] = df_scorecard['lead_date'].dt.to_period('M').dt.to_timestamp()
                else:
                    df_scorecard['period'] = df_scorecard['lead_date']
                
                # Aggregate overall metrics by period
                scorecard_agg = df_scorecard.groupby('period').agg({
                    'lead_count': 'sum',
                    'sent_to_sales_qty': 'sum',
                    'fas_count': 'sum',
                    'funded_count': 'sum',
                    'fas_loan_amount': 'sum',
                    'funded_loan_amount': 'sum',
                    'contribution_margin': 'sum'
                }).reset_index()
                
                # Calculate per-lead averages
                scorecard_agg['avg_fas_loan'] = scorecard_agg.apply(
                    lambda x: x['fas_loan_amount'] / x['fas_count'] if x['fas_count'] > 0 else 0, axis=1
                )
                scorecard_agg['avg_funded_loan'] = scorecard_agg.apply(
                    lambda x: x['funded_loan_amount'] / x['funded_count'] if x['funded_count'] > 0 else 0, axis=1
                )
                scorecard_agg['fas_rate'] = scorecard_agg.apply(
                    lambda x: x['fas_count'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                scorecard_agg['fas_per_sts'] = scorecard_agg.apply(
                    lambda x: x['fas_loan_amount'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                scorecard_agg['cm_per_sts'] = scorecard_agg.apply(
                    lambda x: x['contribution_margin'] / x['sent_to_sales_qty'] if x['sent_to_sales_qty'] > 0 else 0, axis=1
                )
                scorecard_agg['pull_through'] = scorecard_agg.apply(
                    lambda x: x['funded_count'] / x['fas_count'] if x['fas_count'] > 0 else 0, axis=1
                )
                
                # Create scorecard table
                scorecard_display = scorecard_agg.copy()
                scorecard_display['Period'] = scorecard_display['period'].dt.strftime('%Y-%m-%d')
                scorecard_display['Sent to Sales'] = scorecard_display['sent_to_sales_qty'].apply(lambda x: f"{x:,}")
                scorecard_display['FAS Qty'] = scorecard_display['fas_count'].apply(lambda x: f"{x:,}")
                scorecard_display['FAS Rate'] = scorecard_display['fas_rate'].apply(lambda x: f"{x:.1%}")
                scorecard_display['FAS $'] = scorecard_display['fas_loan_amount'].apply(lambda x: f"${x:,.0f}")
                scorecard_display['Avg FAS Loan'] = scorecard_display['avg_fas_loan'].apply(lambda x: f"${x:,.0f}")
                scorecard_display['Avg Funded Loan'] = scorecard_display['avg_funded_loan'].apply(lambda x: f"${x:,.0f}")
                scorecard_display['Pull Through'] = scorecard_display['pull_through'].apply(lambda x: f"{x:.1%}")
                scorecard_display['FAS $ per StS'] = scorecard_display['fas_per_sts'].apply(lambda x: f"${x:,.0f}")
                scorecard_display['CM per StS'] = scorecard_display['cm_per_sts'].apply(lambda x: f"${x:,.0f}")
                
                scorecard_final = scorecard_display[['Period', 'Sent to Sales', 'FAS Qty', 'FAS Rate', 'FAS $', 
                                                      'Avg FAS Loan', 'Avg Funded Loan', 'Pull Through',
                                                      'FAS $ per StS', 'CM per StS']]
                
                st.dataframe(scorecard_final, use_container_width=True, hide_index=True)
                
                # Trend indicators
                st.markdown("#### 📈 Trend Analysis")
                
                if len(scorecard_agg) >= 2:
                    first_period = scorecard_agg.head(2).mean(numeric_only=True)
                    last_period = scorecard_agg.tail(2).mean(numeric_only=True)
                    
                    trend_col1, trend_col2, trend_col3, trend_col4 = st.columns(4)
                    
                    with trend_col1:
                        fas_loan_trend = last_period['avg_fas_loan'] - first_period['avg_fas_loan']
                        st.metric(
                            "Avg FAS Loan Trend",
                            f"${last_period['avg_fas_loan']:,.0f}",
                            f"${fas_loan_trend:+,.0f}",
                            delta_color="normal"
                        )
                    
                    with trend_col2:
                        pull_through_trend = last_period['pull_through'] - first_period['pull_through']
                        st.metric(
                            "Pull Through Trend",
                            f"{last_period['pull_through']:.1%}",
                            f"{pull_through_trend:+.1%}",
                            delta_color="normal"
                        )
                    
                    with trend_col3:
                        fas_per_sts_trend = last_period['fas_per_sts'] - first_period['fas_per_sts']
                        st.metric(
                            "FAS $ per StS Trend",
                            f"${last_period['fas_per_sts']:,.0f}",
                            f"${fas_per_sts_trend:+,.0f}",
                            delta_color="normal"
                        )
                    
                    with trend_col4:
                        cm_trend = last_period['cm_per_sts'] - first_period['cm_per_sts']
                        st.metric(
                            "CM per StS Trend",
                            f"${last_period['cm_per_sts']:,.0f}",
                            f"${cm_trend:+,.0f}",
                            delta_color="normal"
                        )
                    
                    # Summary insight
                    improvements = []
                    if fas_loan_trend > 0:
                        improvements.append(f"Avg FAS Loan up ${fas_loan_trend:,.0f}")
                    if fas_per_sts_trend > 0:
                        improvements.append(f"FAS $ per lead up ${fas_per_sts_trend:,.0f}")
                    if cm_trend > 0:
                        improvements.append(f"Contribution Margin per lead up ${cm_trend:,.0f}")
                    if pull_through_trend > 0:
                        improvements.append(f"Pull Through up {pull_through_trend:+.1%}")
                    
                    if improvements:
                        st.success(f"✅ **Improvements detected:** {' | '.join(improvements)}")
                    else:
                        st.info("📊 Metrics are relatively stable over the period")
                
                # Scorecard Charts
                st.markdown("#### 📊 Scorecard Trend Charts")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Avg Loan Amounts (FAS vs Funded)
                    loan_data = scorecard_agg[['period', 'avg_fas_loan', 'avg_funded_loan']].melt(
                        id_vars=['period'], 
                        value_vars=['avg_fas_loan', 'avg_funded_loan'],
                        var_name='Metric', 
                        value_name='Amount'
                    )
                    loan_data['Metric'] = loan_data['Metric'].replace({
                        'avg_fas_loan': 'Avg FAS Loan',
                        'avg_funded_loan': 'Avg Funded Loan'
                    })
                    
                    loan_chart = alt.Chart(loan_data).mark_line(point=True).encode(
                        x=alt.X('period:T', title=ff_grain, axis=alt.Axis(
                            format='%b %d', labelColor='white', labelFontWeight='bold',
                            titleColor='white', titleFontWeight='bold'
                        )),
                        y=alt.Y('Amount:Q', title='Avg Loan Amount ($)', axis=alt.Axis(
                            format='$,.0f', labelColor='white', labelFontWeight='bold',
                            titleColor='white', titleFontWeight='bold'
                        )),
                        color=alt.Color('Metric:N', scale=alt.Scale(
                            domain=['Avg FAS Loan', 'Avg Funded Loan'],
                            range=['#2ca02c', '#1f77b4']
                        )),
                        tooltip=[
                            alt.Tooltip('period:T', title='Period', format='%b %d, %Y'),
                            alt.Tooltip('Metric:N', title='Metric'),
                            alt.Tooltip('Amount:Q', title='Amount', format='$,.0f')
                        ]
                    ).properties(
                        title='Avg Loan Amount: FAS vs Funded',
                        height=300
                    ).interactive()
                    
                    st.altair_chart(loan_chart, use_container_width=True)
                
                with chart_col2:
                    # Pull Through Rate Over Time
                    pull_chart = alt.Chart(scorecard_agg).mark_line(point=True, color='#ff7f0e').encode(
                        x=alt.X('period:T', title=ff_grain, axis=alt.Axis(
                            format='%b %d', labelColor='white', labelFontWeight='bold',
                            titleColor='white', titleFontWeight='bold'
                        )),
                        y=alt.Y('pull_through:Q', title='Pull Through Rate (%)', axis=alt.Axis(
                            format='.0%', labelColor='white', labelFontWeight='bold',
                            titleColor='white', titleFontWeight='bold'
                        )),
                        tooltip=[
                            alt.Tooltip('period:T', title='Period', format='%b %d, %Y'),
                            alt.Tooltip('pull_through:Q', title='Pull Through', format='.1%'),
                            alt.Tooltip('funded_count:Q', title='Funded Count', format=',d'),
                            alt.Tooltip('fas_count:Q', title='FAS Count', format=',d')
                        ]
                    ).properties(
                        title='Pull Through Rate (Funded / FAS)',
                        height=300
                    ).interactive()
                    
                    st.altair_chart(pull_chart, use_container_width=True)
                
                # Second row of charts
                chart_col3, chart_col4 = st.columns(2)
                
                with chart_col3:
                    # FAS $ and FAS Rate dual axis
                    fas_metrics = alt.Chart(scorecard_agg).mark_line(point=True, color='#2ca02c').encode(
                        x=alt.X('period:T', title=ff_grain, axis=alt.Axis(
                            format='%b %d', labelColor='white', labelFontWeight='bold',
                            titleColor='white', titleFontWeight='bold'
                        )),
                        y=alt.Y('fas_per_sts:Q', title='FAS $ per StS', axis=alt.Axis(
                            format='$,.0f', labelColor='white', labelFontWeight='bold',
                            titleColor='white', titleFontWeight='bold'
                        )),
                        tooltip=[
                            alt.Tooltip('period:T', title='Period', format='%b %d, %Y'),
                            alt.Tooltip('fas_per_sts:Q', title='FAS $ per StS', format='$,.0f'),
                            alt.Tooltip('fas_rate:Q', title='FAS Rate', format='.1%')
                        ]
                    ).properties(
                        title='FAS $ per Lead Sent to Sales',
                        height=300
                    ).interactive()
                    
                    st.altair_chart(fas_metrics, use_container_width=True)
                
                with chart_col4:
                    # Contribution Margin per StS
                    cm_chart_scorecard = alt.Chart(scorecard_agg).mark_area(
                        color='#9467bd', opacity=0.7, line=True
                    ).encode(
                        x=alt.X('period:T', title=ff_grain, axis=alt.Axis(
                            format='%b %d', labelColor='white', labelFontWeight='bold',
                            titleColor='white', titleFontWeight='bold'
                        )),
                        y=alt.Y('cm_per_sts:Q', title='CM per StS ($)', axis=alt.Axis(
                            format='$,.0f', labelColor='white', labelFontWeight='bold',
                            titleColor='white', titleFontWeight='bold'
                        )),
                        tooltip=[
                            alt.Tooltip('period:T', title='Period', format='%b %d, %Y'),
                            alt.Tooltip('cm_per_sts:Q', title='CM per StS', format='$,.0f'),
                            alt.Tooltip('contribution_margin:Q', title='Total CM', format='$,.0f')
                        ]
                    ).properties(
                        title='Contribution Margin per Lead',
                        height=300
                    ).interactive()
                    
                    st.altair_chart(cm_chart_scorecard, use_container_width=True)
                
                with st.expander("View Raw Data"):
                    st.dataframe(df_ff, use_container_width=True)
            
            else:
                st.warning("No persona data found for the analysis period.")
        
        except Exception as e:
            st.error(f"Frank & Felix Analysis Error: {e}")
            st.exception(e)

    # --- TAB: Digital Intent Analysis ---
    with tab_digital:
        st.header("📱 Digital Intent Analysis")
        st.caption("Analyzing leads who started a digital/express app (express_app_started_at IS NOT NULL) - High intent indicators")
        
        st.markdown("""
        **Why Digital App Starts Matter:**
        - Leads who start a digital app show **higher intent** to complete the process
        - They've already invested time entering information
        - These leads typically have **higher conversion rates** than non-digital starters
        """)
        
        # --- FILTERS ---
        st.subheader("🔍 Filters")
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Sent to Sales Date Range
            digital_date_range = st.date_input(
                "Sent to Sales Date Range",
                value=(date(2025, 10, 1), date(2026, 1, 30)),
                min_value=date(2024, 1, 1),
                max_value=date.today(),
                key="digital_date_range"
            )
            
            # Handle single date vs range
            if isinstance(digital_date_range, tuple) and len(digital_date_range) == 2:
                digital_start_date = digital_date_range[0].strftime('%Y-%m-%d')
                digital_end_date = digital_date_range[1].strftime('%Y-%m-%d')
            else:
                digital_start_date = digital_date_range.strftime('%Y-%m-%d') if digital_date_range else '2025-10-01'
                digital_end_date = digital_start_date
        
        # First, get distinct values for filters
        filter_options_query = """
        SELECT DISTINCT 
            persona,
            adjusted_lead_value_cohort
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE lead_created_date >= '2024-01-01'
        AND persona IS NOT NULL
        AND adjusted_lead_value_cohort IS NOT NULL
        """
        
        try:
            filter_options_df = client.query(filter_options_query).to_dataframe()
            persona_options = sorted(filter_options_df['persona'].dropna().unique().tolist())
            lvc_options_raw = sorted(filter_options_df['adjusted_lead_value_cohort'].dropna().unique().tolist())
        except:
            persona_options = ['Frank', 'Felix', 'Fiona', 'Freddie', 'Unknown']
            lvc_options_raw = [str(i) for i in range(1, 11)] + ['X']
        
        with filter_col2:
            # Persona Filter
            selected_personas = st.multiselect(
                "Persona",
                options=persona_options,
                default=[],
                placeholder="All Personas",
                key="digital_persona_filter"
            )
        
        with filter_col3:
            # Adjusted Lead Value Cohort Filter
            selected_lvc_raw = st.multiselect(
                "Adjusted Lead Value Cohort",
                options=lvc_options_raw,
                default=[],
                placeholder="All LVC Values",
                key="digital_lvc_filter"
            )
        
        # Build WHERE clause conditions
        digital_where_conditions = [f"DATE(sent_to_sales_date) BETWEEN '{digital_start_date}' AND '{digital_end_date}'"]
        digital_where_conditions.append("sent_to_sales_date IS NOT NULL")  # Only StS leads
        
        if selected_personas:
            persona_list = ", ".join([f"'{p}'" for p in selected_personas])
            digital_where_conditions.append(f"persona IN ({persona_list})")
        
        if selected_lvc_raw:
            lvc_list = ", ".join([f"'{l}'" for l in selected_lvc_raw])
            digital_where_conditions.append(f"adjusted_lead_value_cohort IN ({lvc_list})")
        
        digital_where_clause = "WHERE " + " AND ".join(digital_where_conditions)
        
        st.divider()
        
        # Query for digital app starts data
        digital_query = f"""
        WITH digital_data AS (
            SELECT 
                DATE(sent_to_sales_date) as lead_date,
                CASE 
                    WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
                    WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
                    WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
                    WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
                    ELSE 'Other'
                END as lvc_group,
                adjusted_lead_value_cohort as lvc_raw,
                COALESCE(persona, 'Unknown') as persona,
                COALESCE(finance_group, 'Unknown') as finance_group,
                COUNT(DISTINCT lendage_guid) as total_leads,
                COUNT(DISTINCT lendage_guid) as sent_to_sales,
                COUNT(DISTINCT CASE WHEN express_app_started_at IS NOT NULL THEN lendage_guid END) as digital_app_starts,
                COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
                COUNT(DISTINCT CASE WHEN express_app_started_at IS NOT NULL AND full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as digital_to_fas,
                COUNT(DISTINCT CASE WHEN express_app_started_at IS NULL AND full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as non_digital_to_fas,
                COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted,
                SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_loan_amount,
                SUM(CASE WHEN express_app_started_at IS NOT NULL AND full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as digital_fas_loan_amount
            FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
            {digital_where_clause}
            GROUP BY 1, 2, 3, 4, 5
        )
        SELECT * FROM digital_data
        ORDER BY lead_date
        """
        
        with st.expander("View Digital Intent SQL"):
            st.code(digital_query)
        
        try:
            with st.spinner("Fetching Digital Intent Data..."):
                df_digital = client.query(digital_query).to_dataframe()
                
                if not df_digital.empty:
                    df_digital['lead_date'] = pd.to_datetime(df_digital['lead_date'])
                    
                    # Overall metrics
                    total_sts = df_digital['sent_to_sales'].sum()
                    total_digital = df_digital['digital_app_starts'].sum()
                    total_fas = df_digital['fas_count'].sum()
                    digital_fas = df_digital['digital_to_fas'].sum()
                    non_digital_fas = df_digital['non_digital_to_fas'].sum()
                    non_digital_pool = total_sts - total_digital
                    
                    digital_rate = total_digital / total_sts if total_sts > 0 else 0
                    digital_conv = digital_fas / total_digital if total_digital > 0 else 0
                    non_digital_conv = non_digital_fas / non_digital_pool if non_digital_pool > 0 else 0
                    overall_conv = total_fas / total_sts if total_sts > 0 else 0
                    
                    # KPI Cards
                    st.subheader("📊 Overall Digital Intent Metrics")
                    
                    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                    
                    with kpi_col1:
                        st.metric(
                            "Digital App Starts",
                            f"{total_digital:,}",
                            f"{digital_rate:.1%} of StS"
                        )
                    
                    with kpi_col2:
                        st.metric(
                            "Digital → FAS Rate",
                            f"{digital_conv:.1%}",
                            f"+{(digital_conv - non_digital_conv):.1%} vs Non-Digital",
                            delta_color="normal"
                        )
                    
                    with kpi_col3:
                        st.metric(
                            "Non-Digital → FAS Rate",
                            f"{non_digital_conv:.1%}",
                            "Baseline"
                        )
                    
                    with kpi_col4:
                        lift = (digital_conv / non_digital_conv - 1) * 100 if non_digital_conv > 0 else 0
                        st.metric(
                            "Conversion Lift",
                            f"{lift:+.0f}%",
                            "Digital vs Non-Digital"
                        )
                    
                    # Breakdown row
                    kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)
                    
                    with kpi_col5:
                        digital_fas_pct = digital_fas / total_fas if total_fas > 0 else 0
                        st.metric(
                            "% of FAS from Digital",
                            f"{digital_fas_pct:.1%}",
                            f"{digital_fas:,} of {total_fas:,}"
                        )
                    
                    with kpi_col6:
                        digital_loan_amt = df_digital['digital_fas_loan_amount'].sum()
                        total_loan_amt = df_digital['fas_loan_amount'].sum()
                        st.metric(
                            "Digital FAS $ Volume",
                            f"${digital_loan_amt:,.0f}",
                            f"{digital_loan_amt/total_loan_amt:.1%} of Total" if total_loan_amt > 0 else "N/A"
                        )
                    
                    with kpi_col7:
                        avg_digital_loan = digital_loan_amt / digital_fas if digital_fas > 0 else 0
                        non_digital_loan = total_loan_amt - digital_loan_amt
                        avg_non_digital_loan = non_digital_loan / non_digital_fas if non_digital_fas > 0 else 0
                        st.metric(
                            "Avg Digital FAS Loan",
                            f"${avg_digital_loan:,.0f}",
                            f"${avg_digital_loan - avg_non_digital_loan:+,.0f} vs Non-Digital"
                        )
                    
                    with kpi_col8:
                        st.metric(
                            "Sent to Sales",
                            f"{total_sts:,}",
                            "Total Volume"
                        )
                    
                    st.divider()
                    
                    # --- BY PERSONA ---
                    st.subheader("👤 Digital Intent by Persona")
                    
                    persona_digital = df_digital.groupby('persona').agg({
                        'sent_to_sales': 'sum',
                        'digital_app_starts': 'sum',
                        'fas_count': 'sum',
                        'digital_to_fas': 'sum',
                        'non_digital_to_fas': 'sum',
                        'digital_fas_loan_amount': 'sum'
                    }).reset_index()
                    
                    persona_digital['digital_rate'] = persona_digital['digital_app_starts'] / persona_digital['sent_to_sales'].replace(0, 1)
                    persona_digital['digital_conv'] = persona_digital['digital_to_fas'] / persona_digital['digital_app_starts'].replace(0, 1)
                    persona_digital['non_digital_pool'] = persona_digital['sent_to_sales'] - persona_digital['digital_app_starts']
                    persona_digital['non_digital_conv'] = persona_digital['non_digital_to_fas'] / persona_digital['non_digital_pool'].replace(0, 1)
                    persona_digital['overall_conv'] = persona_digital['fas_count'] / persona_digital['sent_to_sales'].replace(0, 1)
                    persona_digital['lift'] = (persona_digital['digital_conv'] / persona_digital['non_digital_conv'].replace(0, 1) - 1) * 100
                    persona_digital = persona_digital.sort_values('sent_to_sales', ascending=False)
                    
                    # Chart: Digital Rate by Persona
                    col_p1, col_p2 = st.columns(2)
                    
                    with col_p1:
                        persona_chart_data = persona_digital.head(10).copy()
                        fig_persona_digital = alt.Chart(persona_chart_data).mark_bar(color='#4cc9f0').encode(
                            x=alt.X('digital_rate:Q', title='Digital App Start Rate', axis=alt.Axis(format='.0%', labelColor='white', titleColor='white')),
                            y=alt.Y('persona:N', title='Persona', sort='-x', axis=alt.Axis(labelColor='white', titleColor='white')),
                            tooltip=[
                                alt.Tooltip('persona:N', title='Persona'),
                                alt.Tooltip('digital_rate:Q', title='Digital Rate', format='.1%'),
                                alt.Tooltip('sent_to_sales:Q', title='Sent to Sales', format=',d'),
                                alt.Tooltip('digital_app_starts:Q', title='Digital Starts', format=',d')
                            ]
                        ).properties(
                            title='Digital App Start Rate by Persona',
                            height=350
                        )
                        st.altair_chart(fig_persona_digital, use_container_width=True)
                    
                    with col_p2:
                        # Conversion comparison
                        persona_melt = persona_chart_data[['persona', 'digital_conv', 'non_digital_conv']].melt(
                            id_vars=['persona'],
                            value_vars=['digital_conv', 'non_digital_conv'],
                            var_name='Type',
                            value_name='Conversion'
                        )
                        persona_melt['Type'] = persona_melt['Type'].replace({
                            'digital_conv': 'Digital Starters',
                            'non_digital_conv': 'Non-Digital'
                        })
                        
                        fig_persona_conv = alt.Chart(persona_melt).mark_bar().encode(
                            x=alt.X('persona:N', title='Persona', axis=alt.Axis(labelAngle=-45, labelColor='white', titleColor='white')),
                            y=alt.Y('Conversion:Q', title='FAS Conversion Rate', axis=alt.Axis(format='.0%', labelColor='white', titleColor='white')),
                            color=alt.Color('Type:N', scale=alt.Scale(
                                domain=['Digital Starters', 'Non-Digital'],
                                range=['#4cc9f0', '#8892b0']
                            )),
                            xOffset='Type:N',
                            tooltip=[
                                alt.Tooltip('persona:N', title='Persona'),
                                alt.Tooltip('Type:N', title='Type'),
                                alt.Tooltip('Conversion:Q', title='Conversion', format='.1%')
                            ]
                        ).properties(
                            title='Digital vs Non-Digital Conversion by Persona',
                            height=350
                        )
                        st.altair_chart(fig_persona_conv, use_container_width=True)
                    
                    # Persona Table
                    persona_display = persona_digital[['persona', 'sent_to_sales', 'digital_app_starts', 'digital_rate', 
                                                        'digital_to_fas', 'digital_conv', 'non_digital_conv', 'lift']].copy()
                    persona_display.columns = ['Persona', 'Sent to Sales', 'Digital Starts', 'Digital Rate', 
                                               'Digital FAS', 'Digital Conv', 'Non-Digital Conv', 'Lift %']
                    persona_display['Sent to Sales'] = persona_display['Sent to Sales'].apply(lambda x: f"{x:,}")
                    persona_display['Digital Starts'] = persona_display['Digital Starts'].apply(lambda x: f"{x:,}")
                    persona_display['Digital Rate'] = persona_display['Digital Rate'].apply(lambda x: f"{x:.1%}")
                    persona_display['Digital FAS'] = persona_display['Digital FAS'].apply(lambda x: f"{x:,}")
                    persona_display['Digital Conv'] = persona_display['Digital Conv'].apply(lambda x: f"{x:.1%}")
                    persona_display['Non-Digital Conv'] = persona_display['Non-Digital Conv'].apply(lambda x: f"{x:.1%}")
                    persona_display['Lift %'] = persona_display['Lift %'].apply(lambda x: f"{x:+.0f}%")
                    
                    st.dataframe(persona_display, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # --- BY FINANCE GROUP ---
                    st.subheader("💰 Digital Intent by Finance Group")
                    
                    fg_digital = df_digital.groupby('finance_group').agg({
                        'sent_to_sales': 'sum',
                        'digital_app_starts': 'sum',
                        'fas_count': 'sum',
                        'digital_to_fas': 'sum',
                        'non_digital_to_fas': 'sum',
                        'digital_fas_loan_amount': 'sum'
                    }).reset_index()
                    
                    fg_digital['digital_rate'] = fg_digital['digital_app_starts'] / fg_digital['sent_to_sales'].replace(0, 1)
                    fg_digital['digital_conv'] = fg_digital['digital_to_fas'] / fg_digital['digital_app_starts'].replace(0, 1)
                    fg_digital['non_digital_pool'] = fg_digital['sent_to_sales'] - fg_digital['digital_app_starts']
                    fg_digital['non_digital_conv'] = fg_digital['non_digital_to_fas'] / fg_digital['non_digital_pool'].replace(0, 1)
                    fg_digital['lift'] = (fg_digital['digital_conv'] / fg_digital['non_digital_conv'].replace(0, 1) - 1) * 100
                    fg_digital = fg_digital.sort_values('sent_to_sales', ascending=False)
                    
                    col_f1, col_f2 = st.columns(2)
                    
                    with col_f1:
                        fg_chart_data = fg_digital.head(10).copy()
                        fig_fg_digital = alt.Chart(fg_chart_data).mark_bar(color='#f9c74f').encode(
                            x=alt.X('digital_rate:Q', title='Digital App Start Rate', axis=alt.Axis(format='.0%', labelColor='white', titleColor='white')),
                            y=alt.Y('finance_group:N', title='Finance Group', sort='-x', axis=alt.Axis(labelColor='white', titleColor='white')),
                            tooltip=[
                                alt.Tooltip('finance_group:N', title='Finance Group'),
                                alt.Tooltip('digital_rate:Q', title='Digital Rate', format='.1%'),
                                alt.Tooltip('sent_to_sales:Q', title='Sent to Sales', format=',d')
                            ]
                        ).properties(
                            title='Digital App Start Rate by Finance Group',
                            height=350
                        )
                        st.altair_chart(fig_fg_digital, use_container_width=True)
                    
                    with col_f2:
                        # Conversion lift chart
                        fig_fg_lift = alt.Chart(fg_chart_data).mark_bar().encode(
                            x=alt.X('lift:Q', title='Conversion Lift (%)', axis=alt.Axis(labelColor='white', titleColor='white')),
                            y=alt.Y('finance_group:N', title='Finance Group', sort='-x', axis=alt.Axis(labelColor='white', titleColor='white')),
                            color=alt.condition(
                                alt.datum.lift > 0,
                                alt.value('#2ca02c'),
                                alt.value('#d62728')
                            ),
                            tooltip=[
                                alt.Tooltip('finance_group:N', title='Finance Group'),
                                alt.Tooltip('lift:Q', title='Lift %', format='+.0f'),
                                alt.Tooltip('digital_conv:Q', title='Digital Conv', format='.1%'),
                                alt.Tooltip('non_digital_conv:Q', title='Non-Digital Conv', format='.1%')
                            ]
                        ).properties(
                            title='Digital vs Non-Digital Conversion Lift by Finance Group',
                            height=350
                        )
                        st.altair_chart(fig_fg_lift, use_container_width=True)
                    
                    # Finance Group Table
                    fg_display = fg_digital[['finance_group', 'sent_to_sales', 'digital_app_starts', 'digital_rate',
                                             'digital_to_fas', 'digital_conv', 'non_digital_conv', 'lift']].copy()
                    fg_display.columns = ['Finance Group', 'Sent to Sales', 'Digital Starts', 'Digital Rate',
                                          'Digital FAS', 'Digital Conv', 'Non-Digital Conv', 'Lift %']
                    fg_display['Sent to Sales'] = fg_display['Sent to Sales'].apply(lambda x: f"{x:,}")
                    fg_display['Digital Starts'] = fg_display['Digital Starts'].apply(lambda x: f"{x:,}")
                    fg_display['Digital Rate'] = fg_display['Digital Rate'].apply(lambda x: f"{x:.1%}")
                    fg_display['Digital FAS'] = fg_display['Digital FAS'].apply(lambda x: f"{x:,}")
                    fg_display['Digital Conv'] = fg_display['Digital Conv'].apply(lambda x: f"{x:.1%}")
                    fg_display['Non-Digital Conv'] = fg_display['Non-Digital Conv'].apply(lambda x: f"{x:.1%}")
                    fg_display['Lift %'] = fg_display['Lift %'].apply(lambda x: f"{x:+.0f}%")
                    
                    st.dataframe(fg_display, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # --- BY LVC GROUP ---
                    st.subheader("📊 Digital Intent by LVC Group")
                    
                    lvc_digital = df_digital.groupby('lvc_group').agg({
                        'sent_to_sales': 'sum',
                        'digital_app_starts': 'sum',
                        'fas_count': 'sum',
                        'digital_to_fas': 'sum',
                        'non_digital_to_fas': 'sum'
                    }).reset_index()
                    
                    lvc_digital['digital_rate'] = lvc_digital['digital_app_starts'] / lvc_digital['sent_to_sales'].replace(0, 1)
                    lvc_digital['digital_conv'] = lvc_digital['digital_to_fas'] / lvc_digital['digital_app_starts'].replace(0, 1)
                    lvc_digital['non_digital_pool'] = lvc_digital['sent_to_sales'] - lvc_digital['digital_app_starts']
                    lvc_digital['non_digital_conv'] = lvc_digital['non_digital_to_fas'] / lvc_digital['non_digital_pool'].replace(0, 1)
                    lvc_digital['lift'] = (lvc_digital['digital_conv'] / lvc_digital['non_digital_conv'].replace(0, 1) - 1) * 100
                    
                    col_l1, col_l2, col_l3 = st.columns(3)
                    
                    for idx, (_, row) in enumerate(lvc_digital.iterrows()):
                        col = [col_l1, col_l2, col_l3][idx % 3]
                        with col:
                            st.markdown(f"### {row['lvc_group']}")
                            st.metric("Digital Rate", f"{row['digital_rate']:.1%}", f"{row['digital_app_starts']:,} starts")
                            st.metric("Digital → FAS", f"{row['digital_conv']:.1%}", f"{row['lift']:+.0f}% lift")
                    
                    st.divider()
                    
                    # --- WEEKLY TREND ---
                    st.subheader("📈 Digital Intent Trend Over Time")
                    
                    df_digital['week'] = df_digital['lead_date'] - pd.to_timedelta(df_digital['lead_date'].dt.weekday, unit='D')
                    weekly_digital = df_digital.groupby('week').agg({
                        'sent_to_sales': 'sum',
                        'digital_app_starts': 'sum',
                        'digital_to_fas': 'sum',
                        'non_digital_to_fas': 'sum',
                        'fas_count': 'sum'
                    }).reset_index()
                    
                    weekly_digital['digital_rate'] = weekly_digital['digital_app_starts'] / weekly_digital['sent_to_sales'].replace(0, 1)
                    weekly_digital['digital_conv'] = weekly_digital['digital_to_fas'] / weekly_digital['digital_app_starts'].replace(0, 1)
                    weekly_digital['non_digital_pool'] = weekly_digital['sent_to_sales'] - weekly_digital['digital_app_starts']
                    weekly_digital['non_digital_conv'] = weekly_digital['non_digital_to_fas'] / weekly_digital['non_digital_pool'].replace(0, 1)
                    
                    col_t1, col_t2 = st.columns(2)
                    
                    with col_t1:
                        fig_trend_rate = alt.Chart(weekly_digital).mark_line(point=True, color='#4cc9f0').encode(
                            x=alt.X('week:T', title='Week', axis=alt.Axis(format='%b %d', labelColor='white', titleColor='white')),
                            y=alt.Y('digital_rate:Q', title='Digital Start Rate', axis=alt.Axis(format='.0%', labelColor='white', titleColor='white')),
                            tooltip=[
                                alt.Tooltip('week:T', title='Week', format='%b %d, %Y'),
                                alt.Tooltip('digital_rate:Q', title='Digital Rate', format='.1%'),
                                alt.Tooltip('digital_app_starts:Q', title='Digital Starts', format=',d')
                            ]
                        ).properties(
                            title='Digital App Start Rate Over Time',
                            height=300
                        ).interactive()
                        st.altair_chart(fig_trend_rate, use_container_width=True)
                    
                    with col_t2:
                        trend_conv = weekly_digital[['week', 'digital_conv', 'non_digital_conv']].melt(
                            id_vars=['week'],
                            value_vars=['digital_conv', 'non_digital_conv'],
                            var_name='Type',
                            value_name='Conversion'
                        )
                        trend_conv['Type'] = trend_conv['Type'].replace({
                            'digital_conv': 'Digital',
                            'non_digital_conv': 'Non-Digital'
                        })
                        
                        fig_trend_conv = alt.Chart(trend_conv).mark_line(point=True).encode(
                            x=alt.X('week:T', title='Week', axis=alt.Axis(format='%b %d', labelColor='white', titleColor='white')),
                            y=alt.Y('Conversion:Q', title='FAS Conversion Rate', axis=alt.Axis(format='.0%', labelColor='white', titleColor='white')),
                            color=alt.Color('Type:N', scale=alt.Scale(
                                domain=['Digital', 'Non-Digital'],
                                range=['#4cc9f0', '#8892b0']
                            )),
                            tooltip=[
                                alt.Tooltip('week:T', title='Week', format='%b %d, %Y'),
                                alt.Tooltip('Type:N', title='Type'),
                                alt.Tooltip('Conversion:Q', title='Conversion', format='.1%')
                            ]
                        ).properties(
                            title='Digital vs Non-Digital Conversion Over Time',
                            height=300
                        ).interactive()
                        st.altair_chart(fig_trend_conv, use_container_width=True)
                    
                    st.divider()
                    
                    # --- KEY INSIGHTS ---
                    st.subheader("💡 Key Insights")
                    
                    # Find best performers
                    best_persona_digital = persona_digital.loc[persona_digital['digital_rate'].idxmax()]
                    best_persona_conv = persona_digital.loc[persona_digital['digital_conv'].idxmax()]
                    best_fg_digital = fg_digital.loc[fg_digital['digital_rate'].idxmax()]
                    best_fg_conv = fg_digital.loc[fg_digital['digital_conv'].idxmax()]
                    
                    col_i1, col_i2 = st.columns(2)
                    
                    with col_i1:
                        st.markdown("### 🏆 Top Performers")
                        st.markdown(f"""
                        **Highest Digital Start Rate (Persona):**
                        - `{best_persona_digital['persona']}` at **{best_persona_digital['digital_rate']:.1%}**
                        
                        **Best Digital Conversion (Persona):**
                        - `{best_persona_conv['persona']}` at **{best_persona_conv['digital_conv']:.1%}**
                        
                        **Highest Digital Start Rate (Finance Group):**
                        - `{best_fg_digital['finance_group']}` at **{best_fg_digital['digital_rate']:.1%}**
                        
                        **Best Digital Conversion (Finance Group):**
                        - `{best_fg_conv['finance_group']}` at **{best_fg_conv['digital_conv']:.1%}**
                        """)
                    
                    with col_i2:
                        st.markdown("### 📌 Recommendations")
                        st.markdown(f"""
                        1. **Digital starters convert at {lift:+.0f}% higher rates** - prioritize follow-up on these leads
                        
                        2. **{digital_fas_pct:.1%} of all FAS come from digital starters** - invest in digital experience
                        
                        3. **Focus on personas with high digital rates but lower conversion** - opportunity to improve
                        
                        4. **Route digital starters to best closers** - they're pre-qualified by their own intent
                        
                        5. **Analyze drop-off in digital flow** - why do some start but not complete?
                        """)
                    
                    # Additional insight
                    st.info(f"""
                    📊 **Summary:** Of {total_sts:,} leads sent to sales, {total_digital:,} ({digital_rate:.1%}) started a digital app.
                    These digital starters convert at **{digital_conv:.1%}** vs **{non_digital_conv:.1%}** for non-digital, 
                    representing a **{lift:+.0f}%** conversion lift.
                    """)
                    
                    st.divider()
                    
                    # --- PERSONA x LVC HEATMAP ---
                    st.subheader("🔥 Digital Intent: Persona × LVC Heatmap")
                    
                    # Create pivot for heatmap - Digital Start Rate
                    heatmap_data = df_digital.groupby(['persona', 'lvc_group']).agg({
                        'sent_to_sales': 'sum',
                        'digital_app_starts': 'sum',
                        'digital_to_fas': 'sum'
                    }).reset_index()
                    
                    heatmap_data['digital_rate'] = heatmap_data['digital_app_starts'] / heatmap_data['sent_to_sales'].replace(0, 1)
                    heatmap_data['digital_conv'] = heatmap_data['digital_to_fas'] / heatmap_data['digital_app_starts'].replace(0, 1)
                    
                    # Filter to top personas by volume
                    top_personas = heatmap_data.groupby('persona')['sent_to_sales'].sum().nlargest(12).index.tolist()
                    heatmap_filtered = heatmap_data[heatmap_data['persona'].isin(top_personas)]
                    
                    # Choose metric to display
                    heatmap_metric = st.radio(
                        "Heatmap Metric",
                        ["Digital Start Rate", "Digital → FAS Conversion", "Digital App Starts (Volume)"],
                        horizontal=True,
                        key="digital_heatmap_metric"
                    )
                    
                    if heatmap_metric == "Digital Start Rate":
                        heatmap_col = 'digital_rate'
                        heatmap_format = '.0%'
                        heatmap_title = 'Digital Start Rate by Persona × LVC'
                        color_scheme = 'blues'
                    elif heatmap_metric == "Digital → FAS Conversion":
                        heatmap_col = 'digital_conv'
                        heatmap_format = '.0%'
                        heatmap_title = 'Digital → FAS Conversion by Persona × LVC'
                        color_scheme = 'greens'
                    else:
                        heatmap_col = 'digital_app_starts'
                        heatmap_format = ',d'
                        heatmap_title = 'Digital App Starts Volume by Persona × LVC'
                        color_scheme = 'oranges'
                    
                    # Create label column for heatmap
                    if heatmap_metric == "Digital App Starts (Volume)":
                        heatmap_filtered['label'] = heatmap_filtered.apply(
                            lambda r: f"{int(r['digital_app_starts']):,}\n({r['digital_rate']:.0%})", axis=1
                        )
                    else:
                        heatmap_filtered['label'] = heatmap_filtered.apply(
                            lambda r: f"{r[heatmap_col]:.1%}\n({int(r['digital_app_starts']):,})", axis=1
                        )
                    
                    # Define LVC order
                    lvc_order = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']
                    
                    # Base heatmap
                    heatmap_base = alt.Chart(heatmap_filtered).encode(
                        x=alt.X('lvc_group:N', title='LVC Group', sort=lvc_order, axis=alt.Axis(labelColor='white', titleColor='white', labelAngle=0)),
                        y=alt.Y('persona:N', title='Persona', sort='-x', axis=alt.Axis(labelColor='white', titleColor='white'))
                    )
                    
                    # Heatmap rectangles
                    heatmap_rect = heatmap_base.mark_rect().encode(
                        color=alt.Color(f'{heatmap_col}:Q', 
                                       title=heatmap_metric,
                                       scale=alt.Scale(scheme=color_scheme)),
                        tooltip=[
                            alt.Tooltip('persona:N', title='Persona'),
                            alt.Tooltip('lvc_group:N', title='LVC Group'),
                            alt.Tooltip('digital_app_starts:Q', title='Digital Starts', format=',d'),
                            alt.Tooltip('digital_rate:Q', title='Digital Rate', format='.1%'),
                            alt.Tooltip('digital_conv:Q', title='Digital→FAS', format='.1%'),
                            alt.Tooltip('sent_to_sales:Q', title='Sent to Sales', format=',d')
                        ]
                    )
                    
                    # Text labels
                    heatmap_text = heatmap_base.mark_text(fontSize=10, color='white').encode(
                        text='label:N'
                    )
                    
                    fig_heatmap = (heatmap_rect + heatmap_text).properties(
                        title=heatmap_title,
                        height=450
                    )
                    
                    st.altair_chart(fig_heatmap, use_container_width=True)
                    
                    # Summary table for heatmap
                    with st.expander("View Persona × LVC Breakdown Table"):
                        pivot_display = heatmap_filtered.pivot_table(
                            index='persona',
                            columns='lvc_group',
                            values=['digital_app_starts', 'digital_rate', 'digital_conv'],
                            aggfunc='first'
                        ).round(3)
                        st.dataframe(pivot_display, use_container_width=True)
                    
                    with st.expander("View Raw Data"):
                        st.dataframe(df_digital, use_container_width=True)
                
                else:
                    st.warning("No digital intent data found for the analysis period.")
        
        except Exception as e:
            st.error(f"Digital Intent Analysis Error: {e}")
            st.exception(e)

    # --- TAB: MA Call Metrics Analysis ---
    with tab_call_metrics:
        st.header("📞 MA Call Metrics Analysis")
        st.caption("Understanding Mortgage Advisor call behavior changes pre/post selected date (default: 1/27/2026)")
        st.info("📅 **Note:** Speed-to-lead metrics are filtered to leads sent to sales during business hours (7:30 AM - 4:30 PM) only.")
        
        st.markdown("""
        **Context:** We use a dynamic comparison date to see how MAs adjusted to process changes:
        - **Before:** Prior to the selected date
        - **After:** On or after the selected date
        
        **Key Questions:**
        1. Are MAs making as many call attempts as before?
        2. Are they calling leads as quickly after assignment?
        3. What changed in call patterns post-transition?
        """)
        
        # --- FILTERS ---
        st.subheader("🔍 Filters")
        
        # First, get distinct values for MA and LVC filters
        call_filter_options_query = """
        SELECT DISTINCT 
            mortgage_advisor,
            adjusted_lead_value_cohort
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE sent_to_sales_date >= '2024-01-01'
        AND mortgage_advisor IS NOT NULL
        AND mortgage_advisor IN ('Chris Figueroa', 'Cody Longfield', 'Cody Sorells', 'James Cook', 'Jon Wolff', 'Robert Hart', 'Ryan Johnston', 'Sean McKinney', 'Todd Raccuglia', 'Aimee Johnson', 'Anthony Lawrence', 'Cheryl Lee', 'Christian Elza', 'Corey Henson', 'David Kim', 'Elizabeth Graney', 'Hugo Mejia Romo', 'Jonathan Niesen', 'Joshua Durant', 'Justin Howard', 'Kevin Campbell', 'Paul Weir', 'Raul Flores', 'Steven Droege', 'Anthony Payne', 'Jason Cook', 'Jeremiah Johnson', 'Joseph Freedman', 'Justin Brooks', 'Nic Reed', 'Richard Miller', 'Tanner Zimmerman', 'Allen Hong', 'Ardis Palmer', 'Edgar Acuna', 'Rebecca Monelll', 'Jacob Armstrong', 'Misael Avila')
        """
        
        try:
            call_filter_df = client.query(call_filter_options_query).to_dataframe()
            ma_options = sorted(call_filter_df['mortgage_advisor'].dropna().unique().tolist())
            call_lvc_options = sorted(call_filter_df['adjusted_lead_value_cohort'].dropna().unique().tolist())
        except:
            ma_options = []
            call_lvc_options = [str(i) for i in range(1, 11)] + ['X']
        
        call_filter_col1, call_filter_col2, call_filter_col3 = st.columns(3)
        
        with call_filter_col1:
            call_date_range = st.date_input(
                "Date Range (Sent to Sales Date)",
                value=(date(2025, 11, 1), date(2026, 3, 6)),
                min_value=date(2024, 1, 1),
                max_value=date.today(),
                key="call_metrics_date_range"
            )
            
            if isinstance(call_date_range, tuple) and len(call_date_range) == 2:
                call_start_date = call_date_range[0].strftime('%Y-%m-%d')
                call_end_date = call_date_range[1].strftime('%Y-%m-%d')
            else:
                call_start_date = call_date_range.strftime('%Y-%m-%d') if call_date_range else '2025-11-01'
                call_end_date = call_start_date
            
            comparison_date = st.date_input(
                "Comparison Date (Pre/Post Split)",
                value=date(2026, 1, 27),
                key="call_comparison_date"
            )
            comparison_date_str = comparison_date.strftime('%Y-%m-%d')
            comp_label = comparison_date.strftime('%-m/%-d')
        
        with call_filter_col2:
            selected_mas = st.multiselect(
                "Mortgage Advisors",
                options=ma_options,
                default=[],
                placeholder="All MAs",
                key="call_ma_filter"
            )
        
        with call_filter_col3:
            selected_call_lvc = st.multiselect(
                "Lead Value Cohort",
                options=call_lvc_options,
                default=[],
                placeholder="All LVC Values",
                key="call_lvc_filter"
            )
        
        # Build additional WHERE conditions
        call_extra_conditions = []
        if selected_mas:
            ma_list = ", ".join([f"'{m}'" for m in selected_mas])
            call_extra_conditions.append(f"mortgage_advisor IN ({ma_list})")
        if selected_call_lvc:
            lvc_list = ", ".join([f"'{l}'" for l in selected_call_lvc])
            call_extra_conditions.append(f"adjusted_lead_value_cohort IN ({lvc_list})")
        
        call_extra_where = " AND " + " AND ".join(call_extra_conditions) if call_extra_conditions else ""
        
        st.divider()
        
        # Query for MA call metrics
        call_query = f"""
        WITH call_data AS (
            SELECT 
                lendage_guid,
                DATE(sent_to_sales_date) as sts_date,
                mortgage_advisor,
                call_attempts,
                first_call_attempt_datetime,
                sf__contacted_guid,
                full_app_submit_datetime,
                initial_sales_assigned_datetime,
                CASE 
                    WHEN DATE(sent_to_sales_date) < '{comparison_date_str}' THEN 'Pre'
                    ELSE 'Post'
                END as period,
                TIMESTAMP_DIFF(first_call_attempt_datetime, initial_sales_assigned_datetime, MINUTE) as minutes_to_first_call,
                TIMESTAMP_DIFF(first_call_attempt_datetime, initial_sales_assigned_datetime, HOUR) as hours_to_first_call
            FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
            WHERE DATE(sent_to_sales_date) BETWEEN '{call_start_date}' AND '{call_end_date}'
            AND sent_to_sales_date IS NOT NULL
            AND mortgage_advisor IS NOT NULL
            AND mortgage_advisor IN ('Chris Figueroa', 'Cody Longfield', 'Cody Sorells', 'James Cook', 'Jon Wolff', 'Robert Hart', 'Ryan Johnston', 'Sean McKinney', 'Todd Raccuglia', 'Aimee Johnson', 'Anthony Lawrence', 'Cheryl Lee', 'Christian Elza', 'Corey Henson', 'David Kim', 'Elizabeth Graney', 'Hugo Mejia Romo', 'Jonathan Niesen', 'Joshua Durant', 'Justin Howard', 'Kevin Campbell', 'Paul Weir', 'Raul Flores', 'Steven Droege', 'Anthony Payne', 'Jason Cook', 'Jeremiah Johnson', 'Joseph Freedman', 'Justin Brooks', 'Nic Reed', 'Richard Miller', 'Tanner Zimmerman', 'Allen Hong', 'Ardis Palmer', 'Edgar Acuna', 'Rebecca Monelll', 'Jacob Armstrong', 'Misael Avila')
            AND TIME(sent_to_sales_datetime) BETWEEN '07:30:00' AND '16:30:00'
            {call_extra_where}
        )
        SELECT 
            sts_date,
            period,
            mortgage_advisor as ma_name,
            COUNT(DISTINCT lendage_guid) as total_leads,
            SUM(COALESCE(call_attempts, 0)) as total_call_attempts,
            AVG(COALESCE(call_attempts, 0)) as avg_call_attempts,
            COUNT(DISTINCT CASE WHEN first_call_attempt_datetime IS NOT NULL THEN lendage_guid END) as leads_with_call,
            COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted,
            COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
            AVG(CASE WHEN minutes_to_first_call > 0 AND minutes_to_first_call < 10080 THEN minutes_to_first_call END) as avg_minutes_to_first_call,
            APPROX_QUANTILES(CASE WHEN minutes_to_first_call > 0 AND minutes_to_first_call < 10080 THEN minutes_to_first_call END, 100)[OFFSET(50)] as median_minutes_to_first_call,
            COUNT(DISTINCT CASE WHEN minutes_to_first_call <= 5 THEN lendage_guid END) as called_within_5min,
            COUNT(DISTINCT CASE WHEN minutes_to_first_call <= 15 THEN lendage_guid END) as called_within_15min,
            COUNT(DISTINCT CASE WHEN minutes_to_first_call <= 60 THEN lendage_guid END) as called_within_1hr
        FROM call_data
        GROUP BY 1, 2, 3
        ORDER BY sts_date, ma_name
        """
        
        with st.expander("View Call Metrics SQL"):
            st.code(call_query)
        
        try:
            with st.spinner("Fetching MA Call Metrics Data..."):
                df_calls = client.query(call_query).to_dataframe()
                
                if not df_calls.empty:
                    df_calls['sts_date'] = pd.to_datetime(df_calls['sts_date'])
                    
                    # === OVERALL PRE/POST COMPARISON ===
                    st.subheader(f"📊 Pre/Post {comp_label} Comparison - Overall")
                    
                    period_summary = df_calls.groupby('period').agg({
                        'total_leads': 'sum',
                        'total_call_attempts': 'sum',
                        'leads_with_call': 'sum',
                        'contacted': 'sum',
                        'fas_count': 'sum',
                        'called_within_5min': 'sum',
                        'called_within_15min': 'sum',
                        'called_within_1hr': 'sum'
                    }).reset_index()
                    
                    # Calculate weighted averages for time to first call
                    for period in ['Pre', 'Post']:
                        period_data = df_calls[df_calls['period'] == period]
                        weighted_avg = (period_data['avg_minutes_to_first_call'] * period_data['leads_with_call']).sum() / period_data['leads_with_call'].sum() if period_data['leads_with_call'].sum() > 0 else 0
                        period_summary.loc[period_summary['period'] == period, 'avg_minutes_to_first_call'] = weighted_avg
                    
                    period_summary['avg_call_attempts'] = period_summary['total_call_attempts'] / period_summary['total_leads']
                    period_summary['call_rate'] = period_summary['leads_with_call'] / period_summary['total_leads']
                    period_summary['contact_rate'] = period_summary['contacted'] / period_summary['total_leads']
                    period_summary['fas_rate'] = period_summary['fas_count'] / period_summary['total_leads']
                    period_summary['speed_5min_rate'] = period_summary['called_within_5min'] / period_summary['leads_with_call']
                    period_summary['speed_15min_rate'] = period_summary['called_within_15min'] / period_summary['leads_with_call']
                    period_summary['speed_1hr_rate'] = period_summary['called_within_1hr'] / period_summary['leads_with_call']
                    
                    # Get pre and post values
                    pre_row = period_summary[period_summary['period'] == 'Pre'].iloc[0] if len(period_summary[period_summary['period'] == 'Pre']) > 0 else None
                    post_row = period_summary[period_summary['period'] == 'Post'].iloc[0] if len(period_summary[period_summary['period'] == 'Post']) > 0 else None
                    
                    if pre_row is not None and post_row is not None:
                        # KPI Cards Row 1 - Call Volume
                        st.markdown("### 📞 Call Volume Metrics")
                        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                        
                        with kpi_col1:
                            delta_attempts = post_row['avg_call_attempts'] - pre_row['avg_call_attempts']
                            st.metric(
                                "Avg Call Attempts/Lead",
                                f"{post_row['avg_call_attempts']:.2f}",
                                f"{delta_attempts:+.2f} vs Pre",
                                delta_color="normal"
                            )
                            st.caption(f"Pre: {pre_row['avg_call_attempts']:.2f}")
                            st.caption("📐 `SUM(call_attempts) / COUNT(lendage_guid)`")
                        
                        with kpi_col2:
                            delta_call_rate = (post_row['call_rate'] - pre_row['call_rate']) * 100
                            st.metric(
                                "% Leads Called",
                                f"{post_row['call_rate']:.1%}",
                                f"{delta_call_rate:+.1f}pp vs Pre",
                                delta_color="normal"
                            )
                            st.caption(f"Pre: {pre_row['call_rate']:.1%}")
                            st.caption("📐 `COUNT(first_call_attempt_datetime NOT NULL) / COUNT(lendage_guid)`")
                        
                        with kpi_col3:
                            delta_contact = (post_row['contact_rate'] - pre_row['contact_rate']) * 100
                            st.metric(
                                "Contact Rate",
                                f"{post_row['contact_rate']:.1%}",
                                f"{delta_contact:+.1f}pp vs Pre",
                                delta_color="normal"
                            )
                            st.caption(f"Pre: {pre_row['contact_rate']:.1%}")
                            st.caption("📐 `COUNT(sf__contacted_guid NOT NULL) / COUNT(lendage_guid)`")
                        
                        with kpi_col4:
                            delta_fas = (post_row['fas_rate'] - pre_row['fas_rate']) * 100
                            st.metric(
                                "FAS Rate",
                                f"{post_row['fas_rate']:.1%}",
                                f"{delta_fas:+.1f}pp vs Pre",
                                delta_color="normal"
                            )
                            st.caption(f"Pre: {pre_row['fas_rate']:.1%}")
                            st.caption("📐 `COUNT(full_app_submit_datetime NOT NULL) / COUNT(lendage_guid)`")
                        
                        # KPI Cards Row 2 - Speed to Call
                        st.markdown("### ⏱️ Speed to First Call")
                        kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)
                        
                        with kpi_col5:
                            delta_time = post_row['avg_minutes_to_first_call'] - pre_row['avg_minutes_to_first_call']
                            st.metric(
                                "Avg Minutes to 1st Call",
                                f"{post_row['avg_minutes_to_first_call']:.0f}",
                                f"{delta_time:+.0f} min vs Pre",
                                delta_color="inverse"  # Lower is better
                            )
                            st.caption(f"Pre: {pre_row['avg_minutes_to_first_call']:.0f} min")
                            st.caption("📐 `AVG(TIMESTAMP_DIFF(first_call_attempt_datetime, initial_sales_assigned_datetime, MINUTE))`")
                        
                        with kpi_col6:
                            delta_5min = (post_row['speed_5min_rate'] - pre_row['speed_5min_rate']) * 100
                            st.metric(
                                "Called within 5 min",
                                f"{post_row['speed_5min_rate']:.1%}",
                                f"{delta_5min:+.1f}pp vs Pre",
                                delta_color="normal"
                            )
                            st.caption(f"Pre: {pre_row['speed_5min_rate']:.1%}")
                            st.caption("📐 `COUNT(minutes_to_first_call ≤ 5) / COUNT(leads_with_call)`")
                        
                        with kpi_col7:
                            delta_15min = (post_row['speed_15min_rate'] - pre_row['speed_15min_rate']) * 100
                            st.metric(
                                "Called within 15 min",
                                f"{post_row['speed_15min_rate']:.1%}",
                                f"{delta_15min:+.1f}pp vs Pre",
                                delta_color="normal"
                            )
                            st.caption(f"Pre: {pre_row['speed_15min_rate']:.1%}")
                            st.caption("📐 `COUNT(minutes_to_first_call ≤ 15) / COUNT(leads_with_call)`")
                        
                        with kpi_col8:
                            delta_1hr = (post_row['speed_1hr_rate'] - pre_row['speed_1hr_rate']) * 100
                            st.metric(
                                "Called within 1 hour",
                                f"{post_row['speed_1hr_rate']:.1%}",
                                f"{delta_1hr:+.1f}pp vs Pre",
                                delta_color="normal"
                            )
                            st.caption(f"Pre: {pre_row['speed_1hr_rate']:.1%}")
                            st.caption("📐 `COUNT(minutes_to_first_call ≤ 60) / COUNT(leads_with_call)`")
                    
                    st.divider()

                    # === TREND CHARTS ===
                    st.subheader("📈 Call Metrics Trends Over Time")
                    
                    # Weekly aggregation for trends
                    df_calls['week'] = df_calls['sts_date'] - pd.to_timedelta(df_calls['sts_date'].dt.weekday, unit='D')
                    weekly_calls = df_calls.groupby('week').agg({
                        'total_leads': 'sum',
                        'total_call_attempts': 'sum',
                        'leads_with_call': 'sum',
                        'contacted': 'sum',
                        'fas_count': 'sum',
                        'called_within_5min': 'sum',
                        'called_within_15min': 'sum',
                        'called_within_1hr': 'sum'
                    }).reset_index()
                    
                    weekly_calls['avg_call_attempts'] = weekly_calls['total_call_attempts'] / weekly_calls['total_leads']
                    weekly_calls['call_rate'] = weekly_calls['leads_with_call'] / weekly_calls['total_leads']
                    weekly_calls['contact_rate'] = weekly_calls['contacted'] / weekly_calls['total_leads']
                    weekly_calls['speed_5min_rate'] = weekly_calls['called_within_5min'] / weekly_calls['leads_with_call'].replace(0, 1)
                    
                    # Reference line date
                    ref_date = pd.Timestamp(comparison_date_str)
                    
                    col_t1, col_t2 = st.columns(2)
                    
                    with col_t1:
                        # Avg Call Attempts Trend
                        base_chart = alt.Chart(weekly_calls).mark_line(point=True, color='#4cc9f0').encode(
                            x=alt.X('week:T', title='Week (by Sent to Sales Date)', axis=alt.Axis(format='%b %d')),
                            y=alt.Y('avg_call_attempts:Q', title='Avg Call Attempts'),
                            tooltip=[
                                alt.Tooltip('week:T', title='Week', format='%b %d, %Y'),
                                alt.Tooltip('avg_call_attempts:Q', title='Avg Attempts', format='.2f'),
                                alt.Tooltip('total_leads:Q', title='Total Leads', format=',d')
                            ]
                        )
                        text_labels = base_chart.mark_text(align='center', baseline='bottom', dy=-10, fontSize=11).encode(text=alt.Text('avg_call_attempts:Q', format='.2f'))
                        
                        ref_line = alt.Chart(pd.DataFrame({'x': [ref_date]})).mark_rule(color='#ff6b6b', strokeWidth=2, strokeDash=[5,5]).encode(x='x:T')
                        ref_text = alt.Chart(pd.DataFrame({'x': [ref_date], 'text': [comp_label]})).mark_text(
                            align='left', dx=5, dy=-10, color='#ff6b6b', fontSize=12
                        ).encode(x='x:T', text='text:N')
                        
                        fig_attempts = (base_chart + text_labels + ref_line + ref_text).properties(
                            title='Average Call Attempts per Lead',
                            height=300
                        ).interactive()
                        st.altair_chart(fig_attempts, use_container_width=True)
                        st.caption("**Field:** `call_attempts` | **Calc:** SUM(call_attempts) / COUNT(DISTINCT lendage_guid)")
                    
                    with col_t2:
                        # Speed to Call Trend (% within 5 min)
                        base_chart2 = alt.Chart(weekly_calls).mark_line(point=True, color='#2ca02c').encode(
                            x=alt.X('week:T', title='Week (by Sent to Sales Date)', axis=alt.Axis(format='%b %d')),
                            y=alt.Y('speed_5min_rate:Q', title='% Called within 5 min', axis=alt.Axis(format='.0%')),
                            tooltip=[
                                alt.Tooltip('week:T', title='Week', format='%b %d, %Y'),
                                alt.Tooltip('speed_5min_rate:Q', title='% within 5 min', format='.1%'),
                                alt.Tooltip('called_within_5min:Q', title='Count', format=',d')
                            ]
                        )
                        text_labels2 = base_chart2.mark_text(align='center', baseline='bottom', dy=-10, fontSize=11).encode(text=alt.Text('speed_5min_rate:Q', format='.0%'))
                        
                        fig_speed = (base_chart2 + text_labels2 + ref_line + ref_text).properties(
                            title='Speed to First Call (% within 5 minutes)',
                            height=300
                        ).interactive()
                        st.altair_chart(fig_speed, use_container_width=True)
                        st.caption("**Fields:** `first_call_attempt_datetime`, `initial_sales_assigned_datetime` | **Calc:** COUNT(leads where TIMESTAMP_DIFF ≤ 5 min) / COUNT(leads with call) | **Note:** Filtered to leads sent to sales 7:30AM-4:30PM")
                    
                    col_t3, col_t4 = st.columns(2)
                    
                    with col_t3:
                        # Contact Rate Trend
                        base_chart3 = alt.Chart(weekly_calls).mark_line(point=True, color='#f9c74f').encode(
                            x=alt.X('week:T', title='Week (by Sent to Sales Date)', axis=alt.Axis(format='%b %d')),
                            y=alt.Y('contact_rate:Q', title='Contact Rate', axis=alt.Axis(format='.0%')),
                            tooltip=[
                                alt.Tooltip('week:T', title='Week', format='%b %d, %Y'),
                                alt.Tooltip('contact_rate:Q', title='Contact Rate', format='.1%'),
                                alt.Tooltip('contacted:Q', title='Contacted', format=',d')
                            ]
                        )
                        text_labels3 = base_chart3.mark_text(align='center', baseline='bottom', dy=-10, fontSize=11).encode(text=alt.Text('contact_rate:Q', format='.0%'))
                        
                        fig_contact = (base_chart3 + text_labels3 + ref_line + ref_text).properties(
                            title='Contact Rate Over Time',
                            height=300
                        ).interactive()
                        st.altair_chart(fig_contact, use_container_width=True)
                        st.caption("**Field:** `sf__contacted_guid` | **Calc:** COUNT(DISTINCT sf__contacted_guid IS NOT NULL) / COUNT(DISTINCT lendage_guid)")
                    
                    with col_t4:
                        # % Leads Called Trend
                        base_chart4 = alt.Chart(weekly_calls).mark_line(point=True, color='#9d4edd').encode(
                            x=alt.X('week:T', title='Week (by Sent to Sales Date)', axis=alt.Axis(format='%b %d')),
                            y=alt.Y('call_rate:Q', title='% Leads Called', axis=alt.Axis(format='.0%')),
                            tooltip=[
                                alt.Tooltip('week:T', title='Week', format='%b %d, %Y'),
                                alt.Tooltip('call_rate:Q', title='% Called', format='.1%'),
                                alt.Tooltip('leads_with_call:Q', title='Leads Called', format=',d')
                            ]
                        )
                        text_labels4 = base_chart4.mark_text(align='center', baseline='bottom', dy=-10, fontSize=11).encode(text=alt.Text('call_rate:Q', format='.0%'))
                        
                        fig_called = (base_chart4 + text_labels4 + ref_line + ref_text).properties(
                            title='% of Leads That Received a Call',
                            height=300
                        ).interactive()
                        st.altair_chart(fig_called, use_container_width=True)
                        st.caption("**Field:** `first_call_attempt_datetime` | **Calc:** COUNT(first_call_attempt_datetime IS NOT NULL) / COUNT(DISTINCT lendage_guid)")
                    
                    st.divider()
                    
                    # === MA-LEVEL ANALYSIS ===
                    st.subheader("👤 MA-Level Performance Comparison")
                    
                    # Aggregate by MA and Period
                    ma_comparison = df_calls.groupby(['ma_name', 'period']).agg({
                        'total_leads': 'sum',
                        'total_call_attempts': 'sum',
                        'leads_with_call': 'sum',
                        'contacted': 'sum',
                        'fas_count': 'sum',
                        'called_within_5min': 'sum'
                    }).reset_index()
                    
                    ma_comparison['avg_call_attempts'] = ma_comparison['total_call_attempts'] / ma_comparison['total_leads']
                    ma_comparison['contact_rate'] = ma_comparison['contacted'] / ma_comparison['total_leads']
                    ma_comparison['speed_5min_rate'] = ma_comparison['called_within_5min'] / ma_comparison['leads_with_call'].replace(0, 1)
                    
                    # Pivot for comparison
                    ma_pivot = ma_comparison.pivot(index='ma_name', columns='period', values=['total_leads', 'avg_call_attempts', 'contact_rate', 'speed_5min_rate'])
                    ma_pivot.columns = ['_'.join(col).strip() for col in ma_pivot.columns.values]
                    ma_pivot = ma_pivot.reset_index()
                    
                    # Filter to MAs with significant volume in both periods
                    ma_pivot = ma_pivot[
                        (ma_pivot.get('total_leads_Pre', 0) >= 20) & 
                        (ma_pivot.get('total_leads_Post', 0) >= 20)
                    ]
                    
                    if not ma_pivot.empty and 'avg_call_attempts_Pre' in ma_pivot.columns and 'avg_call_attempts_Post' in ma_pivot.columns:
                        # Convert columns to numeric to avoid dtype issues
                        for col in ma_pivot.columns:
                            if col != 'ma_name':
                                ma_pivot[col] = pd.to_numeric(ma_pivot[col], errors='coerce')
                        
                        ma_pivot['attempts_change'] = ma_pivot['avg_call_attempts_Post'] - ma_pivot['avg_call_attempts_Pre']
                        ma_pivot['contact_change'] = (ma_pivot.get('contact_rate_Post', 0) - ma_pivot.get('contact_rate_Pre', 0)) * 100
                        ma_pivot['speed_change'] = (ma_pivot.get('speed_5min_rate_Post', 0) - ma_pivot.get('speed_5min_rate_Pre', 0)) * 100
                        
                        # Drop rows with NaN in key columns
                        ma_pivot = ma_pivot.dropna(subset=['attempts_change', 'speed_change'])
                        
                        # Top/Bottom performers
                        col_ma1, col_ma2 = st.columns(2)
                        
                        with col_ma1:
                            st.markdown("### 📉 Biggest Drop in Call Attempts")
                            worst_attempts = ma_pivot.nsmallest(10, 'attempts_change')[['ma_name', 'avg_call_attempts_Pre', 'avg_call_attempts_Post', 'attempts_change', 'total_leads_Post']]
                            worst_attempts.columns = ['MA Name', 'Pre Avg Attempts', 'Post Avg Attempts', 'Change', 'Post Volume']
                            worst_attempts['Pre Avg Attempts'] = worst_attempts['Pre Avg Attempts'].apply(lambda x: f"{x:.2f}")
                            worst_attempts['Post Avg Attempts'] = worst_attempts['Post Avg Attempts'].apply(lambda x: f"{x:.2f}")
                            worst_attempts['Change'] = worst_attempts['Change'].apply(lambda x: f"{x:+.2f}")
                            worst_attempts['Post Volume'] = worst_attempts['Post Volume'].apply(lambda x: f"{x:,.0f}")
                            st.dataframe(worst_attempts, use_container_width=True, hide_index=True)
                        
                        with col_ma2:
                            st.markdown("### 📈 Biggest Improvement in Call Attempts")
                            best_attempts = ma_pivot.nlargest(10, 'attempts_change')[['ma_name', 'avg_call_attempts_Pre', 'avg_call_attempts_Post', 'attempts_change', 'total_leads_Post']]
                            best_attempts.columns = ['MA Name', 'Pre Avg Attempts', 'Post Avg Attempts', 'Change', 'Post Volume']
                            best_attempts['Pre Avg Attempts'] = best_attempts['Pre Avg Attempts'].apply(lambda x: f"{x:.2f}")
                            best_attempts['Post Avg Attempts'] = best_attempts['Post Avg Attempts'].apply(lambda x: f"{x:.2f}")
                            best_attempts['Change'] = best_attempts['Change'].apply(lambda x: f"{x:+.2f}")
                            best_attempts['Post Volume'] = best_attempts['Post Volume'].apply(lambda x: f"{x:,.0f}")
                            st.dataframe(best_attempts, use_container_width=True, hide_index=True)
                        
                        col_ma3, col_ma4 = st.columns(2)
                        
                        with col_ma3:
                            st.markdown("### 🐌 Biggest Slowdown in Speed (5 min)")
                            worst_speed = ma_pivot.nsmallest(10, 'speed_change')[['ma_name', 'speed_5min_rate_Pre', 'speed_5min_rate_Post', 'speed_change']]
                            worst_speed.columns = ['MA Name', 'Pre % ≤5min', 'Post % ≤5min', 'Change (pp)']
                            worst_speed['Pre % ≤5min'] = worst_speed['Pre % ≤5min'].apply(lambda x: f"{x:.1%}")
                            worst_speed['Post % ≤5min'] = worst_speed['Post % ≤5min'].apply(lambda x: f"{x:.1%}")
                            worst_speed['Change (pp)'] = worst_speed['Change (pp)'].apply(lambda x: f"{x:+.1f}")
                            st.dataframe(worst_speed, use_container_width=True, hide_index=True)
                        
                        with col_ma4:
                            st.markdown("### 🚀 Biggest Speed Improvement (5 min)")
                            best_speed = ma_pivot.nlargest(10, 'speed_change')[['ma_name', 'speed_5min_rate_Pre', 'speed_5min_rate_Post', 'speed_change']]
                            best_speed.columns = ['MA Name', 'Pre % ≤5min', 'Post % ≤5min', 'Change (pp)']
                            best_speed['Pre % ≤5min'] = best_speed['Pre % ≤5min'].apply(lambda x: f"{x:.1%}")
                            best_speed['Post % ≤5min'] = best_speed['Post % ≤5min'].apply(lambda x: f"{x:.1%}")
                            best_speed['Change (pp)'] = best_speed['Change (pp)'].apply(lambda x: f"{x:+.1f}")
                            st.dataframe(best_speed, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # === DISTRIBUTION ANALYSIS ===
                    st.subheader("📊 Call Attempts Distribution - Pre vs Post")
                    
                    # Histogram of avg call attempts by period
                    ma_hist_data = ma_comparison[['ma_name', 'period', 'avg_call_attempts', 'total_leads']].copy()
                    
                    hist_bars = alt.Chart(ma_hist_data).mark_bar(opacity=0.7).encode(
                        x=alt.X('avg_call_attempts:Q', bin=alt.Bin(maxbins=20), title='Avg Call Attempts per Lead'),
                        y=alt.Y('count():Q', title='Number of MAs'),
                        color=alt.Color('period:N', scale=alt.Scale(
                            domain=['Pre', 'Post'],
                            range=['#4cc9f0', '#ff6b6b']
                        )),
                        tooltip=[
                            alt.Tooltip('period:N', title='Period'),
                            alt.Tooltip('count():Q', title='# MAs')
                        ]
                    )
                    hist_text = hist_bars.mark_text(align='center', baseline='bottom', dy=-5, fontSize=11).encode(
                        text=alt.Text('count():Q', format='d')
                    )
                    
                    fig_hist = (hist_bars + hist_text).properties(
                        title='Distribution of MA Call Attempt Rates',
                        height=300
                    )
                    st.altair_chart(fig_hist, use_container_width=True)
                    
                    st.divider()
                    
                    # === KEY INSIGHTS ===
                    st.subheader("💡 Key Insights & Diagnosis")
                    
                    if pre_row is not None and post_row is not None:
                        col_ins1, col_ins2 = st.columns(2)
                        
                        with col_ins1:
                            st.markdown("### 🔍 What Changed?")
                            
                            insights = []
                            
                            # Call attempts change
                            if delta_attempts < -0.5:
                                insights.append(f"⚠️ **Call attempts dropped by {abs(delta_attempts):.2f}** per lead. The reactive dialer model may be reducing MA initiative to make additional calls.")
                            elif delta_attempts > 0.5:
                                insights.append(f"✅ **Call attempts increased by {delta_attempts:.2f}** per lead. MAs are making more follow-up calls.")
                            else:
                                insights.append(f"➡️ Call attempts remained relatively stable ({delta_attempts:+.2f} change).")
                            
                            # Speed change
                            if delta_5min < -5:
                                insights.append(f"⚠️ **Speed to call slowed significantly** - {abs(delta_5min):.1f}pp fewer leads called within 5 minutes. Queue-based routing may be adding delays.")
                            elif delta_5min > 5:
                                insights.append(f"✅ **Speed to call improved** - {delta_5min:.1f}pp more leads called within 5 minutes.")
                            else:
                                insights.append(f"➡️ Speed to first call remained relatively stable ({delta_5min:+.1f}pp change).")
                            
                            # Contact rate change
                            if delta_contact < -2:
                                insights.append(f"⚠️ **Contact rate dropped by {abs(delta_contact):.1f}pp**. Fewer leads are being successfully reached.")
                            elif delta_contact > 2:
                                insights.append(f"✅ **Contact rate improved by {delta_contact:.1f}pp**. More leads are being successfully reached.")
                            
                            # FAS rate change
                            if delta_fas < -1:
                                insights.append(f"⚠️ **FAS rate dropped by {abs(delta_fas):.1f}pp**. Conversion may be suffering from call behavior changes.")
                            elif delta_fas > 1:
                                insights.append(f"✅ **FAS rate improved by {delta_fas:.1f}pp**. Conversion is up despite changes.")
                            
                            for insight in insights:
                                st.markdown(insight)
                        
                        with col_ins2:
                            st.markdown("### 📋 Recommendations")
                            
                            st.markdown(f"""
                            1. **If call attempts dropped:** 
                               - Consider blending manual dial capability with queue
                               - Set minimum call attempt targets per lead
                               - Track and incentivize call attempt metrics
                            
                            2. **If speed slowed:**
                               - Review queue routing priority logic
                               - Consider "hot lead" bypass for high-intent leads
                               - Analyze queue wait times
                            
                            3. **Monitor MA behavior variance:**
                               - Some MAs adapted well, others didn't
                               - Best practices sharing from top performers
                               - Coaching for MAs with biggest drops
                            
                            4. **Data to track going forward:**
                               - Queue wait time distribution
                               - Calls per hour per MA
                               - First call connect rate
                            """)
                    
                    # Summary box
                    st.info(f"""
                    📊 **Summary:** Comparing Pre vs Post {comp_label}:
                    - **Call Attempts:** {pre_row['avg_call_attempts']:.2f} → {post_row['avg_call_attempts']:.2f} ({delta_attempts:+.2f})
                    - **Speed (≤5 min):** {pre_row['speed_5min_rate']:.1%} → {post_row['speed_5min_rate']:.1%} ({delta_5min:+.1f}pp)
                    - **Contact Rate:** {pre_row['contact_rate']:.1%} → {post_row['contact_rate']:.1%} ({delta_contact:+.1f}pp)
                    - **FAS Rate:** {pre_row['fas_rate']:.1%} → {post_row['fas_rate']:.1%} ({delta_fas:+.1f}pp)
                    """)
                    
                    with st.expander("View Raw MA Data"):
                        st.dataframe(df_calls, use_container_width=True)
                
                else:
                    st.warning("No call metrics data found for the selected period.")
        
        except Exception as e:
            st.error(f"MA Call Metrics Error: {e}")
            st.exception(e)
        
        # === MA vs PHX CALL ANALYSIS ===
        st.divider()
        st.subheader("🔄 MA vs Phoenix (PHX) Call Analysis")
        st.caption("Understanding the relationship between MA calls and Phoenix transfers")
        
        phx_query = f"""
        WITH phx_data AS (
            SELECT 
                lendage_guid,
                DATE(sent_to_sales_date) as sts_date,
                mortgage_advisor,
                first_call_attempt_datetime,
                first_dial_phx,
                phoenix_transfer_flag,
                CASE 
                    WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
                    WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
                    WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
                    WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
                    ELSE 'Other'
                END as lvc_group,
                CASE 
                    WHEN DATE(sent_to_sales_date) < '{comparison_date_str}' THEN 'Pre'
                    ELSE 'Post'
                END as period,
                -- Time analysis
                TIMESTAMP_DIFF(first_call_attempt_datetime, initial_sales_assigned_datetime, MINUTE) as ma_minutes_to_call,
                TIMESTAMP_DIFF(first_dial_phx, initial_sales_assigned_datetime, MINUTE) as phx_minutes_to_dial,
                -- Which came first?
                CASE 
                    WHEN first_call_attempt_datetime IS NOT NULL AND first_dial_phx IS NOT NULL 
                    THEN TIMESTAMP_DIFF(first_call_attempt_datetime, first_dial_phx, MINUTE)
                    ELSE NULL 
                END as ma_vs_phx_minutes,
                -- Categorization
                CASE 
                    WHEN first_call_attempt_datetime IS NOT NULL AND first_dial_phx IS NULL THEN 'MA Only'
                    WHEN first_call_attempt_datetime IS NULL AND first_dial_phx IS NOT NULL THEN 'PHX Only'
                    WHEN first_call_attempt_datetime IS NOT NULL AND first_dial_phx IS NOT NULL THEN 'Both MA & PHX'
                    ELSE 'No Calls'
                END as call_type
            FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
            WHERE DATE(sent_to_sales_date) BETWEEN '{call_start_date}' AND '{call_end_date}'
            AND sent_to_sales_date IS NOT NULL
            AND TIME(sent_to_sales_datetime) BETWEEN '07:30:00' AND '16:30:00'
            {call_extra_where}
        )
        SELECT 
            sts_date,
            period,
            lvc_group,
            call_type,
            phoenix_transfer_flag,
            COUNT(DISTINCT lendage_guid) as lead_count,
            AVG(ma_minutes_to_call) as avg_ma_minutes,
            AVG(phx_minutes_to_dial) as avg_phx_minutes,
            AVG(ma_vs_phx_minutes) as avg_ma_vs_phx_diff,
            COUNT(DISTINCT CASE WHEN ma_vs_phx_minutes < 0 THEN lendage_guid END) as ma_called_first,
            COUNT(DISTINCT CASE WHEN ma_vs_phx_minutes > 0 THEN lendage_guid END) as phx_dialed_first,
            COUNT(DISTINCT CASE WHEN ma_vs_phx_minutes = 0 THEN lendage_guid END) as same_time,
            COUNT(DISTINCT CASE WHEN ABS(ma_vs_phx_minutes) <= 5 THEN lendage_guid END) as within_5min_overlap
        FROM phx_data
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY sts_date, lvc_group, call_type
        """
        
        with st.expander("View MA vs PHX SQL"):
            st.code(phx_query)
        
        try:
            with st.spinner("Analyzing MA vs PHX Call Patterns..."):
                df_phx = client.query(phx_query).to_dataframe()
                
                if not df_phx.empty:
                    df_phx['sts_date'] = pd.to_datetime(df_phx['sts_date'])
                    
                    # Overall Call Type Distribution
                    st.markdown("### 📊 Call Type Distribution")
                    
                    call_type_summary = df_phx.groupby(['period', 'call_type']).agg({
                        'lead_count': 'sum'
                    }).reset_index()
                    
                    # Calculate percentages
                    period_totals = call_type_summary.groupby('period')['lead_count'].sum().reset_index()
                    period_totals.columns = ['period', 'total']
                    call_type_summary = call_type_summary.merge(period_totals, on='period')
                    call_type_summary['pct'] = call_type_summary['lead_count'] / call_type_summary['total']
                    
                    col_ct1, col_ct2 = st.columns(2)
                    
                    with col_ct1:
                        # Chart for call type distribution
                        bars_call_type = alt.Chart(call_type_summary).mark_bar().encode(
                            x=alt.X('period:N', title='Period', sort=['Pre', 'Post']),
                            y=alt.Y('pct:Q', title='% of Leads', axis=alt.Axis(format='.0%')),
                            color=alt.Color('call_type:N', scale=alt.Scale(
                                domain=['MA Only', 'PHX Only', 'Both MA & PHX', 'No Calls'],
                                range=['#4cc9f0', '#f9c74f', '#90be6d', '#8892b0']
                            ), title='Call Type'),
                            xOffset='call_type:N',
                            tooltip=[
                                alt.Tooltip('period:N', title='Period'),
                                alt.Tooltip('call_type:N', title='Call Type'),
                                alt.Tooltip('lead_count:Q', title='Lead Count', format=',d'),
                                alt.Tooltip('pct:Q', title='% of Total', format='.1%')
                            ]
                        )
                        text_call_type = bars_call_type.mark_text(align='center', baseline='bottom', dy=-5, fontSize=11).encode(text=alt.Text('pct:Q', format='.0%'))
                        
                        fig_call_type = (bars_call_type + text_call_type).properties(
                            title=f'Call Type Distribution: Pre vs Post {comp_label}',
                            height=350
                        )
                        st.altair_chart(fig_call_type, use_container_width=True)
                        st.caption("**Fields:** `first_call_attempt_datetime`, `first_dial_phx`")
                    
                    with col_ct2:
                        # Summary table
                        call_type_pivot = call_type_summary.pivot(index='call_type', columns='period', values=['lead_count', 'pct'])
                        call_type_pivot.columns = ['_'.join(col).strip() for col in call_type_pivot.columns.values]
                        call_type_pivot = call_type_pivot.reset_index()
                        
                        # Format for display
                        display_cols = ['call_type']
                        for col in call_type_pivot.columns:
                            if col != 'call_type':
                                display_cols.append(col)
                        
                        call_type_display = call_type_pivot[display_cols].copy()
                        for col in call_type_display.columns:
                            if 'lead_count' in col:
                                call_type_display[col] = call_type_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
                            elif 'pct' in col:
                                call_type_display[col] = call_type_display[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "0%")
                        
                        call_type_display.columns = [c.replace('_Pre', ' (Pre)').replace('_Post', ' (Post)').replace('lead_count', 'Count').replace('pct', '%') for c in call_type_display.columns]
                        st.dataframe(call_type_display, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # === OVERLAP ANALYSIS ===
                    st.markdown("### ⏱️ Call Timing Overlap Analysis")
                    st.caption("For leads with BOTH MA and PHX calls - who called first?")
                    
                    # Filter to only "Both MA & PHX"
                    both_calls = df_phx[df_phx['call_type'] == 'Both MA & PHX']
                    
                    if not both_calls.empty:
                        overlap_summary = both_calls.groupby('period').agg({
                            'lead_count': 'sum',
                            'ma_called_first': 'sum',
                            'phx_dialed_first': 'sum',
                            'same_time': 'sum',
                            'within_5min_overlap': 'sum'
                        }).reset_index()
                        
                        overlap_summary['ma_first_pct'] = overlap_summary['ma_called_first'] / overlap_summary['lead_count']
                        overlap_summary['phx_first_pct'] = overlap_summary['phx_dialed_first'] / overlap_summary['lead_count']
                        overlap_summary['overlap_5min_pct'] = overlap_summary['within_5min_overlap'] / overlap_summary['lead_count']
                        
                        col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
                        
                        for idx, period in enumerate(['Pre', 'Post']):
                            row = overlap_summary[overlap_summary['period'] == period]
                            if not row.empty:
                                row = row.iloc[0]
                                cols = [col_ov1, col_ov2] if period == 'Pre' else [col_ov3, col_ov4]
                                
                                with cols[0]:
                                    st.markdown(f"**{period} {comp_label}** ({row['lead_count']:,.0f} leads)")
                                    st.metric("MA Called First", f"{row['ma_first_pct']:.1%}", f"{row['ma_called_first']:,.0f} leads")
                                
                                with cols[1]:
                                    st.markdown(f"&nbsp;")  # spacer
                                    st.metric("PHX Dialed First", f"{row['phx_first_pct']:.1%}", f"{row['phx_dialed_first']:,.0f} leads")
                        
                        st.caption("**Calc:** `TIMESTAMP_DIFF(first_call_attempt_datetime, first_dial_phx, MINUTE)` - negative = MA first, positive = PHX first")
                    
                    st.divider()
                    
                    # === PHOENIX TRANSFER ANALYSIS ===
                    st.markdown("### 🔀 Phoenix Transfer Analysis")
                    st.caption("When PHX dials a lead, do they transfer it?")
                    
                    # Filter to leads with PHX dial
                    phx_dialed = df_phx[df_phx['call_type'].isin(['PHX Only', 'Both MA & PHX'])]
                    
                    if not phx_dialed.empty:
                        transfer_summary = phx_dialed.groupby(['period', 'phoenix_transfer_flag']).agg({
                            'lead_count': 'sum'
                        }).reset_index()
                        
                        transfer_summary['phoenix_transfer_flag'] = transfer_summary['phoenix_transfer_flag'].fillna('Unknown').astype(str)
                        
                        # Calculate percentages
                        transfer_totals = transfer_summary.groupby('period')['lead_count'].sum().reset_index()
                        transfer_totals.columns = ['period', 'total']
                        transfer_summary = transfer_summary.merge(transfer_totals, on='period')
                        transfer_summary['pct'] = transfer_summary['lead_count'] / transfer_summary['total']
                        
                        col_tr1, col_tr2 = st.columns(2)
                        
                        with col_tr1:
                            bars_transfer = alt.Chart(transfer_summary).mark_bar().encode(
                                x=alt.X('period:N', title='Period'),
                                y=alt.Y('pct:Q', title='% of PHX Dialed Leads', axis=alt.Axis(format='.0%')),
                                color=alt.Color('phoenix_transfer_flag:N', title='Transfer Flag'),
                                xOffset='phoenix_transfer_flag:N',
                                tooltip=[
                                    alt.Tooltip('period:N', title='Period'),
                                    alt.Tooltip('phoenix_transfer_flag:N', title='Transfer Flag'),
                                    alt.Tooltip('lead_count:Q', title='Lead Count', format=',d'),
                                    alt.Tooltip('pct:Q', title='% of Total', format='.1%')
                                ]
                            )
                            text_transfer = bars_transfer.mark_text(align='center', baseline='bottom', dy=-5, fontSize=11).encode(text=alt.Text('pct:Q', format='.0%'))
                            
                            fig_transfer = (bars_transfer + text_transfer).properties(
                                title=f'Phoenix Transfer Rate: Pre vs Post {comp_label}',
                                height=300
                            )
                            st.altair_chart(fig_transfer, use_container_width=True)
                            st.caption("**Field:** `phoenix_transfer_flag` | Shows if PHX transferred the lead after dialing")
                        
                        with col_tr2:
                            # Transfer summary table
                            transfer_pivot = transfer_summary.pivot(index='phoenix_transfer_flag', columns='period', values=['lead_count', 'pct'])
                            transfer_pivot.columns = ['_'.join(col).strip() for col in transfer_pivot.columns.values]
                            transfer_pivot = transfer_pivot.reset_index()
                            
                            for col in transfer_pivot.columns:
                                if 'lead_count' in col:
                                    transfer_pivot[col] = transfer_pivot[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
                                elif 'pct' in col:
                                    transfer_pivot[col] = transfer_pivot[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "0%")
                            
                            transfer_pivot.columns = [c.replace('_Pre', ' (Pre)').replace('_Post', ' (Post)').replace('lead_count', 'Count').replace('pct', '%').replace('phoenix_transfer_flag', 'Transfer Flag') for c in transfer_pivot.columns]
                            st.dataframe(transfer_pivot, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # === WEEKLY TREND ===
                    st.markdown("### 📈 Call Type Trend Over Time")
                    
                    df_phx['week'] = df_phx['sts_date'] - pd.to_timedelta(df_phx['sts_date'].dt.weekday, unit='D')
                    weekly_call_type = df_phx.groupby(['week', 'call_type']).agg({
                        'lead_count': 'sum'
                    }).reset_index()
                    
                    # Calculate weekly percentages
                    weekly_totals = weekly_call_type.groupby('week')['lead_count'].sum().reset_index()
                    weekly_totals.columns = ['week', 'total']
                    weekly_call_type = weekly_call_type.merge(weekly_totals, on='week')
                    weekly_call_type['pct'] = weekly_call_type['lead_count'] / weekly_call_type['total']
                    
                    ref_date = pd.Timestamp(comparison_date_str)
                    
                    fig_trend = alt.Chart(weekly_call_type).mark_area(opacity=0.7).encode(
                        x=alt.X('week:T', title='Week (by Sent to Sales Date)', axis=alt.Axis(format='%b %d')),
                        y=alt.Y('pct:Q', title='% of Leads', stack='normalize', axis=alt.Axis(format='.0%')),
                        color=alt.Color('call_type:N', scale=alt.Scale(
                            domain=['MA Only', 'PHX Only', 'Both MA & PHX', 'No Calls'],
                            range=['#4cc9f0', '#f9c74f', '#90be6d', '#8892b0']
                        ), title='Call Type'),
                        tooltip=[
                            alt.Tooltip('week:T', title='Week', format='%b %d, %Y'),
                            alt.Tooltip('call_type:N', title='Call Type'),
                            alt.Tooltip('lead_count:Q', title='Lead Count', format=',d'),
                            alt.Tooltip('pct:Q', title='% of Total', format='.1%')
                        ]
                    )
                    
                    ref_line = alt.Chart(pd.DataFrame({'x': [ref_date]})).mark_rule(color='#ff6b6b', strokeWidth=2, strokeDash=[5,5]).encode(x='x:T')
                    ref_text = alt.Chart(pd.DataFrame({'x': [ref_date], 'text': [comp_label]})).mark_text(
                        align='left', dx=5, dy=-10, color='#ff6b6b', fontSize=12
                    ).encode(x='x:T', text='text:N')
                    
                    fig_call_trend = (fig_trend + ref_line + ref_text).properties(
                        title='Call Type Mix Over Time (Stacked Area)',
                        height=350
                    ).interactive()
                    st.altair_chart(fig_call_trend, use_container_width=True)
                    
                    st.divider()
                    
                    # === KEY INSIGHTS ===
                    st.markdown("### 💡 MA vs PHX Key Insights")
                    
                    col_ins1, col_ins2 = st.columns(2)
                    
                    with col_ins1:
                        st.markdown("**🔍 What the Data Shows:**")
                        
                        # Calculate some insights
                        pre_both = call_type_summary[(call_type_summary['period'] == 'Pre') & (call_type_summary['call_type'] == 'Both MA & PHX')]
                        post_both = call_type_summary[(call_type_summary['period'] == 'Post') & (call_type_summary['call_type'] == 'Both MA & PHX')]
                        
                        pre_ma_only = call_type_summary[(call_type_summary['period'] == 'Pre') & (call_type_summary['call_type'] == 'MA Only')]
                        post_ma_only = call_type_summary[(call_type_summary['period'] == 'Post') & (call_type_summary['call_type'] == 'MA Only')]
                        
                        insights = []
                        
                        if not pre_both.empty and not post_both.empty:
                            pre_both_pct = pre_both['pct'].values[0]
                            post_both_pct = post_both['pct'].values[0]
                            change = (post_both_pct - pre_both_pct) * 100
                            if abs(change) > 2:
                                direction = "increased" if change > 0 else "decreased"
                                insights.append(f"- **Overlap (Both MA & PHX)** {direction} by {abs(change):.1f}pp post {comp_label}")
                        
                        if not pre_ma_only.empty and not post_ma_only.empty:
                            pre_ma_pct = pre_ma_only['pct'].values[0]
                            post_ma_pct = post_ma_only['pct'].values[0]
                            change = (post_ma_pct - pre_ma_pct) * 100
                            if abs(change) > 2:
                                direction = "increased" if change > 0 else "decreased"
                                insights.append(f"- **MA Only calls** {direction} by {abs(change):.1f}pp post {comp_label}")
                        
                        if insights:
                            for ins in insights:
                                st.markdown(ins)
                        else:
                            st.markdown("- No significant changes detected in call type distribution")
                    
                    with col_ins2:
                        st.markdown("**📋 Interpretation:**")
                        st.markdown("""
                        - **MA Only**: Lead was called by MA but never dialed by PHX
                        - **PHX Only**: PHX dialed but MA never called
                        - **Both MA & PHX**: Both tried to reach the lead
                        - **No Calls**: Neither MA nor PHX attempted contact
                        
                        **Transfer Flag** shows if PHX successfully transferred the call to an MA after their dial.
                        """)
                    
                    st.divider()
                    
                    # === LVC BREAKDOWN ===
                    st.markdown("### 📊 MA vs PHX by LVC Group")
                    st.caption("How does the call type distribution vary by Lead Value Cohort?")
                    
                    # Call type by LVC
                    lvc_call_type = df_phx.groupby(['period', 'lvc_group', 'call_type']).agg({
                        'lead_count': 'sum'
                    }).reset_index()
                    
                    # Calculate percentages within each LVC group
                    lvc_totals = lvc_call_type.groupby(['period', 'lvc_group'])['lead_count'].sum().reset_index()
                    lvc_totals.columns = ['period', 'lvc_group', 'total']
                    lvc_call_type = lvc_call_type.merge(lvc_totals, on=['period', 'lvc_group'])
                    lvc_call_type['pct'] = lvc_call_type['lead_count'] / lvc_call_type['total']
                    
                    # LVC order
                    lvc_order = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']
                    
                    col_lvc1, col_lvc2 = st.columns(2)
                    
                    with col_lvc1:
                        # Pre chart
                        pre_lvc = lvc_call_type[lvc_call_type['period'] == 'Pre']
                        if not pre_lvc.empty:
                            bars_pre_lvc = alt.Chart(pre_lvc).mark_bar().encode(
                                x=alt.X('lvc_group:N', title='LVC Group', sort=lvc_order, axis=alt.Axis(labelAngle=-45)),
                                y=alt.Y('pct:Q', title='% of Leads', stack='normalize', axis=alt.Axis(format='.0%')),
                                color=alt.Color('call_type:N', scale=alt.Scale(
                                    domain=['MA Only', 'PHX Only', 'Both MA & PHX', 'No Calls'],
                                    range=['#4cc9f0', '#f9c74f', '#90be6d', '#8892b0']
                                ), title='Call Type'),
                                tooltip=[
                                    alt.Tooltip('lvc_group:N', title='LVC Group'),
                                    alt.Tooltip('call_type:N', title='Call Type'),
                                    alt.Tooltip('lead_count:Q', title='Lead Count', format=',d'),
                                    alt.Tooltip('pct:Q', title='% of LVC', format='.1%')
                                ]
                            )
                            text_pre_lvc = alt.Chart(pre_lvc).mark_text(dy=15, color='white').encode(
                                x=alt.X('lvc_group:N', sort=lvc_order),
                                y=alt.Y('pct:Q', stack='normalize'),
                                detail='call_type:N',
                                text=alt.Text('pct:Q', format='.0%')
                            )
                            fig_pre_lvc = (bars_pre_lvc + text_pre_lvc).properties(
                                title=f'Pre {comp_label}: Call Type by LVC',
                                height=350
                            )
                            st.altair_chart(fig_pre_lvc, use_container_width=True)
                    
                    with col_lvc2:
                        # Post chart
                        post_lvc = lvc_call_type[lvc_call_type['period'] == 'Post']
                        if not post_lvc.empty:
                            bars_post_lvc = alt.Chart(post_lvc).mark_bar().encode(
                                x=alt.X('lvc_group:N', title='LVC Group', sort=lvc_order, axis=alt.Axis(labelAngle=-45)),
                                y=alt.Y('pct:Q', title='% of Leads', stack='normalize', axis=alt.Axis(format='.0%')),
                                color=alt.Color('call_type:N', scale=alt.Scale(
                                    domain=['MA Only', 'PHX Only', 'Both MA & PHX', 'No Calls'],
                                    range=['#4cc9f0', '#f9c74f', '#90be6d', '#8892b0']
                                ), title='Call Type'),
                                tooltip=[
                                    alt.Tooltip('lvc_group:N', title='LVC Group'),
                                    alt.Tooltip('call_type:N', title='Call Type'),
                                    alt.Tooltip('lead_count:Q', title='Lead Count', format=',d'),
                                    alt.Tooltip('pct:Q', title='% of LVC', format='.1%')
                                ]
                            )
                            text_post_lvc = alt.Chart(post_lvc).mark_text(dy=15, color='white').encode(
                                x=alt.X('lvc_group:N', sort=lvc_order),
                                y=alt.Y('pct:Q', stack='normalize'),
                                detail='call_type:N',
                                text=alt.Text('pct:Q', format='.0%')
                            )
                            fig_post_lvc = (bars_post_lvc + text_post_lvc).properties(
                                title=f'Post {comp_label}: Call Type by LVC',
                                height=350
                            )
                            st.altair_chart(fig_post_lvc, use_container_width=True)
                    
                    # LVC Summary Table
                    st.markdown("#### 📋 Call Type by LVC - Detailed Table")
                    
                    lvc_pivot = lvc_call_type.pivot_table(
                        index=['lvc_group', 'call_type'],
                        columns='period',
                        values=['lead_count', 'pct'],
                        aggfunc='first'
                    )
                    lvc_pivot.columns = ['_'.join(col).strip() for col in lvc_pivot.columns.values]
                    lvc_pivot = lvc_pivot.reset_index()
                    
                    # Format for display
                    for col in lvc_pivot.columns:
                        if 'lead_count' in col:
                            lvc_pivot[col] = lvc_pivot[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
                        elif 'pct' in col:
                            lvc_pivot[col] = lvc_pivot[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "0%")
                    
                    lvc_pivot.columns = [c.replace('_Pre', ' (Pre)').replace('_Post', ' (Post)').replace('lead_count', 'Count').replace('pct', '%') for c in lvc_pivot.columns]
                    st.dataframe(lvc_pivot, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # === TRANSFER RATE BY LVC ===
                    st.markdown("### 🔀 Phoenix Transfer Rate by LVC")
                    st.caption("For leads where PHX dialed - what % were transferred?")
                    
                    # Filter to PHX dialed leads
                    phx_lvc = df_phx[df_phx['call_type'].isin(['PHX Only', 'Both MA & PHX'])]
                    
                    if not phx_lvc.empty:
                        transfer_lvc = phx_lvc.groupby(['period', 'lvc_group', 'phoenix_transfer_flag']).agg({
                            'lead_count': 'sum'
                        }).reset_index()
                        
                        transfer_lvc['phoenix_transfer_flag'] = transfer_lvc['phoenix_transfer_flag'].fillna('Unknown').astype(str)
                        
                        # Calculate transfer rate (assuming 'True' or '1' means transferred)
                        transfer_totals = transfer_lvc.groupby(['period', 'lvc_group'])['lead_count'].sum().reset_index()
                        transfer_totals.columns = ['period', 'lvc_group', 'total']
                        transfer_lvc = transfer_lvc.merge(transfer_totals, on=['period', 'lvc_group'])
                        transfer_lvc['pct'] = transfer_lvc['lead_count'] / transfer_lvc['total']
                        
                        # Filter to just transferred (True/1)
                        transferred = transfer_lvc[transfer_lvc['phoenix_transfer_flag'].isin(['True', 'true', '1', 'Yes', 'yes'])]
                        
                        if not transferred.empty:
                            col_tr_lvc1, col_tr_lvc2 = st.columns(2)
                            
                            with col_tr_lvc1:
                                bars_transfer_lvc = alt.Chart(transferred).mark_bar().encode(
                                    x=alt.X('lvc_group:N', title='LVC Group', sort=lvc_order, axis=alt.Axis(labelAngle=-45)),
                                    y=alt.Y('pct:Q', title='Transfer Rate', axis=alt.Axis(format='.0%')),
                                    color=alt.Color('period:N', scale=alt.Scale(
                                        domain=['Pre', 'Post'],
                                        range=['#4cc9f0', '#ff6b6b']
                                    )),
                                    xOffset='period:N',
                                    tooltip=[
                                        alt.Tooltip('lvc_group:N', title='LVC Group'),
                                        alt.Tooltip('period:N', title='Period'),
                                        alt.Tooltip('lead_count:Q', title='Transferred', format=',d'),
                                        alt.Tooltip('pct:Q', title='Transfer Rate', format='.1%')
                                    ]
                                )
                                text_transfer_lvc = bars_transfer_lvc.mark_text(align='center', baseline='bottom', dy=-5, fontSize=11).encode(text=alt.Text('pct:Q', format='.0%'))
                                
                                fig_transfer_lvc = (bars_transfer_lvc + text_transfer_lvc).properties(
                                    title='PHX Transfer Rate by LVC: Pre vs Post',
                                    height=350
                                )
                                st.altair_chart(fig_transfer_lvc, use_container_width=True)
                            
                            with col_tr_lvc2:
                                # Transfer summary table
                                transfer_pivot = transferred.pivot(index='lvc_group', columns='period', values=['lead_count', 'pct', 'total'])
                                transfer_pivot.columns = ['_'.join(col).strip() for col in transfer_pivot.columns.values]
                                transfer_pivot = transfer_pivot.reset_index()
                                
                                for col in transfer_pivot.columns:
                                    if 'lead_count' in col or 'total' in col:
                                        transfer_pivot[col] = transfer_pivot[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
                                    elif 'pct' in col:
                                        transfer_pivot[col] = transfer_pivot[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "0%")
                                
                                transfer_pivot.columns = [c.replace('_Pre', ' (Pre)').replace('_Post', ' (Post)').replace('lead_count', 'Transferred').replace('pct', 'Rate').replace('total', 'PHX Dialed') for c in transfer_pivot.columns]
                                st.dataframe(transfer_pivot, use_container_width=True, hide_index=True)
                        else:
                            st.info("No transfer data found with 'True' flag. Showing full transfer flag distribution:")
                            st.dataframe(transfer_lvc, use_container_width=True)
                    
                    with st.expander("View Raw PHX Analysis Data"):
                        st.dataframe(df_phx, use_container_width=True)
                
                else:
                    st.warning("No PHX analysis data found for the selected period.")
        
        except Exception as e:
            st.error(f"PHX Analysis Error: {e}")
            st.exception(e)

    # --- TAB: Outreach Metrics (Pre-App / Pre-Fund) — LVC 1-8, PHX 24hr delay context ---
    with tab_outreach:
        st.header("📞 Outreach Metrics (Pre-App / Pre-Fund)")
        st.caption("LVC 1-8 • Period 1: 12/19/25–1/26/26 vs Period 2: 1/27/26–3/6/26 • Weekdays Only • Joined to consumer comms")

        # Context for PowerPoint: PHX delay and AB test next steps
        st.markdown("""
        ### 📌 Context (for slides)
        - **As of 1/27/26:** PHX team does **not** call LVC 1-8 (by `initial_lead_value_cohort`) until **24 hours** after `initial_lead_score_datetime`.
        - In the **first 24 hours** after scoring, only **MAs** call these leads (PHX holds).
        - **Next steps / AB test:** Try alternative delay windows (e.g. **4 hours**, **48 hours**, or values in between) and compare outreach and conversion.
        """)
        st.divider()

        # Fixed periods and LVC filter
        out_period1_start, out_period1_end = "2025-12-19", "2026-01-26"
        out_period2_start, out_period2_end = "2026-01-27", "2026-03-06"

        # LVC filter
        lvc_options_outreach = [str(i) for i in range(1, 9)]
        selected_lvc_outreach = st.multiselect("Filter Initial LVC (default: 1-8)", options=lvc_options_outreach, default=lvc_options_outreach)
        
        if not selected_lvc_outreach:
            st.warning("Please select at least one LVC.")
            st.stop()
            
        lvc_list_str = ", ".join([f"'{l}'" for l in selected_lvc_outreach])

        outreach_base_sql = f"""
        WITH base AS (
            SELECT
                a.lendage_guid,
                DATE(a.sent_to_sales_date) AS sts_date,
                CASE WHEN DATE(a.sent_to_sales_date) BETWEEN '{out_period1_start}' AND '{out_period1_end}' THEN 'Pre (12/19-1/26)'
                     WHEN DATE(a.sent_to_sales_date) BETWEEN '{out_period2_start}' AND '{out_period2_end}' THEN 'Post (1/27-3/6)' END AS period,
                a.initial_lead_value_cohort,
                a.starring,
                a.mortgage_advisor,
                a.initial_sales_assigned_datetime,
                a.first_call_attempt_datetime AS ma_first_call,
                a.first_dial_phx,
                b.first_dial_attempt_datetime AS comms_first_dial,
                CASE
                    WHEN a.first_call_attempt_datetime IS NOT NULL AND (a.first_dial_phx IS NULL OR a.first_call_attempt_datetime <= a.first_dial_phx) AND (b.first_dial_attempt_datetime IS NULL OR a.first_call_attempt_datetime <= b.first_dial_attempt_datetime) THEN a.first_call_attempt_datetime
                    WHEN a.first_dial_phx IS NOT NULL AND (b.first_dial_attempt_datetime IS NULL OR a.first_dial_phx <= b.first_dial_attempt_datetime) THEN a.first_dial_phx
                    ELSE b.first_dial_attempt_datetime
                END AS earliest_dial_datetime,
                CASE WHEN a.sf__contacted_guid IS NOT NULL THEN a.sf_contacted_date ELSE NULL END AS a_sf_contact_datetime,
                b.first_contact_datetime AS b_first_contact,
                CASE
                    WHEN a.sf__contacted_guid IS NOT NULL AND (b.first_contact_datetime IS NULL OR a.sf_contacted_date <= b.first_contact_datetime) THEN a.sf_contacted_date
                    ELSE b.first_contact_datetime
                END AS earliest_contact_datetime,
                b.first_contact_method,
                a.call_attempts,
                a.call_attempts_before_app,
                COALESCE(b.email_sent_count_before_contact, 0) AS email_before_contact,
                COALESCE(b.sms_sent_count_before_contact, 0) AS sms_before_contact,
                COALESCE(b.outbound_phone_calls_count_before_contact, 0) AS calls_before_contact,
                COALESCE(b.email_sent_count, 0) AS emails_sent,
                COALESCE(b.sms_sent_count, 0) AS outbound_sms,
                a.phoenix_fronter_contact_datetime,
                COALESCE(a.phx_total_outbound_attempts, 0) AS phx_total_outbound_attempts,
                COALESCE(b.dialer_campaign_flag, FALSE) AS dialer_campaign_flag,
                COALESCE(b.dialer_contact_flag, FALSE) AS dialer_contact_flag,
                a.sf__contacted_guid IS NOT NULL AS had_contact,
                a.full_app_submit_datetime IS NOT NULL AS had_fas,
                COALESCE(a.e_loan_amount, 0) AS e_loan_amount
            FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table` a
            LEFT JOIN `lendage-data-platform.standardized_data.lendage_consumer_comms_lead_data` b ON a.lendage_guid = b.lendage_guid
            WHERE a.initial_lead_value_cohort IN ({lvc_list_str})
            AND DATE(a.sent_to_sales_date) BETWEEN '{out_period1_start}' AND '{out_period2_end}'
            AND EXTRACT(DAYOFWEEK FROM a.sent_to_sales_date) NOT IN (1, 7)
            AND a.sent_to_sales_date IS NOT NULL
        ),
        metrics AS (
            SELECT
                *,
                TIMESTAMP_DIFF(earliest_dial_datetime, initial_sales_assigned_datetime, MINUTE) AS speed_to_dial_min,
                TIMESTAMP_DIFF(earliest_contact_datetime, initial_sales_assigned_datetime, MINUTE) AS speed_to_first_contact_min
            FROM base
            WHERE period IS NOT NULL
        )
        SELECT * FROM metrics
        """
        try:
            with st.spinner("Loading outreach data (vintages + comms join)…"):
                df_out = client.query(outreach_base_sql).to_dataframe()
        except Exception as e:
            st.error(f"Outreach query failed (check access to both projects): {e}")
            st.exception(e)
            df_out = pd.DataFrame()

        if not df_out.empty:
            df_out["sts_date"] = pd.to_datetime(df_out["sts_date"])
            # Weekly bucketing for time series (week ending Sunday)
            df_out["week_end"] = df_out["sts_date"] - pd.to_timedelta(df_out["sts_date"].dt.dayofweek + 1, unit="d")
            df_out["week_end"] = df_out["week_end"].dt.date

            # --- Context ---
            st.markdown("""
            **Context:** As of 1/27/26, PHX does not call LVC 1-8 until **24 hours** after `initial_lead_score_datetime`.  
            In the first 24 hours only **MAs** call. Goal: compare **Pre** (12/19–1/26) vs **Post** (1/27–3/6) and recommend which delay to test next (e.g. 4 hr, 48 hr).
            """)
            st.divider()

            # --- Pre vs Post aggregate (with FAS) ---
            st.caption("Note: All data in this tab **excludes weekends** (Saturdays/Sundays) to isolate true working behavior.")
            agg = df_out.groupby("period").agg(
                lead_count=("lendage_guid", "nunique"),
                with_dial=("earliest_dial_datetime", lambda x: x.notna().sum()),
                with_contact=("had_contact", "sum"),
                fas_count=("had_fas", "sum"),
                speed_to_dial_median=("speed_to_dial_min", lambda x: x[x.between(0, 10080)].median()),
                speed_to_contact_median=("speed_to_first_contact_min", lambda x: x[x.between(0, 10080)].median()),
                total_call_attempts=("call_attempts", "sum"),
                emails_sent=("emails_sent", "sum"),
                outbound_sms=("outbound_sms", "sum"),
                phx_total_outbound_attempts=("phx_total_outbound_attempts", "sum"),
                phx_contact_count=("phoenix_fronter_contact_datetime", lambda x: x.notna().sum()),
                email_before_contact=("email_before_contact", "sum"),
                sms_before_contact=("sms_before_contact", "sum"),
                calls_before_contact=("calls_before_contact", "sum")
            ).reset_index()
            fas_dol_series = df_out[df_out["had_fas"]].groupby("period")["e_loan_amount"].sum()
            agg["fas_dollars"] = agg["period"].map(lambda p: fas_dol_series.get(p, 0))
            agg["contact_rate"] = agg["with_contact"] / agg["lead_count"]
            agg["dial_rate"] = agg["with_dial"] / agg["lead_count"]
            agg["fas_pct"] = agg["fas_count"] / agg["lead_count"]
            agg["avg_call_attempts"] = agg["total_call_attempts"] / agg["lead_count"]
            agg["avg_emails_sent_per_lead"] = agg["emails_sent"] / agg["lead_count"]
            agg["avg_outbound_sms_per_lead"] = agg["outbound_sms"] / agg["lead_count"]
            agg["avg_phx_attempts_per_lead"] = agg["phx_total_outbound_attempts"] / agg["lead_count"]
            agg["phx_contact_rate"] = agg["phx_contact_count"] / agg["lead_count"]
            agg["avg_email_before_contact_per_lead"] = agg["email_before_contact"] / agg["lead_count"]
            agg["avg_sms_before_contact_per_lead"] = agg["sms_before_contact"] / agg["lead_count"]
            agg["avg_calls_before_contact_per_lead"] = agg["calls_before_contact"] / agg["lead_count"]

            pre_row = agg[agg["period"] == "Pre (12/19-1/26)"].iloc[0] if len(agg[agg["period"] == "Pre (12/19-1/26)"]) else None
            post_row = agg[agg["period"] == "Post (1/27-3/6)"].iloc[0] if len(agg[agg["period"] == "Post (1/27-3/6)"]) else None

            # --- 1. PRE VS POST KPI CARDS (outreach + conversion) ---
            st.subheader("1️⃣ Pre vs Post — KPI comparison")
            st.caption("**Calcs:** Lead count = COUNT(DISTINCT lendage_guid) | Contact rate = COUNT(sf__contacted_guid NOT NULL) / lead_count | FAS % = COUNT(full_app_submit_datetime NOT NULL) / lead_count | FAS $ = SUM(e_loan_amount) for FAS leads only | Speed to dial = median of TIMESTAMP_DIFF(earliest_dial, initial_sales_assigned_datetime, MINUTE) [0–10080 min] | Speed to 1st contact = median of TIMESTAMP_DIFF(earliest_contact, initial_sales_assigned_datetime, MINUTE) [earliest_contact = sf_contacted_date when sf__contacted_guid NOT NULL else comms first_contact_datetime]")
            if pre_row is not None and post_row is not None:
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                with k1:
                    st.metric("Lead count", f"{int(post_row['lead_count']):,}", f"{(post_row['lead_count'] - pre_row['lead_count']):+,} vs Pre")
                with k2:
                    st.metric("Contact rate", f"{post_row['contact_rate']:.1%}", f"{(post_row['contact_rate'] - pre_row['contact_rate']):+.1%}")
                with k3:
                    st.metric("FAS %", f"{post_row['fas_pct']:.1%}", f"{(post_row['fas_pct'] - pre_row['fas_pct']):+.1%}")
                with k4:
                    st.metric("FAS $", f"${post_row['fas_dollars']:,.0f}", f"${(post_row['fas_dollars'] - pre_row['fas_dollars']):+,.0f}")
                with k5:
                    med_d = post_row["speed_to_dial_median"]
                    st.metric("Speed to dial (med min)", f"{med_d:.0f}" if pd.notna(med_d) else "—", f"Pre: {pre_row['speed_to_dial_median']:.0f}" if pd.notna(pre_row['speed_to_dial_median']) else "—")
                with k6:
                    med_c = post_row["speed_to_contact_median"]
                    st.metric("Speed to 1st contact (med min)", f"{med_c:.0f}" if pd.notna(med_c) else "—", f"Pre: {pre_row['speed_to_contact_median']:.0f}" if pd.notna(pre_row['speed_to_contact_median']) else "—")
            st.divider()

            # --- 2. CHARTS OVER TIME (weekly) ---
            st.subheader("2️⃣ Trends over time (weekly)")
            weekly = df_out.groupby(["week_end", "period"]).agg(
                lead_count=("lendage_guid", "nunique"),
                with_contact=("had_contact", "sum"),
                fas_count=("had_fas", "sum"),
                fas_dollars=("e_loan_amount", lambda x: x.sum()),
                speed_to_dial_median=("speed_to_dial_min", lambda x: x[x.between(0, 10080)].median()),
                speed_to_contact_median=("speed_to_first_contact_min", lambda x: x[x.between(0, 10080)].median()),
                total_call_attempts=("call_attempts", "sum"),
                emails_sent=("emails_sent", "sum"),
                outbound_sms=("outbound_sms", "sum"),
                phx_total_outbound_attempts=("phx_total_outbound_attempts", "sum"),
                phx_contact_count=("phoenix_fronter_contact_datetime", lambda x: x.notna().sum()),
                email_before_contact=("email_before_contact", "sum"),
                sms_before_contact=("sms_before_contact", "sum"),
                calls_before_contact=("calls_before_contact", "sum")
            ).reset_index()
            weekly["contact_rate"] = weekly["with_contact"] / weekly["lead_count"]
            weekly["fas_pct"] = weekly["fas_count"] / weekly["lead_count"]
            weekly["phx_contact_rate"] = weekly["phx_contact_count"] / weekly["lead_count"]
            weekly["avg_call_attempts"] = weekly["total_call_attempts"] / weekly["lead_count"]
            weekly["avg_email_before_contact"] = weekly["email_before_contact"] / weekly["lead_count"]
            weekly["avg_sms_before_contact"] = weekly["sms_before_contact"] / weekly["lead_count"]
            weekly["avg_phx_attempts"] = weekly["phx_total_outbound_attempts"] / weekly["lead_count"]
            weekly["avg_calls_before_contact"] = weekly["calls_before_contact"] / weekly["lead_count"]
            # FAS $ only for leads that have FAS (we have lead-level had_fas; at weekly we sum e_loan_amount for FAS leads)
            fas_dollars_by_week = df_out[df_out["had_fas"]].groupby(["week_end", "period"])["e_loan_amount"].sum().reset_index()
            fas_dollars_by_week.columns = ["week_end", "period", "fas_dollars"]
            weekly = weekly.drop(columns=["fas_dollars"], errors="ignore").merge(fas_dollars_by_week, on=["week_end", "period"], how="left")
            weekly["fas_dollars"] = weekly["fas_dollars"].fillna(0)
            weekly["week_label"] = pd.to_datetime(weekly["week_end"]).dt.strftime("%m/%d")
            week_sort = weekly.sort_values("week_end")["week_label"].unique().tolist()

            # Time series: Contact rate
            st.markdown("#### Contact rate by week")
            st.caption("**Calc:** contact_rate = SUM(had_contact) / lead_count per week; had_contact = (sf__contacted_guid IS NOT NULL).")
            chart_contact = alt.Chart(weekly).mark_line(point=True).encode(
                x=alt.X("week_label:N", title="Week ending", sort=week_sort),
                y=alt.Y("contact_rate:Q", title="Contact rate", axis=alt.Axis(format=".0%")),
                color=alt.Color("period:N", scale=alt.Scale(domain=["Pre (12/19-1/26)", "Post (1/27-3/6)"], range=["#1f77b4", "#ff7f0e"])),
                tooltip=["week_label", "period", alt.Tooltip("contact_rate:Q", format=".1%"), "lead_count:Q"]
            ).properties(height=280)
            st.altair_chart(chart_contact, use_container_width=True)

            # Time series: FAS % and FAS $
            col_fas1, col_fas2 = st.columns(2)
            with col_fas1:
                st.markdown("#### FAS % by week")
                st.caption("**Calc:** fas_pct = fas_count / lead_count per week; fas_count = COUNT(full_app_submit_datetime NOT NULL).")
                chart_fas_pct = alt.Chart(weekly).mark_line(point=True).encode(
                    x=alt.X("week_label:N", title="Week ending", sort=week_sort),
                    y=alt.Y("fas_pct:Q", title="FAS %", axis=alt.Axis(format=".0%")),
                    color=alt.Color("period:N", scale=alt.Scale(domain=["Pre (12/19-1/26)", "Post (1/27-3/6)"], range=["#1f77b4", "#ff7f0e"])),
                    tooltip=["week_label", "period", alt.Tooltip("fas_pct:Q", format=".1%"), "fas_count:Q", "lead_count:Q"]
                ).properties(height=280)
                st.altair_chart(chart_fas_pct, use_container_width=True)
            with col_fas2:
                st.markdown("#### FAS $ by week")
                st.caption("**Calc:** fas_dollars = SUM(e_loan_amount) for leads with full_app_submit_datetime NOT NULL, by week.")
                chart_fas_dol = alt.Chart(weekly).mark_line(point=True).encode(
                    x=alt.X("week_label:N", title="Week ending", sort=week_sort),
                    y=alt.Y("fas_dollars:Q", title="FAS $"),
                    color=alt.Color("period:N", scale=alt.Scale(domain=["Pre (12/19-1/26)", "Post (1/27-3/6)"], range=["#1f77b4", "#ff7f0e"])),
                    tooltip=["week_label", "period", alt.Tooltip("fas_dollars:Q", format=",.0f"), "fas_count:Q"]
                ).properties(height=280)
                st.altair_chart(chart_fas_dol, use_container_width=True)

            # Time series: Speed to dial / Speed to contact
            st.markdown("#### Speed to dial & speed to first contact (median minutes) by week")
            st.caption("**Calc:** Speed to dial = median of TIMESTAMP_DIFF(earliest_dial, initial_sales_assigned_datetime, MINUTE) where 0≤min≤10080; earliest_dial = earliest of MA first_call_attempt_datetime, first_dial_phx, comms first_dial_attempt_datetime. Speed to 1st contact = same but from earliest_contact (sf_contacted_date when sf__contacted_guid NOT NULL, else comms first_contact_datetime).")
            weekly_melt = weekly[["week_end", "week_label", "period", "speed_to_dial_median", "speed_to_contact_median"]].copy()
            weekly_melt = weekly_melt.melt(id_vars=["week_end", "week_label", "period"], var_name="metric", value_name="minutes")
            weekly_melt["metric"] = weekly_melt["metric"].map({"speed_to_dial_median": "Speed to dial", "speed_to_contact_median": "Speed to 1st contact"})
            chart_speed = alt.Chart(weekly_melt.dropna(subset=["minutes"])).mark_line(point=True).encode(
                x=alt.X("week_label:N", title="Week ending", sort=week_sort),
                y=alt.Y("minutes:Q", title="Median minutes"),
                color=alt.Color("period:N", scale=alt.Scale(domain=["Pre (12/19-1/26)", "Post (1/27-3/6)"], range=["#1f77b4", "#ff7f0e"])),
                column=alt.Column("metric:N", title=None),
                tooltip=["week_label", "period", "metric", alt.Tooltip("minutes:Q", format=".0f")]
            ).properties(width=350, height=260)
            st.altair_chart(chart_speed, use_container_width=True)

            # Time series: Emails sent, Outbound SMS, PHX outbound attempts, MA Call attempts
            st.markdown("#### Outreach volume per lead by week")
            st.caption("**Calc:** Emails before Contact/Lead = SUM(comms email_sent_count_before_contact) / lead_count. SMS before Contact/Lead = SUM(comms sms_sent_count_before_contact) / lead_count. Outbound Phx attempts = SUM(phx_total_outbound_attempts) / lead_count. Outbound MA calls before contact/Lead = SUM(outbound_phone_calls_count_before_contact) / lead_count.")
            weekly_outreach = weekly[["week_end", "week_label", "period", "avg_email_before_contact", "avg_sms_before_contact", "avg_phx_attempts", "avg_calls_before_contact"]].copy()
            
            col_out1, col_out2 = st.columns(2)
            
            with col_out1:
                st.markdown("**Email before Contact / Lead**")
                base_em = alt.Chart(weekly_outreach).mark_line(point=True).encode(
                    x=alt.X("week_label:N", title="Week ending", sort=week_sort),
                    y=alt.Y("avg_email_before_contact:Q", title="Per Lead"),
                    color=alt.Color("period:N", scale=alt.Scale(domain=["Pre (12/19-1/26)", "Post (1/27-3/6)"], range=["#1f77b4", "#ff7f0e"])),
                    tooltip=["week_label", "period", alt.Tooltip("avg_email_before_contact:Q", format=",.2f")]
                )
                text_em = base_em.mark_text(align='center', baseline='bottom', dy=-10, fontSize=11).encode(text=alt.Text('avg_email_before_contact:Q', format='.2f'))
                chart_em = (base_em + text_em).properties(height=240)
                st.altair_chart(chart_em, use_container_width=True)
                
            with col_out2:
                st.markdown("**SMS before Contact / Lead**")
                base_sms = alt.Chart(weekly_outreach).mark_line(point=True).encode(
                    x=alt.X("week_label:N", title="Week ending", sort=week_sort),
                    y=alt.Y("avg_sms_before_contact:Q", title="Per Lead"),
                    color=alt.Color("period:N", scale=alt.Scale(domain=["Pre (12/19-1/26)", "Post (1/27-3/6)"], range=["#1f77b4", "#ff7f0e"])),
                    tooltip=["week_label", "period", alt.Tooltip("avg_sms_before_contact:Q", format=",.2f")]
                )
                text_sms = base_sms.mark_text(align='center', baseline='bottom', dy=-10, fontSize=11).encode(text=alt.Text('avg_sms_before_contact:Q', format='.2f'))
                chart_sms = (base_sms + text_sms).properties(height=240)
                st.altair_chart(chart_sms, use_container_width=True)
                
            col_out3, col_out4 = st.columns(2)
                
            with col_out3:
                st.markdown("**Outbound Phx attempts**")
                base_phx = alt.Chart(weekly_outreach).mark_line(point=True).encode(
                    x=alt.X("week_label:N", title="Week ending", sort=week_sort),
                    y=alt.Y("avg_phx_attempts:Q", title="Per Lead"),
                    color=alt.Color("period:N", scale=alt.Scale(domain=["Pre (12/19-1/26)", "Post (1/27-3/6)"], range=["#1f77b4", "#ff7f0e"])),
                    tooltip=["week_label", "period", alt.Tooltip("avg_phx_attempts:Q", format=",.2f")]
                )
                text_phx = base_phx.mark_text(align='center', baseline='bottom', dy=-10, fontSize=11).encode(text=alt.Text('avg_phx_attempts:Q', format='.2f'))
                chart_phx = (base_phx + text_phx).properties(height=240)
                st.altair_chart(chart_phx, use_container_width=True)

            with col_out4:
                st.markdown("**Outbound MA calls before contact / Lead**")
                base_ma = alt.Chart(weekly_outreach).mark_line(point=True).encode(
                    x=alt.X("week_label:N", title="Week ending", sort=week_sort),
                    y=alt.Y("avg_calls_before_contact:Q", title="Per Lead"),
                    color=alt.Color("period:N", scale=alt.Scale(domain=["Pre (12/19-1/26)", "Post (1/27-3/6)"], range=["#1f77b4", "#ff7f0e"])),
                    tooltip=["week_label", "period", alt.Tooltip("avg_calls_before_contact:Q", format=",.2f")]
                )
                text_ma = base_ma.mark_text(align='center', baseline='bottom', dy=-10, fontSize=11).encode(text=alt.Text('avg_calls_before_contact:Q', format='.2f'))
                chart_ma = (base_ma + text_ma).properties(height=240)
                st.altair_chart(chart_ma, use_container_width=True)

            st.divider()

            # --- 3. PRE VS POST BAR CHARTS ---
            st.subheader("3️⃣ Pre vs Post — bar comparison")
            st.caption("**Calcs:** Same as section 1 (contact_rate, fas_pct, speed_to_dial_median, speed_to_contact_median).")
            comp = agg[["period", "contact_rate", "fas_pct", "speed_to_dial_median", "speed_to_contact_median"]].copy()
            comp_melt = comp.melt(id_vars="period", var_name="metric", value_name="value")
            comp_melt = comp_melt[comp_melt["metric"].isin(["contact_rate", "fas_pct", "speed_to_dial_median", "speed_to_contact_median"])]
            comp_melt["metric"] = comp_melt["metric"].replace({
                "contact_rate": "Contact rate",
                "fas_pct": "FAS %",
                "speed_to_dial_median": "Speed to dial (median min)",
                "speed_to_contact_median": "Speed to 1st contact (median min)"
            })
            comp_melt = comp_melt.dropna(subset=["value"])
            bar_prepost = alt.Chart(comp_melt).mark_bar().encode(
                x=alt.X("period:N", title=""),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("period:N", scale=alt.Scale(domain=["Pre (12/19-1/26)", "Post (1/27-3/6)"], range=["#1f77b4", "#ff7f0e"])),
                column=alt.Column("metric:N", title=None, header=alt.Header(labelAngle=-45))
            ).properties(width=200, height=220)
            st.altair_chart(bar_prepost, use_container_width=True)

            # FAS $ bar
            st.markdown("#### FAS $ (total by period)")
            st.caption("**Calc:** fas_dollars = SUM(e_loan_amount) where full_app_submit_datetime IS NOT NULL, by period.")
            bar_fas_dol = alt.Chart(agg).mark_bar().encode(
                x=alt.X("period:N", title=""),
                y=alt.Y("fas_dollars:Q", title="FAS $"),
                color=alt.Color("period:N", scale=alt.Scale(domain=["Pre (12/19-1/26)", "Post (1/27-3/6)"], range=["#1f77b4", "#ff7f0e"])),
                tooltip=[alt.Tooltip("fas_dollars:Q", format=",.0f"), "lead_count:Q", "fas_count:Q"]
            ).properties(height=280)
            st.altair_chart(bar_fas_dol, use_container_width=True)
            st.divider()

            # --- 4. BY INITIAL LVC (Pre vs Post) ---
            st.subheader("4️⃣ By initial LVC (Pre vs Post)")
            st.caption("Breakdown by initial_lead_value_cohort. **Calcs:** contact_rate = with_contact/lead_count; fas_pct = fas_count/lead_count; fas_dollars = SUM(e_loan_amount) for FAS leads; phx_contact_rate = phx_contact_count/lead_count; speed medians = same as above; avg_emails/sms/phx per lead = sum/lead_count.")
            by_lvc = df_out.groupby(["period", "initial_lead_value_cohort"]).agg(
                lead_count=("lendage_guid", "nunique"),
                with_contact=("had_contact", "sum"),
                fas_count=("had_fas", "sum"),
                emails_sent=("emails_sent", "sum"),
                outbound_sms=("outbound_sms", "sum"),
                phx_total_outbound_attempts=("phx_total_outbound_attempts", "sum"),
                phx_contact_count=("phoenix_fronter_contact_datetime", lambda x: x.notna().sum()),
                speed_to_dial_median=("speed_to_dial_min", lambda x: x[x.between(0, 10080)].median()),
                speed_to_contact_median=("speed_to_first_contact_min", lambda x: x[x.between(0, 10080)].median()),
            ).reset_index()
            by_lvc["contact_rate"] = by_lvc["with_contact"] / by_lvc["lead_count"]
            by_lvc["fas_pct"] = by_lvc["fas_count"] / by_lvc["lead_count"]
            by_lvc["phx_contact_rate"] = by_lvc["phx_contact_count"] / by_lvc["lead_count"]
            by_lvc["avg_emails_per_lead"] = by_lvc["emails_sent"] / by_lvc["lead_count"]
            by_lvc["avg_sms_per_lead"] = by_lvc["outbound_sms"] / by_lvc["lead_count"]
            by_lvc["avg_phx_attempts_per_lead"] = by_lvc["phx_total_outbound_attempts"] / by_lvc["lead_count"]
            fas_dol_lvc = df_out[df_out["had_fas"]].groupby(["period", "initial_lead_value_cohort"])["e_loan_amount"].sum().reset_index()
            fas_dol_lvc.columns = ["period", "initial_lead_value_cohort", "fas_dollars"]
            by_lvc = by_lvc.merge(fas_dol_lvc, on=["period", "initial_lead_value_cohort"], how="left").fillna(0)
            lvc_pivot = by_lvc.pivot(index="initial_lead_value_cohort", columns="period", values=["lead_count", "contact_rate", "fas_pct", "fas_dollars", "emails_sent", "outbound_sms", "phx_total_outbound_attempts", "phx_contact_rate", "speed_to_dial_median", "speed_to_contact_median", "avg_emails_per_lead", "avg_sms_per_lead", "avg_phx_attempts_per_lead"])
            st.dataframe(by_lvc, use_container_width=True, hide_index=True)
            st.caption("Full table by initial LVC. Use pivot below for Pre vs Post side-by-side.")
            if not lvc_pivot.empty:
                st.dataframe(lvc_pivot, use_container_width=True, hide_index=True)
                
                st.markdown("#### 📊 LVC Comparison Charts")
                by_lvc['initial_lead_value_cohort'] = by_lvc['initial_lead_value_cohort'].astype(str)
                lvc_order = sorted(by_lvc['initial_lead_value_cohort'].unique().tolist())
                domain_periods = ["Pre (12/19-1/26)", "Post (1/27-3/6)"]
                
                col_lvc1, col_lvc2 = st.columns(2)
                
                with col_lvc1:
                    base_cr = alt.Chart(by_lvc).encode(
                        x=alt.X('period:N', title=None, axis=alt.Axis(labels=False, ticks=False), sort=domain_periods),
                        y=alt.Y('contact_rate:Q', title='Contact Rate', axis=alt.Axis(format='.0%')),
                        color=alt.Color('period:N', scale=alt.Scale(domain=domain_periods, range=["#1f77b4", "#ff7f0e"]))
                    ).properties(width=40, height=250)
                    
                    chart_cr = alt.layer(
                        base_cr.mark_bar(),
                        base_cr.mark_text(align='center', baseline='bottom', dy=-5, fontSize=10).encode(text=alt.Text('contact_rate:Q', format='.0%'))
                    ).facet(
                        column=alt.Column('initial_lead_value_cohort:N', title='Initial LVC', sort=lvc_order)
                    ).properties(title="Contact Rate by LVC")
                    
                    st.altair_chart(chart_cr, use_container_width=False)

                    base_spd = alt.Chart(by_lvc.dropna(subset=['speed_to_dial_median'])).encode(
                        x=alt.X('period:N', title=None, axis=alt.Axis(labels=False, ticks=False), sort=domain_periods),
                        y=alt.Y('speed_to_dial_median:Q', title='Median Min'),
                        color=alt.Color('period:N', scale=alt.Scale(domain=domain_periods, range=["#1f77b4", "#ff7f0e"]))
                    ).properties(width=40, height=250)
                    
                    chart_spd = alt.layer(
                        base_spd.mark_bar(),
                        base_spd.mark_text(align='center', baseline='bottom', dy=-5, fontSize=10).encode(text=alt.Text('speed_to_dial_median:Q', format='.0f'))
                    ).facet(
                        column=alt.Column('initial_lead_value_cohort:N', title='Initial LVC', sort=lvc_order)
                    ).properties(title="Speed to Dial by LVC")
                    
                    st.altair_chart(chart_spd, use_container_width=False)

                with col_lvc2:
                    base_fas = alt.Chart(by_lvc).encode(
                        x=alt.X('period:N', title=None, axis=alt.Axis(labels=False, ticks=False), sort=domain_periods),
                        y=alt.Y('fas_pct:Q', title='FAS %', axis=alt.Axis(format='.0%')),
                        color=alt.Color('period:N', scale=alt.Scale(domain=domain_periods, range=["#1f77b4", "#ff7f0e"]))
                    ).properties(width=40, height=250)
                    
                    chart_fas = alt.layer(
                        base_fas.mark_bar(),
                        base_fas.mark_text(align='center', baseline='bottom', dy=-5, fontSize=10).encode(text=alt.Text('fas_pct:Q', format='.1%'))
                    ).facet(
                        column=alt.Column('initial_lead_value_cohort:N', title='Initial LVC', sort=lvc_order)
                    ).properties(title="FAS % by LVC")
                    
                    st.altair_chart(chart_fas, use_container_width=False)
                    
                    base_fas_dol = alt.Chart(by_lvc).encode(
                        x=alt.X('period:N', title=None, axis=alt.Axis(labels=False, ticks=False), sort=domain_periods),
                        y=alt.Y('fas_dollars:Q', title='FAS $'),
                        color=alt.Color('period:N', scale=alt.Scale(domain=domain_periods, range=["#1f77b4", "#ff7f0e"]))
                    ).properties(width=40, height=250)
                    
                    chart_fas_dol = alt.layer(
                        base_fas_dol.mark_bar(),
                        base_fas_dol.mark_text(align='center', baseline='bottom', dy=-5, fontSize=9).encode(text=alt.Text('fas_dollars:Q', format='.2s'))
                    ).facet(
                        column=alt.Column('initial_lead_value_cohort:N', title='Initial LVC', sort=lvc_order)
                    ).properties(title="FAS $ by LVC")
                    
                    st.altair_chart(chart_fas_dol, use_container_width=False)

            st.divider()

            # --- 5. INSIGHTS & WHAT TO TEST NEXT ---
            st.subheader("5️⃣ Insights & what to test next")
            if pre_row is not None and post_row is not None:
                delta_contact = post_row["contact_rate"] - pre_row["contact_rate"]
                delta_fas_pct = post_row["fas_pct"] - pre_row["fas_pct"]
                delta_fas_dol = post_row["fas_dollars"] - pre_row["fas_dollars"]
                delta_dial = (post_row["speed_to_dial_median"] - pre_row["speed_to_dial_median"]) if pd.notna(post_row["speed_to_dial_median"]) and pd.notna(pre_row["speed_to_dial_median"]) else None
                delta_contact_min = (post_row["speed_to_contact_median"] - pre_row["speed_to_contact_median"]) if pd.notna(post_row["speed_to_contact_median"]) and pd.notna(pre_row["speed_to_contact_median"]) else None

                insights = []
                if delta_contact > 0.01:
                    insights.append(f"**Contact rate improved** in Post vs Pre (+{delta_contact:.1%}).")
                elif delta_contact < -0.01:
                    insights.append(f"**Contact rate declined** in Post vs Pre ({delta_contact:.1%}).")
                else:
                    insights.append("**Contact rate** is roughly stable Pre vs Post.")

                if delta_fas_pct > 0.01:
                    insights.append(f"**FAS % improved** in Post vs Pre (+{delta_fas_pct:.1%}).")
                elif delta_fas_pct < -0.01:
                    insights.append(f"**FAS % declined** in Post vs Pre ({delta_fas_pct:.1%}).")
                else:
                    insights.append("**FAS %** is roughly stable Pre vs Post.")

                if delta_fas_dol > 0:
                    insights.append(f"**FAS $** is higher in Post by ${delta_fas_dol:,.0f} (volume/quality effect).")
                elif delta_fas_dol < 0:
                    insights.append(f"**FAS $** is lower in Post by ${abs(delta_fas_dol):,.0f}.")

                if delta_dial is not None:
                    if delta_dial > 5:
                        insights.append(f"**Speed to dial** slowed in Post (median +{delta_dial:.0f} min).")
                    elif delta_dial < -5:
                        insights.append(f"**Speed to dial** improved in Post (median {delta_dial:.0f} min).")
                if delta_contact_min is not None:
                    if delta_contact_min > 5:
                        insights.append(f"**Speed to first contact** slowed in Post (median +{delta_contact_min:.0f} min).")
                    elif delta_contact_min < -5:
                        insights.append(f"**Speed to first contact** improved in Post (median {delta_contact_min:.0f} min).")
                pre_emails = (pre_row.get("emails_sent", 0) or 0)
                post_emails = (post_row.get("emails_sent", 0) or 0)
                pre_sms = (pre_row.get("outbound_sms", 0) or 0)
                post_sms = (post_row.get("outbound_sms", 0) or 0)
                pre_phx_att = (pre_row.get("phx_total_outbound_attempts", 0) or 0)
                post_phx_att = (post_row.get("phx_total_outbound_attempts", 0) or 0)
                pre_phx_contact_pct = (pre_row.get("phx_contact_rate", 0) or 0) * 100
                post_phx_contact_pct = (post_row.get("phx_contact_rate", 0) or 0) * 100
                insights.append(f"**Emails sent:** Pre {pre_emails:,.0f} → Post {post_emails:,.0f}. **Outbound SMS:** Pre {pre_sms:,.0f} → Post {post_sms:,.0f}.")
                insights.append(f"**PHX contact rate:** {pre_phx_contact_pct:.1f}% → {post_phx_contact_pct:.1f}%. **PHX outbound attempts:** Pre {pre_phx_att:,.0f} → Post {post_phx_att:,.0f}.")

                for line in insights:
                    st.markdown(f"- {line}")

                # --- Evidence summary (data supporting the recommendation) ---
                st.markdown("#### Evidence (Pre vs Post — supports recommendation)")
                pre_contact_pct = pre_row["contact_rate"] * 100
                post_contact_pct = post_row["contact_rate"] * 100
                pre_fas_pct = pre_row["fas_pct"] * 100
                post_fas_pct = post_row["fas_pct"] * 100
                med_dial_pre = pre_row["speed_to_dial_median"]
                med_dial_post = post_row["speed_to_dial_median"]
                med_cont_pre = pre_row["speed_to_contact_median"]
                med_cont_post = post_row["speed_to_contact_median"]
                st.markdown(f"""
                | Metric | Pre (12/19–1/26) | Post (1/27–3/6) | Δ (Post − Pre) |
                |--------|------------------|-----------------|----------------|
                | Contact rate | {pre_contact_pct:.1f}% | {post_contact_pct:.1f}% | {delta_contact*100:+.1f}pp |
                | FAS % | {pre_fas_pct:.1f}% | {post_fas_pct:.1f}% | {delta_fas_pct*100:+.1f}pp |
                | FAS $ | ${pre_row['fas_dollars']:,.0f} | ${post_row['fas_dollars']:,.0f} | ${delta_fas_dol:+,.0f} |
                | Speed to dial (median min) | {med_dial_pre:.0f} | {med_dial_post:.0f} | {(med_dial_post - med_dial_pre) if pd.notna(med_dial_post) and pd.notna(med_dial_pre) else '—'} |
                | Speed to 1st contact (median min) | {med_cont_pre:.0f} | {med_cont_post:.0f} | {(med_cont_post - med_cont_pre) if pd.notna(med_cont_post) and pd.notna(med_cont_pre) else '—'} |
                | Emails sent (total) | {pre_emails:,.0f} | {post_emails:,.0f} | {(post_emails - pre_emails):+,.0f} |
                | Outbound SMS (total) | {pre_sms:,.0f} | {post_sms:,.0f} | {(post_sms - pre_sms):+,.0f} |
                | PHX contact rate | {pre_phx_contact_pct:.1f}% | {post_phx_contact_pct:.1f}% | {(post_phx_contact_pct - pre_phx_contact_pct):+.1f}pp |
                | PHX total outbound attempts | {pre_phx_att:,.0f} | {post_phx_att:,.0f} | {(post_phx_att - pre_phx_att):+,.0f} |
                """)
                st.caption("Pre = before 24hr PHX delay; Post = with 24hr PHX delay for LVC 1–8.")

                st.markdown("#### 🌙 The 'Dead Zone' Insight (Why 4-Hour Delay is Risky)")
                st.markdown("""
                A flat 4-hour delay doesn't work for leads scored outside of business hours. Over **35% of leads** arrive in the Evening or Night/Early Morning, and **weekends** (Saturday afternoons / Sundays) have zero MA coverage.
                If a lead is scored at 6:00 PM, a 4-hour delay drops it to PHX at 10:00 PM. MAs only reach ~30% of these before the window expires. The same happens on weekends, when the clock expires before Monday morning.
                *(Data below is filtered to weekdays only to isolate the time-of-day issue from weekend coverage gaps)*
                """)
                dead_zone_data = pd.DataFrame({
                    "Time Scored": ["Morning (7am-Noon)", "Afternoon (Noon-5pm)", "Evening (5pm-Midnight)", "Night (Midnight-7am)"],
                    "% of Leads": ["27.2%", "11.4%", "13.0%", "48.4%"],
                    "MAs calling within 4 hrs": ["71.0%", "0.3%", "82.4%", "91.6%"],
                    "Contact Rate": ["30.9%", "28.1%", "29.7%", "33.7%"]
                })
                st.table(dead_zone_data)
                
                with st.expander("Show Proof: Evening/Night Leads Not Dialed by PHX Until 24+ Hours"):
                    st.markdown("""
                    Because a 24-hour delay pushes evening/night leads into the *next* evening/night, and PHX dialers may not operate at 2 AM, the real delay stretches to 30+ hours.  
                    **Weekends:** A lead scored on Friday night or Saturday will hit the 24-hour mark during the weekend (when teams aren't working), pushing the actual first dial to Monday (70+ hours later).  
                    *Below are real examples of leads scored after 5 PM PT (Post 1/27, **Weekdays Only**) and the actual time it took for PHX to dial them:*
                    """)
                    proof_data = pd.DataFrame({
                        "lendage_guid": [
                            "363ede9e-611c-49b6-a9b2-38482a682e53", 
                            "2331a18a-8727-4d2a-ba41-dd1c31c2fdc5", 
                            "5660ab53-26e0-4036-b08d-ca4025b03d50", 
                            "906cf92f-a71b-483d-89bf-86b1a67d380f"
                        ],
                        "initial_lead_score_datetime": ["2026-03-06 02:09 UTC", "2026-03-06 01:30 UTC", "2026-03-06 01:16 UTC", "2026-03-05 02:14 UTC"],
                        "MA First Dial": ["2026-03-06 09:00 UTC", "2026-03-06 09:00 UTC", "2026-03-06 08:00 UTC", "2026-03-05 09:00 UTC"],
                        "MA Contacted": ["No", "No", "No", "No"],
                        "first_dial_phx": ["2026-03-07 09:42 UTC", "2026-03-07 09:41 UTC", "2026-03-07 08:41 UTC", "2026-03-06 09:42 UTC"],
                        "Hours to PHX Dial": ["31.5 hours", "32.2 hours", "31.4 hours", "31.5 hours"]
                    })
                    st.table(proof_data)
                
                st.markdown("#### ⏳ PHX Time to First Dial (from Score Time)")
                st.markdown("""
                The 24-hour rule caused massive latency. Before the rule, PHX dialed 68% of weekday leads within 4 hours. 
                Now, **over 54% of weekday leads aren't dialed by PHX until after 24 hours** (because if the 24hr clock expires at night, they don't dial until the next day).
                *(Table excludes weekends to show this happens purely due to time-of-day)*
                """)
                phx_timing_data = pd.DataFrame({
                    "Period": ["Pre (12/19-1/26)", "Post (1/27-3/6)"],
                    "Median Hours to Dial": ["0 hours", "24 hours"],
                    "% Dialed < 4 Hours": ["67.9%", "14.8%"],
                    "% Dialed < 24 Hours": ["88.4%", "45.8%"],
                    "% Dialed 24-48 Hours": ["7.7%", "42.5%"],
                    "% Dialed > 48 Hours": ["3.9%", "11.6%"]
                })
                st.table(phx_timing_data)

                st.markdown("#### Recommended next test (data-supported)")
                rec = """**Recommendation: Test a "Same Day / Next Day" routing rule (Next Morning Sweep) instead of a flat hour delay.**
                - **Why (data):** With the current **24hr** PHX delay, conversion and contact declined. MAs slowed down their speed-to-lead, and PHX got locked out for 24-48+ hours, leaving leads untouched during their highest-intent window. However, a flat **4-hour delay** will fail for the ~35% of leads scored at night/evening, as the window will expire while MAs are off the clock. The same applies to weekends where there is zero coverage.
                - **The Solution:** MAs own the lead exclusively for the day it arrives *and* the following working morning. At **12:00 PM (Noon)** every working day, all uncontacted leads from the *previous day* drop into the PHX dialer.
                - **Why it's better:** It guarantees MAs get a fresh morning block to dial yesterday's afternoon/evening leads, avoids 4-hour expirations happening on Sundays or overnight, and ensures PHX gets them by hour 18-24 (of working time) at the latest, while the PHX team is actively staffed and dialing."""
                st.info(rec)
            st.divider()

            # --- 6. BY STARRING & BY MA (with FAS) ---
            st.subheader("6️⃣ By starring (Pre vs Post)")
            st.caption("**Calcs:** contact_rate = with_contact/lead_count; fas_pct = fas_count/lead_count; fas_dollars = SUM(e_loan_amount) for FAS leads; all by period and starring.")
            by_star = df_out.groupby(["period", "starring"]).agg(
                lead_count=("lendage_guid", "nunique"),
                with_contact=("had_contact", "sum"),
                fas_count=("had_fas", "sum"),
            ).reset_index()
            by_star["contact_rate"] = by_star["with_contact"] / by_star["lead_count"]
            by_star["fas_pct"] = by_star["fas_count"] / by_star["lead_count"]
            fas_d_star = df_out[df_out["had_fas"]].groupby(["period", "starring"])["e_loan_amount"].sum().reset_index()
            fas_d_star.columns = ["period", "starring", "fas_dollars"]
            by_star = by_star.merge(fas_d_star, on=["period", "starring"], how="left").fillna(0)
            star_pivot = by_star.pivot(index="starring", columns="period", values=["lead_count", "contact_rate", "fas_pct", "fas_dollars"])
            st.dataframe(star_pivot, use_container_width=True, hide_index=True)

            st.subheader("7️⃣ By mortgage advisor (Pre vs Post)")
            st.caption("**Calcs:** Same as By starring, grouped by period and mortgage_advisor.")
            by_ma = df_out.groupby(["period", "mortgage_advisor"]).agg(
                lead_count=("lendage_guid", "nunique"),
                with_contact=("had_contact", "sum"),
                fas_count=("had_fas", "sum"),
            ).reset_index()
            by_ma["contact_rate"] = by_ma["with_contact"] / by_ma["lead_count"]
            by_ma["fas_pct"] = by_ma["fas_count"] / by_ma["lead_count"]
            fas_d_ma = df_out[df_out["had_fas"]].groupby(["period", "mortgage_advisor"])["e_loan_amount"].sum().reset_index()
            fas_d_ma.columns = ["period", "mortgage_advisor", "fas_dollars"]
            by_ma = by_ma.merge(fas_d_ma, on=["period", "mortgage_advisor"], how="left").fillna(0)
            ma_pivot = by_ma.pivot(index="mortgage_advisor", columns="period", values=["lead_count", "contact_rate", "fas_pct", "fas_dollars"])
            st.dataframe(ma_pivot, use_container_width=True, hide_index=True)

            with st.expander("📋 Suggested slide copy (paste into PowerPoint)"):
                st.markdown("""
                - **Context:** As of 1/27, PHX delays calling LVC 1-8 by 24 hours post `initial_lead_score_datetime`; MAs have the first 24 hours to call.
                - **Pre:** 12/19/25–1/26/26 | **Post:** 1/27/26–3/6/26 (with 24hr PHX delay).
                - Use the **Insights & what to test next** section and the **Recommended next test** for directional guidance on testing 4 hr, 48 hr, or in-between delays.
                """)
        else:
            st.warning("No outreach data returned. Check LVC 1-8 and date range, and access to `lendage-data-platform.standardized_data.lendage_consumer_comms_lead_data`.")

    # --- TAB: Dead Zone Analysis ---
    with tab_dead_zone:
        st.header("💀 Dead Zone Analysis (1/27/2026 - 3/6/2026)")
        st.caption("Deep dive into the 32+ hour PHX delay impact (LVC 1-8, Date Range: 1/27/2026 - 3/6/2026)")
        
        # --- FILTERS ---
        st.subheader("🔍 Filters")
        
        # Fetch distinct sub_group and sent_to_sales_reason
        dz_filter_query = """
        SELECT DISTINCT sub_group, sent_to_sales_reason
        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
        WHERE DATE(sent_to_sales_date) BETWEEN '2026-01-27' AND '2026-03-06'
        """
        try:
            dz_filter_df = client.query(dz_filter_query).to_dataframe()
            dz_sg_options = sorted([str(x) for x in dz_filter_df['sub_group'].dropna().unique()])
            dz_rs_options = sorted([str(x) for x in dz_filter_df['sent_to_sales_reason'].dropna().unique()])
        except:
            dz_sg_options = []
            dz_rs_options = []
        
        col_dz_f1, col_dz_f2, col_dz_f3, col_dz_f4 = st.columns(4)
        
        with col_dz_f1:
            dz_lvc_options = [str(i) for i in range(1, 9)]
            selected_dz_lvc = st.multiselect("Initial LVC (default: 1-8)", options=dz_lvc_options, default=dz_lvc_options, key="dz_lvc")
            
        with col_dz_f2:
            selected_dz_subgroup = st.multiselect("Sub Group", options=dz_sg_options, default=[], key="dz_subgroup")
            
        with col_dz_f3:
            selected_dz_reason = st.multiselect("Sent to Sales Reason", options=dz_rs_options, default=[], key="dz_reason")
            
        with col_dz_f4:
            selected_dz_contact = st.selectbox("MA Contacted?", options=["All", "Yes", "No"], index=0, key="dz_contact")
            
        col_dz_f5, _, _, _ = st.columns(4)
        with col_dz_f5:
            include_weekends = st.toggle("Include Weekend Leads", value=False, key="dz_weekends", help="Toggle to include leads scored on Saturdays and Sundays. Default is OFF to show pure weekday delay behavior.")
            
        if not selected_dz_lvc:
            st.warning("Please select at least one LVC.")
            st.stop()
            
        dz_lvc_str = ", ".join([f"'{l}'" for l in selected_dz_lvc])
        dz_where_clauses = [f"initial_lead_value_cohort IN ({dz_lvc_str})"]
        
        if selected_dz_subgroup:
            sg_list = ", ".join([f"'{x}'" for x in selected_dz_subgroup])
            dz_where_clauses.append(f"sub_group IN ({sg_list})")
            
        if selected_dz_reason:
            rs_list = ", ".join([f"'{x}'" for x in selected_dz_reason])
            dz_where_clauses.append(f"sent_to_sales_reason IN ({rs_list})")
            
        if selected_dz_contact == "Yes":
            dz_where_clauses.append("sf__contacted_guid IS NOT NULL")
        elif selected_dz_contact == "No":
            dz_where_clauses.append("sf__contacted_guid IS NULL")
            
        if not include_weekends:
            dz_where_clauses.append("EXTRACT(DAYOFWEEK FROM initial_lead_score_datetime AT TIME ZONE 'America/Los_Angeles') NOT IN (1, 7)")
            
        dz_where_sql = " AND ".join(dz_where_clauses)
        
        dz_sql = f"""
        WITH phx_timing AS (
            SELECT 
                lendage_guid,
                CASE 
                    WHEN initial_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
                    WHEN initial_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
                    ELSE 'Other'
                END as lvc_group,
                initial_lead_value_cohort,
                EXTRACT(DAYOFWEEK FROM initial_lead_score_datetime AT TIME ZONE "America/Los_Angeles") as dow_num,
                FORMAT_DATETIME('%A', DATETIME(initial_lead_score_datetime, "America/Los_Angeles")) as weekday,
                TIMESTAMP_DIFF(first_dial_phx, initial_lead_score_datetime, MINUTE) / 60.0 as hours_to_phx_dial,
                DATE(lead_created_date) as lead_created_date,
                sub_group,
                initial_lead_score_datetime as score_datetime,
                first_call_attempt_datetime as ma_first_dial,
                first_dial_phx as phx_first_dial,
                CASE WHEN sf__contacted_guid IS NOT NULL THEN 'Yes' ELSE 'No' END as ma_contacted
            FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
            WHERE DATE(sent_to_sales_date) BETWEEN '2026-01-27' AND '2026-03-06'
              AND {dz_where_sql}
              AND first_dial_phx IS NOT NULL
              AND initial_lead_score_datetime IS NOT NULL
        )
        SELECT * FROM phx_timing
        """
        
        try:
            with st.spinner("Loading Dead Zone data..."):
                df_dz = client.query(dz_sql).to_dataframe()
                
            if not df_dz.empty:
                # Create the buckets
                df_dz['bucket'] = pd.cut(df_dz['hours_to_phx_dial'], 
                                        bins=[-1, 24, 28, 32, 36, 999999],
                                        labels=['< 24 hrs', '24 - 28 hrs', '28 - 32 hrs', '32 - 36 hrs', '36+ hrs'])
                
                view_type = st.radio("View tables as:", ["Percentages", "Raw Counts"], horizontal=True, key="dz_view_type")
                
                # --- By LVC Group (Overall) ---
                st.markdown("### 📊 PHX Time to First Dial by LVC Group")
                if not include_weekends:
                    st.caption("Note: This table excludes leads scored on weekends (Saturday/Sunday) to show pure weekday delay behavior. Toggle 'Include Weekend Leads' above to see all data.")
                
                # Filter out weekends for the overall LVC table if toggle is off
                if not include_weekends:
                    df_dz_lvc_source = df_dz[~df_dz['dow_num'].isin([1, 7])]
                else:
                    df_dz_lvc_source = df_dz
                
                dz_agg_lvc = df_dz_lvc_source.groupby('lvc_group').agg(
                    total_leads_dialed_by_phx=('lendage_guid', 'count'),
                    under_24h=('bucket', lambda x: (x == '< 24 hrs').sum()),
                    hours_24_to_28=('bucket', lambda x: (x == '24 - 28 hrs').sum()),
                    hours_28_to_32=('bucket', lambda x: (x == '28 - 32 hrs').sum()),
                    hours_32_to_36=('bucket', lambda x: (x == '32 - 36 hrs').sum()),
                    over_36h=('bucket', lambda x: (x == '36+ hrs').sum())
                ).reset_index()
                
                dz_pct_lvc = dz_agg_lvc.copy()
                for c in ['under_24h', 'hours_24_to_28', 'hours_28_to_32', 'hours_32_to_36', 'over_36h']:
                    dz_pct_lvc[c] = (dz_pct_lvc[c] / dz_pct_lvc['total_leads_dialed_by_phx']).apply(lambda x: f"{x:.1%}")
                
                display_df_lvc = dz_pct_lvc if view_type == "Percentages" else dz_agg_lvc
                st.dataframe(display_df_lvc, use_container_width=True, hide_index=True)
                
                # --- Weekly Trend Chart ---
                st.markdown("#### 📈 % of Leads by PHX Delay Bucket Over Time")
                # Create score_week for the chart
                df_dz_weekday_chart = df_dz_lvc_source.copy()
                df_dz_weekday_chart['score_week'] = pd.to_datetime(df_dz_weekday_chart['score_datetime']).dt.to_period('W-SUN').dt.start_time
                df_dz_weekday_chart['score_week_label'] = df_dz_weekday_chart['score_week'].dt.strftime('%m/%d')
                
                weekly_dz = df_dz_weekday_chart.groupby(['score_week_label', 'bucket'], observed=False).agg(
                    leads=('lendage_guid', 'count')
                ).reset_index()
                
                weekly_totals = weekly_dz.groupby('score_week_label')['leads'].sum().reset_index().rename(columns={'leads': 'total_leads'})
                weekly_dz = weekly_dz.merge(weekly_totals, on='score_week_label')
                weekly_dz['pct'] = weekly_dz['leads'] / weekly_dz['total_leads']
                
                # Filter out weeks with very low volume if necessary or just plot
                week_sort_dz = sorted(weekly_dz['score_week_label'].unique().tolist())
                bucket_order = ['< 24 hrs', '24 - 28 hrs', '28 - 32 hrs', '32 - 36 hrs', '36+ hrs']
                
                chart_dz_trend = alt.Chart(weekly_dz).mark_line(point=True).encode(
                    x=alt.X('score_week_label:N', title='Week Scored', sort=week_sort_dz),
                    y=alt.Y('pct:Q', title='% of Leads', axis=alt.Axis(format='.0%')),
                    color=alt.Color('bucket:N', title='Delay Bucket', sort=bucket_order),
                    tooltip=[
                        alt.Tooltip('score_week_label:N', title='Week'),
                        alt.Tooltip('bucket:N', title='Bucket'),
                        alt.Tooltip('pct:Q', title='% of Leads', format='.1%'),
                        alt.Tooltip('leads:Q', title='Lead Count')
                    ]
                ).properties(height=300)
                st.altair_chart(chart_dz_trend, use_container_width=True)
                
                st.divider()
                
                # --- By Weekday & LVC Group ---
                st.markdown("### 📊 PHX Time to First Dial by Weekday & LVC Group")
                
                # Aggregate for the top table
                dz_agg = df_dz.groupby(['weekday', 'dow_num', 'lvc_group']).agg(
                    total_leads_dialed_by_phx=('lendage_guid', 'count'),
                    under_24h=('bucket', lambda x: (x == '< 24 hrs').sum()),
                    hours_24_to_28=('bucket', lambda x: (x == '24 - 28 hrs').sum()),
                    hours_28_to_32=('bucket', lambda x: (x == '28 - 32 hrs').sum()),
                    hours_32_to_36=('bucket', lambda x: (x == '32 - 36 hrs').sum()),
                    over_36h=('bucket', lambda x: (x == '36+ hrs').sum())
                ).reset_index().sort_values(['dow_num', 'lvc_group'])
                
                dz_agg = dz_agg.drop(columns=['dow_num'])
                
                # Format counts vs percentages
                dz_pct = dz_agg.copy()
                for c in ['under_24h', 'hours_24_to_28', 'hours_28_to_32', 'hours_32_to_36', 'over_36h']:
                    dz_pct[c] = (dz_pct[c] / dz_pct['total_leads_dialed_by_phx']).apply(lambda x: f"{x:.1%}")
                
                display_df = dz_pct if view_type == "Percentages" else dz_agg
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.divider()
                
                # Interactive Details Section
                st.markdown("### 🔎 Deep Dive: Leads Delayed 32+ Hours")
                st.markdown("Filter to see the exact leads that got stuck in the dead zone. You can select a row to see more details, or export the full list.")
                
                # Detail dataframe
                df_32 = df_dz[df_dz['hours_to_phx_dial'] >= 32].copy()
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    sel_day = st.selectbox("Filter by Day Scored", options=["All"] + df_32['weekday'].dropna().unique().tolist())
                with col_f2:
                    sel_lvc = st.selectbox("Filter by Initial LVC", options=["All"] + sorted(df_32['initial_lead_value_cohort'].dropna().unique().tolist()))
                
                if sel_day != "All":
                    df_32 = df_32[df_32['weekday'] == sel_day]
                if sel_lvc != "All":
                    df_32 = df_32[df_32['initial_lead_value_cohort'] == sel_lvc]
                
                df_32_display = df_32[[
                    'lendage_guid', 'lead_created_date', 'sub_group', 'score_datetime', 'weekday',
                    'ma_first_dial', 'phx_first_dial', 'initial_lead_value_cohort', 
                    'ma_contacted', 'hours_to_phx_dial'
                ]].sort_values('hours_to_phx_dial', ascending=False)
                
                df_32_display['hours_to_phx_dial'] = df_32_display['hours_to_phx_dial'].round(1)
                
                st.write(f"Showing **{len(df_32_display)}** leads matching filters:")
                
                # Check streamlit version to see if on_select works, but we can just provide the dataframe normally.
                event = st.dataframe(
                    df_32_display,
                    use_container_width=True,
                    hide_index=True,
                    selection_mode="single-row",
                    on_select="rerun"
                )
                
                if hasattr(event, 'selection') and event.selection.rows:
                    selected_idx = event.selection.rows[0]
                    sel_row = df_32_display.iloc[selected_idx]
                    
                    st.success(f"**Selected Lead:** `{sel_row['lendage_guid']}`")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write(f"**Sub Group:** {sel_row['sub_group']}")
                        st.write(f"**Score Datetime:** {sel_row['score_datetime']}")
                        st.write(f"**Day Scored:** {sel_row['weekday']}")
                        st.write(f"**Initial LVC:** {sel_row['initial_lead_value_cohort']}")
                    with c2:
                        st.write(f"**MA First Dial:** {sel_row['ma_first_dial'] if pd.notna(sel_row['ma_first_dial']) else 'None'}")
                        st.write(f"**MA Contacted:** {sel_row['ma_contacted']}")
                    with c3:
                        st.write(f"**PHX First Dial:** {sel_row['phx_first_dial']}")
                        st.write(f"**Delay:** {sel_row['hours_to_phx_dial']} hours")
                else:
                    st.info("💡 Click on any row in the table above to view its details here.")
                    
            else:
                st.warning("No dead zone data found.")
        except Exception as e:
            st.error(f"Error loading Dead Zone tab: {e}")

    # --- TAB: Volume vs Conversion Analysis ---
    with tab_vol_conv:
        st.header("📈 Volume vs Conversion Impact Analysis")
        st.caption("Understanding whether FAS $ decline is driven by volume drop or conversion decline")
        
        st.markdown("""
        ### What This Analysis Shows
        This analysis compares **September 2025** (high volume month) against **January 2026** (aged cohort)
        to determine what's driving FAS $ differences:
        
        - **Volume Impact**: What if we had Sept volume but Jan conversion rates?
        - **Conversion Impact**: What if we had Jan volume but Sept conversion rates?
        
        *Note: February 2026 excluded as leads are still being worked and haven't fully matured.*
        """)
        
        try:
            # Query for monthly metrics
            vol_conv_query = """
            SELECT
                FORMAT_DATE('%Y-%m', DATE(sent_to_sales_date)) as month,
                DATE_TRUNC(DATE(sent_to_sales_date), MONTH) as month_date,
                COUNT(DISTINCT lendage_guid) as sts_volume,
                COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_count,
                SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars
            FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
            WHERE DATE(sent_to_sales_date) >= '2025-08-01'
            AND DATE(sent_to_sales_date) <= '2026-01-31'
            AND sent_to_sales_date IS NOT NULL
            GROUP BY 1, 2
            ORDER BY month_date
            """
            
            df_vol_conv = client.query(vol_conv_query).to_dataframe()
            
            if len(df_vol_conv) > 0:
                # Calculate metrics
                df_vol_conv['fas_rate'] = df_vol_conv['fas_count'] / df_vol_conv['sts_volume']
                df_vol_conv['fas_per_sts'] = df_vol_conv['fas_dollars'] / df_vol_conv['sts_volume']
                
                # Define periods - Sept 2025 vs Jan 2026 only
                sept_2025 = df_vol_conv[df_vol_conv['month'] == '2025-09']
                jan_2026 = df_vol_conv[df_vol_conv['month'] == '2026-01']
                
                # Aggregated metrics for each period
                sept_agg = {
                    'sts_volume': sept_2025['sts_volume'].sum(),
                    'fas_count': sept_2025['fas_count'].sum(),
                    'fas_dollars': sept_2025['fas_dollars'].sum(),
                }
                sept_agg['fas_rate'] = sept_agg['fas_count'] / sept_agg['sts_volume'] if sept_agg['sts_volume'] > 0 else 0
                sept_agg['fas_per_sts'] = sept_agg['fas_dollars'] / sept_agg['sts_volume'] if sept_agg['sts_volume'] > 0 else 0
                
                jan_agg = {
                    'sts_volume': jan_2026['sts_volume'].sum(),
                    'fas_count': jan_2026['fas_count'].sum(),
                    'fas_dollars': jan_2026['fas_dollars'].sum(),
                }
                jan_agg['fas_rate'] = jan_agg['fas_count'] / jan_agg['sts_volume'] if jan_agg['sts_volume'] > 0 else 0
                jan_agg['fas_per_sts'] = jan_agg['fas_dollars'] / jan_agg['sts_volume'] if jan_agg['sts_volume'] > 0 else 0
                
                st.divider()
                
                # === ACTUAL PERFORMANCE COMPARISON ===
                st.subheader("📊 Actual Performance: Sept 2025 vs Jan 2026")
                
                col_actual1, col_actual2, col_actual3, col_actual4, col_actual5 = st.columns(5)
                
                with col_actual1:
                    delta_vol = jan_agg['sts_volume'] - sept_agg['sts_volume']
                    delta_pct = (delta_vol / sept_agg['sts_volume'] * 100) if sept_agg['sts_volume'] > 0 else 0
                    st.metric(
                        "Sent to Sales Volume",
                        f"{jan_agg['sts_volume']:,.0f}",
                        f"{delta_pct:+.1f}% vs Sept",
                        delta_color="normal"
                    )
                    st.caption(f"Sept 2025: {sept_agg['sts_volume']:,.0f}")
                
                with col_actual2:
                    delta_rate = (jan_agg['fas_rate'] - sept_agg['fas_rate']) * 100
                    st.metric(
                        "FAS Rate",
                        f"{jan_agg['fas_rate']:.1%}",
                        f"{delta_rate:+.1f}pp vs Sept",
                        delta_color="normal"
                    )
                    st.caption(f"Sept 2025: {sept_agg['fas_rate']:.1%}")
                
                with col_actual3:
                    delta_per_sts = jan_agg['fas_per_sts'] - sept_agg['fas_per_sts']
                    st.metric(
                        "FAS $ per StS",
                        f"${jan_agg['fas_per_sts']:,.0f}",
                        f"${delta_per_sts:+,.0f} vs Sept",
                        delta_color="normal"
                    )
                    st.caption(f"Sept 2025: ${sept_agg['fas_per_sts']:,.0f}")
                
                with col_actual4:
                    delta_fas = jan_agg['fas_count'] - sept_agg['fas_count']
                    st.metric(
                        "FAS Count",
                        f"{jan_agg['fas_count']:,.0f}",
                        f"{delta_fas:+,.0f} vs Sept",
                        delta_color="normal"
                    )
                    st.caption(f"Sept 2025: {sept_agg['fas_count']:,.0f}")
                
                with col_actual5:
                    delta_dollars = jan_agg['fas_dollars'] - sept_agg['fas_dollars']
                    delta_dollars_pct = (delta_dollars / sept_agg['fas_dollars'] * 100) if sept_agg['fas_dollars'] > 0 else 0
                    st.metric(
                        "Total FAS $",
                        f"${jan_agg['fas_dollars']:,.0f}",
                        f"{delta_dollars_pct:+.1f}% vs Sept",
                        delta_color="normal"
                    )
                    st.caption(f"Sept 2025: ${sept_agg['fas_dollars']:,.0f}")
                
                st.divider()
                
                # === WHAT-IF SCENARIOS ===
                st.subheader("🔮 What-If Analysis: Volume vs Conversion Impact")
                
                st.markdown("""
                **Key Question:** If FAS $ is down, is it because:
                1. **Volume dropped** (fewer leads to convert)?
                2. **Conversion dropped** (worse FAS rate or lower $ per FAS)?
                """)
                
                # Scenario 1: Sept Volume × Jan Conversion
                scenario1_fas_dollars = sept_agg['sts_volume'] * jan_agg['fas_per_sts']
                scenario1_fas_count = sept_agg['sts_volume'] * jan_agg['fas_rate']
                
                # Scenario 2: Jan Volume × Sept Conversion  
                scenario2_fas_dollars = jan_agg['sts_volume'] * sept_agg['fas_per_sts']
                scenario2_fas_count = jan_agg['sts_volume'] * sept_agg['fas_rate']
                
                col_s1, col_s2, col_s3 = st.columns(3)
                
                with col_s1:
                    st.markdown("### 📉 Actual Jan 2026")
                    st.markdown(f"""
                    - **Volume:** {jan_agg['sts_volume']:,.0f} leads
                    - **FAS Rate:** {jan_agg['fas_rate']:.1%}
                    - **FAS $/StS:** ${jan_agg['fas_per_sts']:,.0f}
                    - **Total FAS $:** ${jan_agg['fas_dollars']:,.0f}
                    """)
                
                with col_s2:
                    st.markdown("### 📊 Scenario 1: Sept Volume + Jan Conversion")
                    st.info("What if we had Sept's volume but Jan's conversion rates?")
                    st.markdown(f"""
                    - **Volume:** {sept_agg['sts_volume']:,.0f} leads *(Sept)*
                    - **FAS Rate:** {jan_agg['fas_rate']:.1%} *(Jan)*
                    - **FAS $/StS:** ${jan_agg['fas_per_sts']:,.0f} *(Jan)*
                    - **Projected FAS $:** ${scenario1_fas_dollars:,.0f}
                    """)
                    diff_s1 = scenario1_fas_dollars - sept_agg['fas_dollars']
                    if diff_s1 < 0:
                        st.error(f"⚠️ Still ${abs(diff_s1):,.0f} below Sept actual → **Conversion is hurting**")
                    else:
                        st.success(f"✅ ${diff_s1:,.0f} above Sept actual → Conversion improved")
                
                with col_s3:
                    st.markdown("### 📊 Scenario 2: Jan Volume + Sept Conversion")
                    st.info("What if we had Jan's volume but Sept's conversion rates?")
                    st.markdown(f"""
                    - **Volume:** {jan_agg['sts_volume']:,.0f} leads *(Jan)*
                    - **FAS Rate:** {sept_agg['fas_rate']:.1%} *(Sept)*
                    - **FAS $/StS:** ${sept_agg['fas_per_sts']:,.0f} *(Sept)*
                    - **Projected FAS $:** ${scenario2_fas_dollars:,.0f}
                    """)
                    diff_s2 = scenario2_fas_dollars - sept_agg['fas_dollars']
                    if diff_s2 < 0:
                        st.error(f"⚠️ Still ${abs(diff_s2):,.0f} below Sept actual → **Volume is hurting**")
                    else:
                        st.success(f"✅ ${diff_s2:,.0f} above Sept actual → Volume improved")
                
                st.divider()
                
                # === IMPACT DECOMPOSITION ===
                st.subheader("🧮 Impact Decomposition")
                
                actual_gap = jan_agg['fas_dollars'] - sept_agg['fas_dollars']
                volume_impact = scenario2_fas_dollars - sept_agg['fas_dollars']  # Holding conversion constant
                conversion_impact = scenario1_fas_dollars - sept_agg['fas_dollars']  # Holding volume constant
                
                # Alternative decomposition
                volume_contribution = (jan_agg['sts_volume'] - sept_agg['sts_volume']) * sept_agg['fas_per_sts']
                conversion_contribution = jan_agg['sts_volume'] * (jan_agg['fas_per_sts'] - sept_agg['fas_per_sts'])
                
                col_d1, col_d2, col_d3 = st.columns(3)
                
                with col_d1:
                    st.metric(
                        "Total FAS $ Gap",
                        f"${actual_gap:,.0f}",
                        "Jan vs Sept"
                    )
                
                with col_d2:
                    vol_pct = (volume_contribution / abs(actual_gap) * 100) if actual_gap != 0 else 0
                    st.metric(
                        "Volume Impact",
                        f"${volume_contribution:,.0f}",
                        f"{vol_pct:.0f}% of gap"
                    )
                    st.caption("(Change in leads × Sept conversion)")
                
                with col_d3:
                    conv_pct = (conversion_contribution / abs(actual_gap) * 100) if actual_gap != 0 else 0
                    st.metric(
                        "Conversion Impact",
                        f"${conversion_contribution:,.0f}",
                        f"{conv_pct:.0f}% of gap"
                    )
                    st.caption("(Jan leads × Change in FAS$/StS)")
                
                # Waterfall-style breakdown
                st.markdown("### 📊 FAS $ Bridge: Sept 2025 → Jan 2026")
                
                waterfall_data = pd.DataFrame({
                    'Category': ['Sept 2025 Actual', 'Volume Impact', 'Conversion Impact', 'Jan 2026 Actual'],
                    'Value': [sept_agg['fas_dollars'], volume_contribution, conversion_contribution, jan_agg['fas_dollars']],
                    'Type': ['Baseline', 'Impact', 'Impact', 'Result']
                })
                
                fig_waterfall = go.Figure(go.Waterfall(
                    name="FAS $",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "total"],
                    x=waterfall_data['Category'],
                    y=[sept_agg['fas_dollars'], volume_contribution, conversion_contribution, 0],
                    text=[f"${sept_agg['fas_dollars']:,.0f}", f"${volume_contribution:+,.0f}", f"${conversion_contribution:+,.0f}", f"${jan_agg['fas_dollars']:,.0f}"],
                    textposition="outside",
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#2ca02c"}},
                    decreasing={"marker": {"color": "#d62728"}},
                    totals={"marker": {"color": "#1f77b4"}}
                ))
                fig_waterfall.update_layout(
                    title="FAS $ Bridge Analysis",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_waterfall, use_container_width=True)
                
                st.divider()
                
                # === MONTHLY BREAKDOWN ===
                st.subheader("📅 Monthly Breakdown")
                
                # Show individual months
                monthly_display = df_vol_conv[['month', 'sts_volume', 'fas_count', 'fas_rate', 'fas_per_sts', 'fas_dollars']].copy()
                monthly_display.columns = ['Month', 'StS Volume', 'FAS Count', 'FAS Rate', 'FAS $/StS', 'Total FAS $']
                monthly_display['FAS Rate'] = monthly_display['FAS Rate'].apply(lambda x: f"{x:.1%}")
                monthly_display['FAS $/StS'] = monthly_display['FAS $/StS'].apply(lambda x: f"${x:,.0f}")
                monthly_display['StS Volume'] = monthly_display['StS Volume'].apply(lambda x: f"{x:,}")
                monthly_display['FAS Count'] = monthly_display['FAS Count'].apply(lambda x: f"{x:,}")
                monthly_display['Total FAS $'] = monthly_display['Total FAS $'].apply(lambda x: f"${x:,.0f}")
                
                st.dataframe(monthly_display, use_container_width=True, hide_index=True)
                
                # Charts
                col_ch1, col_ch2 = st.columns(2)
                
                with col_ch1:
                    fig_vol = alt.Chart(df_vol_conv).mark_bar(color='#4cc9f0').encode(
                        x=alt.X('month:N', title='Month', sort=None),
                        y=alt.Y('sts_volume:Q', title='Sent to Sales Volume'),
                        tooltip=[
                            alt.Tooltip('month:N', title='Month'),
                            alt.Tooltip('sts_volume:Q', title='Volume', format=',')
                        ]
                    ).properties(
                        title='📊 Sent to Sales Volume by Month',
                        height=300
                    )
                    st.altair_chart(fig_vol, use_container_width=True)
                
                with col_ch2:
                    fig_fas_rate = alt.Chart(df_vol_conv).mark_line(point=True, color='#f9c74f').encode(
                        x=alt.X('month:N', title='Month', sort=None),
                        y=alt.Y('fas_rate:Q', title='FAS Rate', axis=alt.Axis(format='.0%')),
                        tooltip=[
                            alt.Tooltip('month:N', title='Month'),
                            alt.Tooltip('fas_rate:Q', title='FAS Rate', format='.1%')
                        ]
                    ).properties(
                        title='📈 FAS Rate by Month',
                        height=300
                    )
                    st.altair_chart(fig_fas_rate, use_container_width=True)
                
                col_ch3, col_ch4 = st.columns(2)
                
                with col_ch3:
                    fig_fas_per = alt.Chart(df_vol_conv).mark_line(point=True, color='#2ca02c').encode(
                        x=alt.X('month:N', title='Month', sort=None),
                        y=alt.Y('fas_per_sts:Q', title='FAS $ per StS', axis=alt.Axis(format='$,.0f')),
                        tooltip=[
                            alt.Tooltip('month:N', title='Month'),
                            alt.Tooltip('fas_per_sts:Q', title='FAS $/StS', format='$,.0f')
                        ]
                    ).properties(
                        title='💵 FAS $ per Sent to Sales by Month',
                        height=300
                    )
                    st.altair_chart(fig_fas_per, use_container_width=True)
                
                with col_ch4:
                    fig_total = alt.Chart(df_vol_conv).mark_bar(color='#9d4edd').encode(
                        x=alt.X('month:N', title='Month', sort=None),
                        y=alt.Y('fas_dollars:Q', title='Total FAS $', axis=alt.Axis(format='$,.0f')),
                        tooltip=[
                            alt.Tooltip('month:N', title='Month'),
                            alt.Tooltip('fas_dollars:Q', title='Total FAS $', format='$,.0f')
                        ]
                    ).properties(
                        title='💰 Total FAS $ by Month',
                        height=300
                    )
                    st.altair_chart(fig_total, use_container_width=True)
                
                st.divider()
                
                # === KEY INSIGHTS ===
                st.subheader("💡 Key Insights")
                
                col_ins1, col_ins2 = st.columns(2)
                
                with col_ins1:
                    st.markdown("### 🔍 Diagnosis")
                    
                    vol_pct_impact = abs(volume_contribution) / (abs(volume_contribution) + abs(conversion_contribution)) * 100 if (abs(volume_contribution) + abs(conversion_contribution)) > 0 else 0
                    conv_pct_impact = abs(conversion_contribution) / (abs(volume_contribution) + abs(conversion_contribution)) * 100 if (abs(volume_contribution) + abs(conversion_contribution)) > 0 else 0
                    
                    if abs(volume_contribution) > abs(conversion_contribution):
                        st.error(f"""
                        **Primary Driver: VOLUME** ({vol_pct_impact:.0f}% of impact)
                        
                        The decline in FAS $ is primarily driven by having fewer leads to work.
                        Even with improved conversion, the lower volume limits FAS $ output.
                        """)
                    else:
                        st.warning(f"""
                        **Primary Driver: CONVERSION** ({conv_pct_impact:.0f}% of impact)
                        
                        The decline in FAS $ is primarily driven by lower conversion (FAS rate or $ per FAS).
                        With the same volume, we would have generated less FAS $.
                        """)
                
                with col_ins2:
                    st.markdown("### 📋 Recommendations")
                    
                    if abs(volume_contribution) > abs(conversion_contribution):
                        st.markdown("""
                        **Since Volume is the main driver:**
                        
                        1. 🎯 **Increase lead acquisition** - Work with marketing to drive more volume
                        2. 📊 **Optimize lead sources** - Focus on channels with higher conversion
                        3. ⏰ **Reduce churn** - Convert more of the leads you do have
                        4. 💰 **Maximize FAS $ per lead** - Focus on higher loan amounts
                        """)
                    else:
                        st.markdown("""
                        **Since Conversion is the main driver:**
                        
                        1. ⚡ **Speed to contact** - Faster calls = higher conversion
                        2. 👤 **Lead quality** - Review persona/LVC mix changes
                        3. 📞 **Sales process** - Training and process optimization
                        4. 🎯 **Prioritization** - Focus on high-value leads first
                        """)
            
            else:
                st.warning("No data found for the selected period.")
                
        except Exception as e:
            st.error(f"Volume vs Conversion Error: {e}")
            st.exception(e)

    # --- TAB: FAS Analysis (In-Period) ---
    with tab_fas_analysis:
        st.header("✅ FAS Analysis (In-Period)")
        st.caption("Analyzing FAS performance based on when FAS occurred (full_app_submit_datetime), not when leads were sent to sales")
        
        st.info("📅 **Note:** All metrics are based on **FAS date** (full_app_submit_datetime), not sent_to_sales_date. This shows actual FAS output per period.")
        
        try:
            # Query for in-period FAS data
            fas_analysis_query = """
            WITH fas_data AS (
                SELECT
                    lendage_guid,
                    DATE(full_app_submit_datetime) as fas_date,
                    DATE_TRUNC(DATE(full_app_submit_datetime), WEEK(SUNDAY)) as fas_week,
                    DATE(sent_to_sales_date) as sts_date,
                    DATE_TRUNC(DATE(sent_to_sales_date), WEEK(SUNDAY)) as sts_week,
                    DATE_DIFF(DATE(full_app_submit_datetime), DATE(sent_to_sales_date), DAY) as days_to_fas,
                    e_loan_amount,
                    adjusted_lead_value_cohort,
                    CASE 
                        WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
                        WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
                        WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
                        WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
                        ELSE 'Other'
                    END as lvc_group,
                    mortgage_advisor,
                    persona
                FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                WHERE full_app_submit_datetime IS NOT NULL
                AND DATE(full_app_submit_datetime) >= '2025-10-01'
                AND DATE(full_app_submit_datetime) <= CURRENT_DATE()
            )
            SELECT
                fas_week,
                lvc_group,
                sts_week,
                mortgage_advisor,
                persona,
                days_to_fas,
                COUNT(DISTINCT lendage_guid) as fas_count,
                SUM(e_loan_amount) as fas_dollars
            FROM fas_data
            GROUP BY 1, 2, 3, 4, 5, 6
            """
            
            df_fas = client.query(fas_analysis_query).to_dataframe()
            
            if len(df_fas) > 0:
                df_fas['fas_week'] = pd.to_datetime(df_fas['fas_week'])
                df_fas['sts_week'] = pd.to_datetime(df_fas['sts_week'])
                
                # Also get STS volume for rate calculation
                sts_volume_query = """
                SELECT
                    DATE_TRUNC(DATE(sent_to_sales_date), WEEK(SUNDAY)) as sts_week,
                    COUNT(DISTINCT lendage_guid) as sts_volume
                FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                WHERE sent_to_sales_date IS NOT NULL
                AND DATE(sent_to_sales_date) >= '2025-10-01'
                AND DATE(sent_to_sales_date) <= CURRENT_DATE()
                GROUP BY 1
                """
                df_sts_vol = client.query(sts_volume_query).to_dataframe()
                df_sts_vol['sts_week'] = pd.to_datetime(df_sts_vol['sts_week'])
                
                # === WEEKLY OVERVIEW CHARTS ===
                st.subheader("📈 Weekly FAS Metrics (In-Period)")
                
                # Aggregate by FAS week
                weekly_fas = df_fas.groupby('fas_week').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                
                # Get STS volume for each FAS week (for rate calculation - using same week STS as denominator)
                weekly_fas = weekly_fas.merge(
                    df_sts_vol.rename(columns={'sts_week': 'fas_week'}), 
                    on='fas_week', 
                    how='left'
                )
                weekly_fas['sts_volume'] = weekly_fas['sts_volume'].fillna(1)
                weekly_fas['fas_rate'] = weekly_fas['fas_count'] / weekly_fas['sts_volume']
                weekly_fas['fas_per_sts'] = weekly_fas['fas_dollars'] / weekly_fas['sts_volume']
                
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    bars_qty = alt.Chart(weekly_fas).mark_bar(color='#4cc9f0').encode(
                        x=alt.X('fas_week:T', title='Week (by FAS Date)', axis=alt.Axis(format='%b %d')),
                        y=alt.Y('fas_count:Q', title='FAS Count'),
                        tooltip=[
                            alt.Tooltip('fas_week:T', title='Week', format='%b %d, %Y'),
                            alt.Tooltip('fas_count:Q', title='FAS Count', format=','),
                            alt.Tooltip('sts_volume:Q', title='StS Volume (same week)', format=',')
                        ]
                    )
                    text_qty = bars_qty.mark_text(dy=-10, fontSize=10).encode(
                        text=alt.Text('fas_count:Q', format=',')
                    )
                    fig_fas_qty = (bars_qty + text_qty).properties(title='📊 FAS Quantity by Week (In-Period)', height=300)
                    st.altair_chart(fig_fas_qty, use_container_width=True)
                    st.caption("**Field:** `full_app_submit_datetime` | **Calc:** COUNT(DISTINCT lendage_guid) grouped by FAS week")
                
                with col_f2:
                    # Format dollars in millions for display
                    weekly_fas['fas_dollars_m'] = weekly_fas['fas_dollars'] / 1000000
                    bars_dollars = alt.Chart(weekly_fas).mark_bar(color='#2ca02c').encode(
                        x=alt.X('fas_week:T', title='Week (by FAS Date)', axis=alt.Axis(format='%b %d')),
                        y=alt.Y('fas_dollars:Q', title='FAS $', axis=alt.Axis(format='$,.0f')),
                        tooltip=[
                            alt.Tooltip('fas_week:T', title='Week', format='%b %d, %Y'),
                            alt.Tooltip('fas_dollars:Q', title='FAS $', format='$,.0f')
                        ]
                    )
                    text_dollars = bars_dollars.mark_text(dy=-10, fontSize=9).encode(
                        text=alt.Text('fas_dollars_m:Q', format='$,.1fM')
                    )
                    fig_fas_dollars = (bars_dollars + text_dollars).properties(title='💰 FAS $ by Week (In-Period)', height=300)
                    st.altair_chart(fig_fas_dollars, use_container_width=True)
                    st.caption("**Field:** `e_loan_amount` | **Calc:** SUM(e_loan_amount) grouped by FAS week")
                
                col_f3, col_f4 = st.columns(2)
                
                with col_f3:
                    line_rate = alt.Chart(weekly_fas).mark_line(point=True, color='#f9c74f').encode(
                        x=alt.X('fas_week:T', title='Week (by FAS Date)', axis=alt.Axis(format='%b %d')),
                        y=alt.Y('fas_rate:Q', title='FAS Rate', axis=alt.Axis(format='.0%')),
                        tooltip=[
                            alt.Tooltip('fas_week:T', title='Week', format='%b %d, %Y'),
                            alt.Tooltip('fas_rate:Q', title='FAS Rate', format='.1%'),
                            alt.Tooltip('fas_count:Q', title='FAS Count', format=','),
                            alt.Tooltip('sts_volume:Q', title='StS Volume', format=',')
                        ]
                    )
                    text_rate = line_rate.mark_text(dy=-12, fontSize=9).encode(
                        text=alt.Text('fas_rate:Q', format='.1%')
                    )
                    fig_fas_rate = (line_rate + text_rate).properties(title='📈 FAS Rate by Week (In-Period FAS / Same Week StS)', height=300)
                    st.altair_chart(fig_fas_rate, use_container_width=True)
                    st.caption("**Calc:** FAS Count (in-period) / StS Volume (same week) — *Note: This is an approximation, not true cohort rate*")
                
                with col_f4:
                    # Format for display
                    weekly_fas['fas_per_sts_k'] = weekly_fas['fas_per_sts'] / 1000
                    line_per_sts = alt.Chart(weekly_fas).mark_line(point=True, color='#9d4edd').encode(
                        x=alt.X('fas_week:T', title='Week (by FAS Date)', axis=alt.Axis(format='%b %d')),
                        y=alt.Y('fas_per_sts:Q', title='FAS $ per StS', axis=alt.Axis(format='$,.0f')),
                        tooltip=[
                            alt.Tooltip('fas_week:T', title='Week', format='%b %d, %Y'),
                            alt.Tooltip('fas_per_sts:Q', title='FAS $/StS', format='$,.0f')
                        ]
                    )
                    text_per_sts = line_per_sts.mark_text(dy=-12, fontSize=9).encode(
                        text=alt.Text('fas_per_sts_k:Q', format='$,.1fK')
                    )
                    fig_fas_per_sts = (line_per_sts + text_per_sts).properties(title='💵 FAS $ per StS Lead by Week', height=300)
                    st.altair_chart(fig_fas_per_sts, use_container_width=True)
                    st.caption("**Calc:** FAS $ (in-period) / StS Volume (same week)")
                
                st.divider()
                
                # === WEEK OF 2/8/2026 DEEP DIVE ===
                st.subheader("🔍 Week of 2/8/2026 Deep Dive: What Drove the Uptick?")
                st.caption("*Week runs Sunday to Saturday. 2/8/2026 is the most recent full week.*")
                
                # Use Sunday-based week start
                target_week = pd.Timestamp('2026-02-08')  # Week starting Sunday Feb 8
                prev_4_weeks = [target_week - pd.Timedelta(weeks=i) for i in range(1, 5)]  # Feb 1, Jan 25, Jan 18, Jan 11
                
                # Filter data
                target_week_data = df_fas[df_fas['fas_week'] == target_week]
                prev_weeks_data = df_fas[df_fas['fas_week'].isin(prev_4_weeks)]
                
                # Aggregate for comparison
                target_agg = {
                    'fas_count': target_week_data['fas_count'].sum(),
                    'fas_dollars': target_week_data['fas_dollars'].sum()
                }
                
                prev_agg = {
                    'fas_count': prev_weeks_data['fas_count'].sum() / 4,  # Average
                    'fas_dollars': prev_weeks_data['fas_dollars'].sum() / 4
                }
                
                # KPI Comparison
                col_k1, col_k2, col_k3, col_k4 = st.columns(4)
                
                with col_k1:
                    delta_count = target_agg['fas_count'] - prev_agg['fas_count']
                    delta_pct = (delta_count / prev_agg['fas_count'] * 100) if prev_agg['fas_count'] > 0 else 0
                    st.metric(
                        "FAS Count (2/8 Week)",
                        f"{target_agg['fas_count']:,.0f}",
                        f"{delta_pct:+.1f}% vs prev 4-wk avg"
                    )
                    st.caption(f"Prev 4-wk avg: {prev_agg['fas_count']:,.0f}")
                
                with col_k2:
                    delta_dollars = target_agg['fas_dollars'] - prev_agg['fas_dollars']
                    delta_dollars_pct = (delta_dollars / prev_agg['fas_dollars'] * 100) if prev_agg['fas_dollars'] > 0 else 0
                    st.metric(
                        "FAS $ (2/8 Week)",
                        f"${target_agg['fas_dollars']:,.0f}",
                        f"{delta_dollars_pct:+.1f}% vs prev 4-wk avg"
                    )
                    st.caption(f"Prev 4-wk avg: ${prev_agg['fas_dollars']:,.0f}")
                
                with col_k3:
                    avg_loan_target = target_agg['fas_dollars'] / target_agg['fas_count'] if target_agg['fas_count'] > 0 else 0
                    avg_loan_prev = prev_agg['fas_dollars'] / prev_agg['fas_count'] if prev_agg['fas_count'] > 0 else 0
                    delta_loan = avg_loan_target - avg_loan_prev
                    st.metric(
                        "Avg FAS Loan (2/8 Week)",
                        f"${avg_loan_target:,.0f}",
                        f"${delta_loan:+,.0f} vs prev avg"
                    )
                    st.caption(f"Prev 4-wk avg: ${avg_loan_prev:,.0f}")
                
                with col_k4:
                    # Days to FAS (weighted average using aggregated data)
                    avg_days_target = (target_week_data['days_to_fas'] * target_week_data['fas_count']).sum() / target_week_data['fas_count'].sum() if target_week_data['fas_count'].sum() > 0 else 0
                    avg_days_prev = (prev_weeks_data['days_to_fas'] * prev_weeks_data['fas_count']).sum() / prev_weeks_data['fas_count'].sum() if prev_weeks_data['fas_count'].sum() > 0 else 0
                    delta_days = avg_days_target - avg_days_prev
                    st.metric(
                        "Avg Days to FAS",
                        f"{avg_days_target:.1f}",
                        f"{delta_days:+.1f} days vs prev avg",
                        delta_color="inverse"
                    )
                    st.caption(f"Prev 4-wk avg: {avg_days_prev:.1f} days")
                
                st.divider()
                
                # === BREAKDOWN: VINTAGE ANALYSIS ===
                st.markdown("### 📅 Vintage Analysis: Which Sent-to-Sales Weeks Are Converting?")
                st.caption("Did the 2/8 FAS uptick come from recent vintages or older leads finally converting?")
                
                # Group by STS week for target and prev
                vintage_target = target_week_data.groupby('sts_week').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                vintage_target['period'] = '2/8 Week'
                
                vintage_prev = prev_weeks_data.groupby('sts_week').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                vintage_prev['fas_count'] = vintage_prev['fas_count'] / 4  # Average
                vintage_prev['fas_dollars'] = vintage_prev['fas_dollars'] / 4
                vintage_prev['period'] = 'Prev 4-wk Avg'
                
                # Calculate days from STS to FAS week
                vintage_target['days_from_sts'] = (target_week - vintage_target['sts_week']).dt.days
                vintage_prev['days_from_sts'] = (prev_4_weeks[0] - vintage_prev['sts_week']).dt.days  # Use most recent prev week as reference
                
                # Bin into vintage buckets
                def vintage_bucket(days):
                    if days <= 7:
                        return '0-7 days (Same Week)'
                    elif days <= 14:
                        return '8-14 days'
                    elif days <= 30:
                        return '15-30 days'
                    elif days <= 60:
                        return '31-60 days'
                    else:
                        return '60+ days'
                
                vintage_target['vintage_bucket'] = vintage_target['days_from_sts'].apply(vintage_bucket)
                
                # Aggregate by bucket
                bucket_target = vintage_target.groupby('vintage_bucket').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                bucket_target['period'] = 'Week of 2/8'
                
                # Do same for prev weeks (simplified)
                vintage_prev_calc = prev_weeks_data.copy()
                vintage_prev_calc['days_to_fas'] = vintage_prev_calc['days_to_fas'].fillna(0)
                
                def days_to_bucket(days):
                    if days <= 7:
                        return '0-7 days (Same Week)'
                    elif days <= 14:
                        return '8-14 days'
                    elif days <= 30:
                        return '15-30 days'
                    elif days <= 60:
                        return '31-60 days'
                    else:
                        return '60+ days'
                
                vintage_prev_calc['vintage_bucket'] = vintage_prev_calc['days_to_fas'].apply(days_to_bucket)
                bucket_prev = vintage_prev_calc.groupby('vintage_bucket').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                bucket_prev['fas_count'] = bucket_prev['fas_count'] / 4
                bucket_prev['fas_dollars'] = bucket_prev['fas_dollars'] / 4
                bucket_prev['period'] = 'Prev 4-wk Avg'
                
                # Combine for comparison
                bucket_comparison = pd.concat([bucket_target, bucket_prev])
                
                bucket_order = ['0-7 days (Same Week)', '8-14 days', '15-30 days', '31-60 days', '60+ days']
                
                col_v1, col_v2 = st.columns(2)
                
                with col_v1:
                    bars_vintage = alt.Chart(bucket_comparison).mark_bar().encode(
                        x=alt.X('vintage_bucket:N', title='Vintage (Days from StS to FAS)', sort=bucket_order),
                        y=alt.Y('fas_count:Q', title='FAS Count'),
                        color=alt.Color('period:N', scale=alt.Scale(
                            domain=['Week of 2/8', 'Prev 4-wk Avg'],
                            range=['#4cc9f0', '#8892b0']
                        )),
                        xOffset='period:N',
                        tooltip=[
                            alt.Tooltip('vintage_bucket:N', title='Vintage'),
                            alt.Tooltip('period:N', title='Period'),
                            alt.Tooltip('fas_count:Q', title='FAS Count', format=',.0f')
                        ]
                    )
                    text_vintage = alt.Chart(bucket_comparison).mark_text(dy=-8, fontSize=8).encode(
                        x=alt.X('vintage_bucket:N', sort=bucket_order),
                        y=alt.Y('fas_count:Q'),
                        xOffset='period:N',
                        text=alt.Text('fas_count:Q', format=',.0f'),
                        color=alt.value('black')
                    )
                    fig_vintage = (bars_vintage + text_vintage).properties(title='FAS Count by Vintage Bucket', height=300)
                    st.altair_chart(fig_vintage, use_container_width=True)
                
                with col_v2:
                    bucket_comparison['fas_dollars_k'] = bucket_comparison['fas_dollars'] / 1000
                    bars_vintage_dollars = alt.Chart(bucket_comparison).mark_bar().encode(
                        x=alt.X('vintage_bucket:N', title='Vintage (Days from StS to FAS)', sort=bucket_order),
                        y=alt.Y('fas_dollars:Q', title='FAS $', axis=alt.Axis(format='$,.0f')),
                        color=alt.Color('period:N', scale=alt.Scale(
                            domain=['Week of 2/8', 'Prev 4-wk Avg'],
                            range=['#2ca02c', '#8892b0']
                        )),
                        xOffset='period:N',
                        tooltip=[
                            alt.Tooltip('vintage_bucket:N', title='Vintage'),
                            alt.Tooltip('period:N', title='Period'),
                            alt.Tooltip('fas_dollars:Q', title='FAS $', format='$,.0f')
                        ]
                    )
                    text_vintage_dollars = alt.Chart(bucket_comparison).mark_text(dy=-8, fontSize=7).encode(
                        x=alt.X('vintage_bucket:N', sort=bucket_order),
                        y=alt.Y('fas_dollars:Q'),
                        xOffset='period:N',
                        text=alt.Text('fas_dollars_k:Q', format='$,.0fK'),
                        color=alt.value('black')
                    )
                    fig_vintage_dollars = (bars_vintage_dollars + text_vintage_dollars).properties(title='FAS $ by Vintage Bucket', height=300)
                    st.altair_chart(fig_vintage_dollars, use_container_width=True)
                
                # === VINTAGE WEEK % CONTRIBUTION TABLE ===
                st.markdown("#### 📊 % FAS $ Contribution by Vintage Week")
                st.caption("Which Sent-to-Sales weeks contributed what % of FAS $ in 2/8 week vs previous 4 weeks?")
                
                # Calculate % contribution for target week
                vintage_target_pct = vintage_target.copy()
                total_target = vintage_target_pct['fas_dollars'].sum()
                vintage_target_pct['pct_contribution'] = (vintage_target_pct['fas_dollars'] / total_target * 100) if total_target > 0 else 0
                vintage_target_pct = vintage_target_pct[['sts_week', 'fas_count', 'fas_dollars', 'pct_contribution', 'days_from_sts']].copy()
                vintage_target_pct.columns = ['Vintage Week (StS)', 'FAS Count (2/8)', 'FAS $ (2/8)', '% of Total (2/8)', 'Days to FAS']
                
                # Calculate % contribution for prev 4 weeks (average)
                vintage_prev_pct = vintage_prev.copy()
                total_prev = vintage_prev_pct['fas_dollars'].sum()
                vintage_prev_pct['pct_contribution'] = (vintage_prev_pct['fas_dollars'] / total_prev * 100) if total_prev > 0 else 0
                vintage_prev_pct = vintage_prev_pct[['sts_week', 'fas_count', 'fas_dollars', 'pct_contribution']].copy()
                vintage_prev_pct.columns = ['Vintage Week (StS)', 'FAS Count (Prev Avg)', 'FAS $ (Prev Avg)', '% of Total (Prev Avg)']
                
                # Merge the two
                vintage_pct_table = vintage_target_pct.merge(
                    vintage_prev_pct, 
                    on='Vintage Week (StS)', 
                    how='outer'
                ).fillna(0)
                
                # Calculate % point change
                vintage_pct_table['% Pt Δ'] = vintage_pct_table['% of Total (2/8)'] - vintage_pct_table['% of Total (Prev Avg)']
                
                # Sort by 2/8 contribution descending
                vintage_pct_table = vintage_pct_table.sort_values('% of Total (2/8)', ascending=False)
                
                # Format for display
                vintage_display = vintage_pct_table.copy()
                vintage_display['Vintage Week (StS)'] = pd.to_datetime(vintage_display['Vintage Week (StS)']).dt.strftime('%b %d, %Y')
                vintage_display['FAS Count (2/8)'] = vintage_display['FAS Count (2/8)'].apply(lambda x: f"{x:,.0f}")
                vintage_display['FAS $ (2/8)'] = vintage_display['FAS $ (2/8)'].apply(lambda x: f"${x:,.0f}")
                vintage_display['% of Total (2/8)'] = vintage_display['% of Total (2/8)'].apply(lambda x: f"{x:.1f}%")
                vintage_display['Days to FAS'] = vintage_display['Days to FAS'].apply(lambda x: f"{x:.0f}")
                vintage_display['FAS Count (Prev Avg)'] = vintage_display['FAS Count (Prev Avg)'].apply(lambda x: f"{x:,.1f}")
                vintage_display['FAS $ (Prev Avg)'] = vintage_display['FAS $ (Prev Avg)'].apply(lambda x: f"${x:,.0f}")
                vintage_display['% of Total (Prev Avg)'] = vintage_display['% of Total (Prev Avg)'].apply(lambda x: f"{x:.1f}%")
                vintage_display['% Pt Δ'] = vintage_display['% Pt Δ'].apply(lambda x: f"{x:+.1f}pp")
                
                # Reorder columns
                vintage_display = vintage_display[[
                    'Vintage Week (StS)', 'Days to FAS',
                    'FAS Count (2/8)', 'FAS $ (2/8)', '% of Total (2/8)',
                    'FAS Count (Prev Avg)', 'FAS $ (Prev Avg)', '% of Total (Prev Avg)',
                    '% Pt Δ'
                ]]
                
                st.dataframe(vintage_display, use_container_width=True, hide_index=True, height=350)
                st.caption("**% Pt Δ** = Percentage point change in contribution. Positive means this vintage contributed MORE to 2/8 week vs historical average.")
                
                # Key insight
                if len(vintage_pct_table) > 0:
                    top_contributor = vintage_pct_table.iloc[0]
                    biggest_gainer = vintage_pct_table.loc[vintage_pct_table['% Pt Δ'].idxmax()]
                    st.info(f"""
                    **Key Takeaway:** 
                    - Largest contributor to 2/8 FAS $: **{pd.to_datetime(top_contributor['Vintage Week (StS)']).strftime('%b %d')}** vintage ({top_contributor['% of Total (2/8)']:.1f}% of total)
                    - Biggest % increase vs historical: **{pd.to_datetime(biggest_gainer['Vintage Week (StS)']).strftime('%b %d')}** vintage (+{biggest_gainer['% Pt Δ']:.1f}pp)
                    """)
                
                # === VINTAGE % BY FAS WEEK - LAST 5 WEEKS COMPARISON ===
                st.markdown("#### 📈 % FAS $ by Vintage Age - Last 5 Weeks Comparison")
                st.caption("Did we see faster conversions (more recent vintages) in the 2/8 week vs previous weeks?")
                
                # Get last 5 weeks of FAS data
                last_5_weeks = [target_week] + prev_4_weeks  # 2/8, 2/1, 1/25, 1/18, 1/11
                
                # For each FAS week, calculate % by vintage bucket
                week_vintage_data = []
                for fas_wk in last_5_weeks:
                    wk_data = df_fas[df_fas['fas_week'] == fas_wk].copy()
                    if len(wk_data) > 0:
                        wk_data['vintage_bucket'] = wk_data['days_to_fas'].apply(days_to_bucket)
                        wk_agg = wk_data.groupby('vintage_bucket').agg({
                            'fas_dollars': 'sum'
                        }).reset_index()
                        total_dollars = wk_agg['fas_dollars'].sum()
                        wk_agg['pct'] = (wk_agg['fas_dollars'] / total_dollars * 100) if total_dollars > 0 else 0
                        wk_agg['fas_week'] = fas_wk.strftime('%b %d')
                        week_vintage_data.append(wk_agg)
                
                if week_vintage_data:
                    all_weeks_vintage = pd.concat(week_vintage_data)
                    
                    # Pivot to create comparison table
                    vintage_pivot = all_weeks_vintage.pivot(
                        index='vintage_bucket', 
                        columns='fas_week', 
                        values='pct'
                    ).fillna(0)
                    
                    # Reorder columns (most recent first)
                    week_order = [w.strftime('%b %d') for w in last_5_weeks]
                    vintage_pivot = vintage_pivot[[c for c in week_order if c in vintage_pivot.columns]]
                    
                    # Reorder rows by vintage bucket order
                    vintage_pivot = vintage_pivot.reindex(bucket_order)
                    vintage_pivot = vintage_pivot.reset_index()
                    vintage_pivot = vintage_pivot.rename(columns={'vintage_bucket': 'Vintage Age'})
                    
                    # Calculate change from avg of prev 4 weeks to 2/8
                    if len(vintage_pivot.columns) > 2:
                        prev_cols = [c for c in week_order[1:] if c in vintage_pivot.columns]
                        if prev_cols:
                            vintage_pivot['Prev 4-Wk Avg'] = vintage_pivot[prev_cols].mean(axis=1)
                            target_col = week_order[0] if week_order[0] in vintage_pivot.columns else None
                            if target_col:
                                vintage_pivot['Δ vs Avg'] = vintage_pivot[target_col] - vintage_pivot['Prev 4-Wk Avg']
                    
                    # Format for display
                    vintage_week_display = vintage_pivot.copy()
                    for col in vintage_week_display.columns:
                        if col != 'Vintage Age':
                            if col == 'Δ vs Avg':
                                vintage_week_display[col] = vintage_week_display[col].apply(lambda x: f"{x:+.1f}pp")
                            else:
                                vintage_week_display[col] = vintage_week_display[col].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(vintage_week_display, use_container_width=True, hide_index=True)
                    st.caption("**Δ vs Avg** = Percentage point difference from previous 4-week average. Positive in recent vintages = faster conversions.")
                    
                    # Quick insight
                    recent_vintages = ['0-7 days (Same Week)', '8-14 days']
                    target_col = week_order[0] if week_order[0] in vintage_pivot.columns else None
                    if target_col and 'Prev 4-Wk Avg' in vintage_pivot.columns:
                        recent_target = vintage_pivot[vintage_pivot['Vintage Age'].isin(recent_vintages)][target_col].sum()
                        recent_prev = vintage_pivot[vintage_pivot['Vintage Age'].isin(recent_vintages)]['Prev 4-Wk Avg'].sum()
                        if recent_target > recent_prev:
                            st.success(f"✅ **Recent vintages (0-14 days) contributed {recent_target:.1f}% of 2/8 FAS $ vs {recent_prev:.1f}% historical avg** — Faster conversions!")
                        else:
                            st.warning(f"⚠️ **Recent vintages (0-14 days) contributed {recent_target:.1f}% of 2/8 FAS $ vs {recent_prev:.1f}% historical avg** — Older leads converting")
                
                st.divider()
                
                # === BREAKDOWN: LVC ANALYSIS ===
                st.markdown("### 📊 LVC Group Analysis: Which Lead Segments Drove Growth?")
                
                lvc_target = target_week_data.groupby('lvc_group').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                lvc_target['period'] = 'Week of 2/8'
                
                lvc_prev = prev_weeks_data.groupby('lvc_group').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                lvc_prev['fas_count'] = lvc_prev['fas_count'] / 4
                lvc_prev['fas_dollars'] = lvc_prev['fas_dollars'] / 4
                lvc_prev['period'] = 'Prev 4-wk Avg'
                
                lvc_comparison = pd.concat([lvc_target, lvc_prev])
                
                lvc_order = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']
                
                col_l1, col_l2 = st.columns(2)
                
                with col_l1:
                    bars_lvc = alt.Chart(lvc_comparison).mark_bar().encode(
                        x=alt.X('lvc_group:N', title='LVC Group', sort=lvc_order),
                        y=alt.Y('fas_count:Q', title='FAS Count'),
                        color=alt.Color('period:N', scale=alt.Scale(
                            domain=['Week of 2/8', 'Prev 4-wk Avg'],
                            range=['#4cc9f0', '#8892b0']
                        )),
                        xOffset='period:N',
                        tooltip=[
                            alt.Tooltip('lvc_group:N', title='LVC Group'),
                            alt.Tooltip('period:N', title='Period'),
                            alt.Tooltip('fas_count:Q', title='FAS Count', format=',.0f')
                        ]
                    )
                    text_lvc = alt.Chart(lvc_comparison).mark_text(dy=-8, fontSize=9).encode(
                        x=alt.X('lvc_group:N', sort=lvc_order),
                        y=alt.Y('fas_count:Q'),
                        xOffset='period:N',
                        text=alt.Text('fas_count:Q', format=',.0f'),
                        color=alt.value('black')
                    )
                    fig_lvc = (bars_lvc + text_lvc).properties(title='FAS Count by LVC Group', height=300)
                    st.altair_chart(fig_lvc, use_container_width=True)
                
                with col_l2:
                    lvc_comparison['fas_dollars_k'] = lvc_comparison['fas_dollars'] / 1000
                    bars_lvc_dollars = alt.Chart(lvc_comparison).mark_bar().encode(
                        x=alt.X('lvc_group:N', title='LVC Group', sort=lvc_order),
                        y=alt.Y('fas_dollars:Q', title='FAS $', axis=alt.Axis(format='$,.0f')),
                        color=alt.Color('period:N', scale=alt.Scale(
                            domain=['Week of 2/8', 'Prev 4-wk Avg'],
                            range=['#2ca02c', '#8892b0']
                        )),
                        xOffset='period:N',
                        tooltip=[
                            alt.Tooltip('lvc_group:N', title='LVC Group'),
                            alt.Tooltip('period:N', title='Period'),
                            alt.Tooltip('fas_dollars:Q', title='FAS $', format='$,.0f')
                        ]
                    )
                    text_lvc_dollars = alt.Chart(lvc_comparison).mark_text(dy=-8, fontSize=8).encode(
                        x=alt.X('lvc_group:N', sort=lvc_order),
                        y=alt.Y('fas_dollars:Q'),
                        xOffset='period:N',
                        text=alt.Text('fas_dollars_k:Q', format='$,.0fK'),
                        color=alt.value('black')
                    )
                    fig_lvc_dollars = (bars_lvc_dollars + text_lvc_dollars).properties(title='FAS $ by LVC Group', height=300)
                    st.altair_chart(fig_lvc_dollars, use_container_width=True)
                
                # LVC Change Table
                lvc_pivot = lvc_comparison.pivot(index='lvc_group', columns='period', values=['fas_count', 'fas_dollars']).reset_index()
                
                # Flatten multi-index columns properly
                new_cols = []
                for col in lvc_pivot.columns:
                    if isinstance(col, tuple):
                        if col[0] == '' or col[1] == '':
                            new_cols.append(col[0] if col[0] else col[1])
                        else:
                            new_cols.append(f"{col[0]}_{col[1]}")
                    else:
                        new_cols.append(col)
                lvc_pivot.columns = new_cols
                
                # Rename columns for clarity
                col_mapping = {
                    'lvc_group': 'LVC Group',
                    'fas_count_Week of 2/8': 'FAS Count (2/8)',
                    'fas_count_Prev 4-wk Avg': 'FAS Count (Prev Avg)',
                    'fas_dollars_Week of 2/8': 'FAS $ (2/8)',
                    'fas_dollars_Prev 4-wk Avg': 'FAS $ (Prev Avg)'
                }
                lvc_pivot = lvc_pivot.rename(columns=col_mapping)
                
                # Handle case where columns might not exist
                required_cols = ['FAS Count (2/8)', 'FAS Count (Prev Avg)', 'FAS $ (2/8)', 'FAS $ (Prev Avg)']
                if all(col in lvc_pivot.columns for col in required_cols):
                    lvc_pivot['Count Change'] = lvc_pivot['FAS Count (2/8)'].fillna(0) - lvc_pivot['FAS Count (Prev Avg)'].fillna(0)
                    lvc_pivot['$ Change'] = lvc_pivot['FAS $ (2/8)'].fillna(0) - lvc_pivot['FAS $ (Prev Avg)'].fillna(0)
                    lvc_pivot['Count % Δ'] = (lvc_pivot['Count Change'] / lvc_pivot['FAS Count (Prev Avg)'].replace(0, 1) * 100).fillna(0)
                    
                    display_cols = [c for c in ['LVC Group', 'FAS Count (2/8)', 'FAS Count (Prev Avg)', 'Count Change', 'Count % Δ', 'FAS $ (2/8)', 'FAS $ (Prev Avg)', '$ Change'] if c in lvc_pivot.columns]
                    lvc_display = lvc_pivot[display_cols].copy()
                    
                    if 'FAS Count (2/8)' in lvc_display.columns:
                        lvc_display['FAS Count (2/8)'] = lvc_display['FAS Count (2/8)'].apply(lambda x: f"{x:,.0f}")
                    if 'FAS Count (Prev Avg)' in lvc_display.columns:
                        lvc_display['FAS Count (Prev Avg)'] = lvc_display['FAS Count (Prev Avg)'].apply(lambda x: f"{x:,.0f}")
                    if 'Count Change' in lvc_display.columns:
                        lvc_display['Count Change'] = lvc_display['Count Change'].apply(lambda x: f"{x:+,.0f}")
                    if 'Count % Δ' in lvc_display.columns:
                        lvc_display['Count % Δ'] = lvc_display['Count % Δ'].apply(lambda x: f"{x:+.1f}%")
                    if 'FAS $ (2/8)' in lvc_display.columns:
                        lvc_display['FAS $ (2/8)'] = lvc_display['FAS $ (2/8)'].apply(lambda x: f"${x:,.0f}")
                    if 'FAS $ (Prev Avg)' in lvc_display.columns:
                        lvc_display['FAS $ (Prev Avg)'] = lvc_display['FAS $ (Prev Avg)'].apply(lambda x: f"${x:,.0f}")
                    if '$ Change' in lvc_display.columns:
                        lvc_display['$ Change'] = lvc_display['$ Change'].apply(lambda x: f"${x:+,.0f}")
                    
                    st.dataframe(lvc_display, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"Missing columns for LVC comparison. Available: {list(lvc_pivot.columns)}")
                
                st.divider()
                
                # === BREAKDOWN: MA PERFORMANCE ===
                st.markdown("### 👤 MA Performance: Week-over-Week Comparison")
                
                ma_target = target_week_data.groupby('mortgage_advisor').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                ma_target['period'] = 'Week of 2/8'
                
                ma_prev = prev_weeks_data.groupby('mortgage_advisor').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                ma_prev['fas_count'] = ma_prev['fas_count'] / 4
                ma_prev['fas_dollars'] = ma_prev['fas_dollars'] / 4
                ma_prev['period'] = 'Prev 4-wk Avg'
                
                # Merge to compare
                ma_comparison = ma_target.merge(ma_prev, on='mortgage_advisor', suffixes=('_target', '_prev'), how='outer').fillna(0)
                ma_comparison['count_change'] = ma_comparison['fas_count_target'] - ma_comparison['fas_count_prev']
                ma_comparison['dollar_change'] = ma_comparison['fas_dollars_target'] - ma_comparison['fas_dollars_prev']
                ma_comparison['count_pct_change'] = (ma_comparison['count_change'] / ma_comparison['fas_count_prev'].replace(0, 1) * 100)
                
                # Full MA Table (sortable)
                st.markdown("#### 📋 All MAs - Week of 2/8 vs Prev 4-Week Avg")
                ma_full = ma_comparison[['mortgage_advisor', 'fas_count_target', 'fas_count_prev', 'count_change', 'count_pct_change', 'fas_dollars_target', 'fas_dollars_prev', 'dollar_change']].copy()
                ma_full = ma_full.sort_values('count_change', ascending=False)
                
                # Create display version with formatting
                ma_display = ma_full.copy()
                ma_display.columns = ['MA', 'FAS Count (2/8)', 'FAS Count (Prev Avg)', 'Count Δ', 'Count % Δ', 'FAS $ (2/8)', 'FAS $ (Prev Avg)', '$ Δ']
                ma_display['FAS Count (2/8)'] = ma_display['FAS Count (2/8)'].apply(lambda x: f"{x:.0f}")
                ma_display['FAS Count (Prev Avg)'] = ma_display['FAS Count (Prev Avg)'].apply(lambda x: f"{x:.1f}")
                ma_display['Count Δ'] = ma_display['Count Δ'].apply(lambda x: f"{x:+.1f}")
                ma_display['Count % Δ'] = ma_display['Count % Δ'].apply(lambda x: f"{x:+.1f}%")
                ma_display['FAS $ (2/8)'] = ma_display['FAS $ (2/8)'].apply(lambda x: f"${x:,.0f}")
                ma_display['FAS $ (Prev Avg)'] = ma_display['FAS $ (Prev Avg)'].apply(lambda x: f"${x:,.0f}")
                ma_display['$ Δ'] = ma_display['$ Δ'].apply(lambda x: f"${x:+,.0f}")
                
                st.dataframe(ma_display, use_container_width=True, hide_index=True, height=400)
                st.caption(f"Showing all {len(ma_display)} MAs with FAS activity. Sorted by Count Change (descending). Click column headers to re-sort.")
                
                # Summary stats
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    gainers = len(ma_comparison[ma_comparison['count_change'] > 0])
                    st.metric("MAs with ↑ FAS Count", f"{gainers}")
                with col_m2:
                    losers = len(ma_comparison[ma_comparison['count_change'] < 0])
                    st.metric("MAs with ↓ FAS Count", f"{losers}")
                with col_m3:
                    flat = len(ma_comparison[ma_comparison['count_change'] == 0])
                    st.metric("MAs Unchanged", f"{flat}")
                
                st.divider()
                
                # === BREAKDOWN: PERSONA ===
                st.markdown("### 🎭 Persona Analysis: Which Personas Drove Growth?")
                
                persona_target = target_week_data.groupby('persona').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                persona_target['period'] = 'Week of 2/8'
                
                persona_prev = prev_weeks_data.groupby('persona').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                persona_prev['fas_count'] = persona_prev['fas_count'] / 4
                persona_prev['fas_dollars'] = persona_prev['fas_dollars'] / 4
                persona_prev['period'] = 'Prev 4-wk Avg'
                
                persona_comparison = pd.concat([persona_target, persona_prev])
                
                bars_persona = alt.Chart(persona_comparison).mark_bar().encode(
                    x=alt.X('persona:N', title='Persona', sort='-y'),
                    y=alt.Y('fas_count:Q', title='FAS Count'),
                    color=alt.Color('period:N', scale=alt.Scale(
                        domain=['Week of 2/8', 'Prev 4-wk Avg'],
                        range=['#4cc9f0', '#8892b0']
                    )),
                    xOffset='period:N',
                    tooltip=[
                        alt.Tooltip('persona:N', title='Persona'),
                        alt.Tooltip('period:N', title='Period'),
                        alt.Tooltip('fas_count:Q', title='FAS Count', format=',.0f'),
                        alt.Tooltip('fas_dollars:Q', title='FAS $', format='$,.0f')
                    ]
                )
                text_persona = alt.Chart(persona_comparison).mark_text(dy=-8, fontSize=8).encode(
                    x=alt.X('persona:N', sort='-y'),
                    y=alt.Y('fas_count:Q'),
                    xOffset='period:N',
                    text=alt.Text('fas_count:Q', format=',.0f'),
                    color=alt.value('black')
                )
                fig_persona = (bars_persona + text_persona).properties(title='FAS Count by Persona', height=350)
                st.altair_chart(fig_persona, use_container_width=True)
                
                st.divider()
                
                # === DAILY FULL FUNNEL ANALYSIS ===
                st.subheader("📊 Daily Full Funnel Analysis: 2/8 Week Lead Quality Deep Dive")
                st.caption("Understanding lead volume and quality at each funnel stage - comparing 2/8 week to previous 4 weeks")
                
                # Query daily funnel data
                funnel_query = f"""
                SELECT
                    DATE(sent_to_sales_date) as sts_date,
                    DATE_TRUNC(DATE(sent_to_sales_date), WEEK(SUNDAY)) as sts_week,
                    DATE(lead_created_date) as lead_created,
                    DATE_TRUNC(DATE(lead_created_date), WEEK(SUNDAY)) as lead_vintage_week,
                    CASE
                        WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
                        WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
                        WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
                        WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
                        ELSE 'Other'
                    END as lvc_group,
                    starring,
                    persona,
                    
                    -- Funnel Metrics
                    COUNT(DISTINCT lendage_guid) as gross_leads,
                    COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales,
                    COUNT(DISTINCT CASE WHEN current_sales_assigned_date IS NOT NULL THEN lendage_guid END) as assigned_leads,
                    COUNT(DISTINCT CASE WHEN contacted_date IS NOT NULL THEN lendage_guid END) as contacted_leads,
                    COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_leads,
                    
                    -- Quality Metrics
                    SUM(e_loan_amount) as total_loan_amount,
                    AVG(e_loan_amount) as avg_loan_amount,
                    SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars
                    
                FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                WHERE DATE(sent_to_sales_date) >= '2026-01-01'
                AND DATE(sent_to_sales_date) <= '2026-02-15'
                GROUP BY 1, 2, 3, 4, 5, 6, 7
                """
                
                try:
                    df_funnel = client.query(funnel_query).to_dataframe()
                    df_funnel['sts_week'] = pd.to_datetime(df_funnel['sts_week'])
                    df_funnel['lead_vintage_week'] = pd.to_datetime(df_funnel['lead_vintage_week'])
                    
                    # Filter for target and prev weeks
                    funnel_target = df_funnel[df_funnel['sts_week'] == target_week]
                    funnel_prev = df_funnel[df_funnel['sts_week'].isin(prev_4_weeks)]
                    
                    # === AGGREGATE VIEW ===
                    st.markdown("### 📈 Aggregate Funnel: 2/8 Week vs Prev 4-Week Avg")
                    
                    # Aggregate metrics
                    agg_target = {
                        'Gross Leads': funnel_target['gross_leads'].sum(),
                        'Sent to Sales': funnel_target['sent_to_sales'].sum(),
                        'Assigned': funnel_target['assigned_leads'].sum(),
                        'Contacted': funnel_target['contacted_leads'].sum(),
                        'FAS': funnel_target['fas_leads'].sum(),
                        'FAS $': funnel_target['fas_dollars'].sum(),
                        'Avg Loan': funnel_target['avg_loan_amount'].mean()
                    }
                    
                    agg_prev = {
                        'Gross Leads': funnel_prev['gross_leads'].sum() / 4,
                        'Sent to Sales': funnel_prev['sent_to_sales'].sum() / 4,
                        'Assigned': funnel_prev['assigned_leads'].sum() / 4,
                        'Contacted': funnel_prev['contacted_leads'].sum() / 4,
                        'FAS': funnel_prev['fas_leads'].sum() / 4,
                        'FAS $': funnel_prev['fas_dollars'].sum() / 4,
                        'Avg Loan': funnel_prev['avg_loan_amount'].mean()
                    }
                    
                    # Create funnel comparison table
                    funnel_table = pd.DataFrame({
                        'Stage': ['Gross Leads', 'Sent to Sales', 'Assigned', 'Contacted', 'FAS'],
                        '2/8 Week': [agg_target['Gross Leads'], agg_target['Sent to Sales'], agg_target['Assigned'], agg_target['Contacted'], agg_target['FAS']],
                        'Prev 4-Wk Avg': [agg_prev['Gross Leads'], agg_prev['Sent to Sales'], agg_prev['Assigned'], agg_prev['Contacted'], agg_prev['FAS']]
                    })
                    funnel_table['Δ'] = funnel_table['2/8 Week'] - funnel_table['Prev 4-Wk Avg']
                    funnel_table['% Δ'] = (funnel_table['Δ'] / funnel_table['Prev 4-Wk Avg'] * 100).fillna(0)
                    
                    # Calculate conversion rates
                    funnel_table['Conv % (2/8)'] = [
                        100,  # Gross
                        agg_target['Sent to Sales'] / agg_target['Gross Leads'] * 100 if agg_target['Gross Leads'] > 0 else 0,
                        agg_target['Assigned'] / agg_target['Sent to Sales'] * 100 if agg_target['Sent to Sales'] > 0 else 0,
                        agg_target['Contacted'] / agg_target['Assigned'] * 100 if agg_target['Assigned'] > 0 else 0,
                        agg_target['FAS'] / agg_target['Contacted'] * 100 if agg_target['Contacted'] > 0 else 0
                    ]
                    funnel_table['Conv % (Prev)'] = [
                        100,
                        agg_prev['Sent to Sales'] / agg_prev['Gross Leads'] * 100 if agg_prev['Gross Leads'] > 0 else 0,
                        agg_prev['Assigned'] / agg_prev['Sent to Sales'] * 100 if agg_prev['Sent to Sales'] > 0 else 0,
                        agg_prev['Contacted'] / agg_prev['Assigned'] * 100 if agg_prev['Assigned'] > 0 else 0,
                        agg_prev['FAS'] / agg_prev['Contacted'] * 100 if agg_prev['Contacted'] > 0 else 0
                    ]
                    
                    # Format for display
                    funnel_display = funnel_table.copy()
                    funnel_display['2/8 Week'] = funnel_display['2/8 Week'].apply(lambda x: f"{x:,.0f}")
                    funnel_display['Prev 4-Wk Avg'] = funnel_display['Prev 4-Wk Avg'].apply(lambda x: f"{x:,.0f}")
                    funnel_display['Δ'] = funnel_display['Δ'].apply(lambda x: f"{x:+,.0f}")
                    funnel_display['% Δ'] = funnel_display['% Δ'].apply(lambda x: f"{x:+.1f}%")
                    funnel_display['Conv % (2/8)'] = funnel_display['Conv % (2/8)'].apply(lambda x: f"{x:.1f}%")
                    funnel_display['Conv % (Prev)'] = funnel_display['Conv % (Prev)'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(funnel_display, use_container_width=True, hide_index=True)
                    
                    # KPI metrics
                    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                    with col_f1:
                        sts_delta = (agg_target['Sent to Sales'] - agg_prev['Sent to Sales']) / agg_prev['Sent to Sales'] * 100 if agg_prev['Sent to Sales'] > 0 else 0
                        st.metric("Sent to Sales", f"{agg_target['Sent to Sales']:,.0f}", f"{sts_delta:+.1f}%")
                    with col_f2:
                        contact_rate_target = agg_target['Contacted'] / agg_target['Assigned'] * 100 if agg_target['Assigned'] > 0 else 0
                        contact_rate_prev = agg_prev['Contacted'] / agg_prev['Assigned'] * 100 if agg_prev['Assigned'] > 0 else 0
                        st.metric("Contact Rate", f"{contact_rate_target:.1f}%", f"{contact_rate_target - contact_rate_prev:+.1f}pp")
                    with col_f3:
                        fas_rate_target = agg_target['FAS'] / agg_target['Sent to Sales'] * 100 if agg_target['Sent to Sales'] > 0 else 0
                        fas_rate_prev = agg_prev['FAS'] / agg_prev['Sent to Sales'] * 100 if agg_prev['Sent to Sales'] > 0 else 0
                        st.metric("FAS Rate (from StS)", f"{fas_rate_target:.1f}%", f"{fas_rate_target - fas_rate_prev:+.1f}pp")
                    with col_f4:
                        loan_delta = agg_target['Avg Loan'] - agg_prev['Avg Loan']
                        st.metric("Avg Loan Amount", f"${agg_target['Avg Loan']:,.0f}", f"${loan_delta:+,.0f}")
                    
                    st.divider()
                    
                    # === BY LVC GROUP ===
                    st.markdown("### 📊 Funnel by LVC Group")
                    
                    lvc_funnel_target = funnel_target.groupby('lvc_group').agg({
                        'gross_leads': 'sum',
                        'sent_to_sales': 'sum',
                        'assigned_leads': 'sum',
                        'contacted_leads': 'sum',
                        'fas_leads': 'sum',
                        'fas_dollars': 'sum',
                        'avg_loan_amount': 'mean'
                    }).reset_index()
                    
                    lvc_funnel_prev = funnel_prev.groupby('lvc_group').agg({
                        'gross_leads': 'sum',
                        'sent_to_sales': 'sum',
                        'assigned_leads': 'sum',
                        'contacted_leads': 'sum',
                        'fas_leads': 'sum',
                        'fas_dollars': 'sum',
                        'avg_loan_amount': 'mean'
                    }).reset_index()
                    lvc_funnel_prev[['gross_leads', 'sent_to_sales', 'assigned_leads', 'contacted_leads', 'fas_leads', 'fas_dollars']] /= 4
                    
                    lvc_funnel = lvc_funnel_target.merge(lvc_funnel_prev, on='lvc_group', suffixes=('_target', '_prev'), how='outer').fillna(0)
                    
                    # Calculate rates
                    lvc_funnel['contact_rate_target'] = lvc_funnel['contacted_leads_target'] / lvc_funnel['assigned_leads_target'].replace(0, 1) * 100
                    lvc_funnel['contact_rate_prev'] = lvc_funnel['contacted_leads_prev'] / lvc_funnel['assigned_leads_prev'].replace(0, 1) * 100
                    lvc_funnel['fas_rate_target'] = lvc_funnel['fas_leads_target'] / lvc_funnel['sent_to_sales_target'].replace(0, 1) * 100
                    lvc_funnel['fas_rate_prev'] = lvc_funnel['fas_leads_prev'] / lvc_funnel['sent_to_sales_prev'].replace(0, 1) * 100
                    
                    lvc_funnel_display = lvc_funnel[['lvc_group', 'sent_to_sales_target', 'sent_to_sales_prev', 'contacted_leads_target', 'contact_rate_target', 'contact_rate_prev', 'fas_leads_target', 'fas_rate_target', 'fas_rate_prev', 'avg_loan_amount_target']].copy()
                    lvc_funnel_display.columns = ['LVC Group', 'StS (2/8)', 'StS (Prev)', 'Contacted (2/8)', 'Contact % (2/8)', 'Contact % (Prev)', 'FAS (2/8)', 'FAS % (2/8)', 'FAS % (Prev)', 'Avg Loan (2/8)']
                    
                    lvc_funnel_display['StS Δ'] = lvc_funnel_display['StS (2/8)'] - lvc_funnel_display['StS (Prev)']
                    lvc_funnel_display['Contact pp Δ'] = lvc_funnel_display['Contact % (2/8)'] - lvc_funnel_display['Contact % (Prev)']
                    lvc_funnel_display['FAS pp Δ'] = lvc_funnel_display['FAS % (2/8)'] - lvc_funnel_display['FAS % (Prev)']
                    
                    # Format
                    for col in ['StS (2/8)', 'StS (Prev)', 'Contacted (2/8)', 'FAS (2/8)']:
                        lvc_funnel_display[col] = lvc_funnel_display[col].apply(lambda x: f"{x:,.0f}")
                    lvc_funnel_display['StS Δ'] = lvc_funnel_display['StS Δ'].apply(lambda x: f"{x:+,.0f}")
                    for col in ['Contact % (2/8)', 'Contact % (Prev)', 'FAS % (2/8)', 'FAS % (Prev)']:
                        lvc_funnel_display[col] = lvc_funnel_display[col].apply(lambda x: f"{x:.1f}%")
                    for col in ['Contact pp Δ', 'FAS pp Δ']:
                        lvc_funnel_display[col] = lvc_funnel_display[col].apply(lambda x: f"{x:+.1f}pp")
                    lvc_funnel_display['Avg Loan (2/8)'] = lvc_funnel_display['Avg Loan (2/8)'].apply(lambda x: f"${x:,.0f}")
                    
                    lvc_funnel_display = lvc_funnel_display[['LVC Group', 'StS (2/8)', 'StS (Prev)', 'StS Δ', 'Contact % (2/8)', 'Contact % (Prev)', 'Contact pp Δ', 'FAS % (2/8)', 'FAS % (Prev)', 'FAS pp Δ', 'Avg Loan (2/8)']]
                    st.dataframe(lvc_funnel_display, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # === BY STARRING ===
                    st.markdown("### ⭐ Funnel by Starring")
                    
                    starring_funnel_target = funnel_target.groupby('starring').agg({
                        'gross_leads': 'sum',
                        'sent_to_sales': 'sum',
                        'assigned_leads': 'sum',
                        'contacted_leads': 'sum',
                        'fas_leads': 'sum',
                        'fas_dollars': 'sum',
                        'avg_loan_amount': 'mean'
                    }).reset_index()
                    
                    starring_funnel_prev = funnel_prev.groupby('starring').agg({
                        'gross_leads': 'sum',
                        'sent_to_sales': 'sum',
                        'assigned_leads': 'sum',
                        'contacted_leads': 'sum',
                        'fas_leads': 'sum',
                        'fas_dollars': 'sum',
                        'avg_loan_amount': 'mean'
                    }).reset_index()
                    starring_funnel_prev[['gross_leads', 'sent_to_sales', 'assigned_leads', 'contacted_leads', 'fas_leads', 'fas_dollars']] /= 4
                    
                    starring_funnel = starring_funnel_target.merge(starring_funnel_prev, on='starring', suffixes=('_target', '_prev'), how='outer').fillna(0)
                    
                    # Calculate rates
                    starring_funnel['contact_rate_target'] = starring_funnel['contacted_leads_target'] / starring_funnel['assigned_leads_target'].replace(0, 1) * 100
                    starring_funnel['contact_rate_prev'] = starring_funnel['contacted_leads_prev'] / starring_funnel['assigned_leads_prev'].replace(0, 1) * 100
                    starring_funnel['fas_rate_target'] = starring_funnel['fas_leads_target'] / starring_funnel['sent_to_sales_target'].replace(0, 1) * 100
                    starring_funnel['fas_rate_prev'] = starring_funnel['fas_leads_prev'] / starring_funnel['sent_to_sales_prev'].replace(0, 1) * 100
                    
                    starring_funnel_display = starring_funnel[['starring', 'sent_to_sales_target', 'sent_to_sales_prev', 'contacted_leads_target', 'contact_rate_target', 'contact_rate_prev', 'fas_leads_target', 'fas_rate_target', 'fas_rate_prev', 'avg_loan_amount_target']].copy()
                    starring_funnel_display.columns = ['Starring', 'StS (2/8)', 'StS (Prev)', 'Contacted (2/8)', 'Contact % (2/8)', 'Contact % (Prev)', 'FAS (2/8)', 'FAS % (2/8)', 'FAS % (Prev)', 'Avg Loan (2/8)']
                    
                    starring_funnel_display['StS Δ'] = starring_funnel_display['StS (2/8)'] - starring_funnel_display['StS (Prev)']
                    starring_funnel_display['Contact pp Δ'] = starring_funnel_display['Contact % (2/8)'] - starring_funnel_display['Contact % (Prev)']
                    starring_funnel_display['FAS pp Δ'] = starring_funnel_display['FAS % (2/8)'] - starring_funnel_display['FAS % (Prev)']
                    
                    # Sort by StS volume
                    starring_funnel_display = starring_funnel_display.sort_values('StS (2/8)', ascending=False)
                    
                    # Format
                    for col in ['StS (2/8)', 'StS (Prev)', 'Contacted (2/8)', 'FAS (2/8)']:
                        starring_funnel_display[col] = starring_funnel_display[col].apply(lambda x: f"{x:,.0f}")
                    starring_funnel_display['StS Δ'] = starring_funnel_display['StS Δ'].apply(lambda x: f"{x:+,.0f}")
                    for col in ['Contact % (2/8)', 'Contact % (Prev)', 'FAS % (2/8)', 'FAS % (Prev)']:
                        starring_funnel_display[col] = starring_funnel_display[col].apply(lambda x: f"{x:.1f}%")
                    for col in ['Contact pp Δ', 'FAS pp Δ']:
                        starring_funnel_display[col] = starring_funnel_display[col].apply(lambda x: f"{x:+.1f}pp")
                    starring_funnel_display['Avg Loan (2/8)'] = starring_funnel_display['Avg Loan (2/8)'].apply(lambda x: f"${x:,.0f}")
                    
                    starring_funnel_display = starring_funnel_display[['Starring', 'StS (2/8)', 'StS (Prev)', 'StS Δ', 'Contact % (2/8)', 'Contact % (Prev)', 'Contact pp Δ', 'FAS % (2/8)', 'FAS % (Prev)', 'FAS pp Δ', 'Avg Loan (2/8)']]
                    st.dataframe(starring_funnel_display, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # === LVC GROUP × STARRING ===
                    st.markdown("### 📊 LVC Group × Starring Matrix")
                    st.caption("Cross-tabulation of LVC Group and Starring - filtered by date based on selected metric")
                    
                    # Filters
                    col_filter1, col_filter2, col_filter3 = st.columns(3)
                    
                    with col_filter1:
                        metric_choice = st.selectbox(
                            "Select Metric:",
                            ['Sent to Sales Volume', 'FAS Volume', 'FAS Rate %', 'FAS $', 'Avg Loan Amount'],
                            key='lvc_starring_metric'
                        )
                    
                    with col_filter2:
                        lvc_star_start = st.date_input(
                            "Start Date:",
                            value=datetime(2026, 1, 1),
                            key='lvc_star_start'
                        )
                    
                    with col_filter3:
                        lvc_star_end = st.date_input(
                            "End Date:",
                            value=datetime(2026, 2, 15),
                            key='lvc_star_end'
                        )
                    
                    # Determine which date field to use based on metric
                    if 'Sent to Sales' in metric_choice:
                        date_field = 'sent_to_sales_date'
                        date_label = 'Sent to Sales Date'
                    else:  # FAS related metrics
                        date_field = 'full_app_submit_datetime'
                        date_label = 'FAS Date'
                    
                    st.info(f"📅 Filtering by **{date_label}** from {lvc_star_start} to {lvc_star_end}")
                    
                    # Query based on selected metric and date
                    lvc_starring_query = f"""
                    SELECT
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
                        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars,
                        AVG(e_loan_amount) as avg_loan_amount
                    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                    WHERE DATE({date_field}) >= '{lvc_star_start}'
                    AND DATE({date_field}) <= '{lvc_star_end}'
                    AND {date_field} IS NOT NULL
                    GROUP BY 1, 2
                    """
                    
                    try:
                        df_lvc_starring = client.query(lvc_starring_query).to_dataframe()
                        
                        # Calculate rates
                        df_lvc_starring['fas_rate'] = df_lvc_starring['fas_count'] / df_lvc_starring['sts_count'].replace(0, 1) * 100
                        
                        # Determine pivot column based on metric
                        if metric_choice == 'Sent to Sales Volume':
                            pivot_col = 'sts_count'
                            value_format = 'volume'
                        elif metric_choice == 'FAS Volume':
                            pivot_col = 'fas_count'
                            value_format = 'volume'
                        elif metric_choice == 'FAS Rate %':
                            pivot_col = 'fas_rate'
                            value_format = 'percent'
                        elif metric_choice == 'FAS $':
                            pivot_col = 'fas_dollars'
                            value_format = 'dollars'
                        else:  # Avg Loan Amount
                            pivot_col = 'avg_loan_amount'
                            value_format = 'dollars'
                        
                        # Pivot table
                        lvc_starring_pivot = df_lvc_starring.pivot(
                            index='lvc_group',
                            columns='starring',
                            values=pivot_col
                        ).fillna(0)
                        
                        # Reorder index
                        lvc_order = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']
                        lvc_starring_pivot = lvc_starring_pivot.reindex([l for l in lvc_order if l in lvc_starring_pivot.index])
                        
                        # Add row totals
                        if value_format in ['volume', 'dollars']:
                            lvc_starring_pivot['TOTAL'] = lvc_starring_pivot.sum(axis=1)
                        else:
                            lvc_starring_pivot['TOTAL'] = lvc_starring_pivot.mean(axis=1)
                        
                        # Add column totals
                        if value_format in ['volume', 'dollars']:
                            col_totals = lvc_starring_pivot.sum(axis=0)
                        else:
                            col_totals = lvc_starring_pivot.mean(axis=0)
                        lvc_starring_pivot.loc['TOTAL'] = col_totals
                        
                        # Format for display
                        lvc_starring_display = lvc_starring_pivot.copy()
                        for col in lvc_starring_display.columns:
                            if value_format == 'volume':
                                lvc_starring_display[col] = lvc_starring_display[col].apply(lambda x: f"{x:,.0f}")
                            elif value_format == 'percent':
                                lvc_starring_display[col] = lvc_starring_display[col].apply(lambda x: f"{x:.1f}%")
                            else:  # dollars
                                lvc_starring_display[col] = lvc_starring_display[col].apply(lambda x: f"${x:,.0f}")
                        
                        lvc_starring_display = lvc_starring_display.reset_index()
                        lvc_starring_display = lvc_starring_display.rename(columns={'lvc_group': 'LVC Group'})
                        
                        st.dataframe(lvc_starring_display, use_container_width=True, hide_index=True)
                        
                        # Detailed table
                        st.markdown("##### Detailed View: LVC × Starring Performance")
                        lvc_starring_detail = df_lvc_starring[['lvc_group', 'starring', 'sts_count', 'fas_count', 'fas_rate', 'fas_dollars', 'avg_loan_amount']].copy()
                        lvc_starring_detail.columns = ['LVC Group', 'Starring', 'Sent to Sales', 'FAS Count', 'FAS Rate %', 'FAS $', 'Avg Loan']
                        lvc_starring_detail = lvc_starring_detail.sort_values(['LVC Group', 'Sent to Sales'], ascending=[True, False])
                        
                        # Format
                        lvc_starring_detail['Sent to Sales'] = lvc_starring_detail['Sent to Sales'].apply(lambda x: f"{x:,.0f}")
                        lvc_starring_detail['FAS Count'] = lvc_starring_detail['FAS Count'].apply(lambda x: f"{x:,.0f}")
                        lvc_starring_detail['FAS Rate %'] = lvc_starring_detail['FAS Rate %'].apply(lambda x: f"{x:.1f}%")
                        lvc_starring_detail['FAS $'] = lvc_starring_detail['FAS $'].apply(lambda x: f"${x:,.0f}")
                        lvc_starring_detail['Avg Loan'] = lvc_starring_detail['Avg Loan'].apply(lambda x: f"${x:,.0f}")
                        
                        st.dataframe(lvc_starring_detail, use_container_width=True, hide_index=True, height=300)
                        st.caption(f"**Date Field Used:** `{date_field}` | **Date Range:** {lvc_star_start} to {lvc_star_end}")
                        
                    except Exception as e:
                        st.error(f"LVC × Starring Query Error: {e}")
                    
                    st.divider()
                    
                    # === 2/8 WEEK DEEP ANALYSIS: WHAT CHANGED? ===
                    st.markdown("### 🔬 2/8 Week Analysis: What Changed?")
                    st.caption("Analyzing allocation shifts and performance changes to understand what made 2/8 so good")
                    
                    # Query for 2/8 vs prev 4 weeks - IN-PERIOD FAS (based on FAS date, not StS vintage)
                    analysis_query = """
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
                            starring,
                            lendage_guid,
                            e_loan_amount as fas_amount,
                            CASE WHEN express_app_started_at IS NOT NULL THEN 1 ELSE 0 END as is_digital_start
                        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                        WHERE full_app_submit_datetime IS NOT NULL
                        AND DATE(full_app_submit_datetime) >= '2026-01-05'
                        AND DATE(full_app_submit_datetime) <= '2026-02-15'
                    ),
                    sts_data AS (
                        SELECT
                            DATE_TRUNC(DATE(sent_to_sales_date), WEEK(SUNDAY)) as sts_week,
                            CASE
                                WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
                                WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
                                WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
                                WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
                                ELSE 'Other'
                            END as lvc_group,
                            starring,
                            lendage_guid,
                            CASE WHEN contacted_date IS NOT NULL THEN 1 ELSE 0 END as is_contacted,
                            CASE WHEN express_app_started_at IS NOT NULL THEN 1 ELSE 0 END as is_digital_start,
                            e_loan_amount
                        FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                        WHERE sent_to_sales_date IS NOT NULL
                        AND DATE(sent_to_sales_date) >= '2026-01-05'
                        AND DATE(sent_to_sales_date) <= '2026-02-15'
                    ),
                    fas_agg AS (
                        SELECT
                            fas_week as week_start,
                            lvc_group,
                            starring,
                            COUNT(DISTINCT lendage_guid) as fas_count,
                            SUM(fas_amount) as fas_dollars,
                            SUM(is_digital_start) as digital_to_fas_count
                        FROM fas_data
                        GROUP BY 1, 2, 3
                    ),
                    sts_agg AS (
                        SELECT
                            sts_week as week_start,
                            lvc_group,
                            starring,
                            COUNT(DISTINCT lendage_guid) as sts_count,
                            SUM(is_contacted) as contacted_count,
                            SUM(is_digital_start) as digital_start_count,
                            AVG(e_loan_amount) as avg_loan
                        FROM sts_data
                        GROUP BY 1, 2, 3
                    )
                    SELECT
                        COALESCE(f.week_start, s.week_start) as week_start,
                        COALESCE(f.lvc_group, s.lvc_group) as lvc_group,
                        COALESCE(f.starring, s.starring) as starring,
                        COALESCE(s.sts_count, 0) as sts_count,
                        COALESCE(s.contacted_count, 0) as contacted_count,
                        COALESCE(f.fas_count, 0) as fas_count,
                        COALESCE(f.fas_dollars, 0) as fas_dollars,
                        COALESCE(s.digital_start_count, 0) as digital_start_count,
                        COALESCE(f.digital_to_fas_count, 0) as digital_to_fas_count,
                        COALESCE(s.avg_loan, 0) as avg_loan
                    FROM fas_agg f
                    FULL OUTER JOIN sts_agg s 
                        ON f.week_start = s.week_start 
                        AND f.lvc_group = s.lvc_group 
                        AND f.starring = s.starring
                    """
                    
                    try:
                        df_analysis = client.query(analysis_query).to_dataframe()
                        df_analysis['week_start'] = pd.to_datetime(df_analysis['week_start'])
                        
                        # Split into 2/8 week and prev 4 weeks
                        target_wk = pd.Timestamp('2026-02-08')
                        prev_wks = [target_wk - pd.Timedelta(weeks=i) for i in range(1, 5)]
                        
                        df_target = df_analysis[df_analysis['week_start'] == target_wk]
                        df_prev = df_analysis[df_analysis['week_start'].isin(prev_wks)]
                        
                        # === SUMMARY KPIs AT TOP ===
                        st.markdown("#### 📊 Week 2/8 Summary vs Previous 4-Week Average")
                        
                        # Calculate totals
                        t_sts = df_target['sts_count'].sum()
                        t_fas = df_target['fas_count'].sum()
                        t_fas_dollars = df_target['fas_dollars'].sum()
                        t_contacted = df_target['contacted_count'].sum()
                        t_digital_start = df_target['digital_start_count'].sum()
                        t_digital_fas = df_target['digital_to_fas_count'].sum()
                        
                        p_sts = df_prev['sts_count'].sum() / 4
                        p_fas = df_prev['fas_count'].sum() / 4
                        p_fas_dollars = df_prev['fas_dollars'].sum() / 4
                        p_contacted = df_prev['contacted_count'].sum() / 4
                        p_digital_start = df_prev['digital_start_count'].sum() / 4
                        p_digital_fas = df_prev['digital_to_fas_count'].sum() / 4
                        
                        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
                        
                        with kpi_col1:
                            st.metric("Sent to Sales", f"{t_sts:,.0f}", f"{t_sts - p_sts:+,.0f}")
                        with kpi_col2:
                            st.metric("FAS Qty", f"{t_fas:,.0f}", f"{t_fas - p_fas:+,.0f}")
                        with kpi_col3:
                            fas_rate_t = t_fas / t_sts * 100
                            fas_rate_p = p_fas / p_sts * 100
                            st.metric("FAS Rate", f"{fas_rate_t:.1f}%", f"{fas_rate_t - fas_rate_p:+.1f}pp")
                        with kpi_col4:
                            st.metric("FAS $", f"${t_fas_dollars:,.0f}", f"${t_fas_dollars - p_fas_dollars:+,.0f}")
                        with kpi_col5:
                            dig_start_rate_t = t_digital_start / t_sts * 100
                            dig_start_rate_p = p_digital_start / p_sts * 100
                            st.metric("Digital Start %", f"{dig_start_rate_t:.1f}%", f"{dig_start_rate_t - dig_start_rate_p:+.1f}pp")
                        with kpi_col6:
                            dig_fas_rate_t = t_digital_fas / t_digital_start * 100 if t_digital_start > 0 else 0
                            dig_fas_rate_p = p_digital_fas / p_digital_start * 100 if p_digital_start > 0 else 0
                            st.metric("Digital→FAS %", f"{dig_fas_rate_t:.1f}%", f"{dig_fas_rate_t - dig_fas_rate_p:+.1f}pp")
                        
                        st.divider()
                        
                        # ==========================================
                        # 1. ALLOCATION SHIFT: Did the MIX change?
                        # ==========================================
                        st.markdown("#### 1️⃣ Allocation Shift: Did the LVC × Starring MIX Change?")
                        st.caption("Comparing % of total Sent-to-Sales by LVC×Starring combination")
                        
                        # Calculate allocation for target week
                        target_total = df_target['sts_count'].sum()
                        alloc_target = df_target.groupby(['lvc_group', 'starring']).agg({
                            'sts_count': 'sum'
                        }).reset_index()
                        alloc_target['pct_of_total'] = (alloc_target['sts_count'] / target_total * 100)
                        alloc_target = alloc_target.rename(columns={'sts_count': 'sts_target', 'pct_of_total': 'pct_target'})
                        
                        # Calculate allocation for prev weeks (avg)
                        prev_total = df_prev['sts_count'].sum() / 4
                        alloc_prev = df_prev.groupby(['lvc_group', 'starring']).agg({
                            'sts_count': 'sum'
                        }).reset_index()
                        alloc_prev['sts_count'] = alloc_prev['sts_count'] / 4
                        alloc_prev['pct_of_total'] = (alloc_prev['sts_count'] / prev_total * 100)
                        alloc_prev = alloc_prev.rename(columns={'sts_count': 'sts_prev', 'pct_of_total': 'pct_prev'})
                        
                        # Merge
                        alloc = alloc_target.merge(alloc_prev, on=['lvc_group', 'starring'], how='outer').fillna(0)
                        alloc['pct_shift'] = alloc['pct_target'] - alloc['pct_prev']
                        alloc['volume_change'] = alloc['sts_target'] - alloc['sts_prev']
                        
                        # Sort by biggest shifts
                        alloc_sorted = alloc.sort_values('pct_shift', ascending=False)
                        
                        # Format for display
                        alloc_display = alloc_sorted.copy()
                        alloc_display['sts_target'] = alloc_display['sts_target'].apply(lambda x: f"{x:,.0f}")
                        alloc_display['sts_prev'] = alloc_display['sts_prev'].apply(lambda x: f"{x:,.0f}")
                        alloc_display['pct_target'] = alloc_display['pct_target'].apply(lambda x: f"{x:.1f}%")
                        alloc_display['pct_prev'] = alloc_display['pct_prev'].apply(lambda x: f"{x:.1f}%")
                        alloc_display['pct_shift'] = alloc_display['pct_shift'].apply(lambda x: f"{x:+.2f}pp")
                        alloc_display['volume_change'] = alloc_display['volume_change'].apply(lambda x: f"{x:+,.0f}")
                        alloc_display.columns = ['LVC Group', 'Starring', 'StS (2/8)', 'Mix % (2/8)', 'StS (Prev Avg)', 'Mix % (Prev)', 'Mix Shift (pp)', 'Volume Δ']
                        alloc_display = alloc_display[['LVC Group', 'Starring', 'StS (2/8)', 'StS (Prev Avg)', 'Volume Δ', 'Mix % (2/8)', 'Mix % (Prev)', 'Mix Shift (pp)']]
                        
                        st.dataframe(alloc_display, use_container_width=True, hide_index=True)
                        
                        # Highlight biggest shifts
                        col_a1, col_a2 = st.columns(2)
                        with col_a1:
                            top_gainers = alloc_sorted.nlargest(3, 'pct_shift')[['lvc_group', 'starring', 'pct_shift']]
                            st.success("**📈 Biggest Mix Increases:**")
                            for _, row in top_gainers.iterrows():
                                st.write(f"• {row['lvc_group']} × {row['starring']}: **+{row['pct_shift']:.2f}pp**")
                        with col_a2:
                            top_losers = alloc_sorted.nsmallest(3, 'pct_shift')[['lvc_group', 'starring', 'pct_shift']]
                            st.warning("**📉 Biggest Mix Decreases:**")
                            for _, row in top_losers.iterrows():
                                st.write(f"• {row['lvc_group']} × {row['starring']}: **{row['pct_shift']:.2f}pp**")
                        
                        st.divider()
                        
                        # ==========================================
                        # 2. PERFORMANCE CHANGE: Did rates improve?
                        # ==========================================
                        st.markdown("#### 2️⃣ Performance Change: Did Starring Performance Improve?")
                        st.caption("Comparing Contact Rate, FAS Rate, and Avg FAS $ by Starring")
                        
                        # Aggregate by starring for target
                        perf_target = df_target.groupby('starring').agg({
                            'sts_count': 'sum',
                            'contacted_count': 'sum',
                            'fas_count': 'sum',
                            'fas_dollars': 'sum',
                            'digital_start_count': 'sum',
                            'digital_to_fas_count': 'sum',
                            'avg_loan': 'mean'
                        }).reset_index()
                        perf_target['contact_rate'] = perf_target['contacted_count'] / perf_target['sts_count'] * 100
                        perf_target['fas_rate'] = perf_target['fas_count'] / perf_target['sts_count'] * 100
                        perf_target['avg_fas'] = perf_target['fas_dollars'] / perf_target['fas_count'].replace(0, 1)
                        perf_target['digital_start_rate'] = perf_target['digital_start_count'] / perf_target['sts_count'] * 100
                        perf_target['digital_fas_rate'] = perf_target['digital_to_fas_count'] / perf_target['digital_start_count'].replace(0, 1) * 100
                        
                        # Aggregate by starring for prev (avg)
                        perf_prev = df_prev.groupby('starring').agg({
                            'sts_count': 'sum',
                            'contacted_count': 'sum',
                            'fas_count': 'sum',
                            'fas_dollars': 'sum',
                            'digital_start_count': 'sum',
                            'digital_to_fas_count': 'sum',
                            'avg_loan': 'mean'
                        }).reset_index()
                        perf_prev[['sts_count', 'contacted_count', 'fas_count', 'fas_dollars', 'digital_start_count', 'digital_to_fas_count']] /= 4
                        perf_prev['contact_rate'] = perf_prev['contacted_count'] / perf_prev['sts_count'] * 100
                        perf_prev['fas_rate'] = perf_prev['fas_count'] / perf_prev['sts_count'] * 100
                        perf_prev['avg_fas'] = perf_prev['fas_dollars'] / perf_prev['fas_count'].replace(0, 1)
                        perf_prev['digital_start_rate'] = perf_prev['digital_start_count'] / perf_prev['sts_count'] * 100
                        perf_prev['digital_fas_rate'] = perf_prev['digital_to_fas_count'] / perf_prev['digital_start_count'].replace(0, 1) * 100
                        
                        # Merge
                        perf = perf_target.merge(perf_prev, on='starring', suffixes=('_target', '_prev'), how='outer').fillna(0)
                        perf['contact_delta'] = perf['contact_rate_target'] - perf['contact_rate_prev']
                        perf['fas_delta'] = perf['fas_rate_target'] - perf['fas_rate_prev']
                        perf['avg_fas_delta'] = perf['avg_fas_target'] - perf['avg_fas_prev']
                        perf['sts_delta'] = perf['sts_count_target'] - perf['sts_count_prev']
                        perf['digital_start_delta'] = perf['digital_start_rate_target'] - perf['digital_start_rate_prev']
                        perf['digital_fas_delta'] = perf['digital_fas_rate_target'] - perf['digital_fas_rate_prev']
                        
                        # Sort by FAS rate improvement
                        perf_sorted = perf.sort_values('fas_delta', ascending=False)
                        
                        # Format
                        perf_display = perf_sorted[['starring', 'sts_count_target', 'sts_count_prev', 'sts_delta', 
                                                     'contact_rate_target', 'contact_rate_prev', 'contact_delta',
                                                     'fas_rate_target', 'fas_rate_prev', 'fas_delta',
                                                     'avg_fas_target', 'avg_fas_prev', 'avg_fas_delta']].copy()
                        perf_display.columns = ['Starring', 'StS (2/8)', 'StS (Prev)', 'StS Δ',
                                                'Contact % (2/8)', 'Contact % (Prev)', 'Contact Δ',
                                                'FAS % (2/8)', 'FAS % (Prev)', 'FAS Δ',
                                                'Avg FAS $ (2/8)', 'Avg FAS $ (Prev)', 'Avg FAS Δ']
                        
                        perf_display['StS (2/8)'] = perf_display['StS (2/8)'].apply(lambda x: f"{x:,.0f}")
                        perf_display['StS (Prev)'] = perf_display['StS (Prev)'].apply(lambda x: f"{x:,.0f}")
                        perf_display['StS Δ'] = perf_display['StS Δ'].apply(lambda x: f"{x:+,.0f}")
                        perf_display['Contact % (2/8)'] = perf_display['Contact % (2/8)'].apply(lambda x: f"{x:.1f}%")
                        perf_display['Contact % (Prev)'] = perf_display['Contact % (Prev)'].apply(lambda x: f"{x:.1f}%")
                        perf_display['Contact Δ'] = perf_display['Contact Δ'].apply(lambda x: f"{x:+.1f}pp")
                        perf_display['FAS % (2/8)'] = perf_display['FAS % (2/8)'].apply(lambda x: f"{x:.1f}%")
                        perf_display['FAS % (Prev)'] = perf_display['FAS % (Prev)'].apply(lambda x: f"{x:.1f}%")
                        perf_display['FAS Δ'] = perf_display['FAS Δ'].apply(lambda x: f"{x:+.1f}pp")
                        perf_display['Avg FAS $ (2/8)'] = perf_display['Avg FAS $ (2/8)'].apply(lambda x: f"${x:,.0f}")
                        perf_display['Avg FAS $ (Prev)'] = perf_display['Avg FAS $ (Prev)'].apply(lambda x: f"${x:,.0f}")
                        perf_display['Avg FAS Δ'] = perf_display['Avg FAS Δ'].apply(lambda x: f"${x:+,.0f}")
                        
                        st.dataframe(perf_display, use_container_width=True, hide_index=True)
                        
                        # === DIGITAL APP ANALYSIS ===
                        st.markdown("##### 📱 Digital App Performance by Starring")
                        st.caption("Digital App Start = customer started app on website | Digital → FAS = % of digital starters who got to FAS")
                        
                        digital_display = perf_sorted[['starring', 'sts_count_target',
                                                        'digital_start_count_target', 'digital_start_rate_target', 'digital_start_rate_prev', 'digital_start_delta',
                                                        'digital_to_fas_count_target', 'digital_fas_rate_target', 'digital_fas_rate_prev', 'digital_fas_delta']].copy()
                        digital_display.columns = ['Starring', 'StS (2/8)',
                                                   'Digital Starts (2/8)', 'Start % (2/8)', 'Start % (Prev)', 'Start Δ',
                                                   'Digital→FAS (2/8)', 'FAS % (2/8)', 'FAS % (Prev)', 'FAS Δ']
                        
                        digital_display['StS (2/8)'] = digital_display['StS (2/8)'].apply(lambda x: f"{x:,.0f}")
                        digital_display['Digital Starts (2/8)'] = digital_display['Digital Starts (2/8)'].apply(lambda x: f"{x:,.0f}")
                        digital_display['Start % (2/8)'] = digital_display['Start % (2/8)'].apply(lambda x: f"{x:.1f}%")
                        digital_display['Start % (Prev)'] = digital_display['Start % (Prev)'].apply(lambda x: f"{x:.1f}%")
                        digital_display['Start Δ'] = digital_display['Start Δ'].apply(lambda x: f"{x:+.1f}pp")
                        digital_display['Digital→FAS (2/8)'] = digital_display['Digital→FAS (2/8)'].apply(lambda x: f"{x:,.0f}")
                        digital_display['FAS % (2/8)'] = digital_display['FAS % (2/8)'].apply(lambda x: f"{x:.1f}%")
                        digital_display['FAS % (Prev)'] = digital_display['FAS % (Prev)'].apply(lambda x: f"{x:.1f}%")
                        digital_display['FAS Δ'] = digital_display['FAS Δ'].apply(lambda x: f"{x:+.1f}pp")
                        
                        st.dataframe(digital_display, use_container_width=True, hide_index=True)
                        
                        # Digital summary KPIs
                        col_dig1, col_dig2, col_dig3, col_dig4 = st.columns(4)
                        
                        total_digital_start_target = df_target['digital_start_count'].sum()
                        total_digital_start_prev = df_prev['digital_start_count'].sum() / 4
                        total_digital_to_fas_target = df_target['digital_to_fas_count'].sum()
                        total_digital_to_fas_prev = df_prev['digital_to_fas_count'].sum() / 4
                        total_sts_target = df_target['sts_count'].sum()
                        total_sts_prev = df_prev['sts_count'].sum() / 4
                        
                        with col_dig1:
                            start_rate_target = total_digital_start_target / total_sts_target * 100
                            start_rate_prev = total_digital_start_prev / total_sts_prev * 100
                            st.metric(
                                "Digital Start Rate",
                                f"{start_rate_target:.1f}%",
                                f"{start_rate_target - start_rate_prev:+.1f}pp"
                            )
                        with col_dig2:
                            st.metric(
                                "Digital Starts (2/8)",
                                f"{total_digital_start_target:,.0f}",
                                f"{total_digital_start_target - total_digital_start_prev:+,.0f}"
                            )
                        with col_dig3:
                            fas_rate_target = total_digital_to_fas_target / total_digital_start_target * 100 if total_digital_start_target > 0 else 0
                            fas_rate_prev = total_digital_to_fas_prev / total_digital_start_prev * 100 if total_digital_start_prev > 0 else 0
                            st.metric(
                                "Digital → FAS Rate",
                                f"{fas_rate_target:.1f}%",
                                f"{fas_rate_target - fas_rate_prev:+.1f}pp"
                            )
                        with col_dig4:
                            st.metric(
                                "Digital → FAS (2/8)",
                                f"{total_digital_to_fas_target:,.0f}",
                                f"{total_digital_to_fas_target - total_digital_to_fas_prev:+,.0f}"
                            )
                        
                        st.divider()
                        
                        # ==========================================
                        # 3. IMPACT DECOMPOSITION: What drove the uplift?
                        # ==========================================
                        st.markdown("#### 3️⃣ Impact Decomposition: What Drove the 2/8 FAS Uplift?")
                        st.caption("Breaking down the FAS improvement into Volume, Rate, and Mix effects")
                        
                        # Calculate overall metrics
                        total_target = {
                            'sts': df_target['sts_count'].sum(),
                            'fas': df_target['fas_count'].sum(),
                            'fas_dollars': df_target['fas_dollars'].sum()
                        }
                        total_prev = {
                            'sts': df_prev['sts_count'].sum() / 4,
                            'fas': df_prev['fas_count'].sum() / 4,
                            'fas_dollars': df_prev['fas_dollars'].sum() / 4
                        }
                        
                        # Calculate changes
                        sts_change = total_target['sts'] - total_prev['sts']
                        fas_change = total_target['fas'] - total_prev['fas']
                        fas_dollar_change = total_target['fas_dollars'] - total_prev['fas_dollars']
                        
                        fas_rate_target = total_target['fas'] / total_target['sts'] * 100
                        fas_rate_prev = total_prev['fas'] / total_prev['sts'] * 100
                        
                        # Decomposition
                        # Volume effect: (new volume - old volume) * old rate
                        volume_effect_fas = sts_change * (fas_rate_prev / 100)
                        
                        # Rate effect: old volume * (new rate - old rate)
                        rate_effect_fas = total_prev['sts'] * ((fas_rate_target - fas_rate_prev) / 100)
                        
                        # Interaction: change in volume * change in rate
                        interaction_fas = sts_change * ((fas_rate_target - fas_rate_prev) / 100)
                        
                        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                        
                        with col_d1:
                            st.metric(
                                "Total FAS Change",
                                f"{fas_change:+,.0f}",
                                f"{(fas_change/total_prev['fas']*100):+.1f}%"
                            )
                        
                        with col_d2:
                            st.metric(
                                "📊 Volume Effect",
                                f"{volume_effect_fas:+,.0f} FAS",
                                f"({volume_effect_fas/fas_change*100:.0f}% of change)" if fas_change != 0 else "N/A"
                            )
                            st.caption("More leads sent to sales")
                        
                        with col_d3:
                            st.metric(
                                "📈 Rate Effect",
                                f"{rate_effect_fas:+,.0f} FAS",
                                f"({rate_effect_fas/fas_change*100:.0f}% of change)" if fas_change != 0 else "N/A"
                            )
                            st.caption("Better conversion rate")
                        
                        with col_d4:
                            st.metric(
                                "🔀 Mix/Interaction",
                                f"{interaction_fas:+,.0f} FAS",
                                f"({interaction_fas/fas_change*100:.0f}% of change)" if fas_change != 0 else "N/A"
                            )
                            st.caption("Volume × Rate interaction")
                        
                        # Summary insight
                        st.markdown("---")
                        st.markdown("#### 💡 Key Findings: What Made 2/8 So Good?")
                        
                        # Determine primary driver
                        if abs(volume_effect_fas) > abs(rate_effect_fas):
                            primary_driver = "**VOLUME**"
                            driver_detail = f"Higher StS volume (+{sts_change:,.0f} leads) was the primary driver"
                        else:
                            primary_driver = "**CONVERSION RATE**"
                            driver_detail = f"Better FAS rate ({fas_rate_target:.1f}% vs {fas_rate_prev:.1f}%) was the primary driver"
                        
                        # Find top performing changes
                        top_mix_gainer = alloc_sorted.iloc[0]
                        top_perf_gainer = perf_sorted.iloc[0]
                        
                        st.info(f"""
                        **Primary Driver:** {primary_driver}
                        - {driver_detail}
                        
                        **Mix Shift:** 
                        - Biggest allocation increase: **{top_mix_gainer['lvc_group']} × {top_mix_gainer['starring']}** (+{top_mix_gainer['pct_shift']:.2f}pp of total mix)
                        
                        **Performance Change:**
                        - Biggest FAS rate improvement: **{top_perf_gainer['starring']}** (+{top_perf_gainer['fas_delta']:.1f}pp)
                        
                        **Overall:**
                        - StS Volume: {total_prev['sts']:,.0f} → {total_target['sts']:,.0f} ({sts_change:+,.0f}, {sts_change/total_prev['sts']*100:+.1f}%)
                        - FAS Rate: {fas_rate_prev:.1f}% → {fas_rate_target:.1f}% ({fas_rate_target-fas_rate_prev:+.1f}pp)
                        - FAS Count: {total_prev['fas']:,.0f} → {total_target['fas']:,.0f} ({fas_change:+,.0f}, {fas_change/total_prev['fas']*100:+.1f}%)
                        """)
                        
                    except Exception as e:
                        st.error(f"2/8 Analysis Error: {e}")
                    
                    st.divider()
                    
                    # === BY LEAD VINTAGE ===
                    st.markdown("### 📅 Funnel by Lead Vintage (Lead Created Week)")
                    st.caption("How do leads from different creation weeks perform when sent to sales in 2/8?")
                    
                    vintage_funnel_target = funnel_target.groupby('lead_vintage_week').agg({
                        'gross_leads': 'sum',
                        'sent_to_sales': 'sum',
                        'assigned_leads': 'sum',
                        'contacted_leads': 'sum',
                        'fas_leads': 'sum',
                        'fas_dollars': 'sum',
                        'avg_loan_amount': 'mean'
                    }).reset_index()
                    
                    # Calculate rates
                    vintage_funnel_target['contact_rate'] = vintage_funnel_target['contacted_leads'] / vintage_funnel_target['assigned_leads'].replace(0, 1) * 100
                    vintage_funnel_target['fas_rate'] = vintage_funnel_target['fas_leads'] / vintage_funnel_target['sent_to_sales'].replace(0, 1) * 100
                    vintage_funnel_target['days_old'] = (target_week - vintage_funnel_target['lead_vintage_week']).dt.days
                    
                    vintage_funnel_display = vintage_funnel_target[['lead_vintage_week', 'days_old', 'sent_to_sales', 'contacted_leads', 'contact_rate', 'fas_leads', 'fas_rate', 'avg_loan_amount']].copy()
                    vintage_funnel_display = vintage_funnel_display.sort_values('lead_vintage_week', ascending=False)
                    vintage_funnel_display.columns = ['Lead Created Week', 'Days Old', 'Sent to Sales', 'Contacted', 'Contact %', 'FAS', 'FAS %', 'Avg Loan']
                    
                    # Format
                    vintage_funnel_display['Lead Created Week'] = pd.to_datetime(vintage_funnel_display['Lead Created Week']).dt.strftime('%b %d')
                    vintage_funnel_display['Sent to Sales'] = vintage_funnel_display['Sent to Sales'].apply(lambda x: f"{x:,.0f}")
                    vintage_funnel_display['Contacted'] = vintage_funnel_display['Contacted'].apply(lambda x: f"{x:,.0f}")
                    vintage_funnel_display['Contact %'] = vintage_funnel_display['Contact %'].apply(lambda x: f"{x:.1f}%")
                    vintage_funnel_display['FAS'] = vintage_funnel_display['FAS'].apply(lambda x: f"{x:,.0f}")
                    vintage_funnel_display['FAS %'] = vintage_funnel_display['FAS %'].apply(lambda x: f"{x:.1f}%")
                    vintage_funnel_display['Avg Loan'] = vintage_funnel_display['Avg Loan'].apply(lambda x: f"${x:,.0f}")
                    
                    st.dataframe(vintage_funnel_display, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # === BY PERSONA ===
                    st.markdown("### 🎭 Funnel by Persona")
                    
                    persona_funnel_target = funnel_target.groupby('persona').agg({
                        'gross_leads': 'sum',
                        'sent_to_sales': 'sum',
                        'assigned_leads': 'sum',
                        'contacted_leads': 'sum',
                        'fas_leads': 'sum',
                        'fas_dollars': 'sum',
                        'avg_loan_amount': 'mean'
                    }).reset_index()
                    
                    persona_funnel_prev = funnel_prev.groupby('persona').agg({
                        'gross_leads': 'sum',
                        'sent_to_sales': 'sum',
                        'assigned_leads': 'sum',
                        'contacted_leads': 'sum',
                        'fas_leads': 'sum',
                        'fas_dollars': 'sum',
                        'avg_loan_amount': 'mean'
                    }).reset_index()
                    persona_funnel_prev[['gross_leads', 'sent_to_sales', 'assigned_leads', 'contacted_leads', 'fas_leads', 'fas_dollars']] /= 4
                    
                    persona_funnel = persona_funnel_target.merge(persona_funnel_prev, on='persona', suffixes=('_target', '_prev'), how='outer').fillna(0)
                    
                    # Calculate rates
                    persona_funnel['contact_rate_target'] = persona_funnel['contacted_leads_target'] / persona_funnel['assigned_leads_target'].replace(0, 1) * 100
                    persona_funnel['contact_rate_prev'] = persona_funnel['contacted_leads_prev'] / persona_funnel['assigned_leads_prev'].replace(0, 1) * 100
                    persona_funnel['fas_rate_target'] = persona_funnel['fas_leads_target'] / persona_funnel['sent_to_sales_target'].replace(0, 1) * 100
                    persona_funnel['fas_rate_prev'] = persona_funnel['fas_leads_prev'] / persona_funnel['sent_to_sales_prev'].replace(0, 1) * 100
                    
                    persona_funnel_display = persona_funnel[['persona', 'sent_to_sales_target', 'sent_to_sales_prev', 'contacted_leads_target', 'contact_rate_target', 'contact_rate_prev', 'fas_leads_target', 'fas_rate_target', 'fas_rate_prev', 'avg_loan_amount_target']].copy()
                    persona_funnel_display.columns = ['Persona', 'StS (2/8)', 'StS (Prev)', 'Contacted (2/8)', 'Contact % (2/8)', 'Contact % (Prev)', 'FAS (2/8)', 'FAS % (2/8)', 'FAS % (Prev)', 'Avg Loan (2/8)']
                    
                    persona_funnel_display['StS Δ'] = persona_funnel_display['StS (2/8)'] - persona_funnel_display['StS (Prev)']
                    persona_funnel_display['FAS pp Δ'] = persona_funnel_display['FAS % (2/8)'] - persona_funnel_display['FAS % (Prev)']
                    
                    # Sort by StS volume
                    persona_funnel_display = persona_funnel_display.sort_values('StS (2/8)', ascending=False)
                    
                    # Format
                    for col in ['StS (2/8)', 'StS (Prev)', 'Contacted (2/8)', 'FAS (2/8)']:
                        persona_funnel_display[col] = persona_funnel_display[col].apply(lambda x: f"{x:,.0f}")
                    persona_funnel_display['StS Δ'] = persona_funnel_display['StS Δ'].apply(lambda x: f"{x:+,.0f}")
                    for col in ['Contact % (2/8)', 'Contact % (Prev)', 'FAS % (2/8)', 'FAS % (Prev)']:
                        persona_funnel_display[col] = persona_funnel_display[col].apply(lambda x: f"{x:.1f}%")
                    persona_funnel_display['FAS pp Δ'] = persona_funnel_display['FAS pp Δ'].apply(lambda x: f"{x:+.1f}pp")
                    persona_funnel_display['Avg Loan (2/8)'] = persona_funnel_display['Avg Loan (2/8)'].apply(lambda x: f"${x:,.0f}")
                    
                    persona_funnel_display = persona_funnel_display[['Persona', 'StS (2/8)', 'StS (Prev)', 'StS Δ', 'Contact % (2/8)', 'Contact % (Prev)', 'FAS % (2/8)', 'FAS % (Prev)', 'FAS pp Δ', 'Avg Loan (2/8)']]
                    st.dataframe(persona_funnel_display, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # === DAILY TREND ===
                    st.markdown("### 📈 Daily Trend: 2/8 Week")
                    
                    daily_target = funnel_target.groupby('sts_date').agg({
                        'sent_to_sales': 'sum',
                        'contacted_leads': 'sum',
                        'assigned_leads': 'sum',
                        'fas_leads': 'sum'
                    }).reset_index()
                    daily_target['contact_rate'] = daily_target['contacted_leads'] / daily_target['assigned_leads'].replace(0, 1) * 100
                    daily_target['fas_rate'] = daily_target['fas_leads'] / daily_target['sent_to_sales'].replace(0, 1) * 100
                    
                    col_d1, col_d2 = st.columns(2)
                    
                    with col_d1:
                        bars_daily_sts = alt.Chart(daily_target).mark_bar(color='#4cc9f0').encode(
                            x=alt.X('sts_date:T', title='Date', axis=alt.Axis(format='%b %d')),
                            y=alt.Y('sent_to_sales:Q', title='Sent to Sales'),
                            tooltip=[
                                alt.Tooltip('sts_date:T', title='Date', format='%b %d'),
                                alt.Tooltip('sent_to_sales:Q', title='Sent to Sales', format=',')
                            ]
                        )
                        text_daily_sts = bars_daily_sts.mark_text(dy=-8, fontSize=9).encode(
                            text=alt.Text('sent_to_sales:Q', format=',')
                        )
                        fig_daily_sts = (bars_daily_sts + text_daily_sts).properties(title='Daily Sent to Sales', height=250)
                        st.altair_chart(fig_daily_sts, use_container_width=True)
                    
                    with col_d2:
                        line_daily_contact = alt.Chart(daily_target).mark_line(point=True, color='#2ca02c').encode(
                            x=alt.X('sts_date:T', title='Date', axis=alt.Axis(format='%b %d')),
                            y=alt.Y('contact_rate:Q', title='Contact Rate %'),
                            tooltip=[
                                alt.Tooltip('sts_date:T', title='Date', format='%b %d'),
                                alt.Tooltip('contact_rate:Q', title='Contact Rate', format='.1f')
                            ]
                        )
                        text_daily_contact = line_daily_contact.mark_text(dy=-12, fontSize=9).encode(
                            text=alt.Text('contact_rate:Q', format='.0f%')
                        )
                        fig_daily_contact = (line_daily_contact + text_daily_contact).properties(title='Daily Contact Rate', height=250)
                        st.altair_chart(fig_daily_contact, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Funnel Analysis Error: {e}")
                
                st.divider()
                
                # === KEY INSIGHTS ===
                st.subheader("💡 Key Insights: What Caused the 2/8 Uptick?")
                
                col_i1, col_i2 = st.columns(2)
                
                with col_i1:
                    st.markdown("### 🔍 Summary")
                    
                    # Calculate top contributors
                    lvc_top = lvc_pivot.nlargest(1, 'Count Change')['LVC Group'].values[0] if len(lvc_pivot) > 0 else "N/A"
                    lvc_top_change = lvc_pivot[lvc_pivot['LVC Group'] == lvc_top]['Count Change'].values[0] if len(lvc_pivot) > 0 else 0
                    
                    ma_top = ma_comparison.nlargest(1, 'count_change')['mortgage_advisor'].values[0] if len(ma_comparison) > 0 else "N/A"
                    ma_top_change = ma_comparison[ma_comparison['mortgage_advisor'] == ma_top]['count_change'].values[0] if len(ma_comparison) > 0 else 0
                    
                    st.markdown(f"""
                    **Week of 2/8 vs Previous 4-Week Average:**
                    
                    - **FAS Count:** {target_agg['fas_count']:,.0f} ({delta_pct:+.1f}%)
                    - **FAS $:** ${target_agg['fas_dollars']:,.0f} ({delta_dollars_pct:+.1f}%)
                    - **Avg Loan Size:** ${avg_loan_target:,.0f} (${delta_loan:+,.0f})
                    
                    **Top Contributors:**
                    - **LVC:** {lvc_top} (+{lvc_top_change:.0f} FAS)
                    - **MA:** {ma_top} (+{ma_top_change:.0f} FAS)
                    """)
                
                with col_i2:
                    st.markdown("### 📋 Potential Drivers")
                    
                    # Analyze vintage distribution
                    same_week_pct_target = bucket_target[bucket_target['vintage_bucket'] == '0-7 days (Same Week)']['fas_count'].sum() / bucket_target['fas_count'].sum() * 100 if bucket_target['fas_count'].sum() > 0 else 0
                    same_week_pct_prev = bucket_prev[bucket_prev['vintage_bucket'] == '0-7 days (Same Week)']['fas_count'].sum() / bucket_prev['fas_count'].sum() * 100 if bucket_prev['fas_count'].sum() > 0 else 0
                    
                    if same_week_pct_target > same_week_pct_prev + 5:
                        vintage_insight = f"✅ **Faster Conversions:** {same_week_pct_target:.0f}% of FAS came from same-week leads (vs {same_week_pct_prev:.0f}% prev)"
                    elif same_week_pct_target < same_week_pct_prev - 5:
                        vintage_insight = f"📅 **Older Vintage Converting:** Only {same_week_pct_target:.0f}% from same-week (vs {same_week_pct_prev:.0f}% prev) - older leads may have finally closed"
                    else:
                        vintage_insight = f"➡️ **Vintage Mix Stable:** {same_week_pct_target:.0f}% same-week (vs {same_week_pct_prev:.0f}% prev)"
                    
                    st.markdown(f"""
                    {vintage_insight}
                    
                    **Possible Explanations:**
                    1. 📅 **Vintage Catch-Up:** Older leads from high-volume weeks finally converting
                    2. 👤 **MA Performance:** Top performers had strong weeks
                    3. 📊 **LVC Mix:** More high-converting LVC segments
                    4. 🎯 **Persona Quality:** Better-fit personas in the pipeline
                    5. ⚡ **Sales Execution:** Faster speed-to-contact or better follow-up
                    """)
            
            else:
                st.warning("No FAS data found for the selected period.")
                
        except Exception as e:
            st.error(f"FAS Analysis Error: {e}")
            st.exception(e)

    # --- TAB: 2/8 Analysis (Executive View) ---
    with tab_28_analysis:
        st.header("🎯 Week of 2/8/2026: Executive Analysis")
        st.caption("Executive summary of the FAS uptick during week of 2/8/2026 with daily performance breakdowns")
        
        try:
            # === KEY TAKEAWAYS ===
            st.markdown("## 📋 Key Takeaways")
            
            # Query for executive summary data - matching the FAS Analysis tab exactly
            exec_summary_query = """
            WITH fas_data AS (
                SELECT
                    DATE_TRUNC(DATE(full_app_submit_datetime), WEEK(SUNDAY)) as fas_week,
                    COUNT(DISTINCT lendage_guid) as fas_count,
                    SUM(e_loan_amount) as fas_dollars,
                    AVG(e_loan_amount) as avg_loan
                FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                WHERE full_app_submit_datetime IS NOT NULL
                AND DATE(full_app_submit_datetime) >= '2025-12-01'
                AND DATE(full_app_submit_datetime) <= '2026-02-15'
                GROUP BY 1
            ),
            sts_data AS (
                SELECT
                    DATE_TRUNC(DATE(sent_to_sales_date), WEEK(SUNDAY)) as sts_week,
                    COUNT(DISTINCT lendage_guid) as sts_count
                FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                WHERE sent_to_sales_date IS NOT NULL
                AND DATE(sent_to_sales_date) >= '2025-12-01'
                AND DATE(sent_to_sales_date) <= '2026-02-15'
                GROUP BY 1
            )
            SELECT 
                COALESCE(f.fas_week, s.sts_week) as week_start,
                COALESCE(f.fas_count, 0) as fas_count,
                COALESCE(f.fas_dollars, 0) as fas_dollars,
                COALESCE(f.avg_loan, 0) as avg_loan,
                COALESCE(s.sts_count, 0) as sts_count
            FROM fas_data f
            FULL OUTER JOIN sts_data s ON f.fas_week = s.sts_week
            WHERE COALESCE(f.fas_week, s.sts_week) IS NOT NULL
            ORDER BY 1
            """
            
            df_exec = client.query(exec_summary_query).to_dataframe()
            df_exec['week_start'] = pd.to_datetime(df_exec['week_start'])
            df_exec['fas_rate'] = df_exec['fas_count'] / df_exec['sts_count'].replace(0, 1)
            df_exec['fas_per_sts'] = df_exec['fas_dollars'] / df_exec['sts_count'].replace(0, 1)
            
            # Filter to last 6 weeks for display
            df_exec = df_exec[df_exec['week_start'] >= '2026-01-05']
            
            # Calculate 2/8 vs prev 4 weeks
            target_week = pd.Timestamp('2026-02-08')
            prev_weeks = [target_week - pd.Timedelta(weeks=i) for i in range(1, 5)]
            
            target_data = df_exec[df_exec['week_start'] == target_week].iloc[0] if len(df_exec[df_exec['week_start'] == target_week]) > 0 else None
            prev_data = df_exec[df_exec['week_start'].isin(prev_weeks)]
            
            if target_data is not None and len(prev_data) > 0:
                prev_avg = {
                    'fas_count': prev_data['fas_count'].mean(),
                    'fas_dollars': prev_data['fas_dollars'].mean(),
                    'avg_loan': prev_data['avg_loan'].mean(),
                    'sts_count': prev_data['sts_count'].mean(),
                    'fas_rate': prev_data['fas_rate'].mean(),
                    'fas_per_sts': prev_data['fas_per_sts'].mean()
                }
                
                fas_pct_change = (target_data['fas_count'] - prev_avg['fas_count']) / prev_avg['fas_count'] * 100 if prev_avg['fas_count'] > 0 else 0
                fas_dollar_change = target_data['fas_dollars'] - prev_avg['fas_dollars']
                
                # Key insights boxes
                col_i1, col_i2 = st.columns(2)
                
                with col_i1:
                    st.success(f"""
                    ### ✅ Week of 2/8 Performance
                    - **FAS Count:** {target_data['fas_count']:,.0f} (+{fas_pct_change:.1f}% vs prev 4-wk avg)
                    - **FAS $:** ${target_data['fas_dollars']:,.0f} (+${fas_dollar_change:,.0f})
                    - **Avg Loan:** ${target_data['avg_loan']:,.0f}
                    - **FAS Rate:** {target_data['fas_rate']*100:.1f}%
                    - **FAS $ per StS:** ${target_data['fas_per_sts']:,.0f}
                    """)
                
                with col_i2:
                    st.info(f"""
                    ### 📊 Previous 4-Week Baseline
                    - **Avg FAS Count:** {prev_avg['fas_count']:,.0f}
                    - **Avg FAS $:** ${prev_avg['fas_dollars']:,.0f}
                    - **Avg Loan:** ${prev_avg['avg_loan']:,.0f}
                    - **Avg FAS Rate:** {prev_avg['fas_rate']*100:.1f}%
                    - **Avg FAS $ per StS:** ${prev_avg['fas_per_sts']:,.0f}
                    """)
                
                # === WHAT CHANGED CALLOUTS ===
                st.markdown("### 🔍 What Changed vs Previous 4 Weeks?")
                
                # Query for Lead Mix, Persona, Channel, and MA changes
                change_query = """
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
                    AND DATE(full_app_submit_datetime) >= '2026-01-12'
                    AND DATE(full_app_submit_datetime) <= '2026-02-14'
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
                
                df_changes = client.query(change_query).to_dataframe()
                df_changes['fas_week'] = pd.to_datetime(df_changes['fas_week'])
                
                target_week_ts = pd.Timestamp('2026-02-08')
                prev_weeks_ts = [target_week_ts - pd.Timedelta(weeks=i) for i in range(1, 5)]
                
                df_target = df_changes[df_changes['fas_week'] == target_week_ts]
                df_prev = df_changes[df_changes['fas_week'].isin(prev_weeks_ts)]
                
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                
                # --- Lead Mix Callout ---
                with col_c1:
                    st.markdown("#### 🎯 Lead Mix (LVC)")
                    
                    # Calculate LVC distribution
                    lvc_target = df_target.groupby('lvc_group')['fas_count'].sum()
                    lvc_prev = df_prev.groupby('lvc_group')['fas_count'].sum() / 4  # 4-week avg
                    
                    target_total = lvc_target.sum()
                    prev_total = lvc_prev.sum()
                    
                    lvc_target_pct = (lvc_target / target_total * 100) if target_total > 0 else lvc_target * 0
                    lvc_prev_pct = (lvc_prev / prev_total * 100) if prev_total > 0 else lvc_prev * 0
                    
                    # Find biggest changes
                    lvc_changes = []
                    for lvc in lvc_target_pct.index:
                        tgt_pct = lvc_target_pct.get(lvc, 0)
                        prv_pct = lvc_prev_pct.get(lvc, 0)
                        change = tgt_pct - prv_pct
                        lvc_changes.append({'lvc': lvc, 'target': tgt_pct, 'prev': prv_pct, 'change': change})
                    
                    lvc_changes = sorted(lvc_changes, key=lambda x: abs(x['change']), reverse=True)
                    
                    for item in lvc_changes[:3]:
                        icon = "📈" if item['change'] > 0 else "📉" if item['change'] < 0 else "➡️"
                        st.markdown(f"- **{item['lvc']}**: {item['target']:.1f}% {icon} ({item['change']:+.1f}pp)")
                    
                    # Key insight
                    biggest = lvc_changes[0] if lvc_changes else None
                    if biggest and abs(biggest['change']) > 2:
                        st.caption(f"*{biggest['lvc']} had biggest shift ({biggest['change']:+.1f}pp)*")
                
                # --- Persona Callout ---
                with col_c2:
                    st.markdown("#### 🎭 Persona Mix")
                    
                    # Calculate Persona distribution
                    persona_target = df_target.groupby('persona')['fas_count'].sum().nlargest(5)
                    persona_prev = df_prev.groupby('persona')['fas_count'].sum() / 4
                    
                    target_total_p = df_target['fas_count'].sum()
                    prev_total_p = df_prev['fas_count'].sum() / 4
                    
                    persona_target_pct = (persona_target / target_total_p * 100) if target_total_p > 0 else persona_target * 0
                    
                    # Find biggest changes
                    persona_changes = []
                    for p in persona_target.index:
                        tgt_pct = persona_target.get(p, 0) / target_total_p * 100 if target_total_p > 0 else 0
                        prv_cnt = persona_prev.get(p, 0)
                        prv_pct = prv_cnt / prev_total_p * 100 if prev_total_p > 0 else 0
                        cnt_change = persona_target.get(p, 0) - prv_cnt
                        persona_changes.append({'persona': p, 'target': persona_target.get(p, 0), 'prev': prv_cnt, 'cnt_change': cnt_change, 'pct_change': tgt_pct - prv_pct})
                    
                    persona_changes = sorted(persona_changes, key=lambda x: x['cnt_change'], reverse=True)
                    
                    for item in persona_changes[:3]:
                        icon = "📈" if item['cnt_change'] > 0 else "📉" if item['cnt_change'] < 0 else "➡️"
                        short_name = item['persona'][:20] + "..." if len(str(item['persona'])) > 20 else item['persona']
                        st.markdown(f"- **{short_name}**: {item['target']:.0f} {icon} ({item['cnt_change']:+.1f})")
                    
                    # Key insight
                    top_gainer = persona_changes[0] if persona_changes else None
                    if top_gainer and top_gainer['cnt_change'] > 0:
                        st.caption(f"*{persona_changes[0]['persona'][:15]}... drove +{top_gainer['cnt_change']:.0f} FAS*")
                
                # --- MA Callout ---
                with col_c3:
                    st.markdown("#### 👤 MA Performance")
                    
                    # Calculate MA performance
                    ma_target = df_target.groupby('mortgage_advisor').agg({'fas_count': 'sum', 'fas_dollars': 'sum'})
                    ma_prev = df_prev.groupby('mortgage_advisor').agg({'fas_count': 'sum', 'fas_dollars': 'sum'}) / 4
                    
                    # Merge and calculate changes
                    ma_comparison = ma_target.merge(ma_prev, left_index=True, right_index=True, 
                                                   suffixes=('_target', '_prev'), how='outer').fillna(0)
                    ma_comparison['fas_change'] = ma_comparison['fas_count_target'] - ma_comparison['fas_count_prev']
                    ma_comparison['dollar_change'] = ma_comparison['fas_dollars_target'] - ma_comparison['fas_dollars_prev']
                    
                    # Top gainers and losers
                    top_gainers = ma_comparison.nlargest(2, 'fas_change')
                    top_losers = ma_comparison.nsmallest(2, 'fas_change')
                    
                    mas_improved = len(ma_comparison[ma_comparison['fas_change'] > 0])
                    mas_declined = len(ma_comparison[ma_comparison['fas_change'] < 0])
                    
                    st.markdown(f"- **Improved:** {mas_improved} MAs")
                    st.markdown(f"- **Declined:** {mas_declined} MAs")
                    
                    # Top gainer callout
                    if len(top_gainers) > 0:
                        top_ma = top_gainers.index[0]
                        top_change = top_gainers.iloc[0]['fas_change']
                        st.markdown(f"- **Top Gainer:** {top_ma[:15]}... (+{top_change:.0f})")
                    
                    # Concentration
                    top_10_target = ma_target.nlargest(10, 'fas_dollars')['fas_dollars'].sum()
                    total_target = ma_target['fas_dollars'].sum()
                    concentration = top_10_target / total_target * 100 if total_target > 0 else 0
                    st.caption(f"*Top 10 MAs = {concentration:.0f}% of FAS $*")
                
                # --- Channel Mix Callout ---
                with col_c4:
                    st.markdown("#### 📢 Channel Mix")
                    
                    # Calculate Channel distribution
                    channel_target = df_target.groupby('channel')['fas_count'].sum()
                    channel_prev = df_prev.groupby('channel')['fas_count'].sum() / 4  # 4-week avg
                    
                    target_total_ch = channel_target.sum()
                    prev_total_ch = channel_prev.sum()
                    
                    channel_target_pct = (channel_target / target_total_ch * 100) if target_total_ch > 0 else channel_target * 0
                    channel_prev_pct = (channel_prev / prev_total_ch * 100) if prev_total_ch > 0 else channel_prev * 0
                    
                    # Find biggest changes
                    channel_changes = []
                    for ch in channel_target_pct.index:
                        tgt_pct = channel_target_pct.get(ch, 0)
                        prv_pct = channel_prev_pct.get(ch, 0)
                        change = tgt_pct - prv_pct
                        cnt_change = channel_target.get(ch, 0) - channel_prev.get(ch, 0)
                        channel_changes.append({'channel': ch, 'target': tgt_pct, 'prev': prv_pct, 'change': change, 'cnt_change': cnt_change})
                    
                    channel_changes = sorted(channel_changes, key=lambda x: abs(x['change']), reverse=True)
                    
                    for item in channel_changes[:3]:
                        icon = "📈" if item['change'] > 0 else "📉" if item['change'] < 0 else "➡️"
                        short_name = str(item['channel'])[:15] + "..." if len(str(item['channel'])) > 15 else item['channel']
                        st.markdown(f"- **{short_name}**: {item['target']:.1f}% {icon} ({item['change']:+.1f}pp)")
                    
                    # Key insight
                    biggest_ch = channel_changes[0] if channel_changes else None
                    if biggest_ch and abs(biggest_ch['change']) > 2:
                        st.caption(f"*{str(biggest_ch['channel'])[:12]}... shifted {biggest_ch['change']:+.1f}pp*")
                
                st.divider()
                
                # === WEEKLY FAS METRICS - MATCHING FAS ANALYSIS TAB ===
                st.markdown("## 📈 Weekly FAS Metrics (In-Period)")
                
                # Query same data as FAS Analysis tab
                weekly_fas_query = """
                WITH fas_data AS (
                    SELECT
                        DATE_TRUNC(DATE(full_app_submit_datetime), WEEK(SUNDAY)) as fas_week,
                        COUNT(DISTINCT lendage_guid) as fas_count,
                        SUM(e_loan_amount) as fas_dollars
                    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                    WHERE full_app_submit_datetime IS NOT NULL
                    AND DATE(full_app_submit_datetime) >= '2025-10-01'
                    AND DATE(full_app_submit_datetime) <= CURRENT_DATE()
                    GROUP BY 1
                ),
                sts_data AS (
                    SELECT
                        DATE_TRUNC(DATE(sent_to_sales_date), WEEK(SUNDAY)) as sts_week,
                        COUNT(DISTINCT lendage_guid) as sts_volume
                    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                    WHERE sent_to_sales_date IS NOT NULL
                    AND DATE(sent_to_sales_date) >= '2025-10-01'
                    AND DATE(sent_to_sales_date) <= CURRENT_DATE()
                    GROUP BY 1
                )
                SELECT 
                    f.fas_week,
                    f.fas_count,
                    f.fas_dollars,
                    COALESCE(s.sts_volume, 1) as sts_volume
                FROM fas_data f
                LEFT JOIN sts_data s ON f.fas_week = s.sts_week
                ORDER BY f.fas_week
                """
                
                weekly_fas = client.query(weekly_fas_query).to_dataframe()
                weekly_fas['fas_week'] = pd.to_datetime(weekly_fas['fas_week'])
                weekly_fas['fas_rate'] = weekly_fas['fas_count'] / weekly_fas['sts_volume']
                weekly_fas['fas_per_sts'] = weekly_fas['fas_dollars'] / weekly_fas['sts_volume']
                weekly_fas['fas_dollars_m'] = weekly_fas['fas_dollars'] / 1000000
                weekly_fas['fas_per_sts_k'] = weekly_fas['fas_per_sts'] / 1000
                
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    bars_qty = alt.Chart(weekly_fas).mark_bar(color='#4cc9f0').encode(
                        x=alt.X('fas_week:T', title='Week (by FAS Date)', axis=alt.Axis(format='%b %d')),
                        y=alt.Y('fas_count:Q', title='FAS Count'),
                        tooltip=[
                            alt.Tooltip('fas_week:T', title='Week', format='%b %d, %Y'),
                            alt.Tooltip('fas_count:Q', title='FAS Count', format=','),
                            alt.Tooltip('sts_volume:Q', title='StS Volume (same week)', format=',')
                        ]
                    )
                    text_qty = bars_qty.mark_text(dy=-10, fontSize=10).encode(
                        text=alt.Text('fas_count:Q', format=',')
                    )
                    fig_fas_qty = (bars_qty + text_qty).properties(title='📊 FAS Quantity by Week (In-Period)', height=300)
                    st.altair_chart(fig_fas_qty, use_container_width=True)
                    st.caption("**Field:** `full_app_submit_datetime` | **Calc:** COUNT(DISTINCT lendage_guid) grouped by FAS week")
                
                with col_f2:
                    bars_dollars = alt.Chart(weekly_fas).mark_bar(color='#2ca02c').encode(
                        x=alt.X('fas_week:T', title='Week (by FAS Date)', axis=alt.Axis(format='%b %d')),
                        y=alt.Y('fas_dollars:Q', title='FAS $', axis=alt.Axis(format='$,.0f')),
                        tooltip=[
                            alt.Tooltip('fas_week:T', title='Week', format='%b %d, %Y'),
                            alt.Tooltip('fas_dollars:Q', title='FAS $', format='$,.0f')
                        ]
                    )
                    text_dollars = bars_dollars.mark_text(dy=-10, fontSize=9).encode(
                        text=alt.Text('fas_dollars_m:Q', format='$,.1fM')
                    )
                    fig_fas_dollars = (bars_dollars + text_dollars).properties(title='💰 FAS $ by Week (In-Period)', height=300)
                    st.altair_chart(fig_fas_dollars, use_container_width=True)
                    st.caption("**Field:** `e_loan_amount` | **Calc:** SUM(e_loan_amount) grouped by FAS week")
                
                col_f3, col_f4 = st.columns(2)
                
                with col_f3:
                    line_rate = alt.Chart(weekly_fas).mark_line(point=True, color='#f9c74f').encode(
                        x=alt.X('fas_week:T', title='Week (by FAS Date)', axis=alt.Axis(format='%b %d')),
                        y=alt.Y('fas_rate:Q', title='FAS Rate', axis=alt.Axis(format='.0%')),
                        tooltip=[
                            alt.Tooltip('fas_week:T', title='Week', format='%b %d, %Y'),
                            alt.Tooltip('fas_rate:Q', title='FAS Rate', format='.1%'),
                            alt.Tooltip('fas_count:Q', title='FAS Count', format=','),
                            alt.Tooltip('sts_volume:Q', title='StS Volume', format=',')
                        ]
                    )
                    text_rate = line_rate.mark_text(dy=-12, fontSize=9).encode(
                        text=alt.Text('fas_rate:Q', format='.1%')
                    )
                    fig_fas_rate = (line_rate + text_rate).properties(title='📈 FAS Rate by Week (In-Period FAS / Same Week StS)', height=300)
                    st.altair_chart(fig_fas_rate, use_container_width=True)
                    st.caption("**Calc:** FAS Count (in-period) / StS Volume (same week) — *Note: This is an approximation, not true cohort rate*")
                
                with col_f4:
                    line_per_sts = alt.Chart(weekly_fas).mark_line(point=True, color='#9d4edd').encode(
                        x=alt.X('fas_week:T', title='Week (by FAS Date)', axis=alt.Axis(format='%b %d')),
                        y=alt.Y('fas_per_sts:Q', title='FAS $ per StS', axis=alt.Axis(format='$,.0f')),
                        tooltip=[
                            alt.Tooltip('fas_week:T', title='Week', format='%b %d, %Y'),
                            alt.Tooltip('fas_per_sts:Q', title='FAS $/StS', format='$,.0f')
                        ]
                    )
                    text_per_sts = line_per_sts.mark_text(dy=-12, fontSize=9).encode(
                        text=alt.Text('fas_per_sts_k:Q', format='$,.1fK')
                    )
                    fig_fas_per_sts = (line_per_sts + text_per_sts).properties(title='💵 FAS $ per StS Lead by Week', height=300)
                    st.altair_chart(fig_fas_per_sts, use_container_width=True)
                    st.caption("**Calc:** FAS $ (in-period) / StS Volume (same week)")
                
                st.divider()
                
                # === 2/8 DEEP DIVE ===
                st.markdown("## 🔍 Week of 2/8 Deep Dive: What Drove the Uptick?")
                
                # KPI Summary
                col_k1, col_k2, col_k3, col_k4 = st.columns(4)
                
                with col_k1:
                    st.metric(
                        "FAS Count",
                        f"{target_data['fas_count']:,.0f}",
                        f"+{fas_pct_change:.0f}% vs prev avg"
                    )
                with col_k2:
                    st.metric(
                        "FAS $",
                        f"${target_data['fas_dollars']/1000000:.1f}M",
                        f"+${fas_dollar_change/1000000:.1f}M"
                    )
                with col_k3:
                    rate_delta = (target_data['fas_rate'] - prev_avg['fas_rate']) * 100
                    st.metric(
                        "FAS Rate",
                        f"{target_data['fas_rate']*100:.1f}%",
                        f"{rate_delta:+.1f}pp"
                    )
                with col_k4:
                    loan_delta = target_data['avg_loan'] - prev_avg['avg_loan']
                    st.metric(
                        "Avg Loan",
                        f"${target_data['avg_loan']:,.0f}",
                        f"${loan_delta:+,.0f}"
                    )
                
                st.divider()
                
                # === LVC GROUP ANALYSIS ===
                st.markdown("### 📊 LVC Group Analysis: Which Lead Segments Drove Growth?")
                
                # Query LVC data for comparison
                lvc_analysis_query = """
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
                    SUM(e_loan_amount) as fas_dollars
                FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                WHERE full_app_submit_datetime IS NOT NULL
                AND DATE(full_app_submit_datetime) >= '2026-01-12'
                AND DATE(full_app_submit_datetime) <= '2026-02-14'
                GROUP BY 1, 2
                """
                
                df_lvc_analysis = client.query(lvc_analysis_query).to_dataframe()
                df_lvc_analysis['fas_week'] = pd.to_datetime(df_lvc_analysis['fas_week'])
                
                lvc_target_week = pd.Timestamp('2026-02-08')
                lvc_prev_weeks = [lvc_target_week - pd.Timedelta(weeks=i) for i in range(1, 5)]
                
                lvc_target = df_lvc_analysis[df_lvc_analysis['fas_week'] == lvc_target_week].copy()
                lvc_target['period'] = 'Week of 2/8'
                
                lvc_prev = df_lvc_analysis[df_lvc_analysis['fas_week'].isin(lvc_prev_weeks)].groupby('lvc_group').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                lvc_prev['fas_count'] = lvc_prev['fas_count'] / 4
                lvc_prev['fas_dollars'] = lvc_prev['fas_dollars'] / 4
                lvc_prev['period'] = 'Prev 4-wk Avg'
                
                lvc_comparison = pd.concat([lvc_target[['lvc_group', 'fas_count', 'fas_dollars', 'period']], lvc_prev])
                
                lvc_order = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']
                
                col_l1, col_l2 = st.columns(2)
                
                with col_l1:
                    bars_lvc = alt.Chart(lvc_comparison).mark_bar().encode(
                        x=alt.X('lvc_group:N', title='LVC Group', sort=lvc_order),
                        y=alt.Y('fas_count:Q', title='FAS Count'),
                        color=alt.Color('period:N', scale=alt.Scale(
                            domain=['Week of 2/8', 'Prev 4-wk Avg'],
                            range=['#4cc9f0', '#8892b0']
                        )),
                        xOffset='period:N',
                        tooltip=[
                            alt.Tooltip('lvc_group:N', title='LVC Group'),
                            alt.Tooltip('period:N', title='Period'),
                            alt.Tooltip('fas_count:Q', title='FAS Count', format=',.0f')
                        ]
                    )
                    text_lvc = alt.Chart(lvc_comparison).mark_text(dy=-8, fontSize=9).encode(
                        x=alt.X('lvc_group:N', sort=lvc_order),
                        y=alt.Y('fas_count:Q'),
                        xOffset='period:N',
                        text=alt.Text('fas_count:Q', format=',.0f'),
                        color=alt.value('black')
                    )
                    fig_lvc = (bars_lvc + text_lvc).properties(title='FAS Count by LVC Group', height=300)
                    st.altair_chart(fig_lvc, use_container_width=True)
                
                with col_l2:
                    lvc_comparison['fas_dollars_k'] = lvc_comparison['fas_dollars'] / 1000
                    bars_lvc_dollars = alt.Chart(lvc_comparison).mark_bar().encode(
                        x=alt.X('lvc_group:N', title='LVC Group', sort=lvc_order),
                        y=alt.Y('fas_dollars:Q', title='FAS $', axis=alt.Axis(format='$,.0f')),
                        color=alt.Color('period:N', scale=alt.Scale(
                            domain=['Week of 2/8', 'Prev 4-wk Avg'],
                            range=['#2ca02c', '#8892b0']
                        )),
                        xOffset='period:N',
                        tooltip=[
                            alt.Tooltip('lvc_group:N', title='LVC Group'),
                            alt.Tooltip('period:N', title='Period'),
                            alt.Tooltip('fas_dollars:Q', title='FAS $', format='$,.0f')
                        ]
                    )
                    text_lvc_dollars = alt.Chart(lvc_comparison).mark_text(dy=-8, fontSize=8).encode(
                        x=alt.X('lvc_group:N', sort=lvc_order),
                        y=alt.Y('fas_dollars:Q'),
                        xOffset='period:N',
                        text=alt.Text('fas_dollars_k:Q', format='$,.0fK'),
                        color=alt.value('black')
                    )
                    fig_lvc_dollars = (bars_lvc_dollars + text_lvc_dollars).properties(title='FAS $ by LVC Group', height=300)
                    st.altair_chart(fig_lvc_dollars, use_container_width=True)
                
                # LVC Change Table
                lvc_pivot = lvc_comparison.pivot(index='lvc_group', columns='period', values=['fas_count', 'fas_dollars']).reset_index()
                
                # Flatten multi-index columns properly
                new_cols = []
                for col in lvc_pivot.columns:
                    if isinstance(col, tuple):
                        if col[0] == '' or col[1] == '':
                            new_cols.append(col[0] if col[0] else col[1])
                        else:
                            new_cols.append(f"{col[0]}_{col[1]}")
                    else:
                        new_cols.append(col)
                lvc_pivot.columns = new_cols
                
                # Rename columns for clarity
                col_mapping = {
                    'lvc_group': 'LVC Group',
                    'fas_count_Week of 2/8': 'FAS Count (2/8)',
                    'fas_count_Prev 4-wk Avg': 'FAS Count (Prev Avg)',
                    'fas_dollars_Week of 2/8': 'FAS $ (2/8)',
                    'fas_dollars_Prev 4-wk Avg': 'FAS $ (Prev Avg)'
                }
                lvc_pivot = lvc_pivot.rename(columns=col_mapping)
                
                # Handle case where columns might not exist
                required_cols = ['FAS Count (2/8)', 'FAS Count (Prev Avg)', 'FAS $ (2/8)', 'FAS $ (Prev Avg)']
                if all(col in lvc_pivot.columns for col in required_cols):
                    lvc_pivot['Count Δ'] = lvc_pivot['FAS Count (2/8)'].fillna(0) - lvc_pivot['FAS Count (Prev Avg)'].fillna(0)
                    lvc_pivot['$ Δ'] = lvc_pivot['FAS $ (2/8)'].fillna(0) - lvc_pivot['FAS $ (Prev Avg)'].fillna(0)
                    lvc_pivot['Count % Δ'] = (lvc_pivot['Count Δ'] / lvc_pivot['FAS Count (Prev Avg)'].replace(0, 1) * 100).fillna(0)
                    
                    # Calculate % of total (share) for each period
                    total_target = lvc_pivot['FAS Count (2/8)'].sum()
                    total_prev = lvc_pivot['FAS Count (Prev Avg)'].sum()
                    lvc_pivot['% of Total (2/8)'] = (lvc_pivot['FAS Count (2/8)'] / total_target * 100) if total_target > 0 else 0
                    lvc_pivot['% of Total (Prev)'] = (lvc_pivot['FAS Count (Prev Avg)'] / total_prev * 100) if total_prev > 0 else 0
                    lvc_pivot['Share Δ (pp)'] = lvc_pivot['% of Total (2/8)'] - lvc_pivot['% of Total (Prev)']
                    
                    display_cols = [c for c in ['LVC Group', 'FAS Count (2/8)', 'FAS Count (Prev Avg)', 'Count Δ', 'Count % Δ', 
                                                '% of Total (2/8)', '% of Total (Prev)', 'Share Δ (pp)',
                                                'FAS $ (2/8)', 'FAS $ (Prev Avg)', '$ Δ'] if c in lvc_pivot.columns]
                    lvc_display = lvc_pivot[display_cols].copy()
                    
                    if 'FAS Count (2/8)' in lvc_display.columns:
                        lvc_display['FAS Count (2/8)'] = lvc_display['FAS Count (2/8)'].apply(lambda x: f"{x:,.0f}")
                    if 'FAS Count (Prev Avg)' in lvc_display.columns:
                        lvc_display['FAS Count (Prev Avg)'] = lvc_display['FAS Count (Prev Avg)'].apply(lambda x: f"{x:.1f}")
                    if 'Count Δ' in lvc_display.columns:
                        lvc_display['Count Δ'] = lvc_display['Count Δ'].apply(lambda x: f"{x:+.1f}")
                    if 'Count % Δ' in lvc_display.columns:
                        lvc_display['Count % Δ'] = lvc_display['Count % Δ'].apply(lambda x: f"{x:+.1f}%")
                    if '% of Total (2/8)' in lvc_display.columns:
                        lvc_display['% of Total (2/8)'] = lvc_display['% of Total (2/8)'].apply(lambda x: f"{x:.1f}%")
                    if '% of Total (Prev)' in lvc_display.columns:
                        lvc_display['% of Total (Prev)'] = lvc_display['% of Total (Prev)'].apply(lambda x: f"{x:.1f}%")
                    if 'Share Δ (pp)' in lvc_display.columns:
                        lvc_display['Share Δ (pp)'] = lvc_display['Share Δ (pp)'].apply(lambda x: f"{x:+.1f}pp")
                    if 'FAS $ (2/8)' in lvc_display.columns:
                        lvc_display['FAS $ (2/8)'] = lvc_display['FAS $ (2/8)'].apply(lambda x: f"${x:,.0f}")
                    if 'FAS $ (Prev Avg)' in lvc_display.columns:
                        lvc_display['FAS $ (Prev Avg)'] = lvc_display['FAS $ (Prev Avg)'].apply(lambda x: f"${x:,.0f}")
                    if '$ Δ' in lvc_display.columns:
                        lvc_display['$ Δ'] = lvc_display['$ Δ'].apply(lambda x: f"${x:+,.0f}")
                    
                    st.dataframe(lvc_display, use_container_width=True, hide_index=True)
                    st.caption("**Share Δ (pp)** = Change in percentage point share of total FAS (mix shift)")
                else:
                    st.warning(f"Missing columns for LVC comparison. Available: {list(lvc_pivot.columns)}")
                
                st.divider()
                
                # === 2/11-2/13 DEEP DIVE ===
                st.markdown("## 🔬 Deep Dive: 2/11-2/13 Performance Drivers")
                st.caption("Analyzing the peak performance days to understand what drove the success")
                
                # Query comprehensive data for 2/11-2/13 vs rest of week
                deep_dive_query = """
                WITH base_data AS (
                    SELECT
                        DATE(full_app_submit_datetime) as fas_date,
                        CASE 
                            WHEN DATE(full_app_submit_datetime) BETWEEN '2026-02-11' AND '2026-02-13' THEN 'Peak (2/11-2/13)'
                            ELSE 'Rest of Week'
                        END as period,
                        CASE
                            WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
                            WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
                            WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
                            WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
                            ELSE 'Other'
                        END as lvc_group,
                        mortgage_advisor,
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
                        e_loan_amount
                    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                    WHERE full_app_submit_datetime IS NOT NULL
                    AND DATE(full_app_submit_datetime) BETWEEN '2026-02-08' AND '2026-02-14'
                )
                SELECT
                    fas_date,
                    period,
                    lvc_group,
                    mortgage_advisor,
                    vintage_date,
                    days_to_fas,
                    speed_to_dial_minutes,
                    is_digital_start,
                    call_attempts,
                    COUNT(DISTINCT lendage_guid) as fas_count,
                    SUM(e_loan_amount) as fas_dollars,
                    AVG(e_loan_amount) as avg_loan
                FROM base_data
                GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9
                """
                
                df_deep = client.query(deep_dive_query).to_dataframe()
                df_deep['fas_date'] = pd.to_datetime(df_deep['fas_date'])
                df_deep['vintage_date'] = pd.to_datetime(df_deep['vintage_date'])
                
                # Separate peak vs rest
                df_peak = df_deep[df_deep['period'] == 'Peak (2/11-2/13)']
                df_rest = df_deep[df_deep['period'] == 'Rest of Week']
                
                # Summary metrics
                peak_fas = df_peak['fas_count'].sum()
                rest_fas = df_rest['fas_count'].sum()
                peak_dollars = df_peak['fas_dollars'].sum()
                rest_dollars = df_rest['fas_dollars'].sum()
                
                st.markdown("### 📊 Peak Days (2/11-2/13) vs Rest of Week")
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                with col_p1:
                    st.metric("Peak FAS Count", f"{peak_fas:,.0f}", f"{peak_fas/(peak_fas+rest_fas)*100:.0f}% of week")
                with col_p2:
                    st.metric("Peak FAS $", f"${peak_dollars:,.0f}", f"{peak_dollars/(peak_dollars+rest_dollars)*100:.0f}% of week")
                with col_p3:
                    peak_avg = df_peak['avg_loan'].mean() if len(df_peak) > 0 else 0
                    rest_avg = df_rest['avg_loan'].mean() if len(df_rest) > 0 else 0
                    st.metric("Peak Avg Loan", f"${peak_avg:,.0f}", f"${peak_avg - rest_avg:+,.0f} vs rest")
                with col_p4:
                    peak_days = 3
                    rest_days = 4
                    peak_daily = peak_fas / peak_days
                    rest_daily = rest_fas / rest_days
                    st.metric("Peak Daily Avg", f"{peak_daily:.1f} FAS/day", f"+{peak_daily - rest_daily:.1f} vs rest")
                
                st.divider()
                
                # === 1. LVC GROUP ANALYSIS ===
                st.markdown("#### 🎯 1. LVC Group Distribution")
                
                lvc_peak = df_peak.groupby('lvc_group')['fas_count'].sum()
                lvc_rest = df_rest.groupby('lvc_group')['fas_count'].sum()
                
                lvc_comp = pd.DataFrame({
                    'LVC Group': lvc_peak.index,
                    'Peak (2/11-13)': lvc_peak.values,
                    'Rest of Week': [lvc_rest.get(lvc, 0) for lvc in lvc_peak.index]
                })
                lvc_comp['Peak %'] = lvc_comp['Peak (2/11-13)'] / lvc_comp['Peak (2/11-13)'].sum() * 100
                lvc_comp['Rest %'] = lvc_comp['Rest of Week'] / lvc_comp['Rest of Week'].sum() * 100
                lvc_comp['% Δ (pp)'] = lvc_comp['Peak %'] - lvc_comp['Rest %']
                
                col_lvc1, col_lvc2 = st.columns([2, 1])
                with col_lvc1:
                    lvc_melt = lvc_comp.melt(id_vars=['LVC Group'], value_vars=['Peak (2/11-13)', 'Rest of Week'], 
                                             var_name='Period', value_name='FAS Count')
                    bars_lvc_deep = alt.Chart(lvc_melt).mark_bar().encode(
                        x=alt.X('LVC Group:N', sort=['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']),
                        y=alt.Y('FAS Count:Q'),
                        color=alt.Color('Period:N', scale=alt.Scale(domain=['Peak (2/11-13)', 'Rest of Week'], range=['#4cc9f0', '#8892b0'])),
                        xOffset='Period:N'
                    ).properties(height=250)
                    st.altair_chart(bars_lvc_deep, use_container_width=True)
                
                with col_lvc2:
                    lvc_display = lvc_comp.copy()
                    lvc_display['Peak %'] = lvc_display['Peak %'].apply(lambda x: f"{x:.1f}%")
                    lvc_display['Rest %'] = lvc_display['Rest %'].apply(lambda x: f"{x:.1f}%")
                    lvc_display['% Δ (pp)'] = lvc_display['% Δ (pp)'].apply(lambda x: f"{x:+.1f}pp")
                    st.dataframe(lvc_display[['LVC Group', 'Peak (2/11-13)', 'Peak %', 'Rest %', '% Δ (pp)']], hide_index=True)
                
                # === 2. MA PERFORMANCE ===
                st.markdown("#### 👤 2. MA Individual Performance")
                
                ma_peak = df_peak.groupby('mortgage_advisor').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                ma_peak.columns = ['MA', 'Peak FAS', 'Peak $']
                
                ma_rest = df_rest.groupby('mortgage_advisor').agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                ma_rest.columns = ['MA', 'Rest FAS', 'Rest $']
                ma_rest['Rest FAS'] = ma_rest['Rest FAS'] * 3/4  # Normalize to 3 days
                ma_rest['Rest $'] = ma_rest['Rest $'] * 3/4
                
                ma_comp = ma_peak.merge(ma_rest, on='MA', how='outer').fillna(0)
                ma_comp['FAS Δ'] = ma_comp['Peak FAS'] - ma_comp['Rest FAS']
                ma_comp = ma_comp.sort_values('Peak FAS', ascending=False)
                
                col_ma1, col_ma2 = st.columns(2)
                with col_ma1:
                    st.markdown("**Top 10 MAs During Peak (2/11-13)**")
                    ma_top = ma_comp.head(10).copy()
                    ma_top['Peak FAS'] = ma_top['Peak FAS'].apply(lambda x: f"{x:.0f}")
                    ma_top['Peak $'] = ma_top['Peak $'].apply(lambda x: f"${x:,.0f}")
                    ma_top['Rest FAS'] = ma_top['Rest FAS'].apply(lambda x: f"{x:.1f}")
                    ma_top['FAS Δ'] = ma_top['FAS Δ'].apply(lambda x: f"{x:+.1f}")
                    st.dataframe(ma_top[['MA', 'Peak FAS', 'Peak $', 'Rest FAS', 'FAS Δ']], hide_index=True, height=350)
                
                with col_ma2:
                    st.markdown("**Biggest Gainers (Peak vs Rest)**")
                    ma_gainers = ma_comp.nlargest(10, 'FAS Δ').copy()
                    ma_gainers['Peak FAS'] = ma_gainers['Peak FAS'].apply(lambda x: f"{x:.0f}")
                    ma_gainers['Rest FAS'] = ma_gainers['Rest FAS'].apply(lambda x: f"{x:.1f}")
                    ma_gainers['FAS Δ'] = ma_gainers['FAS Δ'].apply(lambda x: f"{x:+.1f}")
                    st.dataframe(ma_gainers[['MA', 'Peak FAS', 'Rest FAS', 'FAS Δ']], hide_index=True, height=350)
                
                # === 3. VINTAGE ANALYSIS ===
                st.markdown("#### 📅 3. Vintage (Lead Age)")
                
                # Calculate days to FAS distribution
                peak_days_to_fas = df_peak.groupby('days_to_fas')['fas_count'].sum().reset_index()
                rest_days_to_fas = df_rest.groupby('days_to_fas')['fas_count'].sum().reset_index()
                
                avg_days_peak = (df_peak['days_to_fas'] * df_peak['fas_count']).sum() / df_peak['fas_count'].sum() if df_peak['fas_count'].sum() > 0 else 0
                avg_days_rest = (df_rest['days_to_fas'] * df_rest['fas_count']).sum() / df_rest['fas_count'].sum() if df_rest['fas_count'].sum() > 0 else 0
                
                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    st.metric("Avg Days to FAS (Peak)", f"{avg_days_peak:.1f} days")
                with col_v2:
                    st.metric("Avg Days to FAS (Rest)", f"{avg_days_rest:.1f} days")
                with col_v3:
                    st.metric("Difference", f"{avg_days_peak - avg_days_rest:+.1f} days", 
                             "Faster" if avg_days_peak < avg_days_rest else "Slower")
                
                # Vintage week breakdown
                df_peak['vintage_week'] = df_peak['vintage_date'].dt.strftime('%m/%d')
                vintage_dist = df_peak.groupby('vintage_week')['fas_count'].sum().reset_index()
                vintage_dist = vintage_dist.sort_values('fas_count', ascending=False).head(10)
                
                st.markdown("**Top Vintage Dates Contributing to Peak FAS**")
                st.dataframe(vintage_dist.rename(columns={'vintage_week': 'Vintage Date', 'fas_count': 'FAS Count'}), hide_index=True)
                
                # === 4. SPEED TO DIAL ===
                st.markdown("#### ⏱️ 4. Speed to Dial")
                
                # Filter valid speed to dial data
                df_peak_spd = df_peak[df_peak['speed_to_dial_minutes'].notna() & (df_peak['speed_to_dial_minutes'] >= 0) & (df_peak['speed_to_dial_minutes'] < 1440)]
                df_rest_spd = df_rest[df_rest['speed_to_dial_minutes'].notna() & (df_rest['speed_to_dial_minutes'] >= 0) & (df_rest['speed_to_dial_minutes'] < 1440)]
                
                avg_spd_peak = (df_peak_spd['speed_to_dial_minutes'] * df_peak_spd['fas_count']).sum() / df_peak_spd['fas_count'].sum() if df_peak_spd['fas_count'].sum() > 0 else 0
                avg_spd_rest = (df_rest_spd['speed_to_dial_minutes'] * df_rest_spd['fas_count']).sum() / df_rest_spd['fas_count'].sum() if df_rest_spd['fas_count'].sum() > 0 else 0
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Avg Speed to Dial (Peak)", f"{avg_spd_peak:.0f} min")
                with col_s2:
                    st.metric("Avg Speed to Dial (Rest)", f"{avg_spd_rest:.0f} min")
                with col_s3:
                    diff = avg_spd_peak - avg_spd_rest
                    st.metric("Difference", f"{diff:+.0f} min", "Faster" if diff < 0 else "Slower")
                
                # === 5. DIGITAL APP STARTS ===
                st.markdown("#### 📱 5. Digital App Starts")
                
                digital_peak = df_peak[df_peak['is_digital_start'] == 1]['fas_count'].sum()
                non_digital_peak = df_peak[df_peak['is_digital_start'] == 0]['fas_count'].sum()
                digital_rest = df_rest[df_rest['is_digital_start'] == 1]['fas_count'].sum()
                non_digital_rest = df_rest[df_rest['is_digital_start'] == 0]['fas_count'].sum()
                
                digital_pct_peak = digital_peak / (digital_peak + non_digital_peak) * 100 if (digital_peak + non_digital_peak) > 0 else 0
                digital_pct_rest = digital_rest / (digital_rest + non_digital_rest) * 100 if (digital_rest + non_digital_rest) > 0 else 0
                
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.metric("Digital Start % (Peak)", f"{digital_pct_peak:.1f}%", f"{digital_peak:.0f} FAS")
                with col_d2:
                    st.metric("Digital Start % (Rest)", f"{digital_pct_rest:.1f}%", f"{digital_rest:.0f} FAS")
                with col_d3:
                    st.metric("Difference", f"{digital_pct_peak - digital_pct_rest:+.1f}pp")
                
                # === 6. CALL ATTEMPTS ===
                st.markdown("#### 📞 6. Call Attempts")
                
                # Filter valid call attempts
                df_peak_calls = df_peak[df_peak['call_attempts'].notna()]
                df_rest_calls = df_rest[df_rest['call_attempts'].notna()]
                
                avg_calls_peak = (df_peak_calls['call_attempts'] * df_peak_calls['fas_count']).sum() / df_peak_calls['fas_count'].sum() if df_peak_calls['fas_count'].sum() > 0 else 0
                avg_calls_rest = (df_rest_calls['call_attempts'] * df_rest_calls['fas_count']).sum() / df_rest_calls['fas_count'].sum() if df_rest_calls['fas_count'].sum() > 0 else 0
                
                col_ca1, col_ca2, col_ca3 = st.columns(3)
                with col_ca1:
                    st.metric("Avg Call Attempts (Peak)", f"{avg_calls_peak:.1f}")
                with col_ca2:
                    st.metric("Avg Call Attempts (Rest)", f"{avg_calls_rest:.1f}")
                with col_ca3:
                    diff_calls = avg_calls_peak - avg_calls_rest
                    st.metric("Difference", f"{diff_calls:+.1f}", "More calls" if diff_calls > 0 else "Fewer calls")
                
                # Call attempts distribution
                calls_dist_peak = df_peak_calls.groupby('call_attempts')['fas_count'].sum().reset_index()
                calls_dist_peak['period'] = 'Peak'
                calls_dist_rest = df_rest_calls.groupby('call_attempts')['fas_count'].sum().reset_index()
                calls_dist_rest['period'] = 'Rest'
                calls_dist = pd.concat([calls_dist_peak, calls_dist_rest])
                calls_dist = calls_dist[calls_dist['call_attempts'] <= 10]  # Cap at 10 for readability
                
                bars_calls = alt.Chart(calls_dist).mark_bar().encode(
                    x=alt.X('call_attempts:O', title='Call Attempts'),
                    y=alt.Y('fas_count:Q', title='FAS Count'),
                    color=alt.Color('period:N', scale=alt.Scale(domain=['Peak', 'Rest'], range=['#4cc9f0', '#8892b0'])),
                    xOffset='period:N'
                ).properties(title='FAS by Call Attempts', height=250)
                st.altair_chart(bars_calls, use_container_width=True)
                
                # === 7. CHANNEL MIX (sub_group) ===
                st.markdown("#### 📡 7. Channel Mix (Sub Group)")
                
                # Query channel mix for week of 2/8 vs prev 4 weeks
                channel_mix_query = """
                WITH week_data AS (
                    SELECT
                        CASE 
                            WHEN DATE(full_app_submit_datetime) BETWEEN '2026-02-08' AND '2026-02-14' THEN 'Week 2/8'
                            WHEN DATE(full_app_submit_datetime) >= DATE_SUB(DATE '2026-02-08', INTERVAL 4 WEEK) 
                                 AND DATE(full_app_submit_datetime) < '2026-02-08' THEN 'Prev 4 Weeks'
                        END as period,
                        COALESCE(sub_group, 'Unknown') as sub_group,
                        COUNT(DISTINCT lendage_guid) as fas_count,
                        SUM(e_loan_amount) as fas_dollars
                    FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                    WHERE full_app_submit_datetime IS NOT NULL
                    AND DATE(full_app_submit_datetime) >= DATE_SUB(DATE '2026-02-08', INTERVAL 4 WEEK)
                    AND DATE(full_app_submit_datetime) <= '2026-02-14'
                    GROUP BY 1, 2
                )
                SELECT * FROM week_data WHERE period IS NOT NULL
                ORDER BY period, fas_count DESC
                """
                
                df_channel = client.query(channel_mix_query).to_dataframe()
                
                if len(df_channel) > 0:
                    # Pivot for comparison
                    ch_week = df_channel[df_channel['period'] == 'Week 2/8'].copy()
                    ch_prev = df_channel[df_channel['period'] == 'Prev 4 Weeks'].copy()
                    
                    # Normalize prev 4 weeks to weekly avg
                    ch_prev['fas_count'] = ch_prev['fas_count'] / 4
                    ch_prev['fas_dollars'] = ch_prev['fas_dollars'] / 4
                    
                    week_total = ch_week['fas_count'].sum()
                    prev_total = ch_prev['fas_count'].sum()
                    
                    # Merge
                    all_channels = set(ch_week['sub_group'].tolist() + ch_prev['sub_group'].tolist())
                    channel_comp = []
                    
                    for ch in all_channels:
                        week_fas = ch_week[ch_week['sub_group'] == ch]['fas_count'].sum()
                        prev_fas = ch_prev[ch_prev['sub_group'] == ch]['fas_count'].sum()
                        week_pct = week_fas / week_total * 100 if week_total > 0 else 0
                        prev_pct = prev_fas / prev_total * 100 if prev_total > 0 else 0
                        diff_pp = week_pct - prev_pct
                        channel_comp.append({
                            'Channel': ch,
                            'Week 2/8 FAS': week_fas,
                            'Week 2/8 %': week_pct,
                            'Prev 4-Wk Avg FAS': prev_fas,
                            'Prev 4-Wk %': prev_pct,
                            'Δ (pp)': diff_pp
                        })
                    
                    df_ch_comp = pd.DataFrame(channel_comp)
                    df_ch_comp = df_ch_comp.sort_values('Week 2/8 FAS', ascending=False)
                    
                    # Display table
                    col_ch1, col_ch2 = st.columns([3, 2])
                    
                    with col_ch1:
                        ch_display = df_ch_comp.copy()
                        ch_display['Week 2/8 FAS'] = ch_display['Week 2/8 FAS'].apply(lambda x: f"{x:.0f}")
                        ch_display['Week 2/8 %'] = ch_display['Week 2/8 %'].apply(lambda x: f"{x:.1f}%")
                        ch_display['Prev 4-Wk Avg FAS'] = ch_display['Prev 4-Wk Avg FAS'].apply(lambda x: f"{x:.1f}")
                        ch_display['Prev 4-Wk %'] = ch_display['Prev 4-Wk %'].apply(lambda x: f"{x:.1f}%")
                        ch_display['Δ (pp)'] = ch_display['Δ (pp)'].apply(lambda x: f"{x:+.1f}pp")
                        st.dataframe(ch_display, hide_index=True, height=400)
                    
                    with col_ch2:
                        st.markdown("**📈 Key Channel Shifts vs Prev 4 Weeks**")
                        
                        # Get biggest shifts
                        df_shifts = df_ch_comp[df_ch_comp['Δ (pp)'].abs() >= 1].sort_values('Δ (pp)', key=abs, ascending=False).head(5)
                        
                        for _, row in df_shifts.iterrows():
                            direction = "⬆️" if row['Δ (pp)'] > 0 else "⬇️"
                            st.markdown(f"{direction} **{row['Channel']}**: {row['Δ (pp)']:+.1f}pp ({row['Week 2/8 %']:.1f}% → {row['Prev 4-Wk %']:.1f}%)")
                        
                        # Bar chart for top channels
                        top_channels = df_ch_comp.head(8).copy()
                        ch_melt = top_channels.melt(id_vars=['Channel'], value_vars=['Week 2/8 %', 'Prev 4-Wk %'],
                                                    var_name='Period', value_name='Share %')
                        
                        bars_ch = alt.Chart(ch_melt).mark_bar().encode(
                            x=alt.X('Channel:N', sort='-y'),
                            y=alt.Y('Share %:Q'),
                            color=alt.Color('Period:N', scale=alt.Scale(domain=['Week 2/8 %', 'Prev 4-Wk %'], range=['#4cc9f0', '#8892b0'])),
                            xOffset='Period:N'
                        ).properties(title='Channel Share: Week 2/8 vs Prev 4-Wk Avg', height=250)
                        st.altair_chart(bars_ch, use_container_width=True)
                else:
                    st.info("No channel mix data available.")
                
                # === KEY INSIGHTS SUMMARY ===
                st.divider()
                st.markdown("### 💡 Key Insights: What Drove 2/11-2/13 Performance?")
                
                insights = []
                
                # LVC insight
                lvc_12_peak_pct = lvc_comp[lvc_comp['LVC Group'] == 'LVC 1-2']['Peak %'].values[0] if 'LVC 1-2' in lvc_comp['LVC Group'].values else 0
                lvc_12_rest_pct = lvc_comp[lvc_comp['LVC Group'] == 'LVC 1-2']['Rest %'].values[0] if 'LVC 1-2' in lvc_comp['LVC Group'].values else 0
                if isinstance(lvc_12_peak_pct, str):
                    lvc_12_peak_pct = float(lvc_12_peak_pct.replace('%', ''))
                if isinstance(lvc_12_rest_pct, str):
                    lvc_12_rest_pct = float(lvc_12_rest_pct.replace('%', ''))
                lvc_diff = lvc_12_peak_pct - lvc_12_rest_pct
                if abs(lvc_diff) > 3:
                    insights.append(f"🎯 **LVC Mix:** LVC 1-2 was {lvc_12_peak_pct:.1f}% of peak FAS vs {lvc_12_rest_pct:.1f}% rest ({lvc_diff:+.1f}pp shift)")
                
                # Vintage insight
                if abs(avg_days_peak - avg_days_rest) > 2:
                    insights.append(f"📅 **Vintage:** Peak FAS came from {'newer' if avg_days_peak < avg_days_rest else 'older'} leads ({avg_days_peak:.1f} vs {avg_days_rest:.1f} days to FAS)")
                
                # Speed insight
                if abs(avg_spd_peak - avg_spd_rest) > 10:
                    insights.append(f"⏱️ **Speed to Dial:** Peak leads were dialed {'faster' if avg_spd_peak < avg_spd_rest else 'slower'} ({avg_spd_peak:.0f} vs {avg_spd_rest:.0f} min)")
                
                # Digital insight
                if abs(digital_pct_peak - digital_pct_rest) > 2:
                    insights.append(f"📱 **Digital Apps:** {digital_pct_peak:.1f}% of peak FAS had digital app starts vs {digital_pct_rest:.1f}% rest")
                
                # Call attempts insight
                if abs(avg_calls_peak - avg_calls_rest) > 0.5:
                    insights.append(f"📞 **Call Attempts:** Peak FAS had {'more' if avg_calls_peak > avg_calls_rest else 'fewer'} call attempts ({avg_calls_peak:.1f} vs {avg_calls_rest:.1f})")
                
                # Top MA insight
                top_ma_peak = ma_comp.head(1)['MA'].values[0] if len(ma_comp) > 0 else "N/A"
                top_ma_fas = ma_comp.head(1)['Peak FAS'].values[0] if len(ma_comp) > 0 else 0
                insights.append(f"👤 **Top MA:** {top_ma_peak} led with {top_ma_fas:.0f} FAS during peak days")
                
                for insight in insights:
                    st.markdown(insight)
                
                if not insights:
                    st.info("No major differences detected between peak days and rest of week.")
                
                st.divider()
                
                # === DAILY BREAKOUT ===
                st.markdown("## 📅 Daily Breakout: Week of 2/8/2026")
                st.caption("Daily performance breakdown by LVC Group, Persona, and MA (based on FAS date)")
                
                # Query daily data for 2/8 week - matching FAS Analysis tab
                daily_query = """
                SELECT
                    DATE(full_app_submit_datetime) as fas_date,
                    CASE
                        WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
                        WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
                        WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
                        WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
                        ELSE 'Other'
                    END as lvc_group,
                    persona,
                    mortgage_advisor,
                    COUNT(DISTINCT lendage_guid) as fas_count,
                    SUM(e_loan_amount) as fas_dollars,
                    AVG(e_loan_amount) as avg_loan
                FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                WHERE full_app_submit_datetime IS NOT NULL
                AND DATE(full_app_submit_datetime) >= '2026-02-08'
                AND DATE(full_app_submit_datetime) <= '2026-02-14'
                GROUP BY 1, 2, 3, 4
                """
                
                df_daily = client.query(daily_query).to_dataframe()
                df_daily['fas_date'] = pd.to_datetime(df_daily['fas_date'])
                df_daily['day_name'] = df_daily['fas_date'].dt.strftime('%a %m/%d')
                
                # Sort days properly
                day_order = df_daily.sort_values('fas_date')['day_name'].unique().tolist()
                
                # --- Daily by LVC Group ---
                st.markdown("### 📊 Daily FAS by LVC Group (% of Total)")
                
                daily_lvc = df_daily.groupby(['fas_date', 'day_name', 'lvc_group']).agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum',
                    'avg_loan': 'mean'
                }).reset_index()
                
                # Calculate daily totals for % of total
                daily_totals = daily_lvc.groupby('day_name')['fas_count'].sum().reset_index()
                daily_totals.columns = ['day_name', 'daily_total']
                daily_lvc = daily_lvc.merge(daily_totals, on='day_name')
                daily_lvc['pct_of_total'] = daily_lvc['fas_count'] / daily_lvc['daily_total'] * 100
                
                lvc_order = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']
                
                col_lvc1, col_lvc2 = st.columns(2)
                
                with col_lvc1:
                    # Stacked bar chart showing % of total
                    bars_daily_lvc = alt.Chart(daily_lvc).mark_bar().encode(
                        x=alt.X('day_name:N', title='Day', sort=day_order),
                        y=alt.Y('pct_of_total:Q', title='% of Daily Total', stack='normalize', 
                                axis=alt.Axis(format='.0%')),
                        color=alt.Color('lvc_group:N', title='LVC Group', sort=lvc_order,
                                       scale=alt.Scale(scheme='tableau10')),
                        order=alt.Order('lvc_group:N', sort='ascending'),
                        tooltip=[
                            alt.Tooltip('day_name:N', title='Day'),
                            alt.Tooltip('lvc_group:N', title='LVC Group'),
                            alt.Tooltip('fas_count:Q', title='FAS Count', format=','),
                            alt.Tooltip('pct_of_total:Q', title='% of Daily Total', format='.1f'),
                            alt.Tooltip('fas_dollars:Q', title='FAS $', format='$,.0f')
                        ]
                    ).properties(title='% of Daily FAS by LVC Group', height=300)
                    st.altair_chart(bars_daily_lvc, use_container_width=True)
                
                with col_lvc2:
                    # Table view with % of total
                    daily_lvc_pivot = daily_lvc.pivot_table(
                        index='lvc_group', 
                        columns='day_name', 
                        values='fas_count', 
                        aggfunc='sum'
                    ).fillna(0)
                    # Reorder columns by date
                    ordered_cols = [c for c in day_order if c in daily_lvc_pivot.columns]
                    daily_lvc_pivot = daily_lvc_pivot[ordered_cols]
                    daily_lvc_pivot['Total'] = daily_lvc_pivot.sum(axis=1)
                    grand_total = daily_lvc_pivot['Total'].sum()
                    daily_lvc_pivot['% of Total'] = daily_lvc_pivot['Total'] / grand_total * 100
                    daily_lvc_pivot = daily_lvc_pivot.reset_index()
                    daily_lvc_pivot = daily_lvc_pivot.sort_values('Total', ascending=False)
                    
                    # Format
                    for col in daily_lvc_pivot.columns:
                        if col == '% of Total':
                            daily_lvc_pivot[col] = daily_lvc_pivot[col].apply(lambda x: f"{x:.1f}%")
                        elif col != 'lvc_group':
                            daily_lvc_pivot[col] = daily_lvc_pivot[col].apply(lambda x: f"{int(x):,}")
                    daily_lvc_pivot = daily_lvc_pivot.rename(columns={'lvc_group': 'LVC Group'})
                    st.dataframe(daily_lvc_pivot, use_container_width=True, hide_index=True)
                
                st.divider()
                
                # --- Daily by Persona ---
                st.markdown("### 🎭 Daily FAS by Persona (% of Total)")
                
                daily_persona = df_daily.groupby(['fas_date', 'day_name', 'persona']).agg({
                    'fas_count': 'sum',
                    'fas_dollars': 'sum'
                }).reset_index()
                
                # Calculate daily totals for % of total
                daily_persona_totals = daily_persona.groupby('day_name')['fas_count'].sum().reset_index()
                daily_persona_totals.columns = ['day_name', 'daily_total']
                daily_persona = daily_persona.merge(daily_persona_totals, on='day_name')
                daily_persona['pct_of_total'] = daily_persona['fas_count'] / daily_persona['daily_total'] * 100
                
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    # Top 5 personas - stacked % chart
                    top_personas = daily_persona.groupby('persona')['fas_count'].sum().nlargest(5).index.tolist()
                    daily_persona_top = daily_persona[daily_persona['persona'].isin(top_personas)]
                    
                    bars_daily_persona = alt.Chart(daily_persona_top).mark_bar().encode(
                        x=alt.X('day_name:N', title='Day', sort=day_order),
                        y=alt.Y('pct_of_total:Q', title='% of Daily Total', stack='normalize',
                                axis=alt.Axis(format='.0%')),
                        color=alt.Color('persona:N', title='Persona', scale=alt.Scale(scheme='category10')),
                        order=alt.Order('persona:N'),
                        tooltip=[
                            alt.Tooltip('day_name:N', title='Day'),
                            alt.Tooltip('persona:N', title='Persona'),
                            alt.Tooltip('fas_count:Q', title='FAS Count', format=','),
                            alt.Tooltip('pct_of_total:Q', title='% of Daily Total', format='.1f'),
                            alt.Tooltip('fas_dollars:Q', title='FAS $', format='$,.0f')
                        ]
                    ).properties(title='% of Daily FAS by Persona (Top 5)', height=300)
                    st.altair_chart(bars_daily_persona, use_container_width=True)
                
                with col_p2:
                    # Persona summary table
                    persona_summary = df_daily.groupby('persona').agg({
                        'fas_count': 'sum',
                        'fas_dollars': 'sum',
                        'avg_loan': 'mean'
                    }).reset_index()
                    persona_summary = persona_summary.sort_values('fas_count', ascending=False)
                    persona_summary['% of Total'] = (persona_summary['fas_count'] / persona_summary['fas_count'].sum() * 100)
                    
                    persona_display = persona_summary.copy()
                    persona_display['fas_count'] = persona_display['fas_count'].apply(lambda x: f"{x:,.0f}")
                    persona_display['fas_dollars'] = persona_display['fas_dollars'].apply(lambda x: f"${x:,.0f}")
                    persona_display['avg_loan'] = persona_display['avg_loan'].apply(lambda x: f"${x:,.0f}")
                    persona_display['% of Total'] = persona_display['% of Total'].apply(lambda x: f"{x:.1f}%")
                    persona_display.columns = ['Persona', 'FAS Count', 'FAS $', 'Avg Loan', '% of Total']
                    st.dataframe(persona_display, use_container_width=True, hide_index=True)
                
                st.divider()
                
                # --- Daily MA Performance with Full Funnel ---
                st.markdown("### 👤 Daily MA Performance (Full Funnel)")
                
                # Query MA performance with assigned, contacted, FAS - Target Week
                ma_funnel_query = """
                SELECT
                    mortgage_advisor,
                    COUNT(DISTINCT CASE WHEN current_sales_assigned_date IS NOT NULL 
                          AND DATE(current_sales_assigned_date) BETWEEN '2026-02-08' AND '2026-02-14' 
                          THEN lendage_guid END) as assigned,
                    COUNT(DISTINCT CASE WHEN contacted_date IS NOT NULL 
                          AND DATE(contacted_date) BETWEEN '2026-02-08' AND '2026-02-14' 
                          THEN lendage_guid END) as contacted,
                    COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL 
                          AND DATE(full_app_submit_datetime) BETWEEN '2026-02-08' AND '2026-02-14' 
                          THEN lendage_guid END) as fas_count,
                    SUM(CASE WHEN full_app_submit_datetime IS NOT NULL 
                        AND DATE(full_app_submit_datetime) BETWEEN '2026-02-08' AND '2026-02-14' 
                        THEN e_loan_amount ELSE 0 END) as fas_dollars,
                    AVG(CASE WHEN full_app_submit_datetime IS NOT NULL 
                        AND DATE(full_app_submit_datetime) BETWEEN '2026-02-08' AND '2026-02-14' 
                        THEN e_loan_amount END) as avg_fas_loan,
                    COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL 
                          AND DATE(sent_to_sales_date) BETWEEN '2026-02-08' AND '2026-02-14' 
                          THEN lendage_guid END) as sent_to_sales
                FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                WHERE mortgage_advisor IS NOT NULL
                AND (
                    (current_sales_assigned_date IS NOT NULL AND DATE(current_sales_assigned_date) BETWEEN '2026-02-08' AND '2026-02-14')
                    OR (full_app_submit_datetime IS NOT NULL AND DATE(full_app_submit_datetime) BETWEEN '2026-02-08' AND '2026-02-14')
                    OR (sent_to_sales_date IS NOT NULL AND DATE(sent_to_sales_date) BETWEEN '2026-02-08' AND '2026-02-14')
                )
                GROUP BY 1
                HAVING fas_count > 0 OR assigned > 0
                """
                
                # Query MA performance for Previous 4 Weeks (for comparison)
                ma_prev_query = """
                SELECT
                    mortgage_advisor,
                    COUNT(DISTINCT CASE WHEN current_sales_assigned_date IS NOT NULL 
                          AND DATE(current_sales_assigned_date) BETWEEN '2026-01-12' AND '2026-02-07' 
                          THEN lendage_guid END) / 4.0 as assigned_prev,
                    COUNT(DISTINCT CASE WHEN contacted_date IS NOT NULL 
                          AND DATE(contacted_date) BETWEEN '2026-01-12' AND '2026-02-07' 
                          THEN lendage_guid END) / 4.0 as contacted_prev,
                    COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL 
                          AND DATE(full_app_submit_datetime) BETWEEN '2026-01-12' AND '2026-02-07' 
                          THEN lendage_guid END) / 4.0 as fas_count_prev,
                    SUM(CASE WHEN full_app_submit_datetime IS NOT NULL 
                        AND DATE(full_app_submit_datetime) BETWEEN '2026-01-12' AND '2026-02-07' 
                        THEN e_loan_amount ELSE 0 END) / 4.0 as fas_dollars_prev,
                    AVG(CASE WHEN full_app_submit_datetime IS NOT NULL 
                        AND DATE(full_app_submit_datetime) BETWEEN '2026-01-12' AND '2026-02-07' 
                        THEN e_loan_amount END) as avg_fas_loan_prev,
                    COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL 
                          AND DATE(sent_to_sales_date) BETWEEN '2026-01-12' AND '2026-02-07' 
                          THEN lendage_guid END) / 4.0 as sent_to_sales_prev
                FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                WHERE mortgage_advisor IS NOT NULL
                AND (
                    (current_sales_assigned_date IS NOT NULL AND DATE(current_sales_assigned_date) BETWEEN '2026-01-12' AND '2026-02-07')
                    OR (full_app_submit_datetime IS NOT NULL AND DATE(full_app_submit_datetime) BETWEEN '2026-01-12' AND '2026-02-07')
                    OR (sent_to_sales_date IS NOT NULL AND DATE(sent_to_sales_date) BETWEEN '2026-01-12' AND '2026-02-07')
                )
                GROUP BY 1
                """
                
                df_ma_funnel = client.query(ma_funnel_query).to_dataframe()
                df_ma_prev = client.query(ma_prev_query).to_dataframe()
                
                # Merge target week with previous 4-week avg
                df_ma_combined = df_ma_funnel.merge(df_ma_prev, on='mortgage_advisor', how='left').fillna(0)
                
                # Calculate rates
                df_ma_combined['fas_pct_assigned'] = (df_ma_combined['fas_count'] / df_ma_combined['assigned'].replace(0, 1) * 100)
                df_ma_combined['fas_pct_sts'] = (df_ma_combined['fas_count'] / df_ma_combined['sent_to_sales'].replace(0, 1) * 100)
                df_ma_combined['contact_rate'] = (df_ma_combined['contacted'] / df_ma_combined['assigned'].replace(0, 1) * 100)
                
                # Previous 4-week rates
                df_ma_combined['fas_pct_assigned_prev'] = (df_ma_combined['fas_count_prev'] / df_ma_combined['assigned_prev'].replace(0, 1) * 100)
                df_ma_combined['fas_pct_sts_prev'] = (df_ma_combined['fas_count_prev'] / df_ma_combined['sent_to_sales_prev'].replace(0, 1) * 100)
                
                # Calculate deltas
                df_ma_combined['fas_delta'] = df_ma_combined['fas_count'] - df_ma_combined['fas_count_prev']
                df_ma_combined['fas_pct_delta'] = df_ma_combined['fas_pct_assigned'] - df_ma_combined['fas_pct_assigned_prev']
                df_ma_combined['dollar_delta'] = df_ma_combined['fas_dollars'] - df_ma_combined['fas_dollars_prev']
                
                # Sort by FAS $ descending
                df_ma_combined = df_ma_combined.sort_values('fas_dollars', ascending=False)
                
                # Display table
                st.markdown("##### MA Performance Summary (Week of 2/8 vs Prev 4-Week Avg)")
                
                ma_display = df_ma_combined[['mortgage_advisor', 'assigned', 'contacted', 'contact_rate', 
                                           'fas_count', 'fas_count_prev', 'fas_delta',
                                           'fas_pct_assigned', 'fas_pct_assigned_prev', 'fas_pct_delta',
                                           'fas_pct_sts', 
                                           'fas_dollars', 'fas_dollars_prev', 'dollar_delta',
                                           'avg_fas_loan']].copy()
                ma_display.columns = ['MA', 'Assigned', 'Contacted', 'Contact %', 
                                      'FAS', 'Prev Avg', 'FAS Δ',
                                      'FAS % (Asgn)', 'Prev %', '% Δ',
                                      'FAS % (StS)', 
                                      'FAS $', 'Prev $', '$ Δ',
                                      'Avg FAS $']
                
                # Format
                ma_display['Assigned'] = ma_display['Assigned'].apply(lambda x: f"{x:,.0f}")
                ma_display['Contacted'] = ma_display['Contacted'].apply(lambda x: f"{x:,.0f}")
                ma_display['Contact %'] = ma_display['Contact %'].apply(lambda x: f"{x:.1f}%")
                ma_display['FAS'] = ma_display['FAS'].apply(lambda x: f"{x:,.0f}")
                ma_display['Prev Avg'] = ma_display['Prev Avg'].apply(lambda x: f"{x:.1f}")
                ma_display['FAS Δ'] = ma_display['FAS Δ'].apply(lambda x: f"{x:+.1f}")
                ma_display['FAS % (Asgn)'] = ma_display['FAS % (Asgn)'].apply(lambda x: f"{x:.1f}%")
                ma_display['Prev %'] = ma_display['Prev %'].apply(lambda x: f"{x:.1f}%")
                ma_display['% Δ'] = ma_display['% Δ'].apply(lambda x: f"{x:+.1f}%")
                ma_display['FAS % (StS)'] = ma_display['FAS % (StS)'].apply(lambda x: f"{x:.1f}%")
                ma_display['FAS $'] = ma_display['FAS $'].apply(lambda x: f"${x:,.0f}")
                ma_display['Prev $'] = ma_display['Prev $'].apply(lambda x: f"${x:,.0f}")
                ma_display['$ Δ'] = ma_display['$ Δ'].apply(lambda x: f"${x:+,.0f}")
                ma_display['Avg FAS $'] = ma_display['Avg FAS $'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")
                
                st.dataframe(ma_display.head(25), use_container_width=True, hide_index=True, height=500)
                st.caption(f"Showing top 25 of {len(ma_display)} MAs with activity. Sorted by FAS $ descending. Prev = Previous 4-Week Avg.")
                
                # MA summary stats
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    total_fas = df_ma_combined['fas_count'].sum()
                    st.metric("Total FAS", f"{total_fas:,.0f}")
                with col_stat2:
                    total_fas_dollars = df_ma_combined['fas_dollars'].sum()
                    st.metric("Total FAS $", f"${total_fas_dollars:,.0f}")
                with col_stat3:
                    mas_with_fas = len(df_ma_combined[df_ma_combined['fas_count'] > 0])
                    st.metric("MAs with FAS", f"{mas_with_fas}")
                with col_stat4:
                    top_10_contribution = df_ma_combined.head(10)['fas_dollars'].sum() / total_fas_dollars * 100 if total_fas_dollars > 0 else 0
                    st.metric("Top 10 MA $ %", f"{top_10_contribution:.0f}%")
                
                # Key Takeaways on MA Performance
                st.divider()
                st.markdown("##### 📊 Key Takeaways: MA Performance (2/8 vs Prev 4-Week Avg)")
                
                # Calculate key insights
                gainers = df_ma_combined[df_ma_combined['fas_delta'] > 0]
                losers = df_ma_combined[df_ma_combined['fas_delta'] < 0]
                dollar_gainers = df_ma_combined[df_ma_combined['dollar_delta'] > 0]
                rate_improvers = df_ma_combined[df_ma_combined['fas_pct_delta'] > 0]
                
                # Top performers
                top_5_by_dollars = df_ma_combined.head(5)
                top_fas_delta = df_ma_combined.nlargest(3, 'fas_delta')
                top_dollar_delta = df_ma_combined.nlargest(3, 'dollar_delta')
                biggest_decline = df_ma_combined.nsmallest(3, 'fas_delta')
                
                col_t1, col_t2 = st.columns(2)
                
                with col_t1:
                    st.markdown("**🏆 Top 5 MAs by FAS $ (Week of 2/8)**")
                    for i, row in top_5_by_dollars.iterrows():
                        delta_icon = "📈" if row['dollar_delta'] > 0 else "📉" if row['dollar_delta'] < 0 else "➡️"
                        st.markdown(f"- **{row['mortgage_advisor']}**: ${row['fas_dollars']:,.0f} ({row['fas_count']:.0f} FAS) {delta_icon} ${row['dollar_delta']:+,.0f} vs prev")
                    
                    st.markdown("")
                    st.markdown("**📈 Biggest FAS Count Gainers vs Prev 4-Wk**")
                    for i, row in top_fas_delta.iterrows():
                        if row['fas_delta'] > 0:
                            st.markdown(f"- **{row['mortgage_advisor']}**: +{row['fas_delta']:.1f} FAS ({row['fas_count']:.0f} this week vs {row['fas_count_prev']:.1f} avg)")
                
                with col_t2:
                    st.markdown("**💰 Biggest $ Gainers vs Prev 4-Wk**")
                    for i, row in top_dollar_delta.iterrows():
                        if row['dollar_delta'] > 0:
                            st.markdown(f"- **{row['mortgage_advisor']}**: +${row['dollar_delta']:,.0f}")
                    
                    st.markdown("")
                    st.markdown("**⚠️ Biggest FAS Decliners vs Prev 4-Wk**")
                    for i, row in biggest_decline.iterrows():
                        if row['fas_delta'] < 0:
                            st.markdown(f"- **{row['mortgage_advisor']}**: {row['fas_delta']:.1f} FAS ({row['fas_count']:.0f} this week vs {row['fas_count_prev']:.1f} avg)")
                
                # Summary box
                st.markdown("")
                summary_cols = st.columns(4)
                with summary_cols[0]:
                    st.metric("MAs Improved (Count)", f"{len(gainers)}", help="MAs with more FAS this week than prev 4-wk avg")
                with summary_cols[1]:
                    st.metric("MAs Declined (Count)", f"{len(losers)}", help="MAs with fewer FAS this week than prev 4-wk avg")
                with summary_cols[2]:
                    st.metric("MAs Improved ($)", f"{len(dollar_gainers)}", help="MAs with higher FAS $ this week")
                with summary_cols[3]:
                    avg_rate_change = df_ma_combined[df_ma_combined['assigned'] >= 5]['fas_pct_delta'].mean()
                    st.metric("Avg FAS % Δ", f"{avg_rate_change:+.1f}%", help="Avg change in FAS % (of assigned) for MAs with 5+ assigned")
                
                # === SMS & CALL ATTEMPTS CORRELATION ANALYSIS ===
                st.divider()
                st.markdown("## 📱📞 SMS & Call Attempts Correlation Analysis (Vintage-Based)")
                st.caption("Analyzing the relationship between engagement activities (SMS, Calls) and FAS performance by lead vintage (`lead_created_date`)")
                
                # Methodology
                with st.expander("📋 **Methodology**", expanded=False):
                    st.markdown("""
                    **⚠️ Note: This analysis is VINTAGE-BASED (by `lead_created_date`)**
                    
                    **Date Range:** Last 6 months (rolling)
                    
                    **Approach:**
                    1. Aggregate leads by **vintage week** (`lead_created_date`)
                    2. Calculate FAS rate = FAS from that vintage / Sent to Sales from that vintage
                    3. Calculate SMS per lead and Calls per lead for each vintage week
                    4. Compute correlations between engagement metrics and FAS outcomes
                    5. Control for volume to isolate the engagement effect
                    6. Break down correlations by LVC Group to identify segment-specific patterns
                    
                    **Key Metrics:**
                    - **Sent to Sales**: Leads from that vintage week that were sent to sales
                    - **FAS Count**: FAS from leads created in that vintage week (regardless of when FAS occurred)
                    - **FAS Rate**: Vintage FAS count / Vintage Sent to Sales leads
                    - **SMS per Lead**: Total SMS outbound / Sent to Sales leads
                    - **SMS Pre-Contact per Lead**: SMS before contact / Sent to Sales leads
                    - **Calls per Lead**: Total call attempts / Sent to Sales leads
                    
                    **Why Vintage-Based?** This approach answers: *"For leads created in week X, did more SMS/Calls correlate with higher conversion?"* - tracking lead lifecycle from creation to conversion.
                    
                    **Partial Correlation**: Removes the effect of volume to see if engagement truly predicts FAS, or if high-volume weeks just happen to have more SMS/calls.
                    """)
                
                # Query data for SMS/Calls correlation
                sms_corr_query = """
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
                    SUM(CASE WHEN sent_to_sales_date IS NOT NULL THEN 1 ELSE 0 END) as sent_to_sales,
                    SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN 1 ELSE 0 END) as fas_count,
                    SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount ELSE 0 END) as fas_dollars,
                    AVG(total_sms_outbound_count) as avg_sms_outbound,
                    AVG(total_sms_outbound_before_contact) as avg_sms_before_contact,
                    AVG(call_attempts) as avg_call_attempts,
                    SUM(total_sms_outbound_count) as total_sms,
                    SUM(total_sms_outbound_before_contact) as total_sms_before_contact,
                    SUM(call_attempts) as total_calls
                FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
                WHERE lead_created_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
                AND lead_created_date < CURRENT_DATE()
                GROUP BY 1, 2
                ORDER BY 1, 2
                """
                
                df_sms_corr = client.query(sms_corr_query).to_dataframe()
                df_sms_corr['vintage_week'] = pd.to_datetime(df_sms_corr['vintage_week'])
                
                # Calculate rates
                df_sms_corr['fas_rate'] = df_sms_corr['fas_count'] / df_sms_corr['sent_to_sales'] * 100
                df_sms_corr['sms_per_lead'] = df_sms_corr['total_sms'] / df_sms_corr['sent_to_sales']
                df_sms_corr['sms_before_contact_per_lead'] = df_sms_corr['total_sms_before_contact'] / df_sms_corr['sent_to_sales']
                df_sms_corr['calls_per_lead'] = df_sms_corr['total_calls'] / df_sms_corr['sent_to_sales']
                df_sms_corr = df_sms_corr.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Aggregate weekly (across all LVC groups)
                df_weekly_agg = df_sms_corr.groupby('vintage_week').agg({
                    'total_leads': 'sum',
                    'sent_to_sales': 'sum',
                    'fas_count': 'sum',
                    'fas_dollars': 'sum',
                    'total_sms': 'sum',
                    'total_sms_before_contact': 'sum',
                    'total_calls': 'sum',
                    'avg_sms_outbound': 'mean',
                    'avg_sms_before_contact': 'mean',
                    'avg_call_attempts': 'mean'
                }).reset_index()
                
                df_weekly_agg['fas_rate'] = df_weekly_agg['fas_count'] / df_weekly_agg['sent_to_sales'] * 100
                df_weekly_agg['sms_per_lead'] = df_weekly_agg['total_sms'] / df_weekly_agg['sent_to_sales']
                df_weekly_agg['sms_before_contact_per_lead'] = df_weekly_agg['total_sms_before_contact'] / df_weekly_agg['sent_to_sales']
                df_weekly_agg['calls_per_lead'] = df_weekly_agg['total_calls'] / df_weekly_agg['sent_to_sales']
                df_weekly_agg = df_weekly_agg.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # --- Weekly Summary Table ---
                st.markdown("### 📊 Weekly Summary: SMS, Calls & FAS by Vintage (`lead_created_date`)")
                
                weekly_display = df_weekly_agg.copy()
                weekly_display['Week'] = weekly_display['vintage_week'].dt.strftime('%m/%d/%y')
                weekly_display = weekly_display[['Week', 'sent_to_sales', 'fas_count', 'fas_rate', 'avg_sms_outbound', 'avg_sms_before_contact', 'avg_call_attempts', 'sms_per_lead', 'sms_before_contact_per_lead', 'calls_per_lead']]
                weekly_display.columns = ['Week', 'StS Vol', 'FAS', 'FAS %', 'Avg SMS', 'Avg SMS Pre-Contact', 'Avg Calls', 'SMS/Lead', 'SMS Pre-Cntct/Lead', 'Calls/Lead']
                weekly_display['FAS %'] = weekly_display['FAS %'].apply(lambda x: f"{x:.1f}%")
                weekly_display['Avg SMS'] = weekly_display['Avg SMS'].apply(lambda x: f"{x:.2f}")
                weekly_display['Avg SMS Pre-Contact'] = weekly_display['Avg SMS Pre-Contact'].apply(lambda x: f"{x:.2f}")
                weekly_display['Avg Calls'] = weekly_display['Avg Calls'].apply(lambda x: f"{x:.1f}")
                weekly_display['SMS/Lead'] = weekly_display['SMS/Lead'].apply(lambda x: f"{x:.2f}")
                weekly_display['SMS Pre-Cntct/Lead'] = weekly_display['SMS Pre-Cntct/Lead'].apply(lambda x: f"{x:.2f}")
                weekly_display['Calls/Lead'] = weekly_display['Calls/Lead'].apply(lambda x: f"{x:.2f}")
                weekly_display['StS Vol'] = weekly_display['StS Vol'].apply(lambda x: f"{x:,.0f}")
                weekly_display['FAS'] = weekly_display['FAS'].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(weekly_display, hide_index=True, use_container_width=True, height=400)
                
                # --- Correlation Summary ---
                st.markdown("### 📈 Correlation Analysis")
                
                col_corr1, col_corr2 = st.columns(2)
                
                with col_corr1:
                    st.markdown("#### 🎯 FAS Rate Correlations")
                    
                    # Calculate correlations
                    corr_cols = ['sent_to_sales', 'fas_rate', 'avg_sms_outbound', 'avg_sms_before_contact', 'avg_call_attempts', 'sms_per_lead', 'sms_before_contact_per_lead', 'calls_per_lead']
                    corr_data = df_weekly_agg[corr_cols].copy()
                    corr_matrix = corr_data.corr()
                    
                    # Display key correlations
                    corr_results = pd.DataFrame({
                        'Metric': ['StS Volume', 'Avg SMS Outbound', 'Avg SMS Pre-Contact', 'Avg Call Attempts', 'SMS per Lead', 'SMS Pre-Contact/Lead', 'Calls per Lead'],
                        'Correlation (r)': [
                            corr_matrix.loc['fas_rate', 'sent_to_sales'],
                            corr_matrix.loc['fas_rate', 'avg_sms_outbound'],
                            corr_matrix.loc['fas_rate', 'avg_sms_before_contact'],
                            corr_matrix.loc['fas_rate', 'avg_call_attempts'],
                            corr_matrix.loc['fas_rate', 'sms_per_lead'],
                            corr_matrix.loc['fas_rate', 'sms_before_contact_per_lead'],
                            corr_matrix.loc['fas_rate', 'calls_per_lead']
                        ]
                    })
                    corr_results['Strength'] = corr_results['Correlation (r)'].apply(
                        lambda x: '🟢 Strong' if abs(x) > 0.6 else ('🟡 Moderate' if abs(x) > 0.3 else '🔴 Weak')
                    )
                    corr_results['Correlation (r)'] = corr_results['Correlation (r)'].apply(lambda x: f"{x:.3f}")
                    
                    st.dataframe(corr_results, hide_index=True, use_container_width=True)
                    
                    # Key insight
                    sms_corr = corr_matrix.loc['fas_rate', 'sms_per_lead']
                    sms_pre_contact_corr = corr_matrix.loc['fas_rate', 'sms_before_contact_per_lead']
                    calls_corr = corr_matrix.loc['fas_rate', 'calls_per_lead']
                    
                    if sms_pre_contact_corr > 0.6:
                        st.success(f"✅ **SMS Pre-Contact/Lead** has strong positive correlation ({sms_pre_contact_corr:.2f}) with FAS Rate")
                    elif sms_pre_contact_corr > 0.3:
                        st.info(f"📊 **SMS Pre-Contact/Lead** has moderate positive correlation ({sms_pre_contact_corr:.2f}) with FAS Rate")
                    
                    if sms_corr > 0.6:
                        st.success(f"✅ **SMS per Lead** has strong positive correlation ({sms_corr:.2f}) with FAS Rate")
                    elif sms_corr > 0.3:
                        st.info(f"📊 **SMS per Lead** has moderate positive correlation ({sms_corr:.2f}) with FAS Rate")
                    
                    if calls_corr > 0.6:
                        st.success(f"✅ **Calls per Lead** has strong positive correlation ({calls_corr:.2f}) with FAS Rate")
                    elif calls_corr > 0.3:
                        st.info(f"📊 **Calls per Lead** has moderate positive correlation ({calls_corr:.2f}) with FAS Rate")
                
                with col_corr2:
                    st.markdown("#### 🔬 Volume-Controlled Analysis")
                    st.caption("Partial correlations after controlling for StS volume")
                    
                    # Explanation expander
                    with st.expander("❓ **What is Partial Correlation?**", expanded=False):
                        st.markdown("""
                        **The Problem We're Solving:**
                        
                        When we see a correlation between SMS and FAS Rate, we need to ask: *"Is this real, or is it just because high-volume weeks happen to have both more SMS and higher FAS rates?"*
                        
                        **Example:**
                        - Week A: 7,000 leads → More MAs working → More SMS sent → Higher FAS count (but maybe same rate)
                        - Week B: 4,000 leads → Fewer MAs → Less SMS → Lower FAS count
                        
                        The raw correlation might show "SMS predicts FAS" but it could just be that **volume drives both**.
                        
                        ---
                        
                        **How Partial Correlation Works:**
                        
                        1. **Remove volume's effect on FAS Rate**: Find what FAS Rate would be if all weeks had the same volume
                        2. **Remove volume's effect on SMS**: Find what SMS/Lead would be if all weeks had the same volume  
                        3. **Correlate the residuals**: Now see if the "leftover" variation in SMS still predicts the "leftover" variation in FAS Rate
                        
                        ---
                        
                        **How to Interpret:**
                        
                        | Raw r | Partial r | Meaning |
                        |-------|-----------|---------|
                        | High | High | ✅ **Real effect** - SMS truly drives FAS, independent of volume |
                        | High | Low | ⚠️ **Spurious** - Correlation was due to volume, not SMS itself |
                        | Low | High | 🔍 **Hidden effect** - Volume was masking the SMS-FAS relationship |
                        | Low | Low | ❌ **No effect** - SMS doesn't predict FAS |
                        
                        ---
                        
                        **p-value**: If p < 0.05, the partial correlation is statistically significant (unlikely to be random chance).
                        """)
                    
                    # Calculate partial correlations
                    from scipy import stats
                    
                    def partial_corr(x, y, z):
                        """Partial correlation of x and y controlling for z"""
                        try:
                            slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
                            resid_x = x - (slope_xz * z + intercept_xz)
                            slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
                            resid_y = y - (slope_yz * z + intercept_yz)
                            corr, pval = stats.pearsonr(resid_x, resid_y)
                            return corr, pval
                        except:
                            return 0, 1
                    
                    vol = df_weekly_agg['sent_to_sales'].values
                    fas_rate = df_weekly_agg['fas_rate'].values
                    sms_per_lead = df_weekly_agg['sms_per_lead'].values
                    sms_pre_contact_per_lead = df_weekly_agg['sms_before_contact_per_lead'].values
                    calls_per_lead = df_weekly_agg['calls_per_lead'].values
                    
                    partial_sms, p_sms = partial_corr(fas_rate, sms_per_lead, vol)
                    partial_sms_pre, p_sms_pre = partial_corr(fas_rate, sms_pre_contact_per_lead, vol)
                    partial_calls, p_calls = partial_corr(fas_rate, calls_per_lead, vol)
                    
                    partial_results = pd.DataFrame({
                        'Metric': ['SMS per Lead', 'SMS Pre-Contact/Lead', 'Calls per Lead'],
                        'Raw r': [sms_corr, sms_pre_contact_corr, calls_corr],
                        'Partial r': [partial_sms, partial_sms_pre, partial_calls],
                        'p-value': [p_sms, p_sms_pre, p_calls],
                        'Significant': ['✅ Yes' if p_sms < 0.05 else '❌ No', '✅ Yes' if p_sms_pre < 0.05 else '❌ No', '✅ Yes' if p_calls < 0.05 else '❌ No']
                    })
                    partial_results['Raw r'] = partial_results['Raw r'].apply(lambda x: f"{x:.3f}")
                    partial_results['Partial r'] = partial_results['Partial r'].apply(lambda x: f"{x:.3f}")
                    partial_results['p-value'] = partial_results['p-value'].apply(lambda x: f"{x:.4f}")
                    
                    st.dataframe(partial_results, hide_index=True, use_container_width=True)
                    
                    st.markdown("**Interpretation:**")
                    
                    # Determine if effect is real or spurious
                    sms_pre_raw = sms_pre_contact_corr
                    sms_raw = sms_corr
                    calls_raw = calls_corr
                    
                    if partial_sms_pre > 0.5 and p_sms_pre < 0.05:
                        st.success(f"📱 **SMS Pre-Contact**: Real effect (r stays high: {sms_pre_raw:.2f} → {partial_sms_pre:.2f})")
                    elif partial_sms_pre > 0.3 and p_sms_pre < 0.05:
                        st.info(f"📱 **SMS Pre-Contact**: Moderate real effect (r: {sms_pre_raw:.2f} → {partial_sms_pre:.2f})")
                    elif abs(partial_sms_pre) < abs(sms_pre_raw) * 0.5:
                        st.warning(f"📱 **SMS Pre-Contact**: May be volume-driven (r dropped: {sms_pre_raw:.2f} → {partial_sms_pre:.2f})")
                    
                    if partial_sms > 0.5 and p_sms < 0.05:
                        st.success(f"📱 **Total SMS**: Real effect (r stays high: {sms_raw:.2f} → {partial_sms:.2f})")
                    elif partial_sms > 0.3 and p_sms < 0.05:
                        st.info(f"📱 **Total SMS**: Moderate real effect (r: {sms_raw:.2f} → {partial_sms:.2f})")
                    elif abs(partial_sms) < abs(sms_raw) * 0.5:
                        st.warning(f"📱 **Total SMS**: May be volume-driven (r dropped: {sms_raw:.2f} → {partial_sms:.2f})")
                    
                    if partial_calls > 0.3 and p_calls < 0.05:
                        st.info(f"📞 **Calls**: Real effect (r: {calls_raw:.2f} → {partial_calls:.2f})")
                    elif abs(partial_calls) < abs(calls_raw) * 0.5:
                        st.warning(f"📞 **Calls**: May be volume-driven (r dropped: {calls_raw:.2f} → {partial_calls:.2f})")
                
                # --- Trend Charts ---
                st.markdown("### 📉 Trends Over Time (Oct 2025 - Present)")
                
                col_chart1, col_chart2, col_chart3 = st.columns(3)
                
                # Create base chart data - use temporal encoding for full date range
                df_chart = df_weekly_agg.copy()
                df_chart = df_chart.sort_values('vintage_week')  # Ensure sorted by date
                
                base = alt.Chart(df_chart).encode(
                    x=alt.X('vintage_week:T', title='Vintage Week', axis=alt.Axis(format='%m/%d', labelAngle=-45))
                )
                
                line_fas = base.mark_line(color='#4cc9f0', strokeWidth=2).encode(
                    y=alt.Y('fas_rate:Q', title='FAS Rate %', axis=alt.Axis(titleColor='#4cc9f0'))
                )
                
                with col_chart1:
                    # SMS vs FAS Rate trend
                    line_sms = base.mark_line(color='#f72585', strokeWidth=2, strokeDash=[5,3]).encode(
                        y=alt.Y('sms_per_lead:Q', title='SMS/Lead', axis=alt.Axis(titleColor='#f72585'))
                    )
                    
                    chart_sms = alt.layer(line_fas, line_sms).resolve_scale(y='independent').properties(
                        title='FAS % vs Total SMS', height=250
                    )
                    st.altair_chart(chart_sms, use_container_width=True)
                    st.caption("Total SMS outbound / StS leads")
                
                with col_chart2:
                    # SMS Pre-Contact vs FAS Rate trend
                    line_sms_pre = base.mark_line(color='#ff6b6b', strokeWidth=2, strokeDash=[5,3]).encode(
                        y=alt.Y('sms_before_contact_per_lead:Q', title='SMS Pre-Cntct/Ld', axis=alt.Axis(titleColor='#ff6b6b'))
                    )
                    
                    chart_sms_pre = alt.layer(line_fas, line_sms_pre).resolve_scale(y='independent').properties(
                        title='FAS % vs SMS Pre-Contact', height=250
                    )
                    st.altair_chart(chart_sms_pre, use_container_width=True)
                    st.caption("SMS before contact / StS leads")
                
                with col_chart3:
                    # Calls vs FAS Rate trend
                    line_calls = base.mark_line(color='#7209b7', strokeWidth=2, strokeDash=[5,3]).encode(
                        y=alt.Y('calls_per_lead:Q', title='Calls/Lead', axis=alt.Axis(titleColor='#7209b7'))
                    )
                    
                    chart_calls = alt.layer(line_fas, line_calls).resolve_scale(y='independent').properties(
                        title='FAS % vs Calls', height=250
                    )
                    st.altair_chart(chart_calls, use_container_width=True)
                    st.caption("Call attempts / StS leads")
                
                # --- High vs Low Engagement Comparison ---
                st.markdown("### 📊 High vs Low Engagement Comparison")
                
                median_sms = df_weekly_agg['sms_per_lead'].median()
                median_sms_pre = df_weekly_agg['sms_before_contact_per_lead'].median()
                median_calls = df_weekly_agg['calls_per_lead'].median()
                
                high_sms = df_weekly_agg[df_weekly_agg['sms_per_lead'] > median_sms]
                low_sms = df_weekly_agg[df_weekly_agg['sms_per_lead'] <= median_sms]
                high_sms_pre = df_weekly_agg[df_weekly_agg['sms_before_contact_per_lead'] > median_sms_pre]
                low_sms_pre = df_weekly_agg[df_weekly_agg['sms_before_contact_per_lead'] <= median_sms_pre]
                high_calls = df_weekly_agg[df_weekly_agg['calls_per_lead'] > median_calls]
                low_calls = df_weekly_agg[df_weekly_agg['calls_per_lead'] <= median_calls]
                
                col_eng1, col_eng2, col_eng3 = st.columns(3)
                
                with col_eng1:
                    st.markdown(f"**Total SMS (median: {median_sms:.2f}/lead)**")
                    eng_sms = pd.DataFrame({
                        'Group': [f'High (>{median_sms:.2f})', f'Low (≤{median_sms:.2f})'],
                        'Weeks': [len(high_sms), len(low_sms)],
                        'Avg FAS %': [f"{high_sms['fas_rate'].mean():.2f}%", f"{low_sms['fas_rate'].mean():.2f}%"],
                        'Avg FAS $': [f"${high_sms['fas_dollars'].mean():,.0f}", f"${low_sms['fas_dollars'].mean():,.0f}"]
                    })
                    st.dataframe(eng_sms, hide_index=True, use_container_width=True)
                    
                    delta_sms = high_sms['fas_rate'].mean() - low_sms['fas_rate'].mean()
                    if delta_sms > 0:
                        st.success(f"**+{delta_sms:.2f}pp** FAS rate")
                    else:
                        st.warning(f"**{delta_sms:.2f}pp** FAS rate")
                
                with col_eng2:
                    st.markdown(f"**SMS Pre-Contact (median: {median_sms_pre:.2f}/lead)**")
                    eng_sms_pre = pd.DataFrame({
                        'Group': [f'High (>{median_sms_pre:.2f})', f'Low (≤{median_sms_pre:.2f})'],
                        'Weeks': [len(high_sms_pre), len(low_sms_pre)],
                        'Avg FAS %': [f"{high_sms_pre['fas_rate'].mean():.2f}%", f"{low_sms_pre['fas_rate'].mean():.2f}%"],
                        'Avg FAS $': [f"${high_sms_pre['fas_dollars'].mean():,.0f}", f"${low_sms_pre['fas_dollars'].mean():,.0f}"]
                    })
                    st.dataframe(eng_sms_pre, hide_index=True, use_container_width=True)
                    
                    delta_sms_pre = high_sms_pre['fas_rate'].mean() - low_sms_pre['fas_rate'].mean()
                    if delta_sms_pre > 0:
                        st.success(f"**+{delta_sms_pre:.2f}pp** FAS rate")
                    else:
                        st.warning(f"**{delta_sms_pre:.2f}pp** FAS rate")
                
                with col_eng3:
                    st.markdown(f"**Calls (median: {median_calls:.2f}/lead)**")
                    eng_calls = pd.DataFrame({
                        'Group': [f'High (>{median_calls:.2f})', f'Low (≤{median_calls:.2f})'],
                        'Weeks': [len(high_calls), len(low_calls)],
                        'Avg FAS %': [f"{high_calls['fas_rate'].mean():.2f}%", f"{low_calls['fas_rate'].mean():.2f}%"],
                        'Avg FAS $': [f"${high_calls['fas_dollars'].mean():,.0f}", f"${low_calls['fas_dollars'].mean():,.0f}"]
                    })
                    st.dataframe(eng_calls, hide_index=True, use_container_width=True)
                    
                    delta_calls = high_calls['fas_rate'].mean() - low_calls['fas_rate'].mean()
                    if delta_calls > 0:
                        st.success(f"**+{delta_calls:.2f}pp** FAS rate")
                    else:
                        st.warning(f"**{delta_calls:.2f}pp** FAS rate")
                
                # === LVC GROUP SPECIFIC CORRELATIONS ===
                st.divider()
                st.markdown("### 🎯 Correlation by LVC Group")
                st.caption("Analyzing if SMS/Call impact varies by lead quality segment")
                
                # Aggregate by LVC group for correlation
                lvc_groups = ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer']
                lvc_corr_results = []
                
                for lvc in lvc_groups:
                    df_lvc = df_sms_corr[df_sms_corr['lvc_group'] == lvc].copy()
                    
                    # Aggregate by week for this LVC group
                    df_lvc_weekly = df_lvc.groupby('vintage_week').agg({
                        'sent_to_sales': 'sum',
                        'fas_count': 'sum',
                        'fas_dollars': 'sum',
                        'total_sms': 'sum',
                        'total_sms_before_contact': 'sum',
                        'total_calls': 'sum'
                    }).reset_index()
                    
                    df_lvc_weekly['fas_rate'] = df_lvc_weekly['fas_count'] / df_lvc_weekly['sent_to_sales'] * 100
                    df_lvc_weekly['sms_per_lead'] = df_lvc_weekly['total_sms'] / df_lvc_weekly['sent_to_sales']
                    df_lvc_weekly['sms_pre_contact_per_lead'] = df_lvc_weekly['total_sms_before_contact'] / df_lvc_weekly['sent_to_sales']
                    df_lvc_weekly['calls_per_lead'] = df_lvc_weekly['total_calls'] / df_lvc_weekly['sent_to_sales']
                    df_lvc_weekly = df_lvc_weekly.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(df_lvc_weekly) >= 5:  # Need enough data points
                        sms_corr_lvc = df_lvc_weekly['fas_rate'].corr(df_lvc_weekly['sms_per_lead'])
                        sms_pre_corr_lvc = df_lvc_weekly['fas_rate'].corr(df_lvc_weekly['sms_pre_contact_per_lead'])
                        calls_corr_lvc = df_lvc_weekly['fas_rate'].corr(df_lvc_weekly['calls_per_lead'])
                        avg_fas_rate = df_lvc_weekly['fas_rate'].mean()
                        avg_sms = df_lvc_weekly['sms_per_lead'].mean()
                        avg_sms_pre = df_lvc_weekly['sms_pre_contact_per_lead'].mean()
                        avg_calls = df_lvc_weekly['calls_per_lead'].mean()
                        
                        lvc_corr_results.append({
                            'LVC Group': lvc,
                            'Weeks': len(df_lvc_weekly),
                            'Avg FAS Rate': avg_fas_rate,
                            'Avg SMS/Lead': avg_sms,
                            'Avg SMS Pre/Lead': avg_sms_pre,
                            'Avg Calls/Lead': avg_calls,
                            'SMS↔FAS': sms_corr_lvc,
                            'SMS Pre↔FAS': sms_pre_corr_lvc,
                            'Calls↔FAS': calls_corr_lvc
                        })
                
                if lvc_corr_results:
                    df_lvc_corr = pd.DataFrame(lvc_corr_results)
                    
                    st.markdown("**Correlation by LVC Group**")
                    
                    df_lvc_display = df_lvc_corr.copy()
                    df_lvc_display['Avg FAS Rate'] = df_lvc_display['Avg FAS Rate'].apply(lambda x: f"{x:.2f}%")
                    df_lvc_display['Avg SMS/Lead'] = df_lvc_display['Avg SMS/Lead'].apply(lambda x: f"{x:.2f}")
                    df_lvc_display['Avg SMS Pre/Lead'] = df_lvc_display['Avg SMS Pre/Lead'].apply(lambda x: f"{x:.2f}")
                    df_lvc_display['Avg Calls/Lead'] = df_lvc_display['Avg Calls/Lead'].apply(lambda x: f"{x:.2f}")
                    df_lvc_display['SMS↔FAS'] = df_lvc_display['SMS↔FAS'].apply(lambda x: f"{x:.3f}")
                    df_lvc_display['SMS Pre↔FAS'] = df_lvc_display['SMS Pre↔FAS'].apply(lambda x: f"{x:.3f}")
                    df_lvc_display['Calls↔FAS'] = df_lvc_display['Calls↔FAS'].apply(lambda x: f"{x:.3f}")
                    
                    st.dataframe(df_lvc_display, hide_index=True, use_container_width=True)
                    
                    st.markdown("**📊 Insights by Segment**")
                    
                    col_lvc_ins1, col_lvc_ins2 = st.columns(2)
                    
                    with col_lvc_ins1:
                        for _, row in df_lvc_corr.iterrows():
                            sms_pre_strength = "🟢" if abs(row['SMS Pre↔FAS']) > 0.5 else ("🟡" if abs(row['SMS Pre↔FAS']) > 0.3 else "🔴")
                            sms_strength = "🟢" if abs(row['SMS↔FAS']) > 0.5 else ("🟡" if abs(row['SMS↔FAS']) > 0.3 else "🔴")
                            calls_strength = "🟢" if abs(row['Calls↔FAS']) > 0.5 else ("🟡" if abs(row['Calls↔FAS']) > 0.3 else "🔴")
                            
                            st.markdown(f"**{row['LVC Group']}**: SMS Pre {sms_pre_strength} ({row['SMS Pre↔FAS']:.2f}) | SMS {sms_strength} ({row['SMS↔FAS']:.2f}) | Calls {calls_strength} ({row['Calls↔FAS']:.2f})")
                    
                    with col_lvc_ins2:
                        # Find which LVC benefits most from SMS pre-contact
                        best_sms_pre = df_lvc_corr.loc[df_lvc_corr['SMS Pre↔FAS'].idxmax()]
                        best_sms = df_lvc_corr.loc[df_lvc_corr['SMS↔FAS'].idxmax()]
                        best_calls = df_lvc_corr.loc[df_lvc_corr['Calls↔FAS'].idxmax()]
                        
                        st.markdown("**Best Response by Metric:**")
                        st.markdown(f"- 📱 SMS Pre-Contact: **{best_sms_pre['LVC Group']}** (r={best_sms_pre['SMS Pre↔FAS']:.2f})")
                        st.markdown(f"- 📱 Total SMS: **{best_sms['LVC Group']}** (r={best_sms['SMS↔FAS']:.2f})")
                        st.markdown(f"- 📞 Calls: **{best_calls['LVC Group']}** (r={best_calls['Calls↔FAS']:.2f})")
                    
                    # Bar chart of correlations by LVC
                    df_lvc_melt = df_lvc_corr.melt(id_vars=['LVC Group'], value_vars=['SMS↔FAS', 'SMS Pre↔FAS', 'Calls↔FAS'],
                                                   var_name='Metric', value_name='Correlation')
                    
                    bars_lvc_corr = alt.Chart(df_lvc_melt).mark_bar().encode(
                        x=alt.X('LVC Group:N', sort=['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer']),
                        y=alt.Y('Correlation:Q', title='Correlation with FAS Rate'),
                        color=alt.Color('Metric:N', scale=alt.Scale(domain=['SMS↔FAS', 'SMS Pre↔FAS', 'Calls↔FAS'], range=['#f72585', '#ff6b6b', '#7209b7'])),
                        xOffset='Metric:N'
                    ).properties(title='SMS, SMS Pre-Contact & Call Correlation with FAS Rate by LVC Group', height=280)
                    
                    st.altair_chart(bars_lvc_corr, use_container_width=True)
                
                # === KEY INSIGHTS SUMMARY ===
                st.divider()
                st.markdown("### 💡 Key Insights: SMS & Call Engagement")
                
                col_ins1, col_ins2, col_ins3 = st.columns(3)
                
                with col_ins1:
                    st.markdown("**📱 Total SMS Impact**")
                    raw_sms_corr = corr_matrix.loc['fas_rate', 'sms_per_lead']
                    st.markdown(f"- Raw r: **{raw_sms_corr:.2f}**")
                    st.markdown(f"- Partial r: **{partial_sms:.2f}** {'✅' if p_sms < 0.05 else ''}")
                    st.markdown(f"- High vs Low: **+{delta_sms:.2f}pp**")
                
                with col_ins2:
                    st.markdown("**📱 SMS Pre-Contact Impact**")
                    raw_sms_pre_corr = corr_matrix.loc['fas_rate', 'sms_before_contact_per_lead']
                    st.markdown(f"- Raw r: **{raw_sms_pre_corr:.2f}**")
                    st.markdown(f"- Partial r: **{partial_sms_pre:.2f}** {'✅' if p_sms_pre < 0.05 else ''}")
                    st.markdown(f"- High vs Low: **+{delta_sms_pre:.2f}pp**")
                
                with col_ins3:
                    st.markdown("**📞 Call Attempts Impact**")
                    raw_calls_corr = corr_matrix.loc['fas_rate', 'calls_per_lead']
                    st.markdown(f"- Raw r: **{raw_calls_corr:.2f}**")
                    st.markdown(f"- Partial r: **{partial_calls:.2f}** {'✅' if p_calls < 0.05 else ''}")
                    st.markdown(f"- High vs Low: **{delta_calls:+.2f}pp**")
                
                # Determine the strongest predictor
                strongest_metric = 'SMS Pre-Contact' if raw_sms_pre_corr > max(raw_sms_corr, raw_calls_corr) else ('Total SMS' if raw_sms_corr > raw_calls_corr else 'Call Attempts')
                strongest_corr = max(raw_sms_pre_corr, raw_sms_corr, raw_calls_corr)
                
                # Final summary box
                st.markdown("")
                st.info(f"""
                **🎯 Bottom Line:** 
                - **SMS Pre-Contact/Lead** shows {'strong' if raw_sms_pre_corr > 0.6 else 'moderate' if raw_sms_pre_corr > 0.3 else 'weak'} correlation ({raw_sms_pre_corr:.2f}) with FAS Rate {'✅ Significant after volume control' if p_sms_pre < 0.05 else ''}
                - **Total SMS/Lead** shows {'strong' if raw_sms_corr > 0.6 else 'moderate' if raw_sms_corr > 0.3 else 'weak'} correlation ({raw_sms_corr:.2f}) with FAS Rate {'✅ Significant after volume control' if p_sms < 0.05 else ''}
                - **Calls/Lead** shows {'strong' if raw_calls_corr > 0.6 else 'moderate' if raw_calls_corr > 0.3 else 'weak'} correlation ({raw_calls_corr:.2f}) with FAS Rate {'✅ Significant after volume control' if p_calls < 0.05 else ''}
                
                **Recommendation:** {strongest_metric} shows the strongest relationship (r={strongest_corr:.2f}) with FAS success. {'SMS before contact appears particularly impactful - proactive outreach before reaching the customer correlates with higher conversion.' if raw_sms_pre_corr > raw_sms_corr else 'Focus on overall engagement activity to drive conversions.'}
                """)
            
            else:
                st.warning("No data available for week of 2/8/2026")
                
        except Exception as e:
            st.error(f"2/8 Analysis Error: {e}")
            st.exception(e)

    # --- TAB: Agent Performance Analysis ---
    with tab_agent:
        st.header("⭐ Agent Performance Analysis")
        st.caption("Normalizing performance metrics across star tiers to account for volume and selection bias")
        
        st.markdown("""
        ### The Problem
        Agents are grouped into **5-star, 3-star, and 1-star** tiers. Curated leads are routed with priority:
        - **5-star** gets first pick → High volume, potentially lower conversion rate
        - **3-star** gets overflow → Medium volume
        - **1-star** gets double-filtered leads → Low volume, artificially high conversion rate
        
        **This creates misleading metrics** - a 1-star with 1/3 conversions (33%) looks better than a 5-star with 15/150 (10%), 
        but the 1-star's rate is statistically unreliable and benefits from selection bias.
        """)
        
        st.divider()
        
        # --- EXAMPLE DATA (Simulated for demonstration) ---
        st.subheader("📊 Example: Raw vs Normalized Metrics")
        st.caption("Using sample data to illustrate normalization techniques")
        
        # Create example agent data
        
        example_agents = pd.DataFrame({
            'agent_id': ['Agent A', 'Agent B', 'Agent C', 'Agent D', 'Agent E', 'Agent F', 'Agent G', 'Agent H', 'Agent I'],
            'star_tier': ['5-Star', '5-Star', '5-Star', '3-Star', '3-Star', '3-Star', '1-Star', '1-Star', '1-Star'],
            'leads_received': [150, 142, 138, 48, 52, 45, 8, 5, 3],
            'conversions': [15, 18, 12, 7, 5, 6, 2, 1, 1]
        })
        
        example_agents['raw_rate'] = example_agents['conversions'] / example_agents['leads_received']
        
        # Overall benchmark
        total_conversions = example_agents['conversions'].sum()
        total_volume = example_agents['leads_received'].sum()
        overall_rate = total_conversions / total_volume
        
        st.metric("Overall Benchmark Conversion Rate", f"{overall_rate:.1%}")
        
        # --- SOLUTION 1: Bayesian Shrinkage ---
        st.subheader("🎯 Solution 1: Bayesian Shrinkage")
        st.markdown("""
        **Concept:** Pull small-sample rates toward the overall average. 
        - High volume → Trust the observed rate
        - Low volume → Assume closer to average (more uncertainty)
        
        **Formula:** `Adjusted Rate = (conversions + prior_conversions) / (volume + prior_volume)`
        """)
        
        prior_strength = st.slider("Prior Strength (pseudo-observations)", 10, 100, 50, 
                                   help="Higher = more shrinkage toward average for low-volume agents")
        
        def bayesian_adjusted_rate(conversions, volume, overall_rate, prior_strength):
            return (conversions + (overall_rate * prior_strength)) / (volume + prior_strength)
        
        example_agents['bayesian_rate'] = example_agents.apply(
            lambda x: bayesian_adjusted_rate(x['conversions'], x['leads_received'], overall_rate, prior_strength), axis=1
        )
        
        # Display comparison
        bayes_display = example_agents[['agent_id', 'star_tier', 'leads_received', 'conversions', 'raw_rate', 'bayesian_rate']].copy()
        bayes_display['Raw Rate'] = bayes_display['raw_rate'].apply(lambda x: f"{x:.1%}")
        bayes_display['Bayesian Rate'] = bayes_display['bayesian_rate'].apply(lambda x: f"{x:.1%}")
        bayes_display['Difference'] = (bayes_display['bayesian_rate'] - bayes_display['raw_rate']).apply(lambda x: f"{x:+.1%}")
        bayes_display = bayes_display[['agent_id', 'star_tier', 'leads_received', 'conversions', 'Raw Rate', 'Bayesian Rate', 'Difference']]
        bayes_display.columns = ['Agent', 'Tier', 'Leads', 'Conversions', 'Raw Rate', 'Bayesian Rate', 'Difference']
        
        st.dataframe(bayes_display, use_container_width=True, hide_index=True)
        
        # Chart comparing raw vs bayesian
        bayes_chart_data = example_agents.melt(
            id_vars=['agent_id', 'star_tier', 'leads_received'],
            value_vars=['raw_rate', 'bayesian_rate'],
            var_name='metric_type',
            value_name='rate'
        )
        bayes_chart_data['metric_type'] = bayes_chart_data['metric_type'].replace({
            'raw_rate': 'Raw Rate',
            'bayesian_rate': 'Bayesian Adjusted'
        })
        
        bayes_chart = alt.Chart(bayes_chart_data).mark_bar().encode(
            x=alt.X('agent_id:N', title='Agent', sort=example_agents['agent_id'].tolist()),
            y=alt.Y('rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            color=alt.Color('metric_type:N', title='Metric', scale=alt.Scale(
                domain=['Raw Rate', 'Bayesian Adjusted'],
                range=['#d62728', '#2ca02c']
            )),
            xOffset='metric_type:N',
            tooltip=[
                alt.Tooltip('agent_id:N', title='Agent'),
                alt.Tooltip('star_tier:N', title='Tier'),
                alt.Tooltip('leads_received:Q', title='Leads', format=',d'),
                alt.Tooltip('metric_type:N', title='Metric'),
                alt.Tooltip('rate:Q', title='Rate', format='.1%')
            ]
        ).properties(
            title='Raw vs Bayesian Adjusted Conversion Rates',
            height=350
        )
        
        st.altair_chart(bayes_chart, use_container_width=True)
        
        st.info("""
        **Key Insight:** Notice how the 1-Star agents' rates are pulled down from 33%/25%/20% closer to the average (~11%), 
        while 5-Star agents' rates barely change because their high volume gives us confidence in the observed rate.
        """)
        
        st.divider()
        
        # --- SOLUTION 2: Confidence Intervals (Wilson Score) ---
        st.subheader("📏 Solution 2: Confidence Intervals (Wilson Score)")
        st.markdown("""
        **Concept:** Show the *uncertainty* in each rate. Small samples = wide intervals = less reliable.
        
        **Wilson Score Interval** is preferred over standard confidence intervals for proportions, especially with small samples.
        """)
        
        def wilson_ci(conversions, volume, confidence=0.95):
            """Wilson score interval - better for small samples"""
            if volume == 0:
                return (0, 0, 0)
            
            z = stats.norm.ppf((1 + confidence) / 2)
            p = conversions / volume
            
            denominator = 1 + z**2 / volume
            center = (p + z**2 / (2 * volume)) / denominator
            spread = z * ((p * (1 - p) / volume + z**2 / (4 * volume**2)) ** 0.5) / denominator
            
            return (max(0, center - spread), min(1, center + spread), spread * 2)
        
        example_agents['ci_lower'], example_agents['ci_upper'], example_agents['ci_width'] = zip(*example_agents.apply(
            lambda x: wilson_ci(x['conversions'], x['leads_received']), axis=1
        ))
        
        # Display with confidence intervals
        ci_display = example_agents[['agent_id', 'star_tier', 'leads_received', 'raw_rate', 'ci_lower', 'ci_upper', 'ci_width']].copy()
        ci_display['Rate'] = ci_display['raw_rate'].apply(lambda x: f"{x:.1%}")
        ci_display['95% CI'] = ci_display.apply(lambda x: f"[{x['ci_lower']:.1%}, {x['ci_upper']:.1%}]", axis=1)
        ci_display['CI Width'] = ci_display['ci_width'].apply(lambda x: f"±{x/2:.1%}")
        ci_display = ci_display[['agent_id', 'star_tier', 'leads_received', 'Rate', '95% CI', 'CI Width']]
        ci_display.columns = ['Agent', 'Tier', 'Leads', 'Rate', '95% Confidence Interval', 'Uncertainty']
        
        st.dataframe(ci_display, use_container_width=True, hide_index=True)
        
        # Error bar chart
        ci_chart = alt.Chart(example_agents).mark_point(filled=True, size=100).encode(
            x=alt.X('agent_id:N', title='Agent', sort=example_agents['agent_id'].tolist()),
            y=alt.Y('raw_rate:Q', title='Conversion Rate', axis=alt.Axis(format='.0%')),
            color=alt.Color('star_tier:N', title='Tier', scale=alt.Scale(
                domain=['5-Star', '3-Star', '1-Star'],
                range=['#2ca02c', '#ff7f0e', '#d62728']
            )),
            tooltip=[
                alt.Tooltip('agent_id:N', title='Agent'),
                alt.Tooltip('star_tier:N', title='Tier'),
                alt.Tooltip('leads_received:Q', title='Leads', format=',d'),
                alt.Tooltip('raw_rate:Q', title='Rate', format='.1%'),
                alt.Tooltip('ci_lower:Q', title='CI Lower', format='.1%'),
                alt.Tooltip('ci_upper:Q', title='CI Upper', format='.1%')
            ]
        )
        
        ci_error_bars = alt.Chart(example_agents).mark_errorbar().encode(
            x=alt.X('agent_id:N', sort=example_agents['agent_id'].tolist()),
            y=alt.Y('ci_lower:Q', title=''),
            y2='ci_upper:Q',
            color=alt.Color('star_tier:N', scale=alt.Scale(
                domain=['5-Star', '3-Star', '1-Star'],
                range=['#2ca02c', '#ff7f0e', '#d62728']
            ))
        )
        
        st.altair_chart((ci_error_bars + ci_chart).properties(
            title='Conversion Rates with 95% Confidence Intervals',
            height=400
        ), use_container_width=True)
        
        st.warning("""
        **Key Insight:** Agent I shows 33% conversion, but the true rate is likely **anywhere between 1.8% and 79%**! 
        Meanwhile, Agent B's 12.7% rate is reliably between 8% and 19%. The error bars reveal the truth.
        """)
        
        st.divider()
        
        # --- SOLUTION 3: Volume-Adjusted Score ---
        st.subheader("📈 Solution 3: Volume-Adjusted Composite Score")
        st.markdown("""
        **Concept:** Create a score that rewards BOTH conversion rate AND volume. 
        Low volume = penalized score, even if rate looks high.
        """)
        
        min_volume_threshold = st.slider("Minimum Volume for Full Credit", 20, 100, 50,
                                         help="Agents below this volume get penalized in the score")
        
        def volume_adjusted_score(conversions, volume, min_volume):
            """Score that rewards both conversion AND volume"""
            raw_rate = conversions / volume if volume > 0 else 0
            volume_factor = min(1.0, volume / min_volume)  # 0-1 based on volume
            # Geometric mean of rate and volume factor
            return (raw_rate * volume_factor) ** 0.5
        
        example_agents['volume_factor'] = example_agents['leads_received'].apply(lambda x: min(1.0, x / min_volume_threshold))
        example_agents['volume_adjusted_score'] = example_agents.apply(
            lambda x: volume_adjusted_score(x['conversions'], x['leads_received'], min_volume_threshold), axis=1
        )
        
        # Rank by different methods
        example_agents['rank_raw'] = example_agents['raw_rate'].rank(ascending=False).astype(int)
        example_agents['rank_bayesian'] = example_agents['bayesian_rate'].rank(ascending=False).astype(int)
        example_agents['rank_volume_adj'] = example_agents['volume_adjusted_score'].rank(ascending=False).astype(int)
        
        rank_display = example_agents[['agent_id', 'star_tier', 'leads_received', 'raw_rate', 'volume_factor', 'volume_adjusted_score', 'rank_raw', 'rank_volume_adj']].copy()
        rank_display['Raw Rate'] = rank_display['raw_rate'].apply(lambda x: f"{x:.1%}")
        rank_display['Volume Factor'] = rank_display['volume_factor'].apply(lambda x: f"{x:.0%}")
        rank_display['Adjusted Score'] = rank_display['volume_adjusted_score'].apply(lambda x: f"{x:.3f}")
        rank_display['Rank (Raw)'] = rank_display['rank_raw']
        rank_display['Rank (Adjusted)'] = rank_display['rank_volume_adj']
        rank_display = rank_display[['agent_id', 'star_tier', 'leads_received', 'Raw Rate', 'Volume Factor', 'Adjusted Score', 'Rank (Raw)', 'Rank (Adjusted)']]
        rank_display.columns = ['Agent', 'Tier', 'Leads', 'Raw Rate', 'Volume Factor', 'Adjusted Score', 'Rank (Raw)', 'Rank (Adj)']
        
        st.dataframe(rank_display.sort_values('Rank (Adj)'), use_container_width=True, hide_index=True)
        
        st.success("""
        **Key Insight:** With volume adjustment, the rankings change significantly! 
        1-Star agents drop from top ranks because their low volume means we can't trust their high rates.
        """)
        
        st.divider()
        
        # --- GROWTH FRAMEWORK ---
        st.subheader("🚀 Growing Lower-Tier Agents Fairly")
        st.markdown("""
        ### The Challenge
        Lower-tier agents can't prove themselves because they don't get enough quality leads.
        
        ### Recommended Solution: **Controlled Growth Pool**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Current Flow (Biased)
            ```
            Incoming Lead
                 ↓
            5-Star Available? → Yes → Assign to 5-Star
                 ↓ No
            3-Star Available? → Yes → Assign to 3-Star
                 ↓ No
            1-Star → Gets leftovers
            ```
            
            **Problem:** 1-Star only gets pre-filtered leads
            """)
        
        with col2:
            st.markdown("""
            #### Proposed Flow (Fair)
            ```
            Incoming Lead
                 ↓
            Growth Pool (15%)? → Yes → Random Assignment
                 ↓ No              (all tiers equal chance)
            Normal Priority Flow (85%)
            5-Star → 3-Star → 1-Star
            ```
            
            **Benefit:** Fair evaluation on equal-quality leads
            """)
        
        st.divider()
        
        # --- PROMOTION CRITERIA ---
        st.subheader("📋 Promotion Readiness Framework")
        
        st.markdown("""
        **Criteria for promotion evaluation:**
        1. **Minimum Volume Threshold** - Must have 30+ leads before evaluation
        2. **Evaluate on Growth Pool leads only** - Controls for lead quality bias
        3. **Use Bayesian-adjusted metrics** - Accounts for sample size
        4. **Composite Score** - Weighs multiple metrics
        """)
        
        # Example promotion scorecard
        promotion_data = pd.DataFrame({
            'Agent': ['Agent G', 'Agent H', 'Agent I', 'Agent D', 'Agent E'],
            'Current Tier': ['1-Star', '1-Star', '1-Star', '3-Star', '3-Star'],
            'Growth Pool Leads': [28, 31, 15, 45, 52],
            'GP Conversions': [3, 4, 2, 6, 7],
            'Contact Rate': [0.82, 0.78, 0.75, 0.85, 0.88],
            'Pull Through': [0.65, 0.70, 0.55, 0.72, 0.68]
        })
        
        promotion_data['GP Conv Rate'] = promotion_data['GP Conversions'] / promotion_data['Growth Pool Leads']
        promotion_data['Meets Volume'] = promotion_data['Growth Pool Leads'] >= 30
        
        # Bayesian adjustment for GP rate
        gp_overall = promotion_data['GP Conversions'].sum() / promotion_data['Growth Pool Leads'].sum()
        promotion_data['Adj Conv Rate'] = promotion_data.apply(
            lambda x: bayesian_adjusted_rate(x['GP Conversions'], x['Growth Pool Leads'], gp_overall, 30), axis=1
        )
        
        # Composite score (only if meets volume)
        def promotion_score(row, benchmarks):
            if not row['Meets Volume']:
                return None
            
            contact_score = row['Contact Rate'] / benchmarks['contact']
            conv_score = row['Adj Conv Rate'] / benchmarks['conversion']
            pull_score = row['Pull Through'] / benchmarks['pull_through']
            
            return (contact_score * 0.25 + conv_score * 0.50 + pull_score * 0.25)
        
        benchmarks = {
            'contact': 0.80,
            'conversion': 0.12,
            'pull_through': 0.65
        }
        
        promotion_data['Promotion Score'] = promotion_data.apply(lambda x: promotion_score(x, benchmarks), axis=1)
        promotion_data['Ready?'] = promotion_data.apply(
            lambda x: '✅ Ready' if x['Promotion Score'] and x['Promotion Score'] >= 1.0 else ('⏳ Need More Data' if not x['Meets Volume'] else '❌ Not Yet'),
            axis=1
        )
        
        promo_display = promotion_data.copy()
        promo_display['GP Conv Rate'] = promo_display['GP Conv Rate'].apply(lambda x: f"{x:.1%}")
        promo_display['Adj Conv Rate'] = promo_display['Adj Conv Rate'].apply(lambda x: f"{x:.1%}")
        promo_display['Contact Rate'] = promo_display['Contact Rate'].apply(lambda x: f"{x:.0%}")
        promo_display['Pull Through'] = promo_display['Pull Through'].apply(lambda x: f"{x:.0%}")
        promo_display['Promotion Score'] = promo_display['Promotion Score'].apply(lambda x: f"{x:.2f}" if x else "N/A")
        promo_display['Meets Volume'] = promo_display['Meets Volume'].apply(lambda x: '✅' if x else '❌')
        
        promo_display = promo_display[['Agent', 'Current Tier', 'Growth Pool Leads', 'Meets Volume', 'GP Conv Rate', 'Adj Conv Rate', 'Contact Rate', 'Pull Through', 'Promotion Score', 'Ready?']]
        promo_display.columns = ['Agent', 'Tier', 'GP Leads', 'Vol ✓', 'Raw GP Rate', 'Adj GP Rate', 'Contact', 'Pull Thru', 'Score', 'Status']
        
        st.dataframe(promo_display, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Score Interpretation:**
        - **≥ 1.0** = Performing at or above benchmark → Ready for promotion consideration
        - **0.8 - 1.0** = Close to benchmark → Monitor and develop
        - **< 0.8** = Below benchmark → Needs coaching
        - **N/A** = Insufficient volume → Continue building sample size
        """)
        
        st.divider()
        
        # --- SUMMARY RECOMMENDATIONS ---
        st.subheader("📝 Summary: Implementation Recommendations")
        
        st.markdown("""
        | # | Action | Impact |
        |---|--------|--------|
        | 1 | **Add Bayesian adjustment** to all conversion rate displays | Normalizes for volume differences |
        | 2 | **Show confidence intervals** alongside rates | Makes uncertainty visible |
        | 3 | **Implement 15% Growth Pool** random routing | Provides fair evaluation opportunity |
        | 4 | **Set 30-lead minimum** for promotion evaluation | Ensures statistical reliability |
        | 5 | **Track Growth Pool metrics separately** | Removes selection bias from evaluations |
        | 6 | **Use composite promotion score** | Balances multiple performance dimensions |
        
        ---
        
        ### Code Snippets for Implementation
        
        **Bayesian Adjustment:**
        ```python
        def bayesian_adjusted_rate(conversions, volume, overall_rate, prior_strength=50):
            return (conversions + (overall_rate * prior_strength)) / (volume + prior_strength)
        ```
        
        **Wilson Confidence Interval:**
        ```python
        from scipy import stats
        
        def wilson_ci(conversions, volume, confidence=0.95):
            z = stats.norm.ppf((1 + confidence) / 2)
            p = conversions / volume
            denominator = 1 + z**2 / volume
            center = (p + z**2 / (2 * volume)) / denominator
            spread = z * ((p * (1 - p) / volume + z**2 / (4 * volume**2)) ** 0.5) / denominator
            return (max(0, center - spread), min(1, center + spread))
        ```
        
        **Volume-Adjusted Score:**
        ```python
        def volume_adjusted_score(conversions, volume, min_volume=50):
            raw_rate = conversions / volume if volume > 0 else 0
            volume_factor = min(1.0, volume / min_volume)
            return (raw_rate * volume_factor) ** 0.5
        ```
        """)

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
