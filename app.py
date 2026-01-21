import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import date, datetime
import altair as alt

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
    
    tab_main, tab_insights = st.tabs(["📊 Main Dashboard", "🤖 Comprehensive Insights (Jan 24+)"])
    
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

    # --- TAB 2: Insights ---
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
