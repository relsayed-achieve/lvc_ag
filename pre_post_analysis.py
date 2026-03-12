from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(project="ffn-dw-bigquery-prd")

query = """
WITH metrics AS (
    SELECT 
        CASE 
            WHEN DATE(lead_created_date) < '2025-10-15' THEN 'Pre-Launch'
            ELSE 'Post-Launch'
        END as period,
        CASE 
            WHEN adjusted_lead_value_cohort IN ('1', '2') THEN 'LVC 1-2'
            WHEN adjusted_lead_value_cohort IN ('3', '4', '5', '6', '7', '8') THEN 'LVC 3-8'
            WHEN adjusted_lead_value_cohort IN ('9', '10') THEN 'LVC 9-10'
            WHEN adjusted_lead_value_cohort LIKE '%X%' THEN 'PHX Transfer'
            ELSE 'Other'
        END as lvc_group,
        COUNT(DISTINCT lendage_guid) as lead_qty,
        COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales_qty,
        AVG(CASE WHEN sent_to_sales_date IS NOT NULL THEN initial_lead_score_lp2c END) as avg_lp2c,
        COUNT(DISTINCT CASE WHEN current_sales_assigned_date IS NOT NULL THEN lendage_guid END) as assigned_qty,
        COUNT(DISTINCT CASE WHEN sf__contacted_guid IS NOT NULL THEN lendage_guid END) as contacted_qty,
        COUNT(DISTINCT CASE WHEN full_app_submit_datetime IS NOT NULL THEN lendage_guid END) as fas_qty,
        SUM(CASE WHEN full_app_submit_datetime IS NOT NULL THEN e_loan_amount END) as fas_volume
    FROM \`ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table\`
    WHERE DATE(lead_created_date) >= '2024-01-01'
    GROUP BY 1, 2
)
SELECT 
    period,
    lvc_group,
    lead_qty,
    sent_to_sales_qty,
    ROUND(avg_lp2c, 2) as avg_lp2c,
    assigned_qty,
    contacted_qty,
    fas_qty,
    ROUND(fas_volume, 0) as fas_volume,
    ROUND(SAFE_DIVIDE(contacted_qty, sent_to_sales_qty) * 100, 2) as contact_rate_pct,
    ROUND(SAFE_DIVIDE(fas_qty, sent_to_sales_qty) * 100, 2) as fas_rate_pct
FROM metrics
ORDER BY period DESC, lvc_group
"""

print("Querying BigQuery for Pre/Post 10/15/2025 LVC Analysis...")
df = client.query(query).to_dataframe()
print("\n" + "="*100)
print("PRE vs POST 10/15/2025 LAUNCH - LVC CORE METRICS COMPARISON")
print("="*100)

# Pivot for comparison
for lvc in ['LVC 1-2', 'LVC 3-8', 'LVC 9-10', 'PHX Transfer', 'Other']:
    pre = df[(df['period'] == 'Pre-Launch') & (df['lvc_group'] == lvc)]
    post = df[(df['period'] == 'Post-Launch') & (df['lvc_group'] == lvc)]
    
    if not pre.empty and not post.empty:
        pre = pre.iloc[0]
        post = post.iloc[0]
        
        print(f"\n{'='*50}")
        print(f"  {lvc}")
        print(f"{'='*50}")
        print(f"{'Metric':<25} {'Pre-Launch':>15} {'Post-Launch':>15} {'Change':>15}")
        print(f"{'-'*70}")
        print(f"{'Lead Qty':<25} {pre['lead_qty']:>15,} {post['lead_qty']:>15,} {post['lead_qty']/pre['lead_qty']*100-100:>14.1f}%")
        print(f"{'Sent to Sales':<25} {pre['sent_to_sales_qty']:>15,} {post['sent_to_sales_qty']:>15,} {post['sent_to_sales_qty']/pre['sent_to_sales_qty']*100-100:>14.1f}%")
        print(f"{'Avg LP2C':<25} {pre['avg_lp2c']:>15.2f} {post['avg_lp2c']:>15.2f} {post['avg_lp2c']-pre['avg_lp2c']:>+14.2f}")
        print(f"{'Contacted':<25} {pre['contacted_qty']:>15,} {post['contacted_qty']:>15,} {post['contacted_qty']/pre['contacted_qty']*100-100:>14.1f}%")
        print(f"{'Contact Rate %':<25} {pre['contact_rate_pct']:>14.1f}% {post['contact_rate_pct']:>14.1f}% {post['contact_rate_pct']-pre['contact_rate_pct']:>+13.1f}pp")
        print(f"{'FAS Qty':<25} {pre['fas_qty']:>15,} {post['fas_qty']:>15,} {post['fas_qty']/pre['fas_qty']*100-100:>14.1f}%")
        print(f"{'FAS Rate %':<25} {pre['fas_rate_pct']:>14.1f}% {post['fas_rate_pct']:>14.1f}% {post['fas_rate_pct']-pre['fas_rate_pct']:>+13.1f}pp")
        print(f"{'FAS Volume $':<25} ${pre['fas_volume']:>14,.0f} ${post['fas_volume']:>14,.0f}")

# Summary totals
print(f"\n\n{'='*100}")
print("OVERALL TOTALS")
print(f"{'='*100}")
pre_total = df[df['period'] == 'Pre-Launch']
post_total = df[df['period'] == 'Post-Launch']

print(f"\n{'Metric':<25} {'Pre-Launch':>20} {'Post-Launch':>20}")
print(f"{'-'*65}")
print(f"{'Total Leads':<25} {pre_total['lead_qty'].sum():>20,} {post_total['lead_qty'].sum():>20,}")
print(f"{'Total Sent to Sales':<25} {pre_total['sent_to_sales_qty'].sum():>20,} {post_total['sent_to_sales_qty'].sum():>20,}")
print(f"{'Total FAS':<25} {pre_total['fas_qty'].sum():>20,} {post_total['fas_qty'].sum():>20,}")
print(f"{'Total FAS Volume':<25} ${pre_total['fas_volume'].sum():>19,.0f} ${post_total['fas_volume'].sum():>19,.0f}")
print(f"{'Overall FAS Rate':<25} {pre_total['fas_qty'].sum()/pre_total['sent_to_sales_qty'].sum()*100:>19.2f}% {post_total['fas_qty'].sum()/post_total['sent_to_sales_qty'].sum()*100:>19.2f}%")

# Days comparison
print(f"\n📅 Pre-Launch: Jan 2024 - Oct 14, 2025 (~654 days)")
print(f"📅 Post-Launch: Oct 15, 2025 - Today (~100 days)")
