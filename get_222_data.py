from google.cloud import bigquery

client = bigquery.Client(project='ffn-dw-bigquery-prd')

query = """
SELECT
    COUNT(DISTINCT CASE WHEN sent_to_sales_date IS NOT NULL THEN lendage_guid END) as sent_to_sales_qty,
    COUNT(DISTINCT CASE 
        WHEN full_app_submit_datetime >= TIMESTAMP(DATE(lead_created_date))
        AND DATE(full_app_submit_datetime) <= DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
        THEN lendage_guid 
    END) as fas_day7_qty,
    SUM(CASE 
        WHEN full_app_submit_datetime >= TIMESTAMP(DATE(lead_created_date))
        AND DATE(full_app_submit_datetime) <= DATE_ADD(DATE(lead_created_date), INTERVAL 7 DAY)
        THEN e_loan_amount ELSE 0 
    END) as fas_day7_dollars
FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`
WHERE DATE(lead_created_date) >= '2026-02-22' 
AND DATE(lead_created_date) <= '2026-02-28'
"""

df = client.query(query).to_dataframe()

sts = df['sent_to_sales_qty'].iloc[0]
fas_d7 = df['fas_day7_qty'].iloc[0]
fas_d7_dollars = df['fas_day7_dollars'].iloc[0]
rate = (fas_d7 / sts * 100) if sts > 0 else 0

print(f"--- Aggregate for Week of 2/22 (Based on lead_created_date) ---")
print(f"Sent to Sales (Denominator): {sts:,}")
print(f"FAS Day 7 Qty (Numerator): {fas_d7:,}")
print(f"FAS Day 7 % Rate: {rate:.1f}%")
print(f"FAS Day 7 Dollars: ${fas_d7_dollars:,.0f}")
print(f"-------------------------------------------------------------")
