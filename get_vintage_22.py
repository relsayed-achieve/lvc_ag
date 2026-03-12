import pandas as pd
from lvc_vintage_dashboard import get_day7_summary_data, get_lvc_day7_breakdown_data, get_vintage_deep_dive_data

target_week = '2026-02-22'

print("=== VINTAGE DAY 7 SUMMARY ===")
summary = get_day7_summary_data(target_week)
print(summary[summary['vintage_week'].astype(str).str.contains('2026-02-22')].to_string())

print("\n=== LVC BREAKDOWN ===")
lvc = get_lvc_day7_breakdown_data(target_week)
print(lvc[lvc['vintage_week'].astype(str).str.contains('2026-02-22')].to_string())

print("\n=== CHANNEL BREAKDOWN ===")
deep = get_vintage_deep_dive_data(target_week)
deep_week = deep[deep['vintage_week'].astype(str).str.contains('2026-02-22')]
channel_grp = deep_week.groupby('channel').agg({
    'lead_count': 'sum',
    'sts_eligible': 'sum',
    'day7_eligible_sts': 'sum',
    'fas_day7_count': 'sum',
    'fas_day7_dollars': 'sum'
}).sort_values('fas_day7_count', ascending=False)
print(channel_grp)
