"""
Extract Vintage (FAS Day 7) data for week of 2026-03-01 for FAS diagnosis.
Uses same logic as lvc_day7_vintage_dashboard (day7_eligible_sts = denominator).
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from lvc_day7_vintage_dashboard import (
    get_vintage_summary_data,
    get_lvc_day7_breakdown_data,
    get_vintage_deep_dive_data,
    get_ma_performance_data,
)

TARGET_WEEK = '2026-03-01'

def main():
    print("Vintage (FAS Day 7) data for week of", TARGET_WEEK)
    print("=" * 60)

    # Summary
    v = get_vintage_summary_data(TARGET_WEEK)
    if v.empty:
        print("No summary data (query failed or no data)")
        return
    row = v[v['vintage_week'].astype(str).str.contains('2026-03-01')]
    if row.empty:
        print("No row for week 2026-03-01 in summary")
        print(v.to_string())
        return
    r = row.iloc[0]
    sts = int(r['day7_eligible_sts'])
    fas_count = int(r['fas_day7_count'])
    fas_dollars = float(r['fas_day7_dollars'] or 0)
    rate = (fas_count / sts * 100) if sts else 0
    print(f"Day 7 eligible STS (denominator): {sts}")
    print(f"FAS Day 7 count: {fas_count}")
    print(f"FAS Day 7 $: {fas_dollars:,.0f}")
    print(f"FAS Day 7 rate: {rate:.2f}%")
    print(f"Avg loan: {r.get('avg_loan', 0):,.0f}")
    print(f"Avg LP2C: {r.get('avg_lp2c', 0):.2f}%")
    print()

    # LVC
    lvc = get_lvc_day7_breakdown_data(TARGET_WEEK)
    w = lvc['vintage_week'].astype(str).str.contains('2026-03-01')
    if w.any():
        lvc_week = lvc[w][['lvc_group', 'day7_eligible_sts', 'fas_day7_count', 'fas_day7_dollars', 'avg_lp2c']]
        lvc_week = lvc_week.rename(columns={'day7_eligible_sts': 'sts', 'fas_day7_count': 'fas', 'fas_day7_dollars': 'fas_dollars'})
        print("LVC (day 7 eligible):")
        print(lvc_week.to_string(index=False))
        print()

    # Channel (from deep dive, aggregate)
    deep = get_vintage_deep_dive_data(TARGET_WEEK)
    deep_week = deep[deep['vintage_week'].astype(str).str.contains('2026-03-01')]
    if not deep_week.empty:
        ch = deep_week.groupby('channel').agg(
            sts=('day7_eligible_sts', 'sum'),
            fas=('fas_day7_count', 'sum'),
            fas_dollars=('fas_day7_dollars', 'sum'),
        ).reset_index()
        ch = ch[ch['fas'] > 0].sort_values('fas', ascending=False).head(15)
        print("Top channels (FAS Day 7):")
        print(ch.to_string(index=False))
        print()

    # MA
    ma = get_ma_performance_data(TARGET_WEEK)
    ma_week = ma[ma['vintage_week'].astype(str).str.contains('2026-03-01')]
    if not ma_week.empty:
        ma_agg = ma_week.groupby('mortgage_advisor').agg(
            sts=('day7_eligible_sts', 'sum'),
            fas=('fas_day7_count', 'sum'),
            fas_dollars=('fas_day7_dollars', 'sum'),
        ).reset_index()
        ma_agg = ma_agg[ma_agg['fas'] > 0].sort_values('fas', ascending=False).head(15)
        print("Top MAs (FAS Day 7):")
        print(ma_agg.to_string(index=False))

if __name__ == '__main__':
    main()
