# Outreach Tab — Metrics, Calculations & Fields Reference

**Tab:** 📞 Outreach (Pre-App)  
**Purpose:** Compare Pre (12/19/25–1/26/26) vs Post (1/27/26–3/6/26) for LVC 1–8, with PHX 24hr delay context.  
**Data:** `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table` (a) LEFT JOIN `lendage-data-platform.standardized_data.lendage_consumer_comms_lead_data` (b) ON `a.lendage_guid = b.lendage_guid`.

---

## 1. Filters & scope

| Item | Value |
|------|--------|
| **LVC filter** | `a.initial_lead_value_cohort IN ('1','2','3','4','5','6','7','8')` |
| **Date filter** | `DATE(a.sent_to_sales_date)` between `2025-12-19` and `2026-03-06` |
| **Period** | Non-null `sent_to_sales_date` only |
| **Pre** | `sent_to_sales_date` between 2025-12-19 and 2026-01-26 |
| **Post** | `sent_to_sales_date` between 2026-01-27 and 2026-03-06 |

---

## 2. Source tables — fields used

### Table A: `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table`

| Field | Use |
|-------|-----|
| `lendage_guid` | Join key, distinct lead counts |
| `sent_to_sales_date` | Date filter, period, `sts_date` |
| `initial_lead_value_cohort` | LVC filter and “By initial LVC” breakdown |
| `starring` | “By starring” breakdown |
| `mortgage_advisor` | “By MA” breakdown |
| `initial_sales_assigned_datetime` | Reference time for speed-to-dial and speed-to-first-contact |
| `first_call_attempt_datetime` | MA first call; input to `earliest_dial_datetime` |
| `first_dial_phx` | PHX first dial; input to `earliest_dial_datetime` |
| `sf_contacted_date` | When `sf__contacted_guid IS NOT NULL`, used as SF contact datetime and input to `earliest_contact_datetime` |
| `call_attempts` | Total call attempts (aggregated) |
| `call_attempts_before_app` | Call attempts before app (available in query, not in current UI) |
| `total_email_sent_before_contact` | Part of `email_before_contact` (query only) |
| `total_sms_outbound_before_contact` | Part of `sms_before_contact` (query only) |
| `phoenix_fronter_contact_datetime` | PHX contact; used for PHX contact count/rate |
| `phx_total_outbound_attempts` | PHX outbound attempts (sum / avg per lead) |
| `sf__contacted_guid` | Defines `had_contact` (SF contact only) and, when non-null, `sf_contacted_date` is used for contact datetime |
| `full_app_submit_datetime` | Used to define `had_fas` |
| `e_loan_amount` | Sum for FAS leads only → FAS $ |

### Table B: `lendage-data-platform.standardized_data.lendage_consumer_comms_lead_data`

| Field | Use |
|-------|-----|
| `lendage_guid` | Join key |
| `first_dial_attempt_datetime` | Input to `earliest_dial_datetime` |
| `first_contact_datetime` | Input to `earliest_contact_datetime` |
| `email_sent_count` | Mapped to `emails_sent` (total emails sent) |
| `sms_sent_count` | Mapped to `outbound_sms` (total outbound SMS) |
| `email_sent_count_before_contact` | Part of `email_before_contact` (query only) |
| `sms_sent_count_before_contact` | Part of `sms_before_contact` (query only) |
| `first_contact_method` | Available in query (not in current UI) |
| `dialer_campaign_flag`, `dialer_contact_flag` | Available in query (not in current UI) |

---

## 3. Lead-level derived fields (in SQL / base dataframe)

These are computed once per lead and then aggregated.

| Derived field | Calculation | Source fields |
|---------------|-------------|----------------|
| **period** | `'Pre (12/19-1/26)'` if `sent_to_sales_date` in [2025-12-19, 2026-01-26]; `'Post (1/27-3/6)'` if in [2026-01-27, 2026-03-06] | `a.sent_to_sales_date` |
| **sts_date** | `DATE(a.sent_to_sales_date)` | `a.sent_to_sales_date` |
| **earliest_dial_datetime** | Earliest non-null of: `a.first_call_attempt_datetime`, `a.first_dial_phx`, `b.first_dial_attempt_datetime` (by CASE logic) | `a.first_call_attempt_datetime`, `a.first_dial_phx`, `b.first_dial_attempt_datetime` |
| **earliest_contact_datetime** | When `a.sf__contacted_guid IS NOT NULL` use `a.sf_contacted_date`; else use `b.first_contact_datetime`. Earliest of those two. | `a.sf__contacted_guid`, `a.sf_contacted_date`, `b.first_contact_datetime` |
| **speed_to_dial_min** | `TIMESTAMP_DIFF(earliest_dial_datetime, initial_sales_assigned_datetime, MINUTE)` | `earliest_dial_datetime`, `a.initial_sales_assigned_datetime` |
| **speed_to_first_contact_min** | `TIMESTAMP_DIFF(earliest_contact_datetime, initial_sales_assigned_datetime, MINUTE)` | `earliest_contact_datetime`, `a.initial_sales_assigned_datetime` |
| **email_before_contact** | `COALESCE(a.total_email_sent_before_contact, 0) + COALESCE(b.email_sent_count_before_contact, 0)` | Table A + B (before-contact email counts) |
| **sms_before_contact** | `COALESCE(a.total_sms_outbound_before_contact, 0) + COALESCE(b.sms_sent_count_before_contact, 0)` | Table A + B (before-contact SMS counts) |
| **emails_sent** | `COALESCE(b.email_sent_count, 0)` | `b.email_sent_count` |
| **outbound_sms** | `COALESCE(b.sms_sent_count, 0)` | `b.sms_sent_count` |
| **phx_total_outbound_attempts** | `COALESCE(a.phx_total_outbound_attempts, 0)` | `a.phx_total_outbound_attempts` |
| **had_contact** | `TRUE` if `a.sf__contacted_guid IS NOT NULL` only | `a.sf__contacted_guid` |
| **had_fas** | `TRUE` if `a.full_app_submit_datetime IS NOT NULL` | `a.full_app_submit_datetime` |
| **e_loan_amount** | `COALESCE(a.e_loan_amount, 0)` (FAS $ uses only rows where `had_fas` is TRUE) | `a.e_loan_amount` |

**Week (for time series):**  
`week_end` = `sts_date` minus (day_of_week + 1) days (week ending = previous Sunday).  
`week_label` = `week_end` formatted as `%m/%d`.

---

## 4. Aggregate metrics (Pre vs Post — section 1)

All below are by **period** (Pre or Post), over leads in scope.

| Metric | Calculation | Source fields / derived |
|--------|-------------|--------------------------|
| **Lead count** | `COUNT(DISTINCT lendage_guid)` | `lendage_guid` |
| **with_dial** | Count of leads where `earliest_dial_datetime` is not null | `earliest_dial_datetime` |
| **with_contact** | Sum of `had_contact` (TRUE = 1) | `had_contact` |
| **fas_count** | Sum of `had_fas` (TRUE = 1) | `had_fas` |
| **fas_dollars** | Sum of `e_loan_amount` for leads where `had_fas` is TRUE | `e_loan_amount`, `had_fas` |
| **contact_rate** | `with_contact / lead_count` | `with_contact`, `lead_count` |
| **dial_rate** | `with_dial / lead_count` | `with_dial`, `lead_count` |
| **fas_pct** | `fas_count / lead_count` | `fas_count`, `lead_count` |
| **speed_to_dial_median** | Median of `speed_to_dial_min` where value is in [0, 10080] minutes (excludes nulls and out-of-range) | `speed_to_dial_min` |
| **speed_to_contact_median** | Median of `speed_to_first_contact_min` where value is in [0, 10080] minutes | `speed_to_first_contact_min` |
| **total_call_attempts** | Sum of `call_attempts` | `a.call_attempts` |
| **avg_call_attempts** | `total_call_attempts / lead_count` | `total_call_attempts`, `lead_count` |
| **emails_sent** | Sum of `emails_sent` (lead-level) | `emails_sent` (from `b.email_sent_count`) |
| **outbound_sms** | Sum of `outbound_sms` (lead-level) | `outbound_sms` (from `b.sms_sent_count`) |
| **phx_total_outbound_attempts** | Sum of `phx_total_outbound_attempts` (lead-level) | `phx_total_outbound_attempts` (from `a.phx_total_outbound_attempts`) |
| **phx_contact_count** | Count of leads where `phoenix_fronter_contact_datetime` is not null | `a.phoenix_fronter_contact_datetime` |
| **phx_contact_rate** | `phx_contact_count / lead_count` | `phx_contact_count`, `lead_count` |
| **avg_emails_sent_per_lead** | `emails_sent / lead_count` | `emails_sent`, `lead_count` |
| **avg_outbound_sms_per_lead** | `outbound_sms / lead_count` | `outbound_sms`, `lead_count` |
| **avg_phx_attempts_per_lead** | `phx_total_outbound_attempts / lead_count` | `phx_total_outbound_attempts`, `lead_count` |

---

## 5. Weekly time-series metrics (section 2)

Same logic as above, but grouped by **week_end** and **period**.  
FAS $ is again sum of `e_loan_amount` only for leads with `had_fas` = TRUE in that week/period.

| Metric | Calculation | Same as aggregate? |
|--------|-------------|---------------------|
| lead_count, with_contact, fas_count, contact_rate, fas_pct | By (week_end, period) | Yes, same formulas |
| fas_dollars | Sum of `e_loan_amount` for `had_fas` leads by (week_end, period) | Yes |
| speed_to_dial_median, speed_to_contact_median | Median of speed minutes in [0, 10080] by (week_end, period) | Yes |
| emails_sent, outbound_sms, phx_total_outbound_attempts | Sum by (week_end, period) | Yes |
| phx_contact_count, phx_contact_rate | Count non-null `phoenix_fronter_contact_datetime`; rate = count / lead_count | Yes |

---

## 6. By initial LVC (section 4)

Same metrics as in **section 4**, but grouped by **period** and **initial_lead_value_cohort** (and FAS $ again only over `had_fas` leads).

| Metric | Calculation | Source |
|--------|-------------|--------|
| lead_count, with_contact, fas_count | Same as aggregate, by (period, initial_lead_value_cohort) | Same fields |
| contact_rate, fas_pct | with_contact / lead_count, fas_count / lead_count | Same |
| fas_dollars | Sum of `e_loan_amount` where `had_fas`, by (period, initial_lead_value_cohort) | `e_loan_amount`, `had_fas` |
| emails_sent, outbound_sms, phx_total_outbound_attempts | Sum by (period, initial_lead_value_cohort) | Same as aggregate |
| phx_contact_count, phx_contact_rate | Same as aggregate by (period, initial_lead_value_cohort) | `phoenix_fronter_contact_datetime` |
| speed_to_dial_median, speed_to_contact_median | Median of speed minutes in [0, 10080] by (period, initial_lead_value_cohort) | `speed_to_dial_min`, `speed_to_first_contact_min` |
| avg_emails_per_lead, avg_sms_per_lead, avg_phx_attempts_per_lead | emails_sent / lead_count, etc., by (period, initial_lead_value_cohort) | Same as aggregate |

---

## 7. By starring (section 6)

| Metric | Calculation | Source |
|--------|-------------|--------|
| lead_count | COUNT(DISTINCT lendage_guid) by (period, starring) | `lendage_guid` |
| with_contact | Sum of had_contact by (period, starring) | `had_contact` |
| fas_count | Sum of had_fas by (period, starring) | `had_fas` |
| contact_rate | with_contact / lead_count | — |
| fas_pct | fas_count / lead_count | — |
| fas_dollars | Sum of e_loan_amount for had_fas leads by (period, starring) | `e_loan_amount`, `had_fas` |

---

## 8. By mortgage advisor (section 7)

Same metrics as **By starring**, grouped by **(period, mortgage_advisor)**.

| Metric | Calculation | Source |
|--------|-------------|--------|
| lead_count, with_contact, fas_count, contact_rate, fas_pct, fas_dollars | Same as By starring, by (period, mortgage_advisor) | Same fields |

---

## 9. Evidence table & insights

- **Evidence table:** Pre vs Post values and Δ for: Contact rate, FAS %, FAS $, Speed to dial (median min), Speed to 1st contact (median min), Emails sent (total), Outbound SMS (total), PHX contact rate, PHX total outbound attempts.  
  All use the same definitions as in **section 4**.
- **Insights bullets:** Compare Post vs Pre using the same metrics (contact rate, FAS %, FAS $, speed to dial/contact, emails sent, outbound SMS, PHX contact rate, PHX outbound attempts).
- **Recommendation (e.g. 48 hr):** Uses the same numbers and adds a note to use the “By initial LVC” table to see if decline is concentrated in certain LVCs.

---

## 10. Summary: key definitions

| Term | Definition |
|------|------------|
| **Earliest dial** | Earliest of: MA first call, PHX first dial, comms first dial (all timestamps). |
| **Earliest contact** | From vintages: when `sf__contacted_guid IS NOT NULL` use `sf_contacted_date`; else comms `first_contact_datetime`. Earliest of those two. |
| **Speed to dial** | Minutes from `initial_sales_assigned_datetime` to `earliest_dial_datetime`. Median uses only values in 0–10080 minutes (~7 days). |
| **Speed to first contact** | Minutes from `initial_sales_assigned_datetime` to `earliest_contact_datetime`. Same 0–10080 min cap for median. |
| **Contact (had_contact)** | SF contact only: `a.sf__contacted_guid IS NOT NULL`. |
| **FAS (had_fas)** | Lead has non-null `full_app_submit_datetime`. |
| **FAS $** | Sum of `e_loan_amount` only where `full_app_submit_datetime IS NOT NULL`. |
| **Emails sent** | From comms: `email_sent_count` (total emails sent for the lead). |
| **Outbound SMS** | From comms: `sms_sent_count`. |
| **PHX contact** | Lead has non-null `phoenix_fronter_contact_datetime`. |
| **PHX outbound attempts** | From vintages: `phx_total_outbound_attempts` (total PHX outbound call attempts for the lead). |

---

*Document generated for approval of Outreach tab metrics and field usage. If you want any metric definition or field changed, specify the metric name and the desired calculation/fields.*
