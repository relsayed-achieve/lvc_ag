# Outreach Tab — Every Metric and Its SQL Calculation

Below is **every metric** in the Outreach tab with the **exact SQL expression** that defines it.  
Base data comes from the `base` + `metrics` CTEs; aggregate metrics are expressed as SQL you could run over that result (or in a single query with `GROUP BY`).

---

## Base query: lead-level fields (in SQL)

These columns are produced by the tab’s main SQL. Every other metric is built from these.

### Period & identifiers

| Metric (column) | SQL calculation |
|-----------------|-----------------|
| **period** | `CASE WHEN DATE(a.sent_to_sales_date) BETWEEN '2025-12-19' AND '2026-01-26' THEN 'Pre (12/19-1/26)' WHEN DATE(a.sent_to_sales_date) BETWEEN '2026-01-27' AND '2026-03-06' THEN 'Post (1/27-3/6)' END` |
| **sts_date** | `DATE(a.sent_to_sales_date)` |
| **initial_lead_value_cohort** | `a.initial_lead_value_cohort` |
| **starring** | `a.starring` |
| **mortgage_advisor** | `a.mortgage_advisor` |

### Earliest dial (used for speed to dial)

| Metric (column) | SQL calculation |
|-----------------|-----------------|
| **earliest_dial_datetime** | `CASE WHEN a.first_call_attempt_datetime IS NOT NULL AND (a.first_dial_phx IS NULL OR a.first_call_attempt_datetime <= a.first_dial_phx) AND (b.first_dial_attempt_datetime IS NULL OR a.first_call_attempt_datetime <= b.first_dial_attempt_datetime) THEN a.first_call_attempt_datetime WHEN a.first_dial_phx IS NOT NULL AND (b.first_dial_attempt_datetime IS NULL OR a.first_dial_phx <= b.first_dial_attempt_datetime) THEN a.first_dial_phx ELSE b.first_dial_attempt_datetime END` |

### Earliest contact (used for speed to first contact)

| Metric (column) | SQL calculation |
|-----------------|-----------------|
| **a_sf_contact_datetime** | `CASE WHEN a.sf__contacted_guid IS NOT NULL THEN a.sf_contacted_date ELSE NULL END` |
| **earliest_contact_datetime** | `CASE WHEN a.sf__contacted_guid IS NOT NULL AND (b.first_contact_datetime IS NULL OR a.sf_contacted_date <= b.first_contact_datetime) THEN a.sf_contacted_date ELSE b.first_contact_datetime END` |

### Speed (minutes from assignment)

| Metric (column) | SQL calculation |
|-----------------|-----------------|
| **speed_to_dial_min** | `TIMESTAMP_DIFF(earliest_dial_datetime, a.initial_sales_assigned_datetime, MINUTE)` |
| **speed_to_first_contact_min** | `TIMESTAMP_DIFF(earliest_contact_datetime, a.initial_sales_assigned_datetime, MINUTE)` |

### Calls

| Metric (column) | SQL calculation |
|-----------------|-----------------|
| **call_attempts** | `a.call_attempts` |

### Email & SMS (comms)

| Metric (column) | SQL calculation |
|-----------------|-----------------|
| **emails_sent** | `COALESCE(b.email_sent_count, 0)` |
| **outbound_sms** | `COALESCE(b.sms_sent_count, 0)` |

### PHX

| Metric (column) | SQL calculation |
|-----------------|-----------------|
| **phoenix_fronter_contact_datetime** | `a.phoenix_fronter_contact_datetime` |
| **phx_total_outbound_attempts** | `COALESCE(a.phx_total_outbound_attempts, 0)` |

### Contact & FAS flags

| Metric (column) | SQL calculation |
|-----------------|-----------------|
| **had_contact** | `(a.sf__contacted_guid IS NOT NULL)` |
| **had_fas** | `(a.full_app_submit_datetime IS NOT NULL)` |

### FAS dollars (lead-level)

| Metric (column) | SQL calculation |
|-----------------|-----------------|
| **e_loan_amount** | `COALESCE(a.e_loan_amount, 0)` |

*(FAS $ in the tab is the sum of this only where `had_fas` is TRUE.)*

---

## Aggregate metrics (Pre vs Post) — SQL equivalent

Assume `m` is the result of the base + metrics CTEs above (one row per lead). All of these are by `period` (i.e. `GROUP BY period`).

| Metric | SQL calculation |
|--------|-----------------|
| **lead_count** | `COUNT(DISTINCT m.lendage_guid)` |
| **with_dial** | `COUNTIF(m.earliest_dial_datetime IS NOT NULL)` or `SUM(CASE WHEN m.earliest_dial_datetime IS NOT NULL THEN 1 ELSE 0 END)` |
| **with_contact** | `SUM(CASE WHEN m.had_contact THEN 1 ELSE 0 END)` |
| **fas_count** | `SUM(CASE WHEN m.had_fas THEN 1 ELSE 0 END)` |
| **fas_dollars** | `SUM(CASE WHEN m.had_fas THEN m.e_loan_amount ELSE 0 END)` |
| **contact_rate** | `SUM(CASE WHEN m.had_contact THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **dial_rate** | `COUNTIF(m.earliest_dial_datetime IS NOT NULL) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **fas_pct** | `SUM(CASE WHEN m.had_fas THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **speed_to_dial_median** | `APPROX_QUANTILES(CASE WHEN m.speed_to_dial_min BETWEEN 0 AND 10080 THEN m.speed_to_dial_min END, 100)[OFFSET(50)]` (median of speed_to_dial_min where 0 ≤ value ≤ 10080; nulls excluded) |
| **speed_to_contact_median** | `APPROX_QUANTILES(CASE WHEN m.speed_to_first_contact_min BETWEEN 0 AND 10080 THEN m.speed_to_first_contact_min END, 100)[OFFSET(50)]` |
| **total_call_attempts** | `SUM(m.call_attempts)` |
| **avg_call_attempts** | `SUM(m.call_attempts) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **emails_sent** | `SUM(m.emails_sent)` |
| **outbound_sms** | `SUM(m.outbound_sms)` |
| **phx_total_outbound_attempts** | `SUM(m.phx_total_outbound_attempts)` |
| **phx_contact_count** | `COUNTIF(m.phoenix_fronter_contact_datetime IS NOT NULL)` |
| **phx_contact_rate** | `COUNTIF(m.phoenix_fronter_contact_datetime IS NOT NULL) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **avg_emails_sent_per_lead** | `SUM(m.emails_sent) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **avg_outbound_sms_per_lead** | `SUM(m.outbound_sms) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **avg_phx_attempts_per_lead** | `SUM(m.phx_total_outbound_attempts) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |

---

## Weekly time-series metrics — SQL equivalent

Same as above, but grouped by week and period.  
`week_end` = week-ending date (e.g. `DATE_SUB(DATE(m.sts_date), INTERVAL MOD(EXTRACT(DAYOFWEEK FROM m.sts_date) + 6, 7) DAY)` for Sunday week-end, or equivalent).

| Metric | SQL calculation (per week_end, period) |
|--------|----------------------------------------|
| **lead_count** | `COUNT(DISTINCT m.lendage_guid)` |
| **with_contact** | `SUM(CASE WHEN m.had_contact THEN 1 ELSE 0 END)` |
| **fas_count** | `SUM(CASE WHEN m.had_fas THEN 1 ELSE 0 END)` |
| **fas_dollars** | `SUM(CASE WHEN m.had_fas THEN m.e_loan_amount ELSE 0 END)` |
| **contact_rate** | `SUM(CASE WHEN m.had_contact THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **fas_pct** | `SUM(CASE WHEN m.had_fas THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **speed_to_dial_median** | `APPROX_QUANTILES(CASE WHEN m.speed_to_dial_min BETWEEN 0 AND 10080 THEN m.speed_to_dial_min END, 100)[OFFSET(50)]` |
| **speed_to_contact_median** | `APPROX_QUANTILES(CASE WHEN m.speed_to_first_contact_min BETWEEN 0 AND 10080 THEN m.speed_to_first_contact_min END, 100)[OFFSET(50)]` |
| **emails_sent** | `SUM(m.emails_sent)` |
| **outbound_sms** | `SUM(m.outbound_sms)` |
| **phx_total_outbound_attempts** | `SUM(m.phx_total_outbound_attempts)` |
| **phx_contact_count** | `COUNTIF(m.phoenix_fronter_contact_datetime IS NOT NULL)` |
| **phx_contact_rate** | `COUNTIF(m.phoenix_fronter_contact_datetime IS NOT NULL) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |

---

## By initial LVC — SQL equivalent

Same formulas as the aggregate section, with `GROUP BY period, m.initial_lead_value_cohort`.

| Metric | SQL calculation (per period, initial_lead_value_cohort) |
|--------|----------------------------------------------------------|
| **lead_count** | `COUNT(DISTINCT m.lendage_guid)` |
| **with_contact** | `SUM(CASE WHEN m.had_contact THEN 1 ELSE 0 END)` |
| **fas_count** | `SUM(CASE WHEN m.had_fas THEN 1 ELSE 0 END)` |
| **fas_dollars** | `SUM(CASE WHEN m.had_fas THEN m.e_loan_amount ELSE 0 END)` |
| **contact_rate** | `SUM(CASE WHEN m.had_contact THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **fas_pct** | `SUM(CASE WHEN m.had_fas THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **emails_sent** | `SUM(m.emails_sent)` |
| **outbound_sms** | `SUM(m.outbound_sms)` |
| **phx_total_outbound_attempts** | `SUM(m.phx_total_outbound_attempts)` |
| **phx_contact_count** | `COUNTIF(m.phoenix_fronter_contact_datetime IS NOT NULL)` |
| **phx_contact_rate** | `COUNTIF(m.phoenix_fronter_contact_datetime IS NOT NULL) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **speed_to_dial_median** | `APPROX_QUANTILES(CASE WHEN m.speed_to_dial_min BETWEEN 0 AND 10080 THEN m.speed_to_dial_min END, 100)[OFFSET(50)]` |
| **speed_to_contact_median** | `APPROX_QUANTILES(CASE WHEN m.speed_to_first_contact_min BETWEEN 0 AND 10080 THEN m.speed_to_first_contact_min END, 100)[OFFSET(50)]` |
| **avg_emails_per_lead** | `SUM(m.emails_sent) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **avg_sms_per_lead** | `SUM(m.outbound_sms) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **avg_phx_attempts_per_lead** | `SUM(m.phx_total_outbound_attempts) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |

---

## By starring — SQL equivalent

`GROUP BY period, m.starring`. Same metrics as below.

| Metric | SQL calculation (per period, starring) |
|--------|----------------------------------------|
| **lead_count** | `COUNT(DISTINCT m.lendage_guid)` |
| **with_contact** | `SUM(CASE WHEN m.had_contact THEN 1 ELSE 0 END)` |
| **fas_count** | `SUM(CASE WHEN m.had_fas THEN 1 ELSE 0 END)` |
| **fas_dollars** | `SUM(CASE WHEN m.had_fas THEN m.e_loan_amount ELSE 0 END)` |
| **contact_rate** | `SUM(CASE WHEN m.had_contact THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **fas_pct** | `SUM(CASE WHEN m.had_fas THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |

---

## By mortgage advisor — SQL equivalent

`GROUP BY period, m.mortgage_advisor`. Same metrics as By starring.

| Metric | SQL calculation (per period, mortgage_advisor) |
|--------|-------------------------------------------------|
| **lead_count** | `COUNT(DISTINCT m.lendage_guid)` |
| **with_contact** | `SUM(CASE WHEN m.had_contact THEN 1 ELSE 0 END)` |
| **fas_count** | `SUM(CASE WHEN m.had_fas THEN 1 ELSE 0 END)` |
| **fas_dollars** | `SUM(CASE WHEN m.had_fas THEN m.e_loan_amount ELSE 0 END)` |
| **contact_rate** | `SUM(CASE WHEN m.had_contact THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |
| **fas_pct** | `SUM(CASE WHEN m.had_fas THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT m.lendage_guid), 0)` |

---

## Single-statement example: aggregate Pre/Post

You can reproduce the main Pre/Post aggregates with one query like this (using the same CTEs as the app):

```sql
WITH base AS (
  SELECT
    a.lendage_guid,
    DATE(a.sent_to_sales_date) AS sts_date,
    CASE WHEN DATE(a.sent_to_sales_date) BETWEEN '2025-12-19' AND '2026-01-26' THEN 'Pre (12/19-1/26)'
         WHEN DATE(a.sent_to_sales_date) BETWEEN '2026-01-27' AND '2026-03-06' THEN 'Post (1/27-3/6)' END AS period,
    a.initial_lead_value_cohort,
    a.starring,
    a.mortgage_advisor,
    a.initial_sales_assigned_datetime,
    a.first_call_attempt_datetime,
    a.first_dial_phx,
    b.first_dial_attempt_datetime,
    CASE WHEN a.sf__contacted_guid IS NOT NULL THEN a.sf_contacted_date ELSE NULL END AS a_sf_contact_datetime,
    b.first_contact_datetime,
    CASE WHEN a.first_call_attempt_datetime IS NOT NULL AND (a.first_dial_phx IS NULL OR a.first_call_attempt_datetime <= a.first_dial_phx) AND (b.first_dial_attempt_datetime IS NULL OR a.first_call_attempt_datetime <= b.first_dial_attempt_datetime) THEN a.first_call_attempt_datetime
         WHEN a.first_dial_phx IS NOT NULL AND (b.first_dial_attempt_datetime IS NULL OR a.first_dial_phx <= b.first_dial_attempt_datetime) THEN a.first_dial_phx
         ELSE b.first_dial_attempt_datetime END AS earliest_dial_datetime,
    CASE WHEN a.sf__contacted_guid IS NOT NULL AND (b.first_contact_datetime IS NULL OR a.sf_contacted_date <= b.first_contact_datetime) THEN a.sf_contacted_date
         ELSE b.first_contact_datetime END AS earliest_contact_datetime,
    a.call_attempts,
    COALESCE(b.email_sent_count, 0) AS emails_sent,
    COALESCE(b.sms_sent_count, 0) AS outbound_sms,
    a.phoenix_fronter_contact_datetime,
    COALESCE(a.phx_total_outbound_attempts, 0) AS phx_total_outbound_attempts,
    (a.sf__contacted_guid IS NOT NULL) AS had_contact,
    (a.full_app_submit_datetime IS NOT NULL) AS had_fas,
    COALESCE(a.e_loan_amount, 0) AS e_loan_amount
  FROM `ffn-dw-bigquery-prd.Ramzi.lendage_lead_vintages_table` a
  LEFT JOIN `lendage-data-platform.standardized_data.lendage_consumer_comms_lead_data` b ON a.lendage_guid = b.lendage_guid
  WHERE a.initial_lead_value_cohort IN ('1','2','3','4','5','6','7','8')
    AND DATE(a.sent_to_sales_date) BETWEEN '2025-12-19' AND '2026-03-06'
    AND a.sent_to_sales_date IS NOT NULL
),
metrics AS (
  SELECT *,
    TIMESTAMP_DIFF(earliest_dial_datetime, initial_sales_assigned_datetime, MINUTE) AS speed_to_dial_min,
    TIMESTAMP_DIFF(earliest_contact_datetime, initial_sales_assigned_datetime, MINUTE) AS speed_to_first_contact_min
  FROM base
  WHERE period IS NOT NULL
)
SELECT
  period,
  COUNT(DISTINCT lendage_guid) AS lead_count,
  COUNTIF(earliest_dial_datetime IS NOT NULL) AS with_dial,
  SUM(CASE WHEN had_contact THEN 1 ELSE 0 END) AS with_contact,
  SUM(CASE WHEN had_fas THEN 1 ELSE 0 END) AS fas_count,
  SUM(CASE WHEN had_fas THEN e_loan_amount ELSE 0 END) AS fas_dollars,
  SUM(CASE WHEN had_contact THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT lendage_guid), 0) AS contact_rate,
  COUNTIF(earliest_dial_datetime IS NOT NULL) / NULLIF(COUNT(DISTINCT lendage_guid), 0) AS dial_rate,
  SUM(CASE WHEN had_fas THEN 1 ELSE 0 END) / NULLIF(COUNT(DISTINCT lendage_guid), 0) AS fas_pct,
  APPROX_QUANTILES(CASE WHEN speed_to_dial_min BETWEEN 0 AND 10080 THEN speed_to_dial_min END, 100)[OFFSET(50)] AS speed_to_dial_median,
  APPROX_QUANTILES(CASE WHEN speed_to_first_contact_min BETWEEN 0 AND 10080 THEN speed_to_first_contact_min END, 100)[OFFSET(50)] AS speed_to_contact_median,
  SUM(call_attempts) AS total_call_attempts,
  SUM(call_attempts) / NULLIF(COUNT(DISTINCT lendage_guid), 0) AS avg_call_attempts,
  SUM(emails_sent) AS emails_sent,
  SUM(outbound_sms) AS outbound_sms,
  SUM(phx_total_outbound_attempts) AS phx_total_outbound_attempts,
  COUNTIF(phoenix_fronter_contact_datetime IS NOT NULL) AS phx_contact_count,
  COUNTIF(phoenix_fronter_contact_datetime IS NOT NULL) / NULLIF(COUNT(DISTINCT lendage_guid), 0) AS phx_contact_rate,
  SUM(emails_sent) / NULLIF(COUNT(DISTINCT lendage_guid), 0) AS avg_emails_sent_per_lead,
  SUM(outbound_sms) / NULLIF(COUNT(DISTINCT lendage_guid), 0) AS avg_outbound_sms_per_lead,
  SUM(phx_total_outbound_attempts) / NULLIF(COUNT(DISTINCT lendage_guid), 0) AS avg_phx_attempts_per_lead
FROM metrics
GROUP BY period
ORDER BY period;
```

---

*This file lists every metric and the SQL that defines it. Adjust table/date literals if your environment differs.*
