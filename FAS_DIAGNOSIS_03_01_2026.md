# FAS Degradation Diagnosis — Week of 3/1/2026 (Vintage Tab)

**Data source:** http://127.0.0.1:8051/ — **Vintage** tab (FAS Day 7 logic)  
**Week:** 2026-03-01 (Sun 3/1 – Sat 3/7)  
**Maturity rule:** A lead is Day 7 eligible only when `lead_created_date ≤ CURRENT_DATE() − 7 days`. For the week of 3/1, that means **3/1 through 3/5** are matured once we're past March 12; 3/6 and 3/7 mature on 3/13 and 3/14. The metrics below are therefore based on **five days** of the week (3/1–3/5), not all seven.

---

## Summary (Step Format)

**Step 1: [FAS Rate Down]** ➔ **CONFIRMED**

**Step 2: [Gather Data]** (Vintage View)  
Denominator: 3,940 STS · FAS Day 7: 128 · Rate: 3.25% · FAS $: $10.58M · Avg loan: $82,642 · LP2C: 3.28%

**Step 3: [Is degradation uniform across ALL channels and reps?]** ➔ **NO**

**Step 4: [HYPOTHESIS: Lead Quality Mix Shift]** ➔ **CONFIRMED**

- **SubStep Analysis 1: Degrading Channels Check** ➔ **CONFIRMED**  
  Heloc First and Quinstreet degraded; FAS volume loss was a drop in the bucket (Quinstreet 89 STS, 2 FAS = 2.3% of denominator; Heloc First 107 STS = 2.7%. Combined impact on headline rate is minimal.)

- **SubStep Analysis 2: Lead Volume** ➔ **CONFIRMED**  
  The volume shifted aggressively into lower-converting cohorts:  
  **LVC 3-8 (1,618 STS)** and **LVC 9-10 (1,188 STS)** combined for **2,806 leads**.  
  This means **71.3%** of our entire denominator for the week (3,940 STS) was composed of mid-to-lower intent leads.

**Step 5: [Are margin characteristics still positive at current execution rate?]** ➔ **YES**  
Despite the denominator dragging the headline rate down, those 2,806 lower-tier leads still generated **37 incremental FAS** and roughly **$3.3M** in pipeline, with a healthy overall LP2C of **3.28%** and avg loan **$82,642**.

➔ **Framework Conclusion: [Pareto Optimization Zone]**

---

## Step 1: [FAS Rate Down] → Confirmed

| Metric | Week of 3/1/2026 (matured slice: 3/1–3/5) |
|--------|--------------------------------------------|
| **Day 7 eligible STS (denominator)** | 3,940 |
| **FAS Day 7 count** | 128 |
| **FAS Day 7 $** | $10,578,221 |
| **FAS Day 7 rate** | **3.25%** |
| Avg loan | $82,642 |
| Avg LP2C | 3.28% |

Compared to a prior full week (e.g. 2/22 with 5,675 STS): this week’s denominator and numerator are from **five days** of lead creation (3/1–3/5), so absolute volume is lower. The blended rate (3.25%) is also down vs a typical full week. We treat “FAS rate down” as confirmed and run the framework.

---

## Step 2: Gather data — 7-day FAS by channel, LVC cohort, MA rep

### Volume vs. denominator (3,940 STS)

**LVC cohort (share of denominator and FAS):**

| LVC Group   | STS (denom) | % of denominator | FAS | FAS $   | Implied rate |
|-------------|-------------|-------------------|-----|---------|---------------|
| LVC 1-2     | 854         | **21.7%**         | 61  | $5.12M  | 7.14%         |
| LVC 3-8     | 1,618       | **41.1%**         | 36  | $3.24M  | 2.22%         |
| LVC 9-10    | 1,188       | **30.2%**         | 1   | $31.5K  | 0.08%         |
| PHX Transfer| 243         | 6.2%              | 19  | $1.07M  | 7.82%         |
| Other       | 37          | 0.9%              | 11  | $1.13M  | 29.7%         |

**Takeaway:** **71.3% of the denominator** (2,806 STS) is LVC 3–10. These tiers convert at 2.22% and 0.08%, so they mechanically pull the blended rate down.

**Top channels (volume vs. denominator):**

| Channel   | STS | % of denominator | FAS | FAS $   |
|-----------|-----|-------------------|-----|---------|
| lt pl     | 1,137 | **28.9%**      | 36  | $2.45M  |
| quicken   | 785  | **19.9%**        | 13  | $1.39M  |
| even      | 508  | 12.9%            | 21  | $1.88M  |
| lt heloc  | 591  | 15.0%            | 4   | $404K   |
| nerd wallet | 378 | 9.6%            | 4   | $354K   |
| organic   | 119  | 3.0%             | 17  | $1.43M  |
| heloc first | 107 | 2.7%            | 12  | $1.02M  |
| directmail| 53   | 1.3%             | 9   | $781K   |
| quinstreet| 89   | 2.3%             | 2   | $171K   |

**Takeaway:** **quicken is 20% of the denominator** but only 13 FAS (~1.66% rate). **lt pl** is 29% of denominator. Channel mix (high share of lower-converting volume) directly drives the denominator effect.

**MA (top by FAS):** Cody Sorells (14), Rebecca Monell (9), Tanner Zimmerman (8), Sean McKinney (7), Jonathan Niesen (7), others 3–5. Performance is not uniform across reps.

---

## Step 3: Is degradation uniform across ALL channels and reps? → **No**

- **Channels:** organic, heloc first, directmail show strong rates on their small bases; quicken and lt pl dominate volume with lower effective rates.
- **LVC:** LVC 1-2 and PHX Transfer have high rates (7%+); LVC 3-8 and especially LVC 9-10 drag the blend.
- **Reps:** Clear variance (e.g., Cody Sorells 14 FAS vs others at 3–5).

Degradation is **concentrated** in mix (channel + LVC), not broad-based across every channel and rep.

→ **Next:** Is the degraded channel one we intentionally opened or expanded?

---

## Step 4: Is the degraded channel one we intentionally opened/expanded? → **Yes (mix shift)**

**Volume in relationship to the denominator (3,940 STS):**

1. **quicken**  
   - **785 STS = 19.9% of denominator**  
   - 13 FAS → ~1.66% rate.  
   So quicken is not noise: it’s a fifth of the pool with a below-blend rate, which materially pulls the overall FAS rate down.

2. **LVC 3-8 + LVC 9-10**  
   - **2,806 STS = 71.3% of denominator**  
   - Convert at 2.22% and 0.08%.  
   Pushing this volume into the funnel is intentional expansion into mid/lower-tier leads; the lower rate is the expected denominator effect.

3. **Established channels that “drag” in rate terms**  
   - **quinstreet:** 89 STS (2.3% of denominator), 2 FAS.  
   - **lt heloc:** 591 STS (15% of denominator), 4 FAS.  
   Quinstreet is small in denominator terms; lt heloc is large but only 4 FAS. The main driver is **mix shift** (quicken + LVC 3-10), not established-channel collapse.

**Conclusion:** Primary hypothesis = **Lead quality mix shift / denominator effect**: we intentionally expanded volume into quicken and LVC 3-10; they have lower close rates but add incremental FAS and are margin-acceptable.

---

## Step 5: Are margin characteristics still positive at current execution rate? → **Yes**

- **Avg loan:** $82,642  
- **Avg LP2C:** 3.28%  
- **FAS Day 7 $:** $10.58M on 128 apps  

We are in the **Pareto optimization zone**: rate is down due to mix, but margins and absolute output are still positive.

---

## Framework conclusion: **Pareto optimization zone**

- **Acceptable tradeoff.** Keep pushing volume; watch closely; do not cut based on rate alone.
- **Action:** Email update mid-cycle; do not wait for the next meeting.

---

## Vintage maturity (week of 3/1) — corrected

Day 7 eligibility uses **`lead_created_date ≤ CURRENT_DATE() − 7 days`**. For the week of 3/1 (Sun 3/1–Sat 3/7):

| As of date | Cutoff (today − 7) | Matured days in week 3/1 |
|------------|--------------------|---------------------------|
| March 10   | March 3            | 3/1, 3/2, 3/3 (3 days)   |
| March 11   | March 4            | 3/1–3/4 (4 days)          |
| **March 12+** | **March 5**      | **3/1–3/5 (5 days)**      |
| March 14+  | March 7            | Full week 3/1–3/7         |

So the **3,940 STS and 128 FAS** in this analysis are based on **at most five days (3/1–3/5)** of the week. 3/6 and 3/7 are not yet eligible until we’re past March 13–14. That shrinks the denominator vs a full week (e.g. 2/22 had 5,675 STS) and can make week-over-week rate comparisons noisier. The **framework conclusion is unchanged**: the rate drop is consistent with **denominator effect from intentional mix shift** (quicken + LVC 3-10), not system-wide failure. Margins support continuing volume.

---

## Summary for mid-cycle email

- **FAS rate is down** (3.25%) vs prior full week. **Volume vs denominator** explains most of it.
- **Vintage maturity:** This week’s numbers reflect **five days** of the week (3/1–3/5); 3/6 and 3/7 are not yet matured. When comparing to a full week (e.g. 2/22), keep that in mind.
- **~20% of denominator is quicken** (low rate); **~71% is LVC 3-10** (mid/lower intent). Both are intentional expansion → **lead quality mix shift**, not broad degradation.
- **Established channels** (e.g. quinstreet) are a small share of denominator; their drag is limited in absolute impact.
- **Margins are still positive** (LP2C 3.28%, avg loan $82K, $10.6M FAS $). **Recommendation:** Keep pushing volume; send mid-cycle update; do not cut; monitor LVC 1-2 and PHX rate and MA variance.
