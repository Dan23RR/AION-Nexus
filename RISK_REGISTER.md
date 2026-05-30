# Risk Register — AION-NEXUS

What could kill the project. Severity × Likelihood × Mitigation. Updated quarterly.

> **Public deliverable**. This is the same file as `AION_NEXUS_RD/09_RISK_REGISTER.md` (internal R&D workbook), promoted to production package as part of the credibility hallmark. Investors, partners, and customers can read this directly during due diligence — we publish risks instead of hiding them.

> **Last update**: 2026-05-25. Updates in this revision:
> - R1 partially mitigated: per-bearing F1 breakdown 2026-05-25 (mean 0.9218 ± 0.0426 over 11 bearings) suggests F1=0.884 is NOT heavily inflated by bearing-identity leakage. The v6 LOBO IS measured and published (F1 0.352 ± 0.112 vs 0.934 stratified, −0.58 collapse; see `results/lobo_cv_v6/`); the true LOBO of the shipped v1 checkpoint remains scheduled (~30-60h CPU).
> - AIONTrading explorations archived to `_archive/AIONTrading_explorations_2026/` (R4 partial mitigation — focus discipline).
> - Commercial materials §6.31 audit + v2 rewrites completed in `DefinitiveAION/_v2_post_correction/` (mitigates reputational risk on legacy investor docs).
> - SPS Italia 2026 outreach active (R9 first execution attempt — sensor OEM channel partnership).

---

## Existential risks

These could end the project entirely. Mitigate aggressively.

### R1 — Cross-domain F1 doesn't transfer to industrial signal characteristics
**Severity**: 5/5 (kills the wedge)
**Likelihood**: 3/5
**Source**: FEMTO and MFPT are lab benchmarks. Industrial signals have line interference, mounting variability, multi-RPM, EMI, oil contamination on sensors.
**Mitigation**:
- P0.3 line-frequency notch filter (week 3).
- P1.4 order tracking (weeks 16–17).
- Field pilot data (weeks 19–32).
- If industrial F1 < 0.50 zero-shot, the wedge dies. Pivot to fine-tune-from-scratch service model.

### R2 — Augury / Senseye release competing few-shot story
**Severity**: 4/5 (commodifies our wedge)
**Likelihood**: 3/5
**Source**: Public companies see PdM as growth area. Once they catch the few-shot story, they have brand + distribution + capital advantages.
**Mitigation**:
- Be FAST. v2.0 in market by month 6.
- Build OEM partnership moat (Option B in `08_*`) before competitor moves.
- Methodology paper (the 21-retraction story) as credibility moat — hard to copy.

### R3 — Industrial pilot fails publicly
**Severity**: 5/5 (kills credibility)
**Likelihood**: 2/5
**Source**: First pilot blows up — predictions wrong, integration broken, customer goes vocal.
**Mitigation**:
- Pilot 1 in low-stakes setting (factory tape + 50 motors, no safety-critical).
- Customer NDA + escape clause in contract.
- Heavy human-in-the-loop validation in pilot 1 — analyst reviews every alert.
- Document pilot 1 failures internally; iterate before pilot 2.

---

## High-priority operational risks

### R4 — Founder burnout
**Severity**: 5/5 (no founder = no project)
**Likelihood**: 3/5
**Source**: Solo founder, 50+ hour weeks, parallel projects (NormaAI, AION, Prometheus). Pattern of starting new projects on real-world friction (mentor's note).
**Mitigation**:
- Hard 50-hour weekly cap.
- 1 day per week sabbath.
- Mentor 1:1 every 2 weeks for accountability.
- AION-NEXUS gets 60% of weekly hours; other projects get the rest.
- If by month 6 we're not converting any pilots, decide whether to continue or sunset — don't grind on a dead project.

### R5 — IT/OT integration is harder than expected
**Severity**: 3/5
**Likelihood**: 4/5
**Source**: OPC UA / MQTT integration sounds simple but real-world IT-OT environments have legacy DCS, air-gap networks, IEC 62443 audits, vendor-specific OPC UA dialects.
**Mitigation**:
- Allocate 4 weeks for OPC UA (vs nominal 3) to absorb surprises.
- Find one customer who's "easy" first (modern Siemens/Beckhoff DCS) before targeting harder targets.
- Engage external OPC UA consultant if blocked (~$10K).

### R6 — RUL calibration fails to meet criteria
**Severity**: 3/5
**Likelihood**: 3/5
**Source**: Quantile regression doesn't always calibrate well; FEMTO has only 11 bearings; MC dropout may not capture epistemic uncertainty.
**Mitigation**:
- Pre-registered calibration criteria (`06_RUL_DESIGN_NOTE.md`).
- Fallback: ship classification-only without RUL until calibrated.
- Iterate with deep ensembles if MC dropout fails.

### R7 — Substrate experiment lifts F1 but adds complexity
**Severity**: 2/5
**Likelihood**: 3/5
**Source**: Quadratic readout might give a small lift but make the production package harder to maintain.
**Mitigation**:
- Pre-registered FAIL criterion in `05_*`. If lift < 0.05 F1 OR Cohen's d < 0.5, FAIL it. No "small wins, integrate anyway."
- If PASS but lift is small: keep as optional `head_type="quadratic"` feature, default stays linear.

---

## Medium-priority commercial risks

### R8 — Pricing is wrong (too high or too low)
**Severity**: 3/5
**Likelihood**: 3/5
**Source**: $300/asset/year is mid-market estimate. Could be too high for SME, too low to cover support burden.
**Mitigation**:
- First 3 pilots are FREE (or very cheap): use to establish customer pain + willingness-to-pay.
- Iterate pricing based on first 3 customer conversations.
- Be willing to test 2–3 price points in first year.

### R9 — Sensor OEM partnership doesn't materialize in 8 weeks
**Severity**: 4/5
**Likelihood**: 3/5
**Source**: Sensor OEMs are slow-moving Bs. 8 weeks may not be enough.
**Mitigation**:
- Contact 5 OEMs simultaneously (not 1).
- If by week 8 no traction: pivot to direct SaaS (Option A).
- LinkedIn outreach to 50 reliability engineers per week as parallel funnel.

### R10 — Customer wants RUL "now" but we ship in week 10
**Severity**: 2/5
**Likelihood**: 4/5
**Source**: Customer asks "when do you have RUL?" before we have it. Risk: lose deal.
**Mitigation**:
- Honest about timeline ("RUL in v2.0, q2 2026").
- Pre-orders / waitlist for v2.0 at discount.
- Classification-only as "anomaly screening" tool meanwhile.

---

## Lower-priority risks

### R11 — Foundation model for time series obsoletes us
**Severity**: 3/5
**Likelihood**: 3/5 (24+ month horizon)
**Source**: Chronos / Moirai / TimeGPT eventually do PdM zero-shot.
**Mitigation**: integrate as encoder option in v3.0+. We become a thin specialized layer on top.

### R12 — Italian / EU regulatory burden (AI Act, GDPR)
**Severity**: 2/5
**Likelihood**: 3/5
**Source**: AI Act classifies industrial PdM as low-risk but requires transparency disclosures.
**Mitigation**: model card + benchmarks already cover most. Add EU AI Act compliance note in v2.0.

### R13 — Open-source competitor emerges
**Severity**: 3/5
**Likelihood**: 2/5 (low, no signs)
**Source**: Some research lab releases an open-source PdM model with similar few-shot performance.
**Mitigation**: be that lab. Publish Paper 1 + methodology paper. Be the credible academic reference.

### R14 — Cybersecurity incident at pilot customer
**Severity**: 4/5
**Likelihood**: 1/5
**Source**: Vibration data exfiltration + lateral movement.
**Mitigation**: from day 1, on-prem deployment by default; container with non-root user; no inbound network connections required.

### R15 — Patent / IP claim by SKF or Schaeffler
**Severity**: 4/5
**Likelihood**: 2/5
**Source**: Big bearing OEMs hold thousands of patents. They could claim our cross-domain method.
**Mitigation**:
- File a provisional patent application in Q2 (∼$5K).
- Document our method publicly (blog, paper) to establish prior art.
- Apache 2.0 license = patent retaliation clause if we get sued by a contributor.

---

## Risk-weighted ARR estimate

If all risks materialize at expected likelihood:

- Year 1 base: $175K (Option D primary)
- Adjustments:
  - R1 partial materializes (industrial F1 = 0.55 instead of 0.65): −$25K (slower pilot conversion)
  - R5 (IT/OT delay): −$10K (one pilot delayed)
  - R9 (OEM partnership delay): −$30K (no royalty Year 1)
  - R10 (RUL waiting): no change (waitlisted customers count)
- Year-1 risk-adjusted ARR: ~**$110K**

Year 2 risk-adjusted, similar logic:
- Base: $450K
- Adjustments: −$100K to −$150K
- **~$300–350K Year 2**

This is 2–3× lower than the $1M ARR target. Honest probability of hitting $1M in 24 months: **~15–20%**.

---

## Sunset criteria

If by month 12, all of the following are true, **sunset the project**:

- Year-1 ARR < $50K
- Zero pilot customers paying
- Zero OEM/sensor partnerships in conversation
- Zero pilots converted to paid

If 3 of the 4 are true, **strategically pivot** (probably to OEM-only model).

If 2 of the 4 are true, **stay the course** with adjustments.

If 1 of the 4 is true, **double down**.

Decision gate at month 12 is **mandatory**, not optional. The mentor's "premature invariant pattern" warning applies: don't keep grinding on a dead project.
