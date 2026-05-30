# Task Semantic Mismatch — CWRU Case Study

A short methodological note on why AION-NEXUS does not transfer well to the Case Western Reserve University (CWRU) bearing dataset, and what this teaches.

## TL;DR

Cross-domain transfer learning assumes that the source task (FEMTO severity) and the target task (CWRU fault-location, or CWRU fault-size) measure compatible underlying concepts. **They don't.** The Spearman rank correlation between CWRU fault size and vibration severity is **−0.30**: as fault size grows, vibration severity actually decreases on average, because larger faults are confounded with lighter loads in the CWRU experimental matrix.

Forcing a label mapping FEMTO-severity → CWRU-anything yields F1 ≈ 0.34, not because the model fails to generalize, but because **the labels do not measure the same thing**.

## Setup

CWRU labels available:
- **Fault size**: 0.007", 0.014", 0.021", 0.028" (physical defect diameter)
- **Fault location**: ball, inner race, outer race
- **Load**: 0, 1, 2, 3 hp
- **RPM**: 1797, 1772, 1750, 1730

FEMTO labels:
- **Severity**: continuous 0–100% bearing-life remaining → 4-class (normal / early / medium / advanced)

## Tested mappings

| Mapping rule | Logic | F1 |
|---|---|---|
| Severity-by-size (proportional) | 0.007" → early, 0.014" → medium, 0.021"+ → advanced | 0.359 |
| Severity-by-life-fraction | unknown, no run-to-failure data in CWRU | n/a |
| Location-as-class | ball→0, inner→1, outer→2 | 0.341 |

Both approaches collapse to ~0.34 F1 — barely above chance for 4-class (0.25).

## Why this happens

Vibration severity is a function of:

1. **Defect dimension** — bigger defects → more impact energy
2. **Defect position** — inner-race defects modulate carrier signal differently from outer-race
3. **Operating load** — higher load → bigger contact force → bigger impact
4. **Operating speed** — fault frequencies scale linearly with RPM
5. **Defect type** — pits vs spalls vs cracks have different vibration signatures

The CWRU dataset varies (1) and (2) and (3) and (4) **simultaneously**. A 0.021" outer-race fault at 0 hp can produce LESS vibration than a 0.007" inner-race fault at 3 hp.

The Spearman correlation of fault-size to RMS vibration on the dataset is r = −0.30 (negative). This is a multi-confounder design where the fault-size label captures **only one of five** sources of vibration severity.

## Implication for the field

This is broader than CWRU: any cross-domain bearing study that compares "severity" labels across datasets must verify that the labels measure compatible quantities. Often they don't:

| Dataset | Label semantic |
|---|---|
| FEMTO | Run-to-failure life fraction (continuous degradation) |
| CWRU | Static fault dimension (snapshot defect size) |
| Paderborn | Combined damage + load + speed conditions |
| MFPT | Healthy vs faulty (binary, with type sub-labels) |
| NASA IMS | Run-to-failure (compatible with FEMTO) |
| PHM 2012 | Run-to-failure prediction (RUL regression) |

**Compatible cross-domain pairs** (similar label semantics): FEMTO ↔ NASA IMS ↔ PHM 2012.
**Incompatible** (different semantics): FEMTO ↔ CWRU ↔ MFPT (partially).

## Methodological recommendation

Before any cross-domain bearing study:

1. Define the underlying physical quantity the labels are supposed to measure.
2. Compute Spearman correlation between the source-label and target-label proxies on a small calibration set.
3. If correlation < 0.5, the label spaces are not aligned — do NOT report cross-domain numbers as if they were.
4. If you must transfer, retrain rather than zero-shot evaluate.

## What AION-NEXUS does on CWRU

- **Zero-shot CWRU severity**: F1 ≈ 0.36 (documented poor performance, not a failure of the model)
- **Zero-shot CWRU location**: F1 ≈ 0.34 (different task entirely; not what FEMTO was trained for)
- **Few-shot adaptation on CWRU**: not currently supported in v1.0 (CWRU labels would require a new classification head); planned for v1.1

If you need CWRU support, retrain from scratch on CWRU labels — AION-NEXUS architecture is suitable but the v1.0 trained weights are not.
