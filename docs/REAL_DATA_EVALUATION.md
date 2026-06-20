# Real-data evaluation — what the served layer actually does on FEMTO

This is the honest, reproducible account of running AION-NEXUS on **real** FEMTO
run-to-failure data (the dataset ships under `data/FEMTO+Bearing/`). Two scripts
produce it; both write JSON artifacts under `results/`.

```bash
python -m scripts.eval_real_femto      # served pipeline on real FEMTO + coverage
python -m scripts.sota_real_femto      # post-hoc calibration + selective prediction
```

## 1. The model number, stated honestly

| Protocol | Macro-F1 | What it is |
|---|---|---|
| Stratified-temporal split | **0.884** | windows from all bearings mixed into train/test — the split the package itself flags as **leaky**; reproducible via `verify_per_class.py` + the `aion_data` module |
| Per-bearing on the TRUNCATED `Test_set/Test_set` | **0.92** | `per_bearing_f1.json` — reproducible (regen matches 0.9218 exactly), but the challenge Test_set stops near/before failure → an **easier** distribution |
| Per-window on the COMPLETE run-to-failure (`Full_Test_Set`) | **~0.70** | one window per real `acc_*.csv`, all 11 bearings through actual failure — the model **over-predicts 'advanced'** in the failure phase |

> The published numbers **reproduce** — but on the easier regimes (the leaky
> stratified split; the truncated challenge Test_set). **Do not present 0.884 / 0.92
> as accuracy on complete real run-to-failure data**; that figure is ~0.70.
> Positional FEMTO labels are themselves a noisy proxy (a bearing can stay healthy
> for most of its life then fail fast), so part of the gap is label noise.

## 2. Where the value is: the verifier holds up on real data

The model is mediocre, but the verification layer behaves exactly as designed —
it refuses to certify what it cannot stand behind.

| Metric (real FEMTO, 1320 windows) | Value |
|---|---|
| Conformal coverage, in-distribution (target 0.90) | **0.94** ✅ |
| Conformal coverage, cross-bearing LOBO (target 0.90) | **0.93** ✅ (via larger sets) |
| Selective certification | **CERTIFIED 20% of windows @ 0.86 accuracy** vs 0.70 raw |
| Real signed certificate, verified offline (public key only) | **trusted** ✅ |

The verifier enlarges prediction sets when the model is uncertain, so most windows
go to REVIEW and only the confident-correct ones are CERTIFIED. That selective
behaviour — not the raw accuracy — is the product.

> **Read the certified number honestly (two caveats that must travel with it).**
> 1. **It covers a SLICE of the volume, not the bearings.** "CERTIFIED @ 0.86" applies
>    to the ~1-in-5 windows the verifier is willing to stand behind; the other ~80% are
>    routed to human review. The product is **honest triage** ("here is the fraction I can
>    certify, and I refuse the rest"), not "the system certifies the bearings". Always quote
>    the *coverage fraction* alongside the accuracy.
> 2. **The OPEN cheatbench residual has a real magnitude.** Conformal coverage is MARGINAL,
>    not per-instance (`cheatbench` channel `confident_singleton_unverified`, declared OPEN).
>    On real FEMTO that residual is concrete: among CERTIFIED windows, ~**1 − 0.86 ≈ 14%**
>    are wrong — a confident, signed-and-trusted certificate can still carry an incorrect
>    label. The certificate bounds the *rate*, never the correctness of a single call.

## 3. State-of-the-art post-hoc upgrades (no retraining)

`sota_real_femto.py` fits everything on a calibration split and reports on a
disjoint test split:

| Method | Effect on real data |
|---|---|
| **Temperature scaling** (Guo et al. 2017) | ECE **0.217 → 0.032** (6.8× better calibrated). Wired into the served certificate. |
| **Logit adjustment** (Menon et al. 2021) | macro-F1 +0.017 (debiases the 'advanced' over-prediction) |
| **AdaBN** (Li et al. 2018) | macro-F1 +0.029 (re-estimates BatchNorm stats; modest, in-distribution) |

### Selective prediction (risk-coverage)

The frontier way to present a verifier: accuracy at each coverage gate.

| Coverage | Accuracy (baseline → calibrated) |
|---|---|
| 10% | 0.92 → **0.97** |
| 20% | 0.86 → **0.92** |
| 50% | 0.78 → 0.79 |

AURC (area under risk-coverage, lower is better): **0.203 → 0.175**.

### Marginal vs class-conditional conformal

On the imbalanced classes, marginal conformal under-covers `medium` (0.892);
**class-conditional conformal** restores every class to target (`medium` → 0.938,
all classes ≥ 0.908). For imbalanced deployments, prefer class-conditional.

## 4. What is wired into the served product

- **Temperature scaling** is fit at calibration time and applied to both the
  calibration and the serving probabilities (consistent score transform → coverage
  preserved), and the temperature factor is bound tamper-evidently into the
  certificate's `coverage_guarantee`.
- The real calibration basis is honestly labelled: `real-holdout` (leakage-clean
  vs training, gate-enforced) or `real-indistribution` (real data, but not proven
  disjoint from a globally-stratified checkpoint's training).

## 5. Risk control — bound the catastrophic miss (v2.18.0)

Coverage controls the miscoverage rate; PdM needs a bound on the **asymmetric**
error — calling a degraded bearing healthy. `aion_nexus.verify.risk_control` adds:

- **Conformal Risk Control** (Angelopoulos et al. 2022): `E[false-healthy] <= alpha`,
  distribution-free, finite-sample.
- **RCPS** (Bates et al. 2021): `P(false-healthy <= alpha) >= 1 - delta` (Hoeffding).

On real FEMTO at `alpha = 0.05`: the held-out false-healthy rate is bounded as
guaranteed (CRC realized ≈ 0.05, RCPS ≈ 0.006). The served `/predict_certified`
returns the risk-controlled set and the guarantee string (`AION_RISK_ALPHA`,
default 0.05). Note: the v1 model **over-flags** ('advanced'), so its baseline
miss rate is already low — here CRC's value is the *certifiable guarantee*, not a
large reduction; on a model that *under-flags*, CRC tightens the miss rate
materially (synthetic: 0.08 → 0.04). See `examples/13_risk_controlled_certificate.py`.

## 6. Calibrated RUL with conformal intervals (v2.19.0)

`/predict_degradation` gives a coarse 4-class stage; `aion_nexus.rul` gives a
**time-to-failure with a distribution-free interval** via Conformalized Quantile
Regression (Romano et al. 2019). Labels are the TRUE remaining life
`(n_files-1-i)*10 s` on complete run-to-failure bearings, never the positional proxy.

On real FEMTO (`scripts/eval_rul_femto.py`):

| Protocol | Coverage (target 0.90) | MAE | Interval |
|---|---|---|---|
| In-distribution (same-bearing, exchangeable) | **0.89** ✅ | ~1.0 h | ~4.3 h |
| Cross-bearing LOBO | **0.76** (under-covers) | — | ~4.4 h |

The in-distribution interval holds its coverage on real data; cross-bearing it
under-covers because absolute time-to-failure does not transfer to a never-seen
bearing — and the estimate **says so** (`coverage_caveat`). The honest deployment
recipe is per-asset calibration / online updating. Served via `/predict_rul`
(`AION_RUL_ARTIFACT`); demo in `examples/14_calibrated_rul.py`.

## 7. Honesty notes (workspace 6.31)

- All post-hoc parameters are fit on calibration only and reported on a disjoint
  test split.
- v1 trained globally-stratified (saw all these bearings), so no FEMTO bearing is
  leakage-clean vs training — calibration here is `real-indistribution`. A true
  `real-holdout` artifact requires a dataset the model never trained on.
- Physics on FEMTO is exploratory only: FEMTO does not publish the bearing
  geometry, and the labels are positional (not fault-type), so the physics second
  opinion returns WEAK and is not a sellable claim on this dataset.
