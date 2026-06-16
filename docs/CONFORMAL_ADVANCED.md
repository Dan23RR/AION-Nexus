# Conditional conformal — past the marginal guarantee

`aion_nexus.verify` (v2.8.0) adds four calibrators that go beyond the base
`ConformalCalibrator`'s *marginal* coverage. The red-team's kill-shot #5 was
exactly this: marginal coverage is the floor, and it is valid **only under
exchangeability** — cross-bearing / cross-machine deployment breaks it. These
methods turn that weakness into the claimable frontier (Barber, Candès, Ramdas &
Tibshirani, *"Conformal prediction beyond exchangeability"*, 2023), each with a
**stronger or different guarantee** and each **honest about its own assumption**.

> The guarantees below are not asserted in prose alone. Each is **proven by an
> empirical-coverage simulation** in `tests/test_conformal_advanced.py` that also
> shows the plain marginal calibrator FAILING the same scenario.

## The four methods

| Class | Guarantee | Assumption (`coverage_valid_under`) | When to reach for it |
|---|---|---|---|
| `ClassConditionalConformalCalibrator` | `P(Y∈C(X) \| Y=c) ≥ 1−α` **per class** | within-class exchangeability | a rare-but-critical fault class is silently under-covered by the marginal average (Sadinle, Lei & Wasserman 2019) |
| `MondrianConformalCalibrator` | `P(Y∈C(X) \| group(X)=g) ≥ 1−α` **per group** | within-group exchangeability | calibrate **per bearing / per operating regime**; each group gets its own valid guarantee even across non-exchangeable groups |
| `WeightedConformalCalibrator` | `P(Y∈C(X)) ≥ 1−α` under **covariate shift** | known/estimated likelihood ratio `w(x)=dP_test/dP_cal` | the test regime differs from calibration and you can estimate the shift weights (Tibshirani et al. 2019); `w≡1` reduces to standard CP |
| `AdaptiveConformalGate` (ACI) | empirical miscoverage frequency `→ α` **long-run** | **none** — holds for arbitrary (even adversarial) drift, given online label feedback | a non-stationary stream where degradation evolves over time (Gibbs & Candès 2021) |

## Honesty (workspace 6.31)

None of these is a proof of correctness, and none escapes its own assumption:

- class- / group-conditional CP still needs *within*-class / *within*-group
  exchangeability — it slices the exchangeability requirement, it does not remove it;
- weighted CP is only as good as the weight estimate (it degrades gracefully toward
  the unweighted result, and equals it at `w≡1`);
- ACI guarantees a **long-run average** coverage, not a per-step one, and needs the
  realised labels as feedback to adapt; if the model is so accurate that even
  size-1 sets over-cover, ACI honestly pins at the smallest sets and over-covers
  (you cannot hit α miscoverage without emitting empty sets, which we never do).

Every calibrator exposes its `guarantee` and `coverage_valid_under` strings. All
conformal evidence remains the `EMPIRICAL` assurance tier — statistical, never a
proof — and composes through the assurance lattice unchanged.

## Usage

```python
from aion_nexus.verify import (
    ClassConditionalConformalCalibrator, MondrianConformalCalibrator,
    WeightedConformalCalibrator, AdaptiveConformalGate,
)

# Per-class coverage (rescue a rare hard class).
cc = ClassConditionalConformalCalibrator(alpha=0.1).fit(probs_cal, labels_cal)
sets = cc.predict_set(probs_test)
cc.small_classes          # classes with too few calibration points (flagged, not hidden)

# Per-bearing coverage (the honest answer to the cross-bearing break).
mon = MondrianConformalCalibrator(alpha=0.1, score="aps").fit(probs_cal, labels_cal, bearing_id_cal)
sets = mon.predict_set(probs_test, bearing_id_test)
mon.fell_back_groups      # test groups never seen in calibration (no per-group claim)

# Covariate shift with estimated weights w(x) = dP_test/dP_cal.
wcp = WeightedConformalCalibrator(alpha=0.1).fit(probs_cal, labels_cal, weight_calib=w_cal)
sets = wcp.predict_set(probs_test, weight_test=w_test)

# Online, non-stationary stream (ACI): adapt on realised-label feedback.
aci = AdaptiveConformalGate(alpha=0.1, gamma=0.05).fit(probs_cal, labels_cal)
for probs_t, y_t in stream:
    pred_set = aci.step(probs_t, true_label=y_t)
aci.empirical_miscoverage  # converges to alpha despite drift
```

Runnable walkthrough: [`examples/08_conditional_conformal.py`](../examples/08_conditional_conformal.py).

## Deploy-time covariate shift — estimate the weights (v2.12.0)

`WeightedConformalCalibrator` recovers coverage under covariate shift **if** you
supply the likelihood-ratio weights `w(x)=dP_target/dP_cal`. At install time you do
not have them — only unlabeled windows from the new machine. `aion_nexus.verify`
estimates them:

```python
from aion_nexus.verify import deploy_weighted_calibrator

# cal_features / target_features: model embeddings, signal features, or the
# RPM-invariant physics order-SNR features (aion_nexus.physics.fault_order_energy).
cal, weight_fn = deploy_weighted_calibrator(
    probs_cal, labels_cal, cal_features, target_features, alpha=0.1)
sets = cal.predict_set(probs_test, weight_test=weight_fn(test_features))
```

`estimate_covariate_shift_weights()` does it by probabilistic classification
(label calibration 0 / target 1, fit a logistic regression, `w ∝ odds`; Bickel et
al. 2007). This converts the cross-machine wall **into the product**: a certified,
honestly-widened interval instead of a hidden under-coverage. **Honesty (6.31):**
the estimate is an approximation — recovered coverage is only as good as the
features capturing the shift and the classifier fitting it; with no shift it
reduces to standard split conformal. It is proven by the deploy test in
`tests/test_conformal_advanced.py` (estimated, not oracle, weights restore coverage).

## How it composes with the rest

- **The certificate records the guarantee (v2.9.0).** `Verifier.certify(...,
  conformal_method=, coverage_guarantee=)` stamps WHICH method produced the verdict
  and WHAT it guarantees, bound **tamper-evidently** into `content_hash` (hashed only
  when present, so pre-2.9 certificates are unaffected). Forging or upgrading the
  claim on the wire breaks the hash.
- **The factory bridge surfaces it.** The Sparkplug B payload carries
  `AION/conformal_method` / `AION/coverage_guarantee` metrics and the OPC UA model the
  `ConformalMethod` / `CoverageGuarantee` variables — so a consumer sees not just the
  verdict but the guarantee it holds, and can re-verify the claim offline.
- **The base `Verifier`** wraps the marginal `ConformalCalibrator`; the
  class-conditional calibrator has the same `predict_set(probs)` signature and is a
  drop-in for callers that want per-class coverage in the served verdict. Mondrian /
  weighted / ACI carry extra inputs (group / weight / online feedback) and are used
  directly.
