# Changelog

All notable changes to AION-NEXUS will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.14.0] — 2026-06-16 (Ride a foundation encoder, don't out-pretrain one)

The architecture-leap roadmap's #5. The research showed the binding constraint on cross-machine
generalization is DATA DIVERSITY (number of distinct bearings pretrained on), not model capacity — so a
solo founder cannot out-pretrain a foundation encoder like UniFault (>9B points, 10 datasets,
MIT-licensed). The honest move is to RIDE it and own the verification + adaptation layer. This adds a
model-agnostic adapter that wraps ANY frozen encoder into AION's certified pipeline. No breaking changes.

### Added — `aion_nexus.foundation`
- **`ExternalEncoderAdapter`**: wraps a FROZEN encoder (a torch module or plain callable mapping a
  vibration window `[B, C, N]` to an embedding `[B, D]`) + a few-shot linear head, with the same forward
  contract as v1/v3/v6 (`{"logits", "features"}`). It harmonises AION's `[2, 2560]` @ 25.6 kHz input to
  the encoder's expected length/channels, and keeps the frozen encoder in eval (BatchNorm running stats
  never drift) even while the head trains. Drops into `InferenceEngine`, `FewShotAdapter` (architecture
  `"ext"`), the certified serving path, and the physics second opinion — the same adapter wraps UniFault,
  MOMENT, Mantis, a customer's encoder, or AION's own. *Verify / adapt ANY model, don't compete with it.*
- **`wrap_foundation_encoder(...)`**: convenience factory (freezes the encoder, sets up harmonisation).

### Verification
- 11 tests: forward contract, encoder frozen / head trainable, frozen encoder stays eval in train mode
  (BN running stats unchanged across few-shot adaptation), input harmonisation (resample length, reduce
  channels, reject channel expansion), few-shot adapts the head only, InferenceEngine integration, and a
  plain-callable encoder. A synthetic encoder stands in for UniFault (no weights needed). Suite
  **402 → 413**; ruff clean.

### Honesty (workspace 6.31)
The adapter adds no pretraining diversity of its own — it rides the encoder's. Riding a foundation
encoder only fixes the diversity deficit IF that encoder did not see your held-out bearings in
pretraining; otherwise a "LOBO" number is transductive (exactly v3's flaw). The adapter cannot verify
that for a black-box encoder, so the honest inductive-LOBO number is the caller's to establish (use
`evaluate_leave_one_group_out` from v2.13.0). The few-shot head is a linear probe on a frozen encoder:
the ceiling is the encoder's representation quality, not magic.

## [2.13.0] — 2026-06-16 (Leakage-free evaluation as a feature: the honest number, attested)

The architecture-leap roadmap's #2. Evaluation leakage is endemic in bearing-fault benchmarking
(Hendriks et al. 2022: 40/41 CWRU studies leaked; arXiv:2509.22267: 95-99% under random splits
collapses to 35-60% under bearing-disjoint splits) — so buyers cannot trust vendor accuracy numbers,
the precise opening for an independent verifier. This ships honest evaluation AS a feature. No
breaking changes; zero modeling risk.

### Added — `aion_nexus.evaluation`
- **`check_group_disjoint(train_groups, test_groups)`**: the machine-checkable leakage detector —
  given the bearing/recording/machine id of every sample, it proves whether a claimed split is
  group-disjoint, so a vendor's "99%" can be audited with a yes/no.
- **`evaluate_leave_one_group_out(predict_fn, X, y, groups)`**: a model-agnostic leave-one-group-out
  harness that reports an HONEST INTERVAL (mean ± std across folds), not a single stratified number,
  with prevalence-independent metrics (`macro_auroc`, `macro_f1`, `per_class_recall`, `honest_interval`).
- **`EvaluationReport`** + **`verify_evaluation_report`**: binds the protocol + leakage check + honest
  intervals into a `content_hash`, signable with the same Ed25519/HMAC primitives as the per-decision
  certificate — so the "measured leakage-free" claim is tamper-evident and third-party-verifiable offline.

### Verification
- 9 tests, incl. the headline: the SAME model/data give macro-F1 ~1.0 under a leaky random split but
  ~0.30 ± 0.18 under leave-one-bearing-out (the documented collapse), the detector flags the leak, and
  the signed report verifies + breaks on a forged number. Suite **393 → 402**; ruff clean.
  Example: `examples/10_leakage_free_eval.py`.

### Honesty (workspace 6.31)
The leakage detector verifies disjointness of the GROUP IDS the caller supplies — it cannot detect
leakage those ids do not capture (overlapping sliding windows if you only grouped by bearing,
operating-condition leakage if you only grouped by machine). The report attests the PROTOCOL and
reports the numbers honestly; it does not claim the numbers are good, only that they were measured
the way it says.

## [2.12.0] — 2026-06-16 (Covariate-shift conformal at deploy: estimate the weights from unlabeled target data)

The architecture-leap roadmap's #3: convert the cross-machine generalization wall INTO the product —
a certified, honestly-WIDENED prediction interval under shift instead of a hidden under-coverage. The
`WeightedConformalCalibrator` (v2.8.0) recovers coverage under covariate shift, but it needs the
likelihood-ratio weights `w(x)=dP_target/dP_cal` SUPPLIED — which at install time you do not have. This
release estimates them from UNLABELED target-machine data, the realistic deploy case. No breaking changes.

### Added — `aion_nexus.verify`
- **`estimate_covariate_shift_weights(cal_features, target_features)`**: estimates `w(x)` by probabilistic
  classification (Bickel et al. 2007 / Sugiyama density-ratio-by-classification — label calibration 0 /
  target 1, fit a logistic regression, read `w ∝ odds`), pure-numpy (no sklearn/torch). Returns the
  calibration weights + a `weight_fn` for new test points. Features can be model embeddings, signal
  features, or the RPM-invariant physics order-SNR features from v2.11.0 (`fault_order_energy`).
- **`deploy_weighted_calibrator(probs_cal, labels_cal, cal_features, target_features)`**: one call —
  estimate the weights and return a fitted `WeightedConformalCalibrator` + the `weight_fn`, so a deployed
  model gets coverage-valid sets on a new machine from only unlabeled target windows.

### Verification
- 2 new tests: under a feature-driven covariate shift, vanilla split-CP under-covers while the
  **ESTIMATED-weight** CP (weights inferred from unlabeled features, not oracle-supplied) recovers nominal
  coverage; and with NO shift the estimated weights stay ~uniform (it does not invent a shift). Suite
  **391 → 393**; ruff clean.

### Honesty (workspace 6.31)
The estimate is an APPROXIMATION: recovered coverage is only as good as (a) the features capturing the
shift and (b) the classifier fitting it; with no shift it reduces to standard split conformal. It does
not manufacture a guarantee the data cannot support — the honest, certified, widened interval is the
product, not a claim that the wall is gone.

## [2.11.0] — 2026-06-16 (Physics front-end + model-agnostic second-opinion verifier)

A field-wide research sweep (UniFault, DGFDBenchmark, Hendriks et al., arXiv:2509.22267) converged
on one diagnosis: bearing models collapse cross-machine (in-distribution 0.88 → LOBO 0.35) largely
because they read RAW time-domain windows and IGNORE the physics already available — the shaft rpm
and the bearing geometry — and AION's own `forward()` accepts `rpm`/`geometry` and discards them.
This release closes that gap with classical DSP and turns the physics into a verifier asset. Full
analysis + ranked roadmap: `AION_NEXUS_RD/20_ARCHITECTURE_LEAP.md`. No breaking changes.

### Added — `aion_nexus.physics`
- **RPM-invariant representation.** `BearingGeometry.fault_orders()` gives the BPFO/BPFI/BSF/FTF
  characteristic frequencies in ORDERS (geometry-only, rpm-independent — validated against the
  published SKF 6205 values). `order_spectrum()` (with computed order tracking / angular resampling
  for varying speed) places a fault peak at the SAME order at every speed — invariance *by
  construction*, the most leakage-robust input the field has measured. `envelope_spectrum()`,
  `order_resample()`, `fault_order_energy()` (a speed-independent harmonic-comb SNR per fault family).
- **Model-agnostic second opinion.** `physics_consistency()` asks what no learned model answers about
  itself — *is the energy actually at the claimed defect order on THIS machine?* — using only rpm +
  geometry, so it checks ANY model's claim. Returns CONFIRM / WEAK / CONTRADICT / INDETERMINATE, and
  `PhysicsVerdict.as_component()` composes it with a conformal certificate via `compose_certificates`:
  a CERTIFIED model + a CONTRADICT physics check drops to **REVIEW** — the cross-machine failure mode
  caught and routed to a human, not silently certified.

### Verification
- 11 tests: kinematics vs published SKF 6205, the rpm-invariance proof (same fault, two speeds → same
  order / different Hz), order tracking under a speed sweep, CONFIRM/CONTRADICT/WEAK/INDETERMINATE,
  white-noise gives no false positive (local-prominence + harmonic-comb SNR), and composition with the
  certificate. Suite **380 → 391**; ruff clean. Example: `examples/09_physics_verifier.py`.

### Honesty (workspace 6.31)
Order tracking removes the cross-SPEED (same-machine) shift deterministically; it does NOT by itself
cross the cross-MACHINE wall (sensor placement, transmission path, artificial-vs-real damage remain) —
the literature puts physics-only LOBO around ~0.5-0.6, not a sellable >0.85, because DIVERSITY (number
of distinct bearings), not capacity, is the binding constraint. The physics consistency tier is
EMPIRICAL (a thresholded SNR heuristic), never a proof. The value is the honest second opinion and the
abstention, not an accuracy cure. Note: envelope/order analysis needs more shaft revolutions than the
0.1 s window, so the physics front-end implies longer captures (or aggregation) — an input-contract change.

## [2.10.0] — 2026-06-16 (EU AI Act Annex IV technical-documentation evidence map)

The strategic wedge: the EU AI Act's harmonised standards (CEN-CENELEC JTC 21) are not yet in
the Official Journal, so a provider who can assemble Annex IV technical documentation early
occupies open ground. This release adds an **honest Annex IV evidence map** — for each of the
nine Annex IV points (Article 11), which concrete AION artefacts can supply it and what the
provider must author — built with the same §6.31 discipline as the rest of `aion_nexus.compliance`.

### Added — `aion_nexus.compliance`
- **`annex_iv_dossier(model_metadata=None, *, certificate=None)`**: maps all nine Annex IV
  points to `aion_provides` / `deployer_must_supply` / `status` / a NON-EMPTY `limitation`.
  Caller metadata (intended purpose, architecture, datasets, standards, ...) fills sections;
  anything absent is reported as provider-owned rather than invented. A `readiness` block counts
  AION-supplied sections and **explicitly disclaims** being a conformity measure.
- **`annex_iv_card(...)`**: the same as a Markdown dossier skeleton.
- **`scripts/generate_annex_iv.py`**: CLI to emit the map (Markdown or JSON), optionally from a
  metadata file. Both exported at the top level (`aion_nexus.annex_iv_dossier` / `annex_iv_card`).

### Honesty (workspace 6.31)
This is a documentation SKELETON to accelerate a provider's own dossier — **NOT** the technical
documentation, **NOT** a declaration of conformity, and **NOT** a statement that the system meets
Annex IV. Point 8 (declaration of conformity) and the Article 9 risk management system are marked
provider-owned; the forbidden-claim guard is extended so the generated text can never say
"compliant" / "conforme" / "conformità". High-risk classification and the duty to produce Annex IV
documentation depend on the use context and a formal assessment supported by legal review.

### Verification
- 7 new tests (all nine points present with required keys, no forbidden claims across metadata
  shapes, readiness disclaims conformity, declaration-of-conformity is provider-owned, metadata
  threading, certificate identity threading, card renders). Suite **373 → 380**; ruff clean.

## [2.9.0] — 2026-06-16 (The conformal guarantee, bound into the certificate and surfaced in the factory)

v2.8.0 added conditional conformal as a library — but a verifier whose stronger guarantee
lives only in a library repeats the demolition's kill-shot #3 ("the verifier is not in the
product"). This release WIRES the guarantee into the deployable: the certificate now records
**which** conformal method produced the verdict and **what** coverage it guarantees, bound
**tamper-evidently** into the signature, and the factory bridge publishes it. Backward-compatible
(no signature/hash break for existing certificates).

### Added — the conformal claim, in the certificate
- `Certificate` gains `conformal_method` and `coverage_guarantee` (e.g. `"class-conditional"` /
  `"P(Y∈C|Y=c) ≥ 1−α"`). They are hashed into `content_hash` **ONLY when present**, so a
  certificate minted WITHOUT them hashes **byte-identically** to a pre-2.9 certificate (full
  backward compatibility), while a certificate that DOES claim a conditional guarantee makes
  that claim tamper-evident: forging "per-class coverage" onto a marginal verdict, or upgrading
  the claim on the wire, breaks the hash. `CERT_SCHEMA_VERSION` → `1.2` (provenance only; no
  signing-payload change, so v2.6–2.8 certificates still verify).
- `Verifier.certify(..., conformal_method=, coverage_guarantee=)` stamps the claim — a caller
  using a class-conditional / Mondrian / weighted / ACI calibrator records the guarantee its
  verdict carries.

### Added — surfaced on the factory bus
- `FactoryVerdict` carries `conformal_method` / `coverage_guarantee`; the Sparkplug B payload
  adds `AION/conformal_method` and `AION/coverage_guarantee` metrics, and the OPC UA model adds
  `ConformalMethod` / `CoverageGuarantee` variables. A consumer on the bus sees not just the
  verdict but the GUARANTEE it carries — and can still re-verify the (tamper-evident) claim
  offline with the public key.

### Verification
- 5 new tests: backward-compatible hashing (absent field ⇒ identical hash), the claim changes
  the hash when set, the claim is tamper-evident (forged upgrade ⇒ integrity fails / not
  trusted), `Verifier.certify` stamps it, and the bridge surfaces it tamper-evidently. Suite
  **368 → 373**; ruff clean.

## [2.8.0] — 2026-06-16 (Conditional conformal: past the marginal guarantee)

Closes the demolition's kill-shot #5 on the verifier axis: conformal coverage was
**marginal only**, valid solely under exchangeability — and cross-bearing / cross-machine
deployment breaks it. This release turns that weakness into the claimable frontier (Barber,
Candès, Ramdas & Tibshirani, *"Conformal prediction beyond exchangeability"*, 2023): four
calibrators with **stronger or different** guarantees, each HONEST about its own assumption
and each **proven by empirical-coverage simulation** (not prose). No breaking changes; the
base `ConformalCalibrator` is untouched. See `docs/CONFORMAL_ADVANCED.md`.

### Added — `aion_nexus.verify.conformal_advanced`
- **`ClassConditionalConformalCalibrator`** — per-class coverage
  `P(Y∈C(X) | Y=c) ≥ 1−α` for every class (Sadinle, Lei & Wasserman 2019). Stops a rare,
  critical fault class being silently under-covered to prop up the marginal average.
  `small_classes` flags classes with too few calibration points (never hidden).
- **`MondrianConformalCalibrator`** — per-group coverage
  `P(Y∈C(X) | group(X)=g) ≥ 1−α` for every covariate-defined group (per bearing / regime):
  the honest answer to the cross-bearing break. `fell_back_groups` records test groups never
  calibrated (no per-group claim for those).
- **`WeightedConformalCalibrator`** — coverage under covariate shift with weighted quantiles
  (Tibshirani, Foygel Barber, Candès & Ramdas 2019); reduces EXACTLY to standard split CP at
  `w≡1`.
- **`AdaptiveConformalGate`** (ACI, Gibbs & Candès 2021) — online long-run coverage on a
  NON-stationary stream with **no** exchangeability assumption; the realised miscoverage
  frequency converges to α for arbitrary drift, given online label feedback.
- `finite_sample_level()` helper; all four exported from `aion_nexus.verify`.

### Verification (the guarantees are proven, not claimed)
- `tests/test_conformal_advanced.py` (10 tests) SIMULATES coverage and asserts each method
  delivers its guarantee AND that the marginal calibrator FAILS the same scenario: a rare
  hard class (marginal 0.15 → class-conditional 0.93), a hard regime (marginal 0.80 →
  Mondrian 0.90), covariate shift recovery, and ACI on a drifting stream (fixed gate 0.19 →
  ACI 0.11 miscoverage). Suite **358 → 368**; ruff clean.
- `examples/08_conditional_conformal.py`: the same three contrasts, runnable with no
  checkpoint and no optional dependency.

### Honesty (workspace 6.31)
None of these is a proof of correctness or escapes its own assumption: class/group-conditional
CP still needs *within*-class / *within*-group exchangeability; weighted CP is only as good as
the weight estimate; ACI guarantees a LONG-RUN AVERAGE (not per-step) and needs label feedback.
Each calibrator exposes its `guarantee` and `coverage_valid_under`. All conformal evidence
stays the `EMPIRICAL` assurance tier — statistical, never a proof.

## [2.7.0] — 2026-06-16 (Orizzonte 1: the signed verdict enters the factory's own protocol fabric)

The v2.6.0 demolition's hardest line was kill-shot #1: **0 lines of OPC UA / MQTT
Sparkplug** — produce a certificate and the PLC / historian / UNS does not know AION
exists. This release closes it: `aion_nexus.connect` carries the **signed certificate**
onto Sparkplug B (MQTT / Unified Namespace) and OPC UA (SCADA / historian), so a third
party verifies AION's verdict **offline, off their own bus, with the public key alone** —
the one thing no incumbent ships. See `AION_NEXUS_RD/18_PATH_TO_UNIGNORABLE.md` and
`docs/FACTORY_INTEGRATION.md`. No breaking changes (the verify layer is unchanged).

### Added — `aion_nexus.connect` (the factory bridge)
- **`to_factory_verdict(cert)` → `FactoryVerdict`**: the dependency-free spine that
  separates the **diagnosis** (`normal…advanced`) from the **trust verdict**
  (`CERTIFIED/REVIEW/ABSTAIN`) and carries the whole signed certificate for offline
  re-verification.
- **Sparkplug B**: `build_sparkplug_payload` / `encode_payload` / `decode_payload` emit a
  faithful **Sparkplug B 3.0 `Payload`** protobuf (the full signed cert rides as the
  `AION/certificate` metric), via an in-package protobuf codec with **no runtime
  dependency**. `SparkplugPublisher` publishes to a live MQTT broker over the standard
  `spBv1.0/{group}/{type}/{edge}/{device}` namespace (seq wrapping, `NDEATH` last-will).
- **OPC UA**: `build_condition_model` maps a verdict onto **OPC UA Alarms & Conditions
  (Part 9)** semantics; `CertifiedConditionMonitoringServer` exposes it as a live `asyncua`
  address space (`AionVerification` node, incl. the `Certificate` variable).
- `examples/07_factory_bridge.py`: publish → decode-off-the-bus → verify → tamper-caught →
  honesty gate → OPC UA view, end-to-end with **zero** optional dependencies.

### Added — the honesty gate, in the factory's language (workspace 6.31)
- An **`ABSTAIN`** (OOD / low confidence) or **`REVIEW`** (ambiguous conformal set) can
  **never** present as a high-severity, machine-stop alarm: severity is **capped by the
  trust verdict**. `ABSTAIN → OPC UA Quality = Uncertain` (OPC UA's own "I am not sure"),
  low severity, **not** an active alarm. An unsigned certificate is `actionable=false` /
  Quality Uncertain. Mirrors the serving pipeline's no-escalation-on-abstain rule.

### Verification & supply chain
- **Wire-compatibility proven**: `tests/test_connect.py` decodes our Sparkplug bytes with
  **Google's reference protobuf decoder** and re-verifies the extracted certificate as
  `trusted` (skipped if `protobuf` is absent — never a hard test dependency). 23 new tests;
  suite **335 → 358**.
- **Optional, transport-split extras**: `[factory-mqtt]` (paho-mqtt; permissive licence),
  `[factory-opcua]` (asyncua, **LGPL-3.0** — disclosed in `SECURITY.md`, outside the
  Apache/BSD/MIT core guarantee), `[factory]` (both). Building payloads/models needs neither.

## [2.6.0] — 2026-06-15 (Orizzonte 0: the verification weapon, wired into the product and enterprise-grade)

A demolition against the bar "could Siemens/Beckhoff/SKF ignore this?" found that the one
defensible asset — independent signed verification — was shipped **unloaded and not wired
into the running product** (the server emitted unsigned JSON). This release makes it real,
secure-by-default, and attestable end-to-end. See `AION_NEXUS_RD/18_PATH_TO_UNIGNORABLE.md`.

### ⚠️ Breaking change
- **Signature format**: signatures now cover a `signing_payload` (`content_hash | not_before
  | valid_until | jti | key_id`) instead of the bare `content_hash`, so expiry/identity are
  tamper-evident. A certificate signed by v2.5.0 will **not** verify under v2.6.0.
  `CERT_SCHEMA_VERSION` → `1.1`. The decision `content_hash` itself is unchanged and stays
  deterministic (reproducibility preserved).

### Added — the certificate, wired into the running product
- **`POST /predict_certified`**: runs the `/predict` pipeline, then emits a **signed,
  auditable `Certificate`**, appends it to the hash-chained store, and returns
  `{prediction, certificate, pubkey, verdict}`. With `VERIFY_ED25519_SEED` set it is
  Ed25519-signed and third-party verifiable with the public key alone.
- **`POST /verify`**: re-runs `verify_certificate` (integrity / authenticity / trusted /
  expired) so an auditor can confirm a verdict offline.
- `examples/06_certified_serving.py`: end-to-end — issuer signs, an auditor with **only the
  public key** verifies, tamper/expiry are rejected.

### Added — key management at the enterprise bar
- **Entropy floor enforced at the product boundary**: a weak `VERIFY_ED25519_SEED` (the
  red-team's `1234` kill-shot) is **refused with 503** rather than minting a brute-forceable
  key. `generate_seed()` (CSPRNG), `assert_strong_seed()`, and a pluggable `Signer`
  interface (`LocalEd25519Signer` strict-by-default, `HmacSigner`, KMS/HSM-ready).
- **Signed expiry / anti-replay**: certificates carry `not_before` / `valid_until` / `jti` /
  `key_id` bound into the signature (`AION_CERT_TTL_SECONDS`, default 24h); expired or
  replayed certs fail verification. `valid_until` etc. are NOT in `content_hash`, so
  reproducibility is preserved (the design rule that makes anti-replay + determinism coexist).
- **Secure-by-default**: `AION_REQUIRE_SIGNED_CERT=1` refuses unsigned certs; otherwise an
  unsigned cert ships `authentication=NONE` with an explicit `warning` (never a silent claim).

### Added — attest WHICH weights and WHICH dependencies
- **Checkpoint pinning**: `AION_CHECKPOINT_SHA256` makes the server refuse to start if the
  live checkpoint's hash differs; `AION_REQUIRE_CHECKPOINT_PIN=1` requires the pin. Hash
  exposed in `/health`.
- **Supply chain**: `scripts/generate_sbom.py` (CycloneDX, stdlib fallback),
  `audit_supply_chain.py --strict` is **fail-closed**, `cryptography` upper-bounded,
  `docs/SUPPLY_CHAIN.md` (hash-pinning + cosign/SLSA guidance). "Verifiability applies to us too."

### Fixed
- `/predict_batch`: aggregate byte budget (`AION_MAX_BATCH_BYTES`, default 50 MiB) and an
  explicit `ndim==2` check (a 1-column CSV now returns a clean 400, not an accidental
  `IndexError`).

### Changed
- Version → **2.6.0**. Tests: 289 → **334**. `ruff` clean. §6.31 honesty enforced by tests
  (incl. a security-regression test that the weak-seed boundary stays closed).

## [2.5.0] — 2026-06-15 (Substrate Core convergence: asymmetric signatures, typed assurance, measured cheating surface)

Converges the best of the sibling `substrate_core` verification kernel (same author,
the other "Substrate Core of Verifier Labs") into the public `aion_nexus.verify`. These
three additions close the two weaknesses the red team flagged about the certificate and
add a capability AION had no analog for. See `AION_NEXUS_RD/17_SUBSTRATE_INTEGRATION_ANALYSIS.md`.

### Added — Ed25519 asymmetric signatures (independent verification, for real)
- New `aion_nexus.verify.signing`: `ed25519_sign` / `ed25519_verify` /
  `ed25519_pubkey_from_seed` (seed → deterministic keypair, RFC 8032), plus the existing
  HMAC helpers. **The verifier holds only the public key and cannot forge** — the property
  that makes *independent* verification credible (a customer/insurer/auditor verifies with
  the public key alone). `Certificate` gains `authentication = "Ed25519"` and a `pubkey`
  field; `seal()` precedence is explicit > `VERIFY_ED25519_SEED` > `VERIFY_HMAC_KEY` > NONE.
- **Honesty nuance enforced in code**: verifying against the *embedded* public key proves
  only self-consistency → `authenticity = "SELF-SIGNED"`, **not** trusted. Trust requires
  the issuer's *expected* public key (`verify_certificate(cert, expected_pubkey=...)` or env
  `VERIFY_ED25519_PUBKEY`). A stripped-signature downgrade is detected and never trusted.
- `cryptography` added as a dependency.

### Added — typed assurance lattice (overclaim made structurally impossible)
- New `aion_nexus.verify.assurance`: ordered tiers `none < empirical < bounded <
  proven-spec < proven` with `weakest()` / `strongest()` and a rule-of-three residual-risk
  estimate (with its caveat). A conformal verdict is **always `empirical`** (statistical,
  exchangeability-dependent); the tier is in the **hashed** payload, so it can't be
  silently upgraded. `compose_certificates()` takes the **weakest link**, so an empirical
  component can never be laundered into a "proven" system claim.

### Added — measured cheating surface (`run_cheatbench()`)
- New `aion_nexus.verify.cheatbench`, adapted from `substrate_core.cheatbench`: runs real
  attacks against the certificate gate and **measures** each channel CLOSED/OPEN instead of
  assuming soundness. Four channels (forge-without-key, label-tamper, assurance-overclaim,
  downgrade-strip-sig) measure as CLOSED (regression-tested); the honest **OPEN** residual —
  conformal coverage is marginal, not per-instance correctness — is reported openly, never
  dressed up as closed.

### Changed
- Version → **2.5.0**. Tests: 229 → **289** (signing 14, assurance 17, +23 verify, cheatbench 6).
  `ruff` clean. §6.31 honesty constraints remain enforced by tests.

## [2.4.0] — 2026-06-15 (Substrate Core: verification & certification as a product)

Brings the verification/certification layer — the defensible value layer of the
[2032 market vision](../AION_NEXUS_RD/14_MARKET_VISION_2032.md) — from R&D into the public
package, as a model-agnostic, first-class capability. The honest reframe (degradation-stage /
RUL proxy, not fault-type diagnosis) is now the product's stated intent.

### Added — Substrate Core (`aion_nexus.verify`)
- **Model-agnostic verification layer** that wraps *any* classifier's probabilities:
  `Verifier(alpha).calibrate(probs, labels).certify(probs) -> Certificate`.
  - `ConformalCalibrator` — split-conformal (APS/LAC) prediction sets with finite-sample
    quantile, never an empty set, and an explicit `coverage_valid_under` field stating the
    marginal-coverage guarantee holds **only under exchangeability** (cross-bearing / cross-machine
    breaks it — voids the guarantee).
  - `Certificate` — `content_hash` over a canonical payload that **binds the human-readable labels**
    (`predicted_name`, `conformal_set_names`), so a label-only tamper breaks integrity. HMAC-SHA256
    signature when `VERIFY_HMAC_KEY` is set (`authentication = "HMAC-SHA256"`); otherwise
    `authentication = "NONE"` — integrity hash only, **not tamper-evident** against an adversary.
  - `verify_certificate()` returns a single safe `trusted` flag (`integrity_ok AND authenticity ==
    "VERIFIED"`) so consumers can't be fooled by checking authenticity alone.
  - `CertificateStore` — append-only hash-chained JSONL (HMAC chain when keyed; keyless
    re-concatenation is detected as `FORGED` by a key-holder).
- **Top-level exports**: `Verifier`, `Certificate`, `ConformalCalibrator`, `verify_certificate`,
  `compliance_evidence`, `evidence_card`.

### Added — degradation-stage / RUL (first-class, honest)
- `aion_nexus.degradation` + `InferenceEngine.predict_degradation()` + `POST /predict_degradation`:
  expose a **coarse positional degradation stage** (early/mid/advanced/critical) with a conformal
  stage-set, the honest product of the positional FEMTO labels. Explicitly documented as **not** a
  calibrated time-to-failure / RUL-in-hours.

### Added — EU AI Act evidence mapping (`aion_nexus.compliance`)
- `compliance_evidence(certificate)` / `evidence_card()` map the certificate's fields to
  **EU AI Act Art. 12 (record-keeping), Art. 14 (human oversight), Art. 15 (accuracy/robustness)**,
  ISO 13381-1:2025 and ISO/IEC 42001 — strictly as *"provides evidence toward"*. A `_FORBIDDEN_CLAIMS`
  guard and a test fail the build if the words "compliant"/"conforme"/"certified compliant" ever
  appear. `docs/COMPLIANCE_MAPPING.md` carries a strong disclaimer: this is **not** a declaration or
  third-party conformity assessment.

### Changed
- README + MODEL_CARD: new "Verified inference (Substrate Core)" section; intended use reframed to
  degradation-stage estimation + verified inference; out-of-scope clarified (not fault-type diagnosis,
  not calibrated RUL, not a compliance declaration).
- Version → **2.4.0**. Tests: 160 → **229** (verify 20, degradation 31, compliance 18). `ruff` clean.
  The §6.31 honesty constraints are **enforced by tests**, not only prose.

## [2.3.0] — 2026-06-12 (Red-team rebuild)

An adversarial red team attacked the project from seven angles (security, robustness,
reproducibility, science, business, thesis) and confirmed real kill-shots with reproductions.
This release closes the confirmed engineering kill-shots and discloses the three weaknesses the
project had not yet self-declared. See [`RETRACTIONS.md`](RETRACTIONS.md).

### Security
- **Denial-of-service on `/predict_long_signal` closed**: a single ~200 KB request
  (`window=2560, stride=1` → ~8941 windows) used to peg a CPU worker for >115 s. `LongSignalRequest`
  now caps `signal` length, requires `window ≥ 2560` and `stride > 0`, and rejects any request whose
  estimated window count exceeds `AION_MAX_WINDOWS` (default 5000) **before** any numpy allocation —
  the attack vector now returns `422` in <0.4 s.
- **Ragged-input crash closed**: a body like `{"signal":[[1,2,3],[1]]}` raised `ValueError` inside
  `np.asarray` and leaked a `500` + stack trace. Both `/predict` and `/predict_long_signal` now wrap
  the conversion and return a clean `400`.
- **`/predict_batch` file-count cap**: `AION_MAX_BATCH_FILES` (default 256) → `413` past the limit
  (each file was already individually size-capped).

### Added
- **Signal-plausibility / OOD gate** (`aion_nexus/ood.py`): a lightweight, **honest** heuristic
  (spectral flatness + crest factor + per-channel std) that flags inputs implausible as bearing
  vibration — pure noise, a disconnected/saturated sensor, a near-constant trace. When flagged,
  `predict()`/`predict_batch()` set `ood_flag`/`ood_score`/`ood_reason`/`abstain` and the
  `recommended_action` **abstains** (no automated stop/inspection escalation) while preserving the raw
  class/confidence. This closes the red-team finding that white noise was classified as an imminent
  fault with 0.79 mean confidence. Tunable via `AION_OOD_*` env vars. It is explicitly **not** a
  learned OOD detector — see the module docstring; thresholds are tuned on FEMTO/PRONOSTIA and must be
  re-validated for other sensor classes.

### Fixed
- **NaN-to-"no action" path closed**: `validate_signal` now casts to `float32` **before** the
  finiteness check, so values that overflow `float32` to `Inf` (magnitude ≳ 3.4e38) are rejected with
  an actionable error instead of silently propagating `NaN` into a `class="normal"` verdict.

### Documentation honesty (three previously-undisclosed weaknesses)
- **Substrate v3 LOBO 10-shot 0.783 is transductive, not clean LOBO**: the SSL encoder was pretrained
  on a corpus that *includes* the held-out bearings (unlabeled). Every occurrence of 0.783 now carries
  this caveat; the clean nested-LOBO re-pretrain is pre-registered (in the R&D workbook).
- **Cross-dataset binary-health 0.91–1.00 is matched by a trivial non-ML kurtosis threshold** (≈0.911).
  Documented; the substrate's value claim is moved to 4-class severity, not binary health.
- **The 4-class labels are positional** (`degradation_pct = file_idx/(total−1)`, binned), i.e. a
  degradation-stage / RUL proxy — **not** an independently diagnosed fault type. The README and model
  card now say this next to the headline, not in an appendix. The honest LOBO number (v6
  0.352 ± 0.112) sits next to 0.884 in the first table.

### Reproducibility
- `examples/sample_signal.csv` is now a valid ≥2560-sample 2-channel signal (was a 10-row placeholder
  with a comment on row 11 that crashed the README quickstart).
- `verify_checkpoint.py` no longer prints "VERIFICATION PASSED" when zero samples were evaluated
  (the MFPT `*.mat` vs `*.csv` mismatch produced a deceptive pass); it now exits non-zero with a clear
  message. `docs/reproduce.md` states honestly what runs from a clean public clone vs what needs the
  external FEMTO dataset.

### Changed
- Package version → **2.3.0**. Tests: 127 → **160** (OOD, float32-overflow, and kill-shot regression
  suites added). `ruff` clean.

## [2.2.0] — 2026-06-11 (Enterprise hardening)

### Security
- **Substrate v3 checkpoint loader hardened**: `torch.load(..., weights_only=True)` is now the
  default load path, with an optional `expected_sha256` integrity check before deserialization.
  Closes the threat-model gap tracked in `docs/threat_model.md` ("Critical mitigation gaps").
- **Server authentication**: optional `AION_API_KEY` env var enables `X-API-Key` header auth on
  all endpoints (`/health` stays exempt for orchestrator liveness probes).
- **Request body size limit**: `AION_MAX_BODY_BYTES` (default 10 MB) — oversized requests are
  rejected with `413` before parsing. Makes the long-standing SECURITY.md claim enforceable in
  the app itself, not only at the reverse proxy.
- **CORS**: `AION_CORS_ORIGINS` (comma-separated allowlist). Default: no CORS headers at all.

### Added
- **v3 substrate servable**: `detect_architecture` now recognizes v3 checkpoints; few-shot
  tooling works against the frozen v3 encoder.
- **`/metrics` endpoint** (Prometheus exposition format) — request counts and latency; depends
  on `prometheus-client` (added to `requirements.txt`).
- **Structured JSON logging**: `AION_LOG_JSON=1` switches logs to JSON with a per-request
  `request_id`.
- `HealthResponse` now includes `architecture_version`.
- `docs/reproduce.md` — honest, minimal reproduction guide for the published F1 = 0.884
  (including the globally-stratified-split caveat).

### Changed
- FastAPI startup migrated from deprecated `on_event` hooks to the `lifespan` context manager.
- Package version → **2.2.0** (`aion_nexus/version.py` single source of truth).

### Packaging / CI
- `pyproject.toml`: console entry points fixed (`scripts/` is now a real package included in
  `packages.find`); `py.typed` marker shipped (`Typing :: Typed` classifier +
  `[tool.setuptools.package-data]`); absolute `[project.urls]`; coverage config with
  `fail_under = 70`.
- Dockerfile: torch installed from the CPU wheel index (`download.pytorch.org/whl/cpu`) —
  slim CPU image; OCI version labels 2.2.0. `docker-compose.yml`: obsolete `version:` key
  removed, image tag aligned to 2.2.0.
- CI: coverage gate (`--cov-fail-under=70`), weekly scheduled supply-chain audit job
  (`scripts/audit_supply_chain.py` + `pip-audit` — makes the SECURITY.md "weekly in CI" claim
  true), `python -m build` + `twine check dist/*` job, container smoke test (`docker run` +
  `curl /health`) after the Docker build.

### Documentation reconciliation
- MODEL_CARD / PERFORMANCE_BENCHMARKS / INDEX / README / docs aligned to 2.2.0 and to the
  verified numbers only: v6 "+5.0 pp" advantage row caveated (not a valid head-to-head, see
  §6.31 disclosure); sample-efficiency 10-shot point reconciled to the verified 0.672 ± 0.006;
  cost figures and SOTA/latency tables explicitly labeled as estimates / literature-reported
  where no artifact exists in `results/`; cross-dataset 0.91–1.00 claim qualified inline as
  binary health (2-class) vs a weak random-init control; `FewShotAdapter` quickstart fixed to
  the real signature `FewShotAdapter(engine)`; FAQ pointer to non-existent
  `aion_nexus.utils.cohens_d` corrected.

## [2.1.0] — 2026-06-04 (v3 self-supervised substrate foundation backbone)

### Added
- **v3 substrate** (`aion_nexus/substrate_v3.py`): a PatchTST self-supervised foundation
  encoder (1,220,928 params), pretrained contrastively (NT-Xent) on unlabeled
  FEMTO+MFPT+CWRU vibration. Frozen encoder + per-deployment few-shot head; served via the
  AION-2 verified trust layer (conformal + physics verifier → tamper-evident certificate).
  `create_substrate_v3()` with encoder param-count guard; checkpoint
  `checkpoints/aion_nexus_substrate_v3.pth` (objective `contrastive-ntxent-patchTST`).
- `tests/test_substrate_v3.py` — param guard, drift-refusal, forward contract,
  frozen-encoder/trainable-head, checkpoint load (5 tests).

### Verified (honest, §6.31)
- v3 is NOT a higher in-distribution classifier — v1 FEMTO F1=0.884 stands; v3 full-transfer
  to unseen machines ≈ 0.55.
- v3's value = cross-domain FEW-SHOT: 10-shot LOBO FEMTO severity F1 ≈ 0.78; **cross-DATASET
  10-shot macro-F1 0.91–1.00** (FEMTO↔MFPT↔CWRU) — **binary health (2-class) task, vs a weak
  random-init control** (random-init 0.5–0.8). Zero-shot cross-rig is NOT reliable.
- Scaled run (9,315 windows / 400 epochs, Colab T4): 10-shot 0.760→0.783, full-transfer
  0.554→0.533 — **within noise** → the encoder is **architecture-saturated** at
  d_model 192/depth 4. Promoted (drop-in) as the production v3 checkpoint. The real next
  lever is a bigger **v3.1** (see Roadmap), not more data at this scale.

### Notes
- Backward-compatible: v1 (default) and v6 unchanged; v3 is opt-in.
- Fulfils the 2.1.0 roadmap item (multi-domain pre-training), realized as a self-supervised
  substrate. Scaled pretraining reproducible via the AION-2 `foundation/` scripts (Colab GPU).

## [1.0.0] — 2025-10-10

### Added
- Production architecture: Multi-Scale CNN + Channel Attention + Bidirectional GRU + 3-layer MLP classifier (1,061,724 parameters, 4.1 MB).
- `aion_nexus.InferenceEngine` — checkpoint-loading inference with batch support, latency telemetry, confidence banding.
- `aion_nexus.FewShotAdapter` — 10-sample domain adaptation, encoder frozen, classifier-only fine-tuning.
- `aion_nexus.preprocessing` — strict input validation (shape, NaN/Inf, stuck-sensor detection) + per-channel z-score normalization.
- `server` — FastAPI service with `/predict`, `/predict_batch`, `/predict_long_signal`, `/health`, `/version` endpoints.
- `scripts/verify_checkpoint.py` — independent F1 verification against published numbers.
- `scripts/benchmark_inference.py` — latency / throughput benchmark.
- `scripts/export_onnx.py` — ONNX export for edge deployment.
- Test suite: smoke tests + few-shot tests, no checkpoint required for CI.
- Dockerfile with non-root user, healthcheck, resource limits.
- Documentation: README, MODEL_CARD, PERFORMANCE_BENCHMARKS, DEPLOYMENT, architecture, troubleshooting.

### Verified performance
- FEMTO in-distribution F1 = 0.898 (validation), 0.884 (test)
- MFPT zero-shot cross-domain F1 = 0.615
- MFPT few-shot 10-sample F1 = 0.672 ± 0.006 (3 runs)

All numbers cross-verified across `final_results.json`, `cross_validation_results.json`, and `mfpt_sensitivity_results.json`.

### Documented negative results
- SimCLR contrastive pretraining on FEMTO unlabeled — caused MFPT zero-shot F1 to drop from 0.615 to 0.184 (−70%). NOT included in v1.0 checkpoint.
- AdaBN adaptation with MFPT class imbalance — F1 dropped from 0.615 to 0.488 (−21%). NOT in v1.0.
- CWRU severity-mapped task — F1 ≈ 0.34 due to physically invalid label mapping (Spearman correlation = −0.30). Documented in `docs/task_mismatch.md`.

### Known limitations (see MODEL_CARD.md for details)
- 4-class severity diagnosis only; not validated for fault localization.
- Sampling rate < 10 kHz not supported (resample first).
- Acoustic / thermal sensor inputs not validated.

## [2.0.0] — 2026-04-27 (v6 architecture + critical comparability finding)

### CRITICAL methodological finding
The v6 "F1=0.934" headline number was measured on a **DIFFERENT dataset** than v1's "F1=0.884":
- **v1** trained AND tested on **Test_set/Test_set** = 11 run-to-failure bearings (industrial scenario, 13,959 samples).
- **v6** trained AND tested on **Test_set/Training_set/Learning_set** = 6 short-run calibration bearings (~7,500 samples).

When v6 (trained on Learning_set) is evaluated on the industrial Test_set 11 bearings (cross-bearing transfer), F1 drops from 0.934 to **0.302** (verified 2026-04-27). This is the same §6.31 family F1 (measurement-construct misalignment) pattern that produced 21 retractions in sister projects (motif/, VELLERA/).

**Honest re-statement**: v1 and v6 are NOT comparable head-to-head on the same data — the published F1 numbers come from different test sets. After full cross-evaluation:

|  | Learning_set (calibration) | Test_set (run-to-failure) |
|---|---|---|
| v1 (BiGRU) | 0.382 cross | 0.884 in-dist |
| v6 (TempAttn+TRM) | 0.934 in-dist | 0.302 cross |

**Both architectures fail symmetrically when tested on the OTHER FEMTO subset**. The architecture choice (BiGRU vs TempAttn+TRM) is overshadowed by the train-test distribution shift. There is no "production default architecture"; there is "the right architecture for the regime", with **few-shot adaptation (F1=0.672 with 10 samples)** as the cross-regime tool. Production package supports both architectures with auto-detection from the checkpoint state-dict.

### Added
- **AION-NEXUS v6 architecture** (`AIONNexusV6`): MultiScale CNN + ChannelAttn + **TemporalSelfAttention** + **TinyRecursiveReasoner**.
  - 716,577 parameters (32.5% smaller than v1)
  - 2.73 MB on disk (vs 4.1 MB for v1)
  - **In-distribution Test F1 = 0.934** on Learning_set held-out (verified delta=0.0000)
  - **Cross-domain F1 on Test_set run-to-failure = 0.302** (verified — significant degradation)
  - Noise-robust at training time: SNR+5dB F1 ≈ 0.87
  - Inference latency ~16 ms
- **`TemporalSelfAttention`** module (4-head MHA + learned-query pooling, replaces v1's BiGRU + dual-pool)
- **`TinyRecursiveReasoner`** module (TRM-inspired progressive refinement; Jolicoeur-Martineau 2025)
- **Auto-detection of architecture version** in `InferenceEngine.from_checkpoint`. Inspects state_dict keys to pick v1 or v6; can be forced via `version="v1"|"v6"` flag.
- `architecture_version` attribute on `InferenceEngine`, exposed in `/health` endpoint.
- `tests/test_v6.py` — 14 v6-specific tests (param count, forward, auto-detection, backward-compat).
- v1 baseline (1,061,724 params, F1=0.884) remains fully supported as a backward-compatible mode.

### Changed
- `aion_nexus/__init__.py` re-exports both v1 (`AIONNexus`, `create_aion_nexus`) and v6 (`AIONNexusV6`, `create_aion_nexus_v6`) symbols.
- `scripts/verify_checkpoint.py` adapts published-F1 tolerances based on detected architecture.

### Verification
- v1 architecture: analytical param count = **1,061,724** = training log (delta = 0).
- v6 architecture: analytical param count = **716,577** = training log (delta = 0).
- v1 F1 verification: 0.8843 vs published 0.8840 (delta = 0.0003) — VERIFIED on 2026-04-27.
- v6 F1 verification: pending user run with `verify_checkpoint --checkpoint nexus_ultra_v6/best_model.pth`.

### Breaking changes
- This is a **major version bump** because the default architecture path expanded. Existing v1.0 deployments continue to work unchanged (backward-compatible: v1 checkpoints still detected and loaded). New deployments may use v6 by passing a v6 checkpoint to `from_checkpoint`.

## [1.0.2] — 2026-04-27 (CRITICAL: preprocessing alignment fix)

### Fixed — CRITICAL
- **Preprocessing chain now matches training**. Independent verification on
  the same FEMTO data revealed F1 = 0.279 instead of 0.884: a 60-percentage-
  point delta caused by missing preprocessing steps in the production
  package vs the training-time loader.
  - **Added high-pass Butterworth 1 Hz filter** (2nd order, sosfilt) in
    `aion_nexus.preprocessing.preprocess_signal`. Training in `aion_data.py`
    applies this filter after z-score normalization; without it, the model
    receives out-of-distribution input and predictions collapse.
  - Pipeline order is now: validate → crop → z-score per channel → HP-Butterworth
    (matches `aion_data._process_csv_file` step-for-step).
- **`scripts/verify_checkpoint.py` rewritten to import the original
  `aion_data.AIONDataset` + `create_stratified_temporal_splits` (seed=42)**.
  Required to reproduce the 2,792-sample test set used for the published
  F1=0.884. The previous implementation walked subdirectories, producing
  a different evaluation set and an invalid F1 number.
- Added `diagnose_checkpoint` step that runs without any data: param-count
  match + class distribution sanity on synthetic input. Catches "model
  collapsed to one class" failures before any downstream evaluation.

### Added
- `scipy>=1.10.0,<2.0.0` to `requirements.txt`. Required for
  `scipy.signal.butter` / `sosfilt`. Mean-removal fallback if scipy
  unavailable, but training used scipy — production should match.

### Validation
- Verified that test_preprocessing_edge_cases tests still pass after
  preprocessing change (assertions on mean/std relaxed to reflect HP-
  filter attenuation behavior).

## [1.0.1] — 2026-04-27 (bomb-proofing patch)

### Fixed
- **Stuck-sensor detection threshold**: raised from `1e-9` to `1e-7` and stddev now computed in float64. Float32 precision noise (e.g., `np.asarray([0.1]*N, dtype=float32).std() ≈ 2e-8`) was leaking past the previous threshold; effectively-stuck sensors were not being caught. New threshold aligns with industrial accelerometer noise floors (PCB 352C03 ~5 μg).
- **NaN/Inf check moved before center-crop**: invalid values in any portion of the input signal are now caught regardless of whether they fall within the centered 2,560-sample window. Previously, NaN/Inf in the cropped tail was silently dropped.
- **`/predict` endpoint split into `/predict` (JSON) and `/predict_csv` (file upload)**: FastAPI cannot reliably auto-detect content-type when a single endpoint accepts both `File()` and a Pydantic body. The previous design always rejected JSON bodies as "missing." JSON-only `/predict` is now the canonical path; CSV uploads use `/predict_csv` (same response schema).
- **`/predict_long_signal` now returns 400 (not 500) on invalid `aggregation` method**: `ValueError` from `aggregate_window_predictions` is wrapped in `HTTPException(400, ...)`.

### Breaking changes (within 1.x)
- `POST /predict` now accepts JSON body ONLY. CSV upload moved to `POST /predict_csv`. Clients that posted CSVs to `/predict` need to update endpoint URL.

### Added
- `docs/FAQ.md` — 51 anticipated questions with answers (verification, architecture, deployment, few-shot, edge cases, concurrency, security, comparisons, reproducibility, licensing).
- `docs/threat_model.md` — STRIDE-based threat analysis with mitigation gaps.
- `docs/data_contract.md` — formal input/output spec with stability guarantees.
- `SECURITY.md` — vulnerability disclosure policy + hardening checklist.
- `CONTRIBUTING.md` — workflow, §6.31 discipline, ADR process.
- `.pre-commit-config.yaml` — ruff + secrets detection + smoke-test hook.
- `Makefile` — common operational targets (install, test, lint, audit, etc.).
- `tests/test_preprocessing_edge_cases.py` — 23 edge cases for `validate_signal` / `preprocess_signal`.
- `tests/test_determinism.py` — same-seed reproducibility, eval-mode determinism, batch consistency.
- `tests/test_concurrency.py` — thread-safety of read-only inference.
- `tests/test_api_integration.py` — FastAPI TestClient end-to-end (200, 400, 422, 503 paths).
- `scripts/audit_supply_chain.py` — pip-audit + license allowlist/blocklist.
- `scripts/verify_onnx_parity.py` — numerical equivalence check between PyTorch and ONNX export.
- `scripts/quantize.py` — INT8 dynamic quantization with FP32 parity validation.
- `scripts/generate_manifest.py` — SHA-256 manifest with `--check` mode.

### Verified
- Architecture parameter count analytically derived: 1,061,724 (delta = 0 vs training log).
- 64 of 78 pytest cases pass on user's local environment with torch installed (1.0.0). Remaining 14 traced to two root causes: API integration state-leakage (fixed in this patch) and float32 precision in stuck-sensor threshold (fixed in this patch).

## Roadmap (future)

> Current released version is **2.2.0** (see above). The items below are post-2.2 plans.
> (2.2.0 shipped as enterprise hardening; the multi-task-heads item previously slated for
> 2.2.0 moves to 2.3.0, streaming to 2.4.0.)

- ✅ 2.1.0: Multi-domain pre-training — DONE, realized as the v3 self-supervised substrate (FEMTO+MFPT+CWRU; Paderborn/NASA-IMS to be added in the scaled run).
- v3.1 (bigger arch d_model 256/depth 6 + hybrid masked⊕contrastive): **TESTED 2026-06-04, FAILED** — LOBO 10-shot 0.801 (+0.018 vs v3, within noise), full-transfer down → architecture is NOT the lever on the current 3-rig data; not promoted (v3 stays). The production module is arch-flexible (`AIONNexusV3.from_checkpoint`) for a future data-driven v3.2. See `AION_NEXUS_RD/aion2_verified_substrate/foundation/RETRACTION_substrate_v31.md`.
- **Next real lever = DATA diversity**: add Paderborn (KAt-DataCenter) + NASA-IMS rigs to the unlabeled pretraining corpus, re-pretrain at the v3 architecture.
- 2.3.0: Multi-task heads (severity + location simultaneously).
- 2.4.0: Streaming inference engine (online sequential decisions).
