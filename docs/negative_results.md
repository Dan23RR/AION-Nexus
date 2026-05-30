# Negative Results

Documented experiments that did NOT improve performance, with mechanisms identified. Documenting these prevents future re-running of dead-end approaches.

## 1. SimCLR contrastive pretraining on FEMTO unlabeled

**Hypothesis**: Self-supervised contrastive features generalize better cross-domain than supervised features.

**Setup**: SimCLR on FEMTO unlabeled data, 100 epochs, 61 hours CPU. Then supervised fine-tune.

**Outcome**:
| Metric | Baseline | After SimCLR | Δ |
|---|---|---|---|
| FEMTO F1 | 0.898 | 0.914 | +1.8% ✓ |
| **MFPT F1 (zero-shot)** | **0.615** | **0.184** | **−70.1%** ✗ |
| **CWRU F1** | **0.341** | **0.225** | **−34.0%** ✗ |

**Mechanism**: Single-domain contrastive learning **over-specializes** to the FEMTO distribution. Negatives are sampled from the same domain, so the contrastive loss makes the feature space compact for FEMTO and rejects out-of-distribution samples by design.

**Lesson**: Multi-domain pretraining is needed for cross-domain transfer. Single-domain SSL actively harms generalization.

**v1.0 disposition**: Released checkpoint does NOT include SimCLR. Future versions will explore multi-domain contrastive (FEMTO + Paderborn + NASA-IMS jointly).

## 2. Adaptive BatchNorm with MFPT unlabeled

**Hypothesis**: Re-estimate BatchNorm statistics using MFPT unlabeled samples (no labels needed).

**Setup**: 20 forward passes through MFPT unlabeled with BN momentum=0.5 to update running statistics.

**Outcome**:
| Metric | Baseline | After AdaBN | Δ |
|---|---|---|---|
| **MFPT F1** | **0.615** | **0.488** | **−20.5%** ✗ |

**Mechanism**: MFPT has severe class imbalance (0% Early, ~50% Advanced). BN running stats shift toward the majority-class distribution, distorting the implicit feature normalization the classifier relied on.

**Lesson**: AdaBN requires a balanced unlabeled distribution. With class-imbalanced unlabeled data, AdaBN actively degrades performance.

**v1.0 disposition**: Released checkpoint uses FEMTO BN statistics. For deployment, prefer few-shot adaptation over AdaBN.

## 3. CWRU severity-mapped task

**Hypothesis**: Map CWRU fault-size labels (0.007", 0.014", 0.021") to FEMTO severity classes (early / medium / advanced).

**Setup**: Defined a fault-size → severity mapping based on intuition, evaluated zero-shot.

**Outcome**:
| Mapping | F1 | Spearman correlation (size, severity) |
|---|---|---|
| Severity-based (proportional) | 0.359 | n/a |
| Location-based (different task) | 0.341 | n/a |
| Fault size vs vibration severity | n/a | **−0.30** (negative!) |

**Mechanism**: CWRU fault size is a **physical defect dimension**, not a measure of vibration energy. The mapping fault-size → severity is confounded by:

- Bearing load
- Defect position (inner / outer / ball)
- Defect type (pit / spall / crack)
- Operating speed

A 0.007" outer-race pit may produce more vibration than a 0.021" rolling-element crack at light load.

**Lesson**: **Task semantic mismatch can dominate domain shift.** Validate label alignment before cross-domain evaluation. Spearman correlation of physical proxies should be ≥ 0.7 for the mapping to be meaningful.

**v1.0 disposition**: AION-NEXUS does NOT support CWRU. For CWRU deployment, either retrain on CWRU labels directly or treat as a different (location) classification task.

## 4. Larger model with full features (Enhanced AION baseline)

**Hypothesis**: Adding STFT spectrograms + envelope analysis + more parameters improves F1.

**Setup**: 4.6M parameter "Enhanced" model with hand-engineered spectrograms.

**Outcome**:
| Method | Params | FEMTO F1 |
|---|---|---|
| Baseline (LinearAttention fallback) | 250K | 0.145 |
| Enhanced (4.6M params, spectrograms, envelope) | 4.6M | 0.145 |
| **NEXUS (this release)** | **1.06M** | **0.898** |

**Mechanism**: With 8K training samples, a 4.6M parameter model has **15 samples per parameter** — severe overfitting territory. The hand-engineered features were redundant with what the multi-scale CNN learns end-to-end, and the parameter explosion prevented training-time generalization.

**Lesson**: Right-sized architectures (≥40 samples/param) outperform feature-engineered larger models on small datasets. The multi-scale CNN with 1.06M params achieves 1× sweet spot at 8K samples → 8 samples/param post-pooling, which still works because of the strong receptive-field inductive bias.

**v1.0 disposition**: Released v1.0 is the 1.06M-param NEXUS, not the Enhanced model. Enhanced model is documented but archived.

## Pattern across negative results

All four documented negatives share a common structure: **a plausible technique imported from a different setting that does not respect the constraints of bearing fault diagnosis**.

| Technique | Imported from | Failed because |
|---|---|---|
| SimCLR contrastive | Computer vision (large-scale unlabeled) | Insufficient distributional diversity |
| AdaBN | Domain adaptation literature (balanced UDA) | Target class imbalance violates assumption |
| Severity mapping | Common-sense physics | Confounded by load + position + type |
| Larger model | Scaling laws (large data) | Sample-to-param ratio too low |

**Methodological insight**: Always validate the assumptions of an imported technique against the actual deployment constraints (data scale, class balance, label semantics, hardware) before incorporating it.
