# Per-bearing F1 macro breakdown — production v1 checkpoint

**Checkpoint**: `checkpoints\aion_nexus_v1.pth`

**Bearings evaluated**: 11 / 11

**Generated**: C:\Users\Utente\MODELLIPERSONALIZZATI\DC2026\Startup\AionNexus\AION_NEXUS_PRODUCTION


## Headline summary

- **F1 macro mean** across 11 bearings: **0.9218**
- **F1 macro std** across bearings: **0.0426**
- **F1 macro range** (max - min): **0.1400** (min 0.8242 on `Bearing2_7`, max 0.9641 on `Bearing2_5`)

## Interpretation

High variance across bearings (std > 0.10 or range > 0.30) suggests the model generalizes UNEVENLY across the 11 run-to-failure bearings, which is consistent with the hypothesis that the stratified-random 80/20 split INFLATES the headline F1=0.884 by leaking bearing identity into both train and test. Low variance (std < 0.05) means the headline number is honest. This is NOT a true LOBO measurement (would require retraining); it is a per-bearing breakdown of the existing checkpoint.

## Per-bearing detail

| Bearing | n samples | F1 macro | Accuracy | True class dist | Pred class dist |
|---|---:|---:|---:|---|---|
| Bearing1_3 | 1802 | **0.9623** | 0.963 | [361, 540, 540, 361] | [341, 555, 571, 335] |
| Bearing1_4 | 1139 | **0.9315** | 0.928 | [228, 341, 342, 228] | [236, 335, 328, 240] |
| Bearing1_5 | 2302 | **0.9407** | 0.939 | [461, 690, 690, 461] | [440, 689, 679, 494] |
| Bearing1_6 | 2302 | **0.9390** | 0.935 | [461, 690, 690, 461] | [434, 701, 718, 449] |
| Bearing1_7 | 1502 | **0.9615** | 0.959 | [301, 450, 450, 301] | [307, 458, 442, 295] |
| Bearing2_3 | 1202 | **0.9286** | 0.932 | [241, 360, 360, 241] | [247, 351, 394, 210] |
| Bearing2_4 | 612 | **0.8917** | 0.887 | [123, 183, 183, 123] | [109, 208, 178, 117] |
| Bearing2_5 | 2002 | **0.9641** | 0.964 | [401, 600, 600, 401] | [402, 600, 568, 432] |
| Bearing2_6 | 572 | **0.8785** | 0.872 | [115, 171, 171, 115] | [109, 178, 166, 119] |
| Bearing2_7 | 172 | **0.8242** | 0.820 | [35, 51, 51, 35] | [35, 52, 58, 27] |
| Bearing3_3 | 352 | **0.9173** | 0.909 | [71, 105, 105, 71] | [72, 86, 134, 60] |

## Honest framing for external use

This number is the per-bearing F1 macro variance of the released v1 checkpoint on the 11 FEMTO run-to-failure bearings, evaluated independently. It is the cheapest available proxy for the question 'is F1=0.884 inflated by stratified-random split data leakage?'. A true leave-one-bearing-out (LOBO) F1 number would require retraining the model 11 times and is scheduled for the next iteration.
