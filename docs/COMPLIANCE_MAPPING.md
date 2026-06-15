# Compliance Evidence Mapping — Substrate Core certificates

> **Questo NON è una dichiarazione di conformità né una valutazione di conformità
> di terza parte ai sensi dell'EU AI Act.** Fornisce evidenza tecnica e
> tracciabilità che SUPPORTANO il processo di conformità del deployer/operatore.
> La classificazione high-risk e gli obblighi finali dipendono dal contesto
> d'impiego e da una valutazione formale.

Questo documento spiega come i campi di un
[`Certificate`](../aion_nexus/verify/certificate.py) di Substrate Core si mappano
sull'**evidenza tecnica** verso obblighi regolatori specifici. La mappatura è
implementata in [`aion_nexus/compliance.py`](../aion_nexus/compliance.py)
(`compliance_evidence()` per la struttura dati, `evidence_card()` per la scheda
human-readable).

Il linguaggio è deliberato. Diciamo sempre **"provides evidence toward / supports
/ maps to"**, mai **"compliant with" / "conforme a" / "certified compliant"**. Il
sistema non certifica conformità: produce un artefatto verificabile che il
deployer usa *come input* al proprio processo di conformità.

---

## Cosa è e cosa NON è

| È | NON è |
|---|---|
| Un artefatto per-decisione, ri-eseguibile e verificabile | Una dichiarazione di conformità (Dichiarazione UE di conformità) |
| Evidenza tecnica verso articoli/clausole specifici | Una valutazione di conformità di terza parte / da organismo notificato |
| Tracciabilità e human-in-the-loop documentati | Una prova che un umano abbia effettivamente revisionato |
| Un input al processo di conformità del deployer | La determinazione del rischio (high-risk) del sistema |

La classificazione high-risk ai sensi dell'EU AI Act, e quali obblighi
effettivamente vincolino, **dipendono dal contesto d'uso** e da una valutazione
formale svolta dal deployer/operatore e, ove richiesto, da un organismo
notificato.

---

## Tabella: articolo/clausola → evidenza fornita → limite

| Framework / Riferimento | Evidenza fornita (dai campi del certificato) | Limite (cosa NON copre) |
|---|---|---|
| **EU AI Act — Art. 12** (record-keeping / logging automatico) | `cert_id` + `timestamp_utc` + `input_sha256` + `content_hash`, accodati da `CertificateStore` a un log JSONL append-only con catena di hash → la sequenza di eventi è ricostruibile. Il regime di autenticazione è letto dal campo `authentication`. | Logga il record di decisione, NON il segnale d'ingresso completo (solo lo SHA-256) né il contesto a monte di acquisizione dati. La **tamper-evidence esiste solo con `VERIFY_HMAC_KEY` settata**; con `authentication=NONE` il log prova consistenza ma non autenticità. Ritenzione e integrità nel tempo restano responsabilità del deployer. |
| **EU AI Act — Art. 14** (human oversight) | Il `verdict` codifica l'instradamento: `CERTIFIED` → agibile; `REVIEW` (conformal set >1 etichetta) → revisione umana; `ABSTAIN` (sotto soglia di confidenza) → nessuna azione automatica. Il verdetto è legato in `content_hash`, quindi l'instradamento mostrato non può divergere silenziosamente da quello certificato. | Rende disponibile e registra l'hand-off umano; NON prova che un umano abbia revisionato, né che avesse competenza/autorità/tempo per fare override. L'esistenza e l'efficacia del processo di revisione a valle è un controllo organizzativo che il deployer deve implementare e dimostrare separatamente. |
| **EU AI Act — Art. 15** (accuracy, robustness, cybersecurity) | `alpha` + `qhat` rendono auditabile il target di copertura dichiarato (1 − α) per decisione; `ABSTAIN` aggiunge un floor di confidenza; un gate OOD di plausibilità del segnale (`check_signal_plausibility`) filtra input implausibili a monte; la catena HMAC (se con chiave) evidenzia l'integrità del log. | **La garanzia di copertura conformal vale SOLO sotto scambiabilità** (exchangeability); cross-bearing / cross-machine la rompe e **annulla** la garanzia marginale 1 − α — il target dichiarato diventa aspirazionale, non garantito. Il **gate OOD gira a monte del verifier e NON è un campo del certificato**: il certificato da solo non prova che il gate sia stato eseguito su quell'input. Le accuracy da benchmark non si trasferiscono ad altra macchina/sensore/regime senza ri-validazione. |
| **ISO 13381-1:2025** — Clause 7 (stadi prognostici) | `predicted_name` + `conformal_set_names` riportano lo stadio di degrado stimato e l'insieme di stadi che il target di copertura non riesce a escludere: stima di stadio con banda di incertezza esplicita, non una bare point label. | Gli stadi sono **POSIZIONALI** (degrado/RUL), NON diagnosi del tipo di guasto: una label di stadio non nomina il failure mode. ISO 13381-1 si attende anche stime RUL con confidenza su un orizzonte temporale — qui si riporta uno stadio in un istante, non una traiettoria RUL calibrata. La conformance piena richiede la procedura prognostica più ampia del deployer. |
| **ISO/IEC 42001:2023** — AI management system | `model_id` + `schema_version` + ri-verificabilità via `verify_certificate` legano la decisione a una versione di modello e la rendono auditabile a posteriori: un artefatto di tracciabilità referenziabile da un AIMS. | Un singolo artefatto NON è un sistema di gestione: ISO/IEC 42001 richiede ambito organizzativo, trattamento del rischio, ruoli, monitoraggio e miglioramento continuo. Questo modulo fornisce un input a tale sistema e non certifica nulla sui processi circostanti, sulla data governance o sul lifecycle management. |

---

## I tre campi di ogni voce di evidenza

`compliance_evidence(certificate)` restituisce, per ogni obbligo mappato, tre
campi — e il terzo è **obbligatorio e mai vuoto**:

- `provides_evidence_for` — l'obbligo, sempre nella forma "provides evidence
  toward ...".
- `how` — quali campi del certificato forniscono l'evidenza e come.
- `limitation` — cosa l'evidenza **non** copre. Un item di evidenza senza limite
  dichiarato è esattamente l'overclaim che rifiutiamo di spedire.

---

## Disclaimer sui regimi di autenticazione

Il certificato è **tamper-evident contro un avversario solo con chiave HMAC
settata** (`VERIFY_HMAC_KEY` → `authentication = "HMAC-SHA256"`). Senza chiave,
`authentication = "NONE"`: il `content_hash` è un **integrity hash** che prova
consistenza interna ma **non** autenticità contro un avversario che possiede
questo sorgente. La scheda di evidenza riflette il regime effettivo del singolo
record e adatta il limite di conseguenza.

---

## Disclaimer sulla scambiabilità (conformal)

La predizione conformal garantisce copertura marginale **solo sotto
scambiabilità** dei dati di calibrazione e di serving. In deployment
cross-bearing / cross-machine la scambiabilità si rompe e la garanzia 1 − α non
vale più. Ogni output conformal porta questo caveat (`coverage_valid_under` sul
calibratore); la mappatura Art. 15 lo ripete esplicitamente.

---

## Riferimenti

- Mappatura: [`aion_nexus/compliance.py`](../aion_nexus/compliance.py)
- Certificato: [`aion_nexus/verify/certificate.py`](../aion_nexus/verify/certificate.py)
- Verifier (logica verdetto): [`aion_nexus/verify/verifier.py`](../aion_nexus/verify/verifier.py)
- Gate OOD: [`aion_nexus/ood.py`](../aion_nexus/ood.py)
- Test: [`tests/test_compliance.py`](../tests/test_compliance.py)
