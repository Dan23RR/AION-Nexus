# Contributing to AION-NEXUS

Thank you for your interest. This project follows strict engineering and scientific discipline; please read this document before opening a pull request.

---

## Code of conduct

Be respectful, evidence-based, and direct. We welcome critique of code, claims, and design — frame it constructively. Personal attacks are not tolerated.

## Two principles you should know upfront

### 1. Pre-registered binary criteria for any non-trivial scientific claim

If your contribution involves a quantitative claim (e.g., "this preprocessing improves F1 by X"), you must:

1. Open a draft PR or issue that describes the experiment.
2. Pre-register the **binary** PASS/FAIL criterion before running the experiment.
3. Lock the criterion with a git commit (or comment timestamp).
4. Run the experiment.
5. Report the outcome (PASS or FAIL) — both are acceptable contributions; documented FAILs go in `docs/negative_results.md`.

This discipline (informally called §6.31 head-to-head) is borrowed from sibling research projects with 21+ documented retractions. It is the project's main credibility asset.

### 2. Reproducibility is non-negotiable

Every numerical claim in the project must be re-runnable from `scripts/`. If your PR adds or changes a number, you must:

- Add a script under `scripts/` that produces it.
- Add a test (smoke or integration) that asserts it.
- Update `PERFORMANCE_BENCHMARKS.md` with the new number AND its source script.

PRs that introduce un-re-runnable claims will be requested to add reproducibility before merging.

---

## Workflow

### 1. Open an issue first

For anything more than a typo fix or a doc tweak, open an issue describing:

- What problem you're solving.
- Why it's a problem (customer request? CVE? performance regression?).
- Proposed approach.

This avoids wasted work on changes the maintainer doesn't want to merge.

### 2. Fork + branch

```bash
git checkout -b feat/your-feature
# or fix/, docs/, ci/, refactor/
```

### 3. Make changes following style

- Ruff for lint + format. Run `make lint` and `make format` before committing.
- Type hints on all new public APIs.
- Docstrings on all new public functions/classes (Google style or NumPy style).
- Tests for new code paths (smoke, integration, or edge-case as appropriate).

### 4. Run the test suite

```bash
make test-fast    # ~30 seconds
make test         # ~3 minutes
```

All existing tests must still pass. New tests must be added for new functionality.

### 5. Update documentation

- `README.md` if usage changes.
- `MODEL_CARD.md` if model behavior changes.
- `CHANGELOG.md` with a line under "Unreleased".
- `docs/architecture.md` if architecture changes (will require major version bump).

### 6. Open the PR

Use the template:

```markdown
## What this PR does
(brief description)

## Why
(link to issue or motivation)

## Pre-registered claim (if applicable)
(if the PR makes a quantitative claim, paste the locked PASS/FAIL criterion here BEFORE running the experiment)

## Tests added
- [ ] Smoke
- [ ] Integration
- [ ] Edge case
- [ ] Determinism
- [ ] Concurrency (if relevant)

## Documentation updated
- [ ] README
- [ ] MODEL_CARD
- [ ] CHANGELOG
- [ ] FAQ (if user-visible behavior changes)

## Backward compatibility
- [ ] No breaking changes
- [ ] Breaking change documented (and version bumped accordingly)
```

### 7. Code review

A maintainer will review. Expect:

- Pushback on un-tested code paths.
- Requests for reproducibility on numerical claims.
- Discussion of architectural implications.

We aim for first response within 7 days; full review within 14 days.

---

## Architectural changes

Changes that affect:

- Model parameter count or layer structure
- Input/output contract (`docs/data_contract.md`)
- Class taxonomy (4-class severity)
- Confidence-band semantics
- API endpoints (additive OK; removal/breaking requires major version)

…require an **Architecture Decision Record (ADR)**. Open an issue with the prefix `ADR:` describing:

- Context (what problem)
- Decision (what we chose)
- Alternatives (what we considered and rejected)
- Consequences (what this enables, what it costs)

ADRs are stored in `docs/adr/` once accepted.

---

## Testing philosophy

- **Smoke tests**: must pass on every CI run; no checkpoint required; ~30 seconds total.
- **Integration tests**: run on every PR; no real data required; ~3 minutes.
- **Verification tests**: run nightly + before release; require checkpoint + datasets; up to 30 minutes.
- **Edge-case tests**: comprehensive coverage of preprocessing branches, error handling.
- **Determinism tests**: ensure same input + same seed = same output.
- **Concurrency tests**: ensure read-only inference is thread-safe.

If you add a new feature, you should add at least:
- 1 smoke test (does it instantiate / run?)
- 1 integration test (does it work end-to-end?)
- 2-3 edge-case tests (what could break it?)

---

## Branch naming

- `feat/short-name` — new features
- `fix/short-name` — bug fixes
- `docs/short-name` — documentation only
- `refactor/short-name` — internal refactor, no behavior change
- `ci/short-name` — CI/CD changes
- `release/v1.X.0` — release branches

---

## Commit messages

Imperative mood, concise. Example:

```
feat(preprocessing): add line-frequency notch filter

50/60 Hz notch and harmonics. Configurable per region.
Closes #42.

Pre-registered claim: notch filter does not regress F1 on FEMTO test
(within ±1 pp). Verified: F1 = 0.882 vs 0.884 baseline; delta = -0.002.
```

---

## Releases

Maintainer cuts releases. Process:

1. Update `CHANGELOG.md`.
2. Bump version in `aion_nexus/version.py`.
3. Create release branch `release/vX.Y.Z`.
4. Run full verification: `make test && make audit && make verify`.
5. Tag: `git tag -a vX.Y.Z -m "release X.Y.Z"`.
6. Push tag; CI builds + pushes container image.
7. Update `MODEL_CARD.md` if model changes.

Major versions (X) for breaking changes; minor (Y) for additive features; patch (Z) for fixes.

---

## License

By contributing, you agree your contribution is licensed under Apache 2.0 (the project's license).

---

## Questions?

Open an issue or email daniel.culotta@gmail.com.

For security issues, see [`SECURITY.md`](./SECURITY.md) — DO NOT open public issues.
