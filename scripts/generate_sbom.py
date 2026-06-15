"""Generate a CycloneDX SBOM (JSON) of the installed Python environment.

A Software Bill of Materials lets a customer (or auditor) see exactly which
third-party components ship inside AION-NEXUS, with versions and package URLs,
so known-vulnerability scanners (e.g. Grype, Trivy, Dependency-Track) and
license review can run against the *actual* installed tree — not a hand-curated
list that drifts from reality.

Two code paths:

1. If the `cyclonedx-bom` tool is installed, shell out to it (`cyclonedx-py env`)
   for a richer, spec-complete document.
2. Otherwise fall back to a pure-stdlib generator that walks
   `importlib.metadata` and emits a *minimal but valid* CycloneDX 1.5 JSON
   document (component name, version, `pkg:pypi/...` purl, declared license).

The fallback has NO third-party dependencies, so this script always runs in a
clean environment and in CI without extra install steps.

Honesty note: the fallback SBOM is intentionally minimal. It does NOT include
dependency-graph edges, hashes of the installed artifacts, or full license
expressions for every package. For a release-grade, signed SBOM, install
`cyclonedx-bom` (the script will use it automatically). See docs/SUPPLY_CHAIN.md.

Usage:
    python scripts/generate_sbom.py [--out sbom.cyclonedx.json]
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata as ilmd
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

DEFAULT_OUT = "sbom.cyclonedx.json"
CYCLONEDX_SPEC_VERSION = "1.5"


def _now_utc_iso() -> str:
    """RFC 3339 / ISO 8601 UTC timestamp, e.g. 2026-06-15T10:00:00Z."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _root_version() -> str:
    """Version of the aion-nexus package, best effort (for SBOM metadata).

    Prefers installed-distribution metadata; falls back to the in-tree single
    source of truth (`aion_nexus.version.__version__`) so the value is correct
    even when run as a bare script from the repo root (where the editable-install
    metadata finder is not always on the path).
    """
    for dist_name in ("aion-nexus", "aion_nexus"):
        try:
            return ilmd.version(dist_name)
        except ilmd.PackageNotFoundError:
            continue
    # Fallback: read the in-tree version module directly.
    try:
        repo_root = Path(__file__).resolve().parent.parent
        version_file = repo_root / "aion_nexus" / "version.py"
        for line in version_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("__version__"):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return "0.0.0+unknown"


def _license_of(meta: ilmd.PackageMetadata) -> str | None:
    """Extract a best-effort license string from package metadata.

    PyPI metadata is messy: the `License` field is often empty and the real
    signal lives in the `License :: ...` Trove classifiers. We try both.
    """
    lic = meta.get("License")
    if lic and lic.strip() and lic.strip().upper() != "UNKNOWN":
        # Some packages dump the full license TEXT here; keep it short.
        first_line = lic.strip().splitlines()[0]
        return first_line[:120]
    classifiers = meta.get_all("Classifier") or []
    for c in classifiers:
        if c.startswith("License :: "):
            # e.g. "License :: OSI Approved :: MIT License"
            return c.split(" :: ")[-1]
    return None


def generate_sbom_stdlib() -> dict:
    """Build a minimal valid CycloneDX 1.5 JSON document from importlib.metadata."""
    components: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for dist in ilmd.distributions():
        meta = dist.metadata
        name = meta["Name"]
        version = dist.version or "0"
        if not name:
            continue
        key = (name.lower(), version)
        if key in seen:  # editable + wheel installs can double-list
            continue
        seen.add(key)

        component: dict = {
            "type": "library",
            "name": name,
            "version": version,
            # PEP 503 normalized name in the purl namespace.
            "purl": f"pkg:pypi/{name.lower().replace('_', '-')}@{version}",
            "bom-ref": f"pkg:pypi/{name.lower().replace('_', '-')}@{version}",
        }
        lic = _license_of(meta)
        if lic:
            component["licenses"] = [{"license": {"name": lic}}]
        components.append(component)

    components.sort(key=lambda c: c["name"].lower())

    return {
        "bomFormat": "CycloneDX",
        "specVersion": CYCLONEDX_SPEC_VERSION,
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": _now_utc_iso(),
            "tools": [
                {
                    "vendor": "AION-NEXUS",
                    "name": "generate_sbom.py (stdlib fallback)",
                    "version": _root_version(),
                }
            ],
            "component": {
                "type": "application",
                "name": "aion-nexus",
                "version": _root_version(),
                "purl": f"pkg:pypi/aion-nexus@{_root_version()}",
                "bom-ref": f"pkg:pypi/aion-nexus@{_root_version()}",
            },
        },
        "components": components,
    }


def generate_sbom_cyclonedx_tool(out_path: Path) -> bool:
    """Try the cyclonedx-bom tool. Returns True on success, False to fall back.

    The modern entry point is `cyclonedx-py env`. Older releases expose
    `cyclonedx-py` with positional args; if the invocation fails we fall back
    rather than shipping a broken file.
    """
    if not _have("cyclonedx-py"):
        return False
    cmd = ["cyclonedx-py", "env", "--output-format", "JSON", "--outfile", str(out_path)]
    print(f"Found cyclonedx-py; running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"  cyclonedx-py invocation failed ({exc}); falling back to stdlib.")
        return False
    if result.returncode != 0:
        print(f"  cyclonedx-py exited {result.returncode}; falling back to stdlib.")
        if result.stderr:
            print(f"  stderr: {result.stderr[:400]}")
        return False
    if not out_path.exists():
        print("  cyclonedx-py reported success but no file written; falling back.")
        return False
    # Validate it parses as JSON before trusting it.
    try:
        json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("  cyclonedx-py output is not valid JSON; falling back to stdlib.")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output path for the CycloneDX JSON SBOM (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Skip cyclonedx-py even if installed (use the stdlib generator).",
    )
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    used_tool = False
    if not args.force_fallback:
        used_tool = generate_sbom_cyclonedx_tool(out_path)

    if not used_tool:
        print("Generating SBOM via stdlib fallback (importlib.metadata) ...")
        sbom = generate_sbom_stdlib()
        out_path.write_text(json.dumps(sbom, indent=2), encoding="utf-8")
        n = len(sbom["components"])
        print(f"  Wrote {n} components.")

    # Final sanity check: the file we just wrote must parse and look like CycloneDX.
    try:
        doc = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: generated SBOM is not valid JSON: {exc}")
        return 1
    if doc.get("bomFormat") != "CycloneDX":
        print("ERROR: generated file is not a CycloneDX document.")
        return 1

    print(f"SBOM written to: {out_path}")
    print(f"  format: CycloneDX {doc.get('specVersion', '?')}, "
          f"components: {len(doc.get('components', []))}, "
          f"generator: {'cyclonedx-py' if used_tool else 'stdlib-fallback'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
