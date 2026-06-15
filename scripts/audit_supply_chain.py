"""Supply-chain audit: scan dependencies for known CVEs, license compatibility.

Run weekly in CI. Fail-fast on high-severity CVEs.

Exit semantics:
    Without --strict (permissive / local dev):
        - missing tools (pip-audit, pip-licenses) downgrade to a WARN and the
          run can still PASS; only *found* vulnerabilities/blocked licenses fail.
    With --strict (CI gate, FAIL-CLOSED):
        - pip-audit MUST be installed and MUST run; if it is missing, errors,
          or finds any vulnerability, the audit FAILS (non-zero exit).
        - unknown/unclassified licenses are also treated as failures.
      This closes the previous fail-open hole where a missing scanner returned a
      silent WARN and CI went green without ever checking for CVEs.

Usage:
    python -m scripts.audit_supply_chain [--strict] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Whitelist: only these license families are compatible with Apache 2.0 redistribution
LICENSE_ALLOWLIST = {
    "Apache-2.0", "Apache 2.0", "Apache Software License",
    "MIT", "MIT License",
    "BSD-3-Clause", "BSD-2-Clause", "BSD",
    "Python-2.0", "PSF", "PSF-2.0",
    "ISC",
}

# Block: GPL-family (AGPL especially) is incompatible with our redistribution model
LICENSE_BLOCKLIST = {"AGPL-3.0", "GPL-3.0", "LGPL-3.0", "GPL-2.0"}

# Sentinel meaning "the check could not run" (tool missing, crashed, bad output).
# Kept distinct from a real count of 0 so --strict can fail-close on it.
COULD_NOT_RUN = -1


def have_command(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_pip_audit() -> int:
    """Run pip-audit on installed packages. Returns count of vulnerabilities found."""
    if not have_command("pip-audit"):
        print("WARN: pip-audit not installed. Run `pip install pip-audit` first.")
        return COULD_NOT_RUN

    print("Running pip-audit ...")
    try:
        result = subprocess.run(
            ["pip-audit", "--format=json"],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("ERROR: pip-audit timed out after 120s.")
        return COULD_NOT_RUN

    if result.returncode != 0 and not result.stdout:
        print(f"ERROR: pip-audit failed:\n{result.stderr}")
        return COULD_NOT_RUN

    try:
        data = json.loads(result.stdout) if result.stdout.strip() else []
    except json.JSONDecodeError:
        print("ERROR: pip-audit returned non-JSON output:")
        print(result.stdout[:1000])
        return COULD_NOT_RUN

    vulns = data.get("vulnerabilities", []) if isinstance(data, dict) else data
    if not vulns:
        print("  pip-audit: no known vulnerabilities found.")
        return 0

    print(f"  pip-audit: {len(vulns)} vulnerabilities found:")
    for v in vulns:
        print(f"    - {v.get('name', 'unknown')} {v.get('version', '')} -> "
              f"{v.get('vuln_id', '?')} ({v.get('description', '')[:80]})")
    return len(vulns)


def check_license_compatibility(strict: bool = False) -> int:
    """Check installed package licenses against allow/block lists."""
    if not have_command("pip-licenses"):
        print("WARN: pip-licenses not installed. Run `pip install pip-licenses`.")
        return COULD_NOT_RUN

    print("Checking license compatibility ...")
    try:
        result = subprocess.run(
            ["pip-licenses", "--format=json"],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        print("ERROR: pip-licenses timed out after 60s.")
        return COULD_NOT_RUN
    if result.returncode != 0:
        print(f"ERROR: pip-licenses failed:\n{result.stderr}")
        return COULD_NOT_RUN

    try:
        pkgs = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("ERROR: pip-licenses returned non-JSON output.")
        return COULD_NOT_RUN

    blocked = []
    unknown = []
    for pkg in pkgs:
        name = pkg.get("Name", "?")
        lic = pkg.get("License", "UNKNOWN")
        normalized_lic = lic.strip()
        if normalized_lic in LICENSE_BLOCKLIST:
            blocked.append((name, lic))
        elif not any(allow in normalized_lic for allow in LICENSE_ALLOWLIST):
            unknown.append((name, lic))

    if blocked:
        print(f"  BLOCKED licenses ({len(blocked)}):")
        for pkg, pkg_lic in blocked:
            print(f"    {pkg} -> {pkg_lic}")
    if unknown:
        print(f"  UNKNOWN/UNCLASSIFIED licenses ({len(unknown)}):")
        for pkg, pkg_lic in unknown[:20]:
            print(f"    {pkg} -> {pkg_lic}")

    return len(blocked) + (len(unknown) if strict else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Treat unknown licenses as failures")
    parser.add_argument("--out", default=None, help="Write JSON report to path")
    args = parser.parse_args()

    print("=" * 70)
    print("AION-NEXUS supply-chain audit")
    print("=" * 70)

    n_vulns = run_pip_audit()
    n_lic = check_license_compatibility(strict=args.strict)

    audit_ran = n_vulns != COULD_NOT_RUN
    licenses_ran = n_lic != COULD_NOT_RUN

    summary = {
        "strict": args.strict,
        "vulnerabilities_found": n_vulns,
        "license_issues": n_lic,
        "pip_audit_ran": audit_ran,
        "pip_licenses_ran": licenses_ran,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nSaved report to {args.out}")

    print("=" * 70)

    # --- FAIL-CLOSED gate (only in --strict) -------------------------------
    # The CVE scanner is the load-bearing control. In --strict it MUST have
    # run; a missing/broken pip-audit is a hard failure, never a silent pass.
    if args.strict and not audit_ran:
        print("FAIL (strict): pip-audit could not run - refusing to certify the "
              "dependency tree as clean. Install pip-audit and retry.")
        return 3

    # Real findings always fail (with or without --strict).
    found_vulns = n_vulns if audit_ran else 0
    found_lic = n_lic if licenses_ran else 0
    if found_vulns > 0 or found_lic > 0:
        print(f"FAIL: {found_vulns} vulnerabilit(ies) + {found_lic} license issue(s).")
        return 1

    # Some checks could not run but we are NOT strict: warn, don't fail.
    if not audit_ran or not licenses_ran:
        missing = []
        if not audit_ran:
            missing.append("pip-audit (CVE scan)")
        if not licenses_ran:
            missing.append("pip-licenses (license scan)")
        print(f"WARN: skipped checks - {', '.join(missing)}. "
              "Run with the tooling installed (or use --strict in CI to fail-close).")
        return 2

    print("PASS: no vulnerabilities or license issues.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
