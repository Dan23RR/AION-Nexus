"""Emit an EU AI Act Annex IV technical-documentation EVIDENCE MAP.

This produces a documentation SKELETON (Markdown or JSON) showing which artefacts
AION-NEXUS supplies toward each of the nine Annex IV points and what the provider
must author. It is NOT the technical documentation, NOT a declaration of
conformity, and NOT a statement that the system meets Annex IV — it accelerates a
provider's own dossier. See `aion_nexus.compliance.annex_iv_dossier` for the full
disclaimer.

Usage:
    python -m scripts.generate_annex_iv                         # Markdown to stdout
    python -m scripts.generate_annex_iv --json                  # JSON to stdout
    python -m scripts.generate_annex_iv --metadata meta.json    # fill known facts
    python -m scripts.generate_annex_iv --out annex_iv.md       # write to a file

The optional metadata JSON may set: name, version, intended_purpose, provider,
architecture, datasets, harmonised_standards, documentation_date, model_id.
Anything omitted is reported as provider-owned rather than invented.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aion_nexus.compliance import annex_iv_card, annex_iv_dossier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an EU AI Act Annex IV evidence map.")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="path to a JSON file of known system facts")
    parser.add_argument("--json", action="store_true",
                        help="emit the structured dict as JSON instead of Markdown")
    parser.add_argument("--out", type=Path, default=None,
                        help="write to this file instead of stdout")
    args = parser.parse_args(argv)

    metadata = None
    if args.metadata is not None:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            parser.error("--metadata must contain a JSON object")

    if args.json:
        output = json.dumps(annex_iv_dossier(metadata), indent=2, ensure_ascii=False)
    else:
        output = annex_iv_card(metadata)

    if args.out is not None:
        args.out.write_text(output, encoding="utf-8")
        print(f"Wrote Annex IV evidence map to {args.out}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
