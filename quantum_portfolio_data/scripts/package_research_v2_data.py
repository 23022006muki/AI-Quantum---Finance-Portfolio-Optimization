from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "research_v2" / "raw" / "corporate_actions"
ARCHIVE = ROOT / "outputs" / "research_v2" / "raw" / "corporate_actions_raw_archive.zip"
MANIFEST = ROOT / "outputs" / "research_v2" / "raw" / "RAW_ARCHIVE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
    if not SOURCE.exists():
        raise SystemExit(f"Missing raw corporate-action directory: {SOURCE}")
    files = sorted(path for path in SOURCE.rglob("*") if path.is_file())
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(SOURCE.parent).as_posix())
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(SOURCE),
        "archive": str(ARCHIVE),
        "file_count": len(files),
        "uncompressed_bytes": sum(path.stat().st_size for path in files),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "note": "Public raw HTML and checkpoint payloads; no cookies, tokens or credentials.",
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
