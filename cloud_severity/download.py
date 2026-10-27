"""Download the public RICE archive with resumable transfer and safe extraction."""

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

RICE_ID = "1CricZtIj28BGFvkD_x-W8fSexPiDtgHk"


def extract_archive(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            # Reject symbolic-link members instead of materializing ambiguous paths.
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Symbolic link not supported: {info.filename}")
        bad = z.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP member: {bad}")
        z.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description="Download the official ~540 MB RICE archive")
    parser.add_argument("--output", default="data")
    parser.add_argument("--archive", help="Use an already downloaded ZIP instead")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive) if args.archive else root / "RICE_DATASET.zip"
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        import gdown
        partial = root / "RICE_DATASET.zip.part"
        for attempt in range(3):
            try:
                result = gdown.download(id=RICE_ID, output=str(partial), use_cookies=False, resume=True)
                if not result or not zipfile.is_zipfile(partial):
                    raise RuntimeError("Provider did not return a ZIP archive")
                with zipfile.ZipFile(partial) as z:
                    if z.testzip(): raise RuntimeError("Downloaded ZIP failed its CRC check")
                partial.replace(archive)
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError("RICE direct download failed after 3 attempts. Google Drive may be rate-limited. Retry later, or supply the official ZIP with --archive.") from exc
                time.sleep(3)
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''): digest.update(chunk)
    extract_archive(archive, root / "RICE_DATASET")
    provenance={"source":"https://github.com/BUPTLdy/RICE_DATASET", "google_drive_id":RICE_ID,
                "archive":str(archive.resolve()),"sha256":digest.hexdigest(),"bytes":archive.stat().st_size}
    (root/"archive_provenance.json").write_text(json.dumps(provenance,indent=2))
    from .discovery import discover_rice
    sources=discover_rice(root/"RICE_DATASET")
    (root/"sources.json").write_text(json.dumps(sources,indent=2))
    print(f"Extracted to {root / 'RICE_DATASET'}. Inspect folders before preparing a manifest.")


if __name__ == "__main__":
    main()