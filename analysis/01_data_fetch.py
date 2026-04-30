"""
Step 1: Data acquisition from Kidney Cell Atlas
Source: Stewart et al. Science 2019 / Lake et al. Nature 2023
URL: https://www.kidneycellatlas.org/
File: Mature_Full_v3.h5ad (cellgeni.cog.sanger.ac.uk)
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

H5AD_URL = "https://cellgeni.cog.sanger.ac.uk/kidneycellatlas/Mature_Full_v3.h5ad"
H5AD_PATH = DATA_DIR / "Mature_Full_v3.h5ad"
EXPECTED_SIZE = 1_150_421_376

def download_with_resume(url: str, dest: Path):
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
    print(f"  Downloading {url}")
    print(f"  Resuming from byte {existing:,}" if existing else "  Starting fresh")
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing > 0 else "wb"
    with urllib.request.urlopen(req) as resp, open(dest, mode) as out:
        total = int(resp.headers.get("Content-Length", 0)) + existing
        downloaded = existing
        chunk = 1024 * 1024
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            downloaded += len(buf)
            pct = downloaded / max(total, 1) * 100
            print(f"\r  Progress: {downloaded/(1024**2):.1f} MB / {total/(1024**2):.1f} MB ({pct:.1f}%)", end="")
    print()

def validate_file(path: Path) -> bool:
    size = path.stat().st_size
    print(f"  File size: {size:,} bytes (expected {EXPECTED_SIZE:,})")
    return size == EXPECTED_SIZE

def main():
    print("=== Step 1: Kidney Cell Atlas Data Fetch ===")
    if H5AD_PATH.exists() and H5AD_PATH.stat().st_size == EXPECTED_SIZE:
        print("  File already complete. Skipping download.")
    else:
        download_with_resume(H5AD_URL, H5AD_PATH)

    if not validate_file(H5AD_PATH):
        print("  WARNING: File size mismatch. Re-run to resume.")
        sys.exit(1)
    print("  Data acquisition complete.")

if __name__ == "__main__":
    main()
