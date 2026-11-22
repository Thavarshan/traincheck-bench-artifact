"""Download and verify the exact public UCI files used by the held-out study."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
DATA_FILES = {
    "optdigits.tra": (
        UCI + "optdigits/optdigits.tra",
        "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd",
    ),
    "optdigits.tes": (
        UCI + "optdigits/optdigits.tes",
        "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8",
    ),
    "wine.data": (
        UCI + "wine/wine.data",
        "6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659",
    ),
}


def read_verified(data_dir: Path, name: str) -> bytes:
    path = data_dir / name
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != DATA_FILES[name][1]:
        raise ValueError(f"Dataset checksum mismatch: {path}")
    return data


def _download(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:
                return response.read(1_000_001)
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
    raise RuntimeError("Download retry loop exited unexpectedly")


def fetch(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in DATA_FILES.items():
        path = data_dir / name
        if path.is_file():
            read_verified(data_dir, name)
        else:
            data = _download(url)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Downloaded data failed checksum verification: {url}")
            with tempfile.NamedTemporaryFile(dir=data_dir, delete=False) as staged:
                temporary = Path(staged.name)
                try:
                    staged.write(data)
                    staged.close()
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        print(f"{name}: {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    fetch(args.data_dir)


if __name__ == "__main__":
    main()
