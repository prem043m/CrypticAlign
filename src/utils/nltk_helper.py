import shutil
import time
from pathlib import Path

import nltk


REQUIRED_RESOURCES = (
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
)


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if they are not already available."""
    for resource_path, package_name in REQUIRED_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                # Check for zip version specifically before cleaning up/redownloading
                nltk.data.find(f"{resource_path}.zip")
            except LookupError:
                _recover_resource(package_name)
        except Exception:
            _recover_resource(package_name)


def _recover_resource(package_name: str) -> None:
    _cleanup_resource(package_name)

    for _ in range(3):
        try:
            nltk.download(package_name, quiet=True, force=True)
            return
        except PermissionError:
            time.sleep(0.25)
        except OSError:
            time.sleep(0.25)


def _cleanup_resource(package_name: str) -> None:
    """Remove broken local NLTK package files before redownloading."""
    for base_dir in nltk.data.path:
        base_path = Path(base_dir)
        candidates = [
            base_path / f"{package_name}.zip",
            base_path / package_name,
            base_path / "corpora" / f"{package_name}.zip",
            base_path / "corpora" / package_name,
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            if candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)
            else:
                try:
                    candidate.unlink()
                except OSError:
                    pass
