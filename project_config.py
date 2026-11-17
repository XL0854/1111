"""Machine-local paths used by the training and evaluation entry point."""

from pathlib import Path


DATA_ROOT = Path("/workspace/python_pro/newropstitch/Data")
UDIS_ROOT = DATA_ROOT / "UDIS-D"
TRAIN_DATA = UDIS_ROOT / "training"
TEST_DATA = UDIS_ROOT / "testing"
DEPTH_DATA = Path("/workspace/python_pro/test1/sxl/depth_cache/UDIS-D-training")


def require_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise SystemExit(f"Configured {label} directory does not exist: {path}")
    return path
