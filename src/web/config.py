"""Web server configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WebSettings:
    """Paths and defaults for the ML Studio API."""

    project_root: Path
    images_dir: Path
    csv_path: Path
    dataset_dir: Path
    dataset_yaml: Path
    stages_dir: Path
    frontend_dist: Path
    low_confidence_threshold: float = 0.7
    host: str = "0.0.0.0"
    port: int = 8765

    @classmethod
    def default(cls) -> WebSettings:
        root = _project_root()
        return cls(
            project_root=root,
            images_dir=root / "training" / "test_images",
            csv_path=root / "output" / "results.csv",
            dataset_dir=root / "dataset",
            dataset_yaml=root / "dataset" / "dataset.yaml",
            stages_dir=root / "output" / "stages",
            frontend_dist=root / "frontend" / "dist",
        )
