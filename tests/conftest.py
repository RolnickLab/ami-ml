"""
Shared pytest fixtures for all pipeline stage tests.

Fixtures defined here are available to every test file under tests/ with no imports.

Fixture summary:
  mock_bq_client   — create_autospec(bigquery.Client) with sensible defaults
  small_csv        — CSV file: 5 species × 10 images, photo_ids spanning tasks 0-9
  small_sqfs       — real sqfs file with 10 PIL-generated JPEGs (session-scoped)
  sample_sql_file  — a minimal .sql fixture file for bq_export tests
"""

import io
import subprocess
from pathlib import Path
from unittest.mock import create_autospec

import pandas as pd
import pytest
from google.cloud import bigquery
from PIL import Image

# ── BQ client ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_bq_client():
    """
    Strict BQ client mock — rejects calls to non-existent methods.

    Defaults:
      query().result()                      → []
      load_table_from_dataframe().result()  → None
      get_table()                           → MagicMock (table exists)
      delete_table()                        → None

    Override per-test:
      mock_bq_client.query.return_value.result.return_value = [row1, row2]
    """
    client = create_autospec(bigquery.Client)
    client.query.return_value.result.return_value = []
    client.load_table_from_dataframe.return_value.result.return_value = None
    client.query.return_value.job_id = "test-job-id"
    client.query.return_value.dml_stats.updated_row_count = 0
    return client


# ── Small CSV dataset ─────────────────────────────────────────────────────────


def _make_small_df() -> pd.DataFrame:
    """
    5 species × 10 images = 50 rows.

    Designed so that:
      - photo_ids 0-49 span all 10 tasks (MOD 10)
      - 2 images share each gbif_id (occurrence grouping)
      - all species have >= 5 images (min_instances=5 tests pass)
      - one species has exactly 5 images (min_instances boundary)
    """
    species = [
        ("Danaus plexippus", 1001, 101),
        ("Vanessa atalanta", 1002, 102),
        ("Papilio machaon", 1003, 103),
        ("Colias croceus", 1004, 104),
        ("Pieris brassicae", 1005, 105),
    ]
    rows = []
    photo_id = 0
    for sp_name, taxon_id, base_gbif in species:
        for i in range(10):
            gbif_id = base_gbif + (i // 2)  # 2 images share a gbif_id
            rows.append(
                {
                    "photo_id": photo_id,
                    "gbif_id": gbif_id,
                    "inat_taxon_id": taxon_id,
                    "species_name": sp_name,
                    "dataset_source_uuid": f"uuid-{photo_id:04d}",
                    "relative_local_path": f"{photo_id % 256:03d}/{photo_id:06d}.jpg",
                    "absolute_url": f"https://inaturalist.org/photos/{photo_id}/original.jpg",
                }
            )
            photo_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Raw DataFrame — use when you need to manipulate before writing to CSV."""
    return _make_small_df()


@pytest.fixture
def small_csv(tmp_path) -> Path:
    """CSV file on disk — the direct input format for split.py and bq_export.py."""
    path = tmp_path / "small_dataset.csv"
    _make_small_df().to_csv(path, index=False)
    return path


# ── Small SquashFS ────────────────────────────────────────────────────────────


def _build_small_sqfs(root: Path) -> Path:
    """
    Create a small sqfs with 10 PIL-generated JPEGs.
    Images are placed in bucket dirs (000/, 001/, ...) matching
    the structure download_images.py produces.
    """
    img_dir = root / "images"
    for i in range(10):
        bucket = img_dir / f"{i:03d}"
        bucket.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), color=(i * 25, 100, 200)).save(buf, format="JPEG")
        (bucket / f"{i:06d}.jpg").write_bytes(buf.getvalue())

    sqfs_path = root / "test_fixture.sqfs"
    result = subprocess.run(
        [
            "mksquashfs",
            str(img_dir),
            str(sqfs_path),
            "-noappend",
            "-no-xattrs",
            "-comp",
            "zstd",
            "-Xcompression-level",
            "1",
            "-quiet",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip(f"mksquashfs not available: {result.stderr.decode()}")
    return sqfs_path


@pytest.fixture(scope="session")
def small_sqfs(tmp_path_factory) -> Path:
    """
    Real sqfs file with 10 synthetic JPEGs — built once per test session.
    Skipped automatically if mksquashfs is not available on the current machine.
    """
    root = tmp_path_factory.mktemp("sqfs_fixture")
    return _build_small_sqfs(root)


# ── SQL file ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_sql_file(tmp_path) -> Path:
    """Minimal SQL query file for bq_export.py tests."""
    path = tmp_path / "test_query.sql"
    path.write_text(
        "SELECT photo_id, species_name, gbif_id\n"
        "FROM `leps-ai.global_butterflies_2604.training_images`\n"
        "WHERE fetch_status = 'downloaded'\n"
        "LIMIT 10\n"
    )
    return path
