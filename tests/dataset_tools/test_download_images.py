"""
Failure scenario tests for download_images.py.

All tests use mocks — no network, no BQ, no filesystem writes beyond tmp_path.
Run with: pytest tests/dataset_tools/test_download_images.py -v
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import requests

import src.dataset_tools.bq_squashfs.download_images as di


# ── helpers ───────────────────────────────────────────────────────────────────

def make_response(status_code: int, content: bytes = b"\xff\xd8\xff\xe0JFIF"):
    """Build a minimal mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.iter_content.return_value = [content]
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"HTTP {status_code}"
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def make_mock_session(*responses):
    """Return a mock session whose .get() yields responses in order."""
    session = MagicMock()
    session.get.side_effect = list(responses)
    return session


# ── _fetch_with_retry ─────────────────────────────────────────────────────────

class TestFetchWithRetry:

    def test_success_on_first_attempt(self, tmp_path):
        dest = tmp_path / "img.jpg"
        session = make_mock_session(make_response(200, b"IMAGE"))
        with patch.object(di, "_get_session", return_value=session), \
             patch("time.sleep"):
            di._fetch_with_retry("http://example.com/img.jpg", dest)
        assert dest.read_bytes() == b"IMAGE"
        assert session.get.call_count == 1

    def test_429_retries_then_succeeds(self, tmp_path):
        dest = tmp_path / "img.jpg"
        session = make_mock_session(
            make_response(429),
            make_response(429),
            make_response(200, b"IMAGE"),
        )
        with patch.object(di, "_get_session", return_value=session), \
             patch("time.sleep"):
            di._fetch_with_retry("http://example.com/img.jpg", dest)
        assert session.get.call_count == 3
        assert dest.read_bytes() == b"IMAGE"

    def test_503_retries_then_succeeds(self, tmp_path):
        dest = tmp_path / "img.jpg"
        session = make_mock_session(
            make_response(503),
            make_response(200, b"IMAGE"),
        )
        with patch.object(di, "_get_session", return_value=session), \
             patch("time.sleep"):
            di._fetch_with_retry("http://example.com/img.jpg", dest)
        assert session.get.call_count == 2

    def test_connection_error_errno16_retries(self, tmp_path):
        """Errno 16 (device/resource busy — too many sockets) retries."""
        dest = tmp_path / "img.jpg"
        errno16 = requests.exceptions.ConnectionError("[Errno 16] Device or resource busy")
        session = make_mock_session(errno16, errno16, make_response(200, b"IMAGE"))
        with patch.object(di, "_get_session", return_value=session), \
             patch("time.sleep"):
            di._fetch_with_retry("http://example.com/img.jpg", dest)
        assert session.get.call_count == 3

    def test_timeout_retries_then_succeeds(self, tmp_path):
        dest = tmp_path / "img.jpg"
        session = make_mock_session(
            requests.exceptions.Timeout(),
            make_response(200, b"IMAGE"),
        )
        with patch.object(di, "_get_session", return_value=session), \
             patch("time.sleep"):
            di._fetch_with_retry("http://example.com/img.jpg", dest)
        assert session.get.call_count == 2

    def test_404_raises_immediately_no_retry(self, tmp_path):
        """404 is not in RETRY_STATUSES — raises without retrying."""
        dest = tmp_path / "img.jpg"
        session = make_mock_session(make_response(404))
        with patch.object(di, "_get_session", return_value=session), \
             patch("time.sleep"), \
             pytest.raises(requests.exceptions.HTTPError):
            di._fetch_with_retry("http://example.com/img.jpg", dest)
        assert session.get.call_count == 1

    def test_exhausted_retries_on_connection_error_raises(self, tmp_path):
        """After MAX_RETRIES connection errors, raises the last one."""
        dest = tmp_path / "img.jpg"
        err = requests.exceptions.ConnectionError("connection refused")
        session = make_mock_session(*([err] * (di._MAX_RETRIES + 1)))
        with patch.object(di, "_get_session", return_value=session), \
             patch("time.sleep"), \
             pytest.raises(requests.exceptions.ConnectionError):
            di._fetch_with_retry("http://example.com/img.jpg", dest)
        assert session.get.call_count == di._MAX_RETRIES + 1

    def test_exhausted_retries_on_timeout_raises(self, tmp_path):
        dest = tmp_path / "img.jpg"
        session = make_mock_session(*([requests.exceptions.Timeout()] * (di._MAX_RETRIES + 1)))
        with patch.object(di, "_get_session", return_value=session), \
             patch("time.sleep"), \
             pytest.raises(requests.exceptions.Timeout):
            di._fetch_with_retry("http://example.com/img.jpg", dest)


# ── download_and_verify ───────────────────────────────────────────────────────

class TestDownloadAndVerify:

    ROW = {
        "dataset_source_uuid": "uuid-001",
        "absolute_url": "http://example.com/img.jpg",
        "relative_local_path": "000/img.jpg",
    }

    def test_success(self, tmp_path):
        """Valid JPEG → fetch_status=downloaded, dimensions populated."""
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (64, 48)).save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        with patch.object(di, "_fetch_with_retry") as mock_fetch:
            def write_file(url, dest):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(jpeg_bytes)
            mock_fetch.side_effect = write_file

            result = di.download_and_verify(self.ROW, tmp_path)

        assert result["fetch_status"] == "downloaded"
        assert result["image_width"] == 64
        assert result["image_height"] == 48
        assert result["corrupted"] is False
        assert result["image_size"] > 0

    def test_network_failure_recorded_as_failed(self, tmp_path):
        """Network error → fetch_status=failed, no image on disk."""
        with patch.object(di, "_fetch_with_retry",
                          side_effect=Exception("connection refused")):
            result = di.download_and_verify(self.ROW, tmp_path)

        assert result["fetch_status"] == "failed"
        assert result["image_width"] is None

    def test_corrupted_image_recorded_as_corrupted(self, tmp_path):
        """Truncated/invalid image bytes → fetch_status=corrupted."""
        with patch.object(di, "_fetch_with_retry") as mock_fetch:
            def write_garbage(url, dest):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"not an image at all")
            mock_fetch.side_effect = write_garbage

            result = di.download_and_verify(self.ROW, tmp_path)

        assert result["fetch_status"] == "corrupted"
        assert result["corrupted"] is True
        assert result["image_width"] is None


# ── write_results_to_bq ───────────────────────────────────────────────────────

class TestWriteResultsToBq:

    RESULTS = [
        {"dataset_source_uuid": "u1", "fetch_status": "downloaded",
         "image_width": 100, "image_height": 80, "image_size": 5000, "corrupted": False},
    ]

    def test_success(self):
        client = MagicMock()
        client.load_table_from_dataframe.return_value.result.return_value = None
        di.write_results_to_bq(client, self.RESULTS, "test_table")
        assert client.load_table_from_dataframe.call_count == 1

    def test_retries_on_transient_error(self):
        """First write fails, second succeeds — should not raise."""
        client = MagicMock()
        client.load_table_from_dataframe.side_effect = [
            Exception("BQ transient error"),
            MagicMock(result=MagicMock(return_value=None)),
        ]
        with patch("time.sleep"):
            di.write_results_to_bq(client, self.RESULTS, "test_table", max_retries=2)
        assert client.load_table_from_dataframe.call_count == 2

    def test_exhausted_retries_raises(self):
        """All retries fail → raises the last exception."""
        client = MagicMock()
        client.load_table_from_dataframe.side_effect = Exception("BQ down")
        with patch("time.sleep"), pytest.raises(Exception, match="BQ down"):
            di.write_results_to_bq(client, self.RESULTS, "test_table", max_retries=2)
        assert client.load_table_from_dataframe.call_count == 2


# ── pack_chunk_to_sqfs ────────────────────────────────────────────────────────

class TestPackChunkToSqfs:

    def test_success(self, tmp_path):
        staging = tmp_path / "staging"
        (staging / "000").mkdir(parents=True)
        (staging / "000" / "img.jpg").write_bytes(b"x")
        fake_sqfs = staging / "chunk_0001.sqfs"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # Simulate mksquashfs creating the output file
            fake_sqfs.write_bytes(b"sqfs")
            result = di.pack_chunk_to_sqfs(staging, chunk_num=1)

        assert result == fake_sqfs
        assert mock_run.call_count == 1

    def test_mksquashfs_failure_raises_runtime_error(self, tmp_path):
        """Non-zero mksquashfs exit → RuntimeError so SLURM marks task failed."""
        staging = tmp_path / "staging"
        (staging / "000").mkdir(parents=True)
        (staging / "000" / "img.jpg").write_bytes(b"x")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            with pytest.raises(RuntimeError, match="mksquashfs failed"):
                di.pack_chunk_to_sqfs(staging, chunk_num=1)

    def test_empty_staging_returns_none(self, tmp_path):
        """No images in staging → returns None without calling mksquashfs."""
        staging = tmp_path / "staging"
        staging.mkdir()
        with patch("subprocess.run") as mock_run:
            result = di.pack_chunk_to_sqfs(staging, chunk_num=1)
        assert result is None
        mock_run.assert_not_called()


# ── merge_chunk_into_training_images ─────────────────────────────────────────

class TestMergeChunkIntoTrainingImages:

    def test_empty_results_skips_merge(self):
        """No successful results → no BQ calls."""
        client = MagicMock()
        results = [{"dataset_source_uuid": "u1", "fetch_status": "failed",
                    "image_width": None, "image_height": None,
                    "image_size": None, "corrupted": None}]
        n = di.merge_chunk_into_training_images(
            client, results, "training_table", "downloads_table"
        )
        assert n == 0
        client.load_table_from_dataframe.assert_not_called()

    def test_successful_results_trigger_merge(self):
        """Downloaded rows → temp table load + MERGE + temp table delete."""
        client = MagicMock()
        client.load_table_from_dataframe.return_value.result.return_value = None
        job = MagicMock()
        job.dml_stats.updated_row_count = 2
        client.query.return_value = job

        results = [
            {"dataset_source_uuid": "u1", "fetch_status": "downloaded",
             "image_width": 100, "image_height": 80, "image_size": 5000, "corrupted": False},
            {"dataset_source_uuid": "u2", "fetch_status": "corrupted",
             "image_width": None, "image_height": None, "image_size": 500, "corrupted": True},
        ]
        n = di.merge_chunk_into_training_images(
            client, results, "training_table", "downloads_table"
        )
        assert n == 2
        assert client.load_table_from_dataframe.call_count == 1  # temp table load
        assert client.query.call_count == 1                       # MERGE
        assert client.delete_table.call_count == 1               # cleanup

    def test_temp_table_deleted_even_on_merge_failure(self):
        """Temp table must be cleaned up even if the MERGE query fails."""
        client = MagicMock()
        client.load_table_from_dataframe.return_value.result.return_value = None
        client.query.side_effect = Exception("MERGE failed")

        results = [{"dataset_source_uuid": "u1", "fetch_status": "downloaded",
                    "image_width": 100, "image_height": 80,
                    "image_size": 5000, "corrupted": False}]

        with pytest.raises(Exception, match="MERGE failed"):
            di.merge_chunk_into_training_images(
                client, results, "training_table", "downloads_table"
            )
        client.delete_table.assert_called_once()  # cleanup still ran


# ── warn_chunk_accumulation ───────────────────────────────────────────────────

class TestWarnChunkAccumulation:

    def test_no_warning_below_threshold(self, tmp_path, capsys):
        for i in range(5):
            (tmp_path / f"chunk_{i:04d}.sqfs").write_bytes(b"x")
        di.warn_chunk_accumulation(tmp_path)
        assert "WARNING" not in capsys.readouterr().out

    def test_warning_at_threshold(self, tmp_path, capsys):
        for i in range(di._CHUNK_ACCUMULATION_WARN):
            (tmp_path / f"chunk_{i:04d}.sqfs").write_bytes(b"x")
        di.warn_chunk_accumulation(tmp_path)
        assert "WARNING" in capsys.readouterr().out
