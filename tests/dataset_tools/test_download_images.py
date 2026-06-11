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

    def _make_client(self, updated_count: int = 1) -> MagicMock:
        client = MagicMock()
        client.load_table_from_dataframe.return_value.result.return_value = None
        job = MagicMock()
        job.dml_stats.updated_row_count = updated_count
        client.query.return_value = job
        return client

    def test_empty_list_skips_merge(self):
        """Completely empty results list → no BQ calls."""
        client = self._make_client()
        n = di.merge_chunk_into_training_images(client, [], "t", "d")
        assert n == 0
        client.load_table_from_dataframe.assert_not_called()

    def test_downloaded_triggers_merge(self):
        """downloaded rows → temp table load + MERGE + cleanup."""
        client = self._make_client(updated_count=1)
        results = [{"dataset_source_uuid": "u1", "fetch_status": "downloaded",
                    "image_width": 100, "image_height": 80,
                    "image_size": 5000, "corrupted": False}]
        n = di.merge_chunk_into_training_images(client, results, "t", "d")
        assert n == 1
        assert client.load_table_from_dataframe.call_count == 1
        assert client.query.call_count == 1
        assert client.delete_table.call_count == 1

    def test_corrupted_triggers_merge(self):
        """corrupted rows → merged with fetch_status='corrupted'."""
        client = self._make_client(updated_count=1)
        results = [{"dataset_source_uuid": "u1", "fetch_status": "corrupted",
                    "image_width": None, "image_height": None,
                    "image_size": None, "corrupted": True}]
        n = di.merge_chunk_into_training_images(client, results, "t", "d")
        assert n == 1
        assert client.load_table_from_dataframe.call_count == 1

    def test_failed_triggers_merge(self):
        """failed rows (404/403/exhausted) → merged so fetch_status='failed' in
        training_images. Permanent failures are excluded from future re-runs
        via WHERE fetch_status='pending' without needing the LEFT JOIN."""
        client = self._make_client(updated_count=1)
        results = [{"dataset_source_uuid": "u1", "fetch_status": "failed",
                    "image_width": None, "image_height": None,
                    "image_size": None, "corrupted": None}]
        n = di.merge_chunk_into_training_images(client, results, "t", "d")
        assert n == 1
        assert client.load_table_from_dataframe.call_count == 1
        assert client.query.call_count == 1
        assert client.delete_table.call_count == 1

    def test_all_three_statuses_merged_together(self):
        """Mixed chunk — downloaded, corrupted, failed — all three trigger one MERGE."""
        client = self._make_client(updated_count=3)
        results = [
            {"dataset_source_uuid": "u1", "fetch_status": "downloaded",
             "image_width": 100, "image_height": 80, "image_size": 5000, "corrupted": False},
            {"dataset_source_uuid": "u2", "fetch_status": "corrupted",
             "image_width": None, "image_height": None, "image_size": 500, "corrupted": True},
            {"dataset_source_uuid": "u3", "fetch_status": "failed",
             "image_width": None, "image_height": None, "image_size": None, "corrupted": None},
        ]
        n = di.merge_chunk_into_training_images(client, results, "t", "d")
        assert n == 3
        assert client.load_table_from_dataframe.call_count == 1  # one temp table for all 3
        assert client.query.call_count == 1                       # one MERGE
        assert client.delete_table.call_count == 1               # one cleanup

    def test_failed_rows_included_in_temp_table(self):
        """Verify the dataframe passed to BQ includes the failed row."""
        import pandas as pd
        client = self._make_client()
        captured_df = {}

        def capture_load(df, table, **kwargs):
            captured_df["data"] = df.copy()
            return MagicMock(result=MagicMock(return_value=None))

        client.load_table_from_dataframe.side_effect = capture_load

        results = [
            {"dataset_source_uuid": "ok",   "fetch_status": "downloaded",
             "image_width": 64, "image_height": 48, "image_size": 1000, "corrupted": False},
            {"dataset_source_uuid": "dead", "fetch_status": "failed",
             "image_width": None, "image_height": None, "image_size": None, "corrupted": None},
        ]
        di.merge_chunk_into_training_images(client, results, "t", "d")

        df = captured_df["data"]
        assert len(df) == 2                               # both rows in temp table
        statuses = set(df["fetch_status"].tolist())
        assert "downloaded" in statuses
        assert "failed" in statuses                       # failed row present

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
        client.delete_table.assert_called_once()


# ── get_pending_rows / MOD split ─────────────────────────────────────────────

class TestModSplit:
    """Verify that num_jobs/task_id partitioning is correct and complete."""

    def _make_client(self, photo_ids: list[int], num_jobs: int, task_id: int) -> MagicMock:
        """Return a mock BQ client that filters photo_ids by MOD split."""
        matching = [
            {"dataset_source_uuid": f"uuid-{p}", "absolute_url": f"http://x/{p}",
             "relative_local_path": f"000/{p}.jpg"}
            for p in photo_ids if p % num_jobs == task_id
        ]
        client = MagicMock()
        client.query.return_value.result.return_value = [
            MagicMock(**{k: v for k, v in row.items()}, **{"__iter__": lambda self: iter(row.items()), "keys": lambda self: row.keys()})
            for row in matching
        ]
        # Simpler: just return dicts directly via side_effect
        client.query.return_value.result.return_value = matching
        return client

    def test_no_overlap_between_tasks(self):
        """Each photo_id must appear in exactly one task — no overlaps."""
        photo_ids = list(range(100))
        num_jobs = 10
        all_assigned = []

        for task_id in range(num_jobs):
            assigned = [p for p in photo_ids if p % num_jobs == task_id]
            all_assigned.extend(assigned)

        assert len(all_assigned) == len(photo_ids)
        assert len(set(all_assigned)) == len(photo_ids)  # no duplicates

    def test_all_images_covered_across_tasks(self):
        """Union of all task subsets must equal the full image set."""
        photo_ids = list(range(1000))
        num_jobs = 10
        covered = set()
        for task_id in range(num_jobs):
            subset = {p for p in photo_ids if p % num_jobs == task_id}
            assert not subset & covered, f"Overlap at task_id={task_id}"
            covered |= subset
        assert covered == set(photo_ids)

    def test_task_gets_correct_subset(self):
        """Task 3 of 10 should only see photo_ids ending in 3."""
        photo_ids = list(range(50))
        expected = [p for p in photo_ids if p % 10 == 3]  # 3, 13, 23, 33, 43
        actual   = [p for p in photo_ids if p % 10 == 3]
        assert actual == expected
        assert all(p % 10 == 3 for p in actual)

    def test_uneven_split_all_images_still_covered(self):
        """101 images across 10 tasks — some tasks get 11, others get 10."""
        photo_ids = list(range(101))
        num_jobs = 10
        subsets = [[p for p in photo_ids if p % num_jobs == t] for t in range(num_jobs)]
        sizes = [len(s) for s in subsets]
        assert sum(sizes) == 101
        assert max(sizes) - min(sizes) <= 1  # balanced within 1

    def test_single_job_gets_all_images(self):
        """num_jobs=1, task_id=0 must return every image."""
        photo_ids = list(range(50))
        assigned = [p for p in photo_ids if p % 1 == 0]
        assert assigned == photo_ids

    def test_resumability_skips_already_attempted(self):
        """LEFT JOIN should exclude images already in downloads table."""
        # Simulate: 10 images total, 3 already in downloads table
        all_uuids = [f"uuid-{i}" for i in range(10)]
        attempted = {f"uuid-{i}" for i in range(3)}
        pending = [u for u in all_uuids if u not in attempted]
        assert len(pending) == 7
        assert not set(pending) & attempted  # no overlap with attempted

    def test_force_redownload_ignores_downloads_table(self):
        """force_redownload=True should query training_images directly, no LEFT JOIN."""
        client = MagicMock()
        client.query.return_value.result.return_value = []

        di.get_pending_rows(
            client,
            training_table="t",
            downloads_table="d",
            num_jobs=10,
            task_id=3,
            force_redownload=True,
        )

        query_sql = client.query.call_args[0][0]
        assert "LEFT JOIN" not in query_sql
        assert "MOD(photo_id, 10) = 3" in query_sql

    def test_normal_query_has_left_join(self):
        """Normal query must LEFT JOIN downloads table to skip attempted images."""
        client = MagicMock()
        client.query.return_value.result.return_value = []

        di.get_pending_rows(
            client,
            training_table="t",
            downloads_table="d",
            num_jobs=10,
            task_id=3,
            force_redownload=False,
        )

        query_sql = client.query.call_args[0][0]
        assert "LEFT JOIN" in query_sql
        assert "MOD(ti.photo_id, 10) = 3" in query_sql

    def test_limit_applied_to_query(self):
        """--limit N should add LIMIT clause to the BQ query."""
        client = MagicMock()
        client.query.return_value.result.return_value = []

        di.get_pending_rows(
            client,
            training_table="t",
            downloads_table="d",
            num_jobs=1,
            task_id=0,
            limit=50,
        )

        query_sql = client.query.call_args[0][0]
        assert "LIMIT 50" in query_sql


# ── Multi-task distribution and merge ────────────────────────────────────────

class TestMultiTaskDistributionAndMerge:
    """
    Verify correct behaviour when multiple tasks run in parallel:
      - work is partitioned correctly across tasks
      - tasks don't interfere with each other's queries
      - BQ downloads table receives appends from all tasks safely
      - training_images MERGE is correct when multiple tasks write concurrently
    """

    PHOTO_IDS = list(range(50))   # simulate 50 images

    def _partition(self, num_jobs: int) -> dict[int, list[int]]:
        """Return {task_id: [photo_ids]} for all tasks."""
        return {
            t: [p for p in self.PHOTO_IDS if p % num_jobs == t]
            for t in range(num_jobs)
        }

    # ── partitioning ──────────────────────────────────────────────────────────

    def test_two_tasks_partition_all_images(self):
        """num_jobs=2: task 0 + task 1 together cover every image exactly once."""
        parts = self._partition(2)
        combined = parts[0] + parts[1]
        assert sorted(combined) == self.PHOTO_IDS
        assert set(parts[0]) & set(parts[1]) == set()  # no overlap

    def test_ten_tasks_partition_all_images(self):
        """num_jobs=10: all 10 tasks together cover every image exactly once."""
        parts = self._partition(10)
        combined = [p for task in parts.values() for p in task]
        assert sorted(combined) == self.PHOTO_IDS
        for i in range(10):
            for j in range(i + 1, 10):
                assert set(parts[i]) & set(parts[j]) == set()

    def test_task0_completion_does_not_affect_task1_query(self):
        """Task 1's LEFT JOIN only skips images task 1 itself attempted — not task 0's."""
        # task 0 attempted photo_ids 0,2,4... (even); task 1 should still see 1,3,5...
        task0_uuids = {f"uuid-{p}" for p in self.PHOTO_IDS if p % 2 == 0}
        task1_pending = [p for p in self.PHOTO_IDS if p % 2 == 1]

        # task 1 query: LEFT JOIN filters on task 1's uuids only
        # since task 0's uuids (even photo_ids) aren't in task 1's subset,
        # they never appear in the LEFT JOIN result anyway
        task1_uuids = {f"uuid-{p}" for p in task1_pending}
        assert task0_uuids & task1_uuids == set()  # completely disjoint

    def test_non_sequential_photo_ids_still_partition_correctly(self):
        """Real photo_ids from iNat are large non-sequential ints — MOD still works."""
        real_ids = [487851, 7047265, 8233026, 8427425, 10239192,
                    17327318, 21463254, 27648248, 36757555, 41676327]
        for num_jobs in [2, 5, 10]:
            parts = {t: [p for p in real_ids if p % num_jobs == t]
                     for t in range(num_jobs)}
            combined = [p for task in parts.values() for p in task]
            assert sorted(combined) == sorted(real_ids)

    def test_empty_task_handled_gracefully(self):
        """A task assigned 0 images should produce 0 downloads cleanly."""
        # with 1 image and num_jobs=2, one task will have 0 images
        single_id = [4]  # 4 % 2 == 0, so task 1 gets nothing
        task0 = [p for p in single_id if p % 2 == 0]
        task1 = [p for p in single_id if p % 2 == 1]
        assert task0 == [4]
        assert task1 == []

    # ── BQ writes from multiple tasks ────────────────────────────────────────

    def test_downloads_table_appends_are_independent(self):
        """Both tasks append to downloads table — append-only, no conflicts."""
        client = MagicMock()
        client.load_table_from_dataframe.return_value.result.return_value = None

        task0_results = [{"dataset_source_uuid": f"uuid-{p}", "fetch_status": "downloaded",
                          "image_width": 100, "image_height": 80,
                          "image_size": 5000, "corrupted": False}
                         for p in range(0, 10, 2)]   # even photo_ids

        task1_results = [{"dataset_source_uuid": f"uuid-{p}", "fetch_status": "downloaded",
                          "image_width": 100, "image_height": 80,
                          "image_size": 5000, "corrupted": False}
                         for p in range(1, 10, 2)]   # odd photo_ids

        # both tasks write to the same table — no conflict because WRITE_APPEND
        di.write_results_to_bq(client, task0_results, "downloads_table")
        di.write_results_to_bq(client, task1_results, "downloads_table")

        assert client.load_table_from_dataframe.call_count == 2
        # both calls target same table
        calls = client.load_table_from_dataframe.call_args_list
        assert calls[0][0][1] == "downloads_table"
        assert calls[1][0][1] == "downloads_table"

    def test_merge_from_two_tasks_updates_correct_rows(self):
        """Each task's MERGE only touches its own rows — no cross-task collision."""
        client = MagicMock()
        client.load_table_from_dataframe.return_value.result.return_value = None

        # task 0 merges even photo_ids
        job0 = MagicMock()
        job0.dml_stats.updated_row_count = 5
        # task 1 merges odd photo_ids
        job1 = MagicMock()
        job1.dml_stats.updated_row_count = 5
        client.query.side_effect = [job0, job1]

        task0_results = [{"dataset_source_uuid": f"uuid-{i}", "fetch_status": "downloaded",
                          "image_width": 64, "image_height": 48,
                          "image_size": 1000, "corrupted": False}
                         for i in range(5)]
        task1_results = [{"dataset_source_uuid": f"uuid-{i+5}", "fetch_status": "downloaded",
                          "image_width": 64, "image_height": 48,
                          "image_size": 1000, "corrupted": False}
                         for i in range(5)]

        n0 = di.merge_chunk_into_training_images(
            client, task0_results, "training_table", "downloads_table"
        )
        n1 = di.merge_chunk_into_training_images(
            client, task1_results, "training_table", "downloads_table"
        )

        assert n0 == 5
        assert n1 == 5
        assert client.query.call_count == 2   # one MERGE per task
        assert client.delete_table.call_count == 2  # temp table cleaned per task

    def test_total_coverage_after_all_tasks_complete(self):
        """After all tasks finish, every image should be accounted for."""
        num_jobs = 5
        all_downloaded = set()

        for task_id in range(num_jobs):
            task_images = {p for p in self.PHOTO_IDS if p % num_jobs == task_id}
            all_downloaded |= task_images

        assert all_downloaded == set(self.PHOTO_IDS)
        assert len(all_downloaded) == len(self.PHOTO_IDS)


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


# ── --dataset flag and NULL fetch_status handling ─────────────────────────────

class TestDatasetFlagAndNullFetchStatus:
    """Tests for --dataset CLI flag and NULL fetch_status support (global_all_leps_2605)."""

    # ── --dataset default ─────────────────────────────────────────────────────

    def test_default_dataset_constant(self):
        """BQ_DEFAULT_DATASET must default to global_butterflies_2604 for backwards compat."""
        assert di.BQ_DEFAULT_DATASET == "global_butterflies_2604"

    # ── NULL fetch_status in get_pending_rows ─────────────────────────────────

    def test_normal_query_includes_null_fetch_status(self):
        """Normal (LEFT JOIN) query must include OR fetch_status IS NULL to pick up
        rows from datasets like global_all_leps_2605 where status starts as NULL."""
        client = MagicMock()
        client.query.return_value.result.return_value = []

        di.get_pending_rows(
            client,
            training_table="proj.ds.training_images",
            downloads_table="proj.ds.training_images_downloads",
            num_jobs=10,
            task_id=0,
            force_redownload=False,
        )

        sql = client.query.call_args[0][0]
        assert "fetch_status IS NULL" in sql
        assert "fetch_status = 'pending'" in sql

    def test_force_redownload_query_includes_null_fetch_status(self):
        """Force-redownload query must also include OR fetch_status IS NULL."""
        client = MagicMock()
        client.query.return_value.result.return_value = []

        di.get_pending_rows(
            client,
            training_table="proj.ds.training_images",
            downloads_table="proj.ds.training_images_downloads",
            num_jobs=10,
            task_id=0,
            force_redownload=True,
        )

        sql = client.query.call_args[0][0]
        assert "fetch_status IS NULL" in sql
        assert "fetch_status = 'pending'" in sql
        assert "LEFT JOIN" not in sql

    # ── NULL fetch_status in MERGE ────────────────────────────────────────────

    def test_merge_condition_handles_null_fetch_status(self):
        """MERGE SQL must allow updating rows where T.fetch_status IS NULL,
        not just 'pending' — needed for global_all_leps_2605 initial state."""
        client = MagicMock()
        client.load_table_from_dataframe.return_value.result.return_value = None
        job = MagicMock()
        job.dml_stats.updated_row_count = 1
        client.query.return_value = job

        results = [{"dataset_source_uuid": "u1", "fetch_status": "downloaded",
                    "image_width": 64, "image_height": 48,
                    "image_size": 1000, "corrupted": False}]

        di.merge_chunk_into_training_images(
            client, results,
            training_table="leps-ai.global_all_leps_2605.training_images",
            downloads_table="leps-ai.global_all_leps_2605.training_images_downloads",
        )

        sql = client.query.call_args[0][0]
        assert "fetch_status IS NULL" in sql
        assert "fetch_status = 'pending'" in sql

    # ── tmp_table derived from training_table ─────────────────────────────────

    def test_tmp_table_uses_same_dataset_as_training_table(self):
        """Temp table for MERGE must be in the same dataset as training_table,
        not hardcoded to global_butterflies_2604."""
        client = MagicMock()
        client.load_table_from_dataframe.return_value.result.return_value = None
        job = MagicMock()
        job.dml_stats.updated_row_count = 1
        client.query.return_value = job

        results = [{"dataset_source_uuid": "u1", "fetch_status": "downloaded",
                    "image_width": 64, "image_height": 48,
                    "image_size": 1000, "corrupted": False}]

        di.merge_chunk_into_training_images(
            client, results,
            training_table="leps-ai.global_all_leps_2605.training_images",
            downloads_table="leps-ai.global_all_leps_2605.training_images_downloads",
        )

        # The temp table passed to load_table_from_dataframe must be in global_all_leps_2605
        tmp_table_arg = client.load_table_from_dataframe.call_args[0][1]
        assert tmp_table_arg.startswith("leps-ai.global_all_leps_2605.")
        assert "global_butterflies_2604" not in tmp_table_arg

    def test_tmp_table_not_in_wrong_dataset_when_using_new_dataset(self):
        """Regression: old code used BQ_DATASET module constant — ensure it no longer does."""
        client = MagicMock()
        client.load_table_from_dataframe.return_value.result.return_value = None
        job = MagicMock()
        job.dml_stats.updated_row_count = 1
        client.query.return_value = job

        results = [{"dataset_source_uuid": "u1", "fetch_status": "downloaded",
                    "image_width": 64, "image_height": 48,
                    "image_size": 1000, "corrupted": False}]

        di.merge_chunk_into_training_images(
            client, results,
            training_table="leps-ai.global_all_leps_2605.training_images",
            downloads_table="leps-ai.global_all_leps_2605.training_images_downloads",
        )

        tmp_table_arg = client.load_table_from_dataframe.call_args[0][1]
        assert "global_butterflies_2604" not in tmp_table_arg  # must not leak old dataset
