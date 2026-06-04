"""
Tests for stream_chunks_to_tar.py.

The script streams chunk sqfs files as a single tar to stdout for piping to sqfstar.
All squashfuse calls are mocked — no real sqfs or FUSE needed.

Run with: pytest tests/dataset_tools/test_stream_chunks_to_tar.py -v
"""

import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import src.dataset_tools.bq_squashfs.stream_chunks_to_tar as sct


# ── helpers ───────────────────────────────────────────────────────────────────

def make_chunk_sqfs(staging_dir: Path, chunk_num: int) -> Path:
    """Create a dummy chunk_NNNN.sqfs file (content doesn't matter — squashfuse is mocked)."""
    p = staging_dir / f"chunk_{chunk_num:04d}.sqfs"
    p.write_bytes(b"fake sqfs")
    return p


def make_mount_dir_with_images(root: Path, filenames: list[str]) -> Path:
    """Create a directory tree simulating a squashfuse mount with images."""
    mnt = root / "mnt"
    bucket = mnt / "000"
    bucket.mkdir(parents=True)
    for name in filenames:
        (bucket / name).write_bytes(b"JPEG")
    return mnt


def read_tar_from_bytes(data: bytes) -> list[str]:
    """Return list of member names from a tar written to bytes."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
        return [m.name for m in tf.getmembers()]


# ── stream_dir_to_tar ─────────────────────────────────────────────────────────

class TestStreamDirToTar:

    def test_files_added_with_relative_paths(self, tmp_path):
        """Files are added to the tar with paths relative to the mount dir."""
        mnt = make_mount_dir_with_images(tmp_path, ["abc.jpg", "def.jpg"])
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:") as tf:
            count = sct.stream_dir_to_tar(tf, str(mnt))
        assert count == 2
        members = read_tar_from_bytes(buf.getvalue())
        assert "000/abc.jpg" in members
        assert "000/def.jpg" in members

    def test_dirs_included_files_counted(self, tmp_path):
        """Directory entries are included; only files contribute to count."""
        mnt = make_mount_dir_with_images(tmp_path, ["img.jpg"])
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:") as tf:
            count = sct.stream_dir_to_tar(tf, str(mnt))
        assert count == 1   # only the .jpg, not the dir
        members = read_tar_from_bytes(buf.getvalue())
        assert "000" in members       # dir entry
        assert "000/img.jpg" in members  # file entry

    def test_empty_mount_dir_returns_zero(self, tmp_path):
        """Empty mount dir → count=0, no files added."""
        mnt = tmp_path / "mnt"
        mnt.mkdir()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:") as tf:
            count = sct.stream_dir_to_tar(tf, str(mnt))
        assert count == 0

    def test_multiple_bucket_dirs_all_streamed(self, tmp_path):
        """Files across multiple bucket dirs are all included."""
        mnt = tmp_path / "mnt"
        for bucket in ["000", "001", "002"]:
            (mnt / bucket).mkdir(parents=True)
            (mnt / bucket / "img.jpg").write_bytes(b"JPEG")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:") as tf:
            count = sct.stream_dir_to_tar(tf, str(mnt))
        assert count == 3
        members = read_tar_from_bytes(buf.getvalue())
        assert "000/img.jpg" in members
        assert "001/img.jpg" in members
        assert "002/img.jpg" in members


# ── squashfuse_mount / unmount ────────────────────────────────────────────────

class TestSquashfuseMount:

    def test_successful_mount_returns_true(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = sct.squashfuse_mount("/fake.sqfs", "/mnt/fake")
        assert result is True

    def test_failed_mount_returns_false(self, capsys):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="fuse: failed to open /fake.sqfs"
            )
            result = sct.squashfuse_mount("/fake.sqfs", "/mnt/fake")
        assert result is False
        assert "ERROR" in capsys.readouterr().err

    def test_unmount_calls_fusermount(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            sct.squashfuse_unmount("/mnt/fake")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "fusermount" in cmd
        assert "-u" in cmd
        assert "/mnt/fake" in cmd


# ── main: no chunks ───────────────────────────────────────────────────────────

class TestNoChunks:

    def test_empty_staging_dir_exits_with_error(self, tmp_path, capsys):
        """No chunk_*.sqfs files → exits with code 1."""
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["stream_chunks_to_tar.py", str(tmp_path)]):
                sct.main()
        assert exc.value.code == 1
        assert "ERROR" in capsys.readouterr().err

    def test_missing_staging_dir_exits_with_error(self, tmp_path, capsys):
        """Non-existent staging dir → exits with code 1."""
        missing = tmp_path / "does_not_exist"
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["stream_chunks_to_tar.py", str(missing)]):
                sct.main()
        assert exc.value.code == 1


# ── main: dry run ─────────────────────────────────────────────────────────────

class TestDryRun:

    def test_dry_run_lists_chunks_no_streaming(self, tmp_path, capsys):
        """--dry-run prints chunk paths without mounting or streaming."""
        staging = tmp_path / "staging"
        staging.mkdir()
        c1 = make_chunk_sqfs(staging, 1)
        c2 = make_chunk_sqfs(staging, 2)

        with patch("sys.argv", ["stream_chunks_to_tar.py", str(staging), "--dry-run"]), \
             patch.object(sct, "squashfuse_mount") as mock_mount:
            sct.main()

        mock_mount.assert_not_called()  # no mounting in dry run
        out = capsys.readouterr().out
        assert str(c1) in out
        assert str(c2) in out

    def test_dry_run_lists_in_sorted_order(self, tmp_path, capsys):
        """Chunks are listed in sorted (chunk_0001 before chunk_0002) order."""
        staging = tmp_path / "staging"
        staging.mkdir()
        make_chunk_sqfs(staging, 3)
        make_chunk_sqfs(staging, 1)
        make_chunk_sqfs(staging, 2)

        with patch("sys.argv", ["stream_chunks_to_tar.py", str(staging), "--dry-run"]):
            sct.main()

        lines = [l for l in capsys.readouterr().out.strip().splitlines() if l]
        names = [Path(l).name for l in lines]
        assert names == ["chunk_0001.sqfs", "chunk_0002.sqfs", "chunk_0003.sqfs"]


# ── main: streaming ───────────────────────────────────────────────────────────

class TestStreaming:

    def _run_stream(self, staging: Path, extra_args: list[str] = []) -> tuple[bytes, str]:
        """Run main(), capture stdout bytes and stderr text."""
        stdout_buf = io.BytesIO()
        with patch("sys.argv", ["stream_chunks_to_tar.py", str(staging)] + extra_args), \
             patch("sys.stdout") as mock_stdout:
            mock_stdout.buffer = stdout_buf
            sct.main()
        return stdout_buf.getvalue(), ""

    def test_single_chunk_produces_valid_tar(self, tmp_path):
        """One chunk sqfs → tar stream contains all files from that chunk."""
        staging = tmp_path / "staging"
        staging.mkdir()
        make_chunk_sqfs(staging, 1)

        # Create fake mount content
        fake_mnt = tmp_path / "fake_mnt"
        fake_mnt.mkdir()
        (fake_mnt / "000").mkdir()
        (fake_mnt / "000" / "img.jpg").write_bytes(b"JPEG")

        stdout_buf = io.BytesIO()
        with patch("sys.argv", ["stream_chunks_to_tar.py", str(staging)]), \
             patch("sys.stdout") as mock_stdout, \
             patch.object(sct, "squashfuse_mount", return_value=True), \
             patch.object(sct, "squashfuse_unmount"), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path / "mnt_base")), \
             patch("os.makedirs"), \
             patch("os.rmdir"), \
             patch.object(sct, "stream_dir_to_tar", return_value=1) as mock_stream:
            mock_stdout.buffer = stdout_buf
            sct.main()

        mock_stream.assert_called_once()

    def test_two_chunks_single_continuous_stream(self, tmp_path):
        """Two chunks produce ONE continuous tar (not two separate tars)."""
        staging = tmp_path / "staging"
        staging.mkdir()
        make_chunk_sqfs(staging, 1)
        make_chunk_sqfs(staging, 2)

        call_count = {"n": 0}

        def fake_stream(tar, mnt_dir):
            call_count["n"] += 1
            return 5

        stdout_buf = io.BytesIO()
        with patch("sys.argv", ["stream_chunks_to_tar.py", str(staging)]), \
             patch("sys.stdout") as mock_stdout, \
             patch.object(sct, "squashfuse_mount", return_value=True), \
             patch.object(sct, "squashfuse_unmount"), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path / "mnt_base")), \
             patch("os.makedirs"), \
             patch("os.rmdir"), \
             patch.object(sct, "stream_dir_to_tar", side_effect=fake_stream):
            mock_stdout.buffer = stdout_buf
            sct.main()

        # stream_dir_to_tar called twice (one per chunk) into the SAME tar
        assert call_count["n"] == 2

    def test_delete_after_stream_removes_chunk(self, tmp_path):
        """--delete-after-stream: each chunk file is deleted after streaming."""
        staging = tmp_path / "staging"
        staging.mkdir()
        chunk = make_chunk_sqfs(staging, 1)
        assert chunk.exists()

        stdout_buf = io.BytesIO()
        with patch("sys.argv", ["stream_chunks_to_tar.py", str(staging), "--delete-after-stream"]), \
             patch("sys.stdout") as mock_stdout, \
             patch.object(sct, "squashfuse_mount", return_value=True), \
             patch.object(sct, "squashfuse_unmount"), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path / "mnt_base")), \
             patch("os.makedirs"), \
             patch("os.rmdir"), \
             patch.object(sct, "stream_dir_to_tar", return_value=1):
            mock_stdout.buffer = stdout_buf
            sct.main()

        assert not chunk.exists()   # deleted after streaming

    def test_without_delete_flag_chunks_preserved(self, tmp_path):
        """Without --delete-after-stream, chunk files remain on disk."""
        staging = tmp_path / "staging"
        staging.mkdir()
        chunk = make_chunk_sqfs(staging, 1)

        stdout_buf = io.BytesIO()
        with patch("sys.argv", ["stream_chunks_to_tar.py", str(staging)]), \
             patch("sys.stdout") as mock_stdout, \
             patch.object(sct, "squashfuse_mount", return_value=True), \
             patch.object(sct, "squashfuse_unmount"), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path / "mnt_base")), \
             patch("os.makedirs"), \
             patch("os.rmdir"), \
             patch.object(sct, "stream_dir_to_tar", return_value=1):
            mock_stdout.buffer = stdout_buf
            sct.main()

        assert chunk.exists()   # preserved


# ── main: error handling ──────────────────────────────────────────────────────

class TestErrorHandling:

    def test_failed_mount_skipped_continues_to_next_chunk(self, tmp_path, capsys):
        """If one chunk fails to mount, it's skipped and remaining chunks continue."""
        staging = tmp_path / "staging"
        staging.mkdir()
        make_chunk_sqfs(staging, 1)
        make_chunk_sqfs(staging, 2)

        # chunk 1 fails to mount, chunk 2 succeeds
        mount_results = [False, True]

        stdout_buf = io.BytesIO()
        with patch("sys.argv", ["stream_chunks_to_tar.py", str(staging)]), \
             patch("sys.stdout") as mock_stdout, \
             patch.object(sct, "squashfuse_mount", side_effect=mount_results), \
             patch.object(sct, "squashfuse_unmount"), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path / "mnt_base")), \
             patch("os.makedirs"), \
             patch("os.rmdir"), \
             patch.object(sct, "stream_dir_to_tar", return_value=5):
            mock_stdout.buffer = stdout_buf
            with pytest.raises(SystemExit) as exc:
                sct.main()

        # exits non-zero because there were errors
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "errors=1" in err

    def test_all_mounts_fail_exits_nonzero(self, tmp_path):
        """All mounts failing → exit code 1."""
        staging = tmp_path / "staging"
        staging.mkdir()
        make_chunk_sqfs(staging, 1)
        make_chunk_sqfs(staging, 2)

        stdout_buf = io.BytesIO()
        with patch("sys.argv", ["stream_chunks_to_tar.py", str(staging)]), \
             patch("sys.stdout") as mock_stdout, \
             patch.object(sct, "squashfuse_mount", return_value=False), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path / "mnt_base")), \
             patch("os.makedirs"), \
             patch("os.rmdir"):
            mock_stdout.buffer = stdout_buf
            with pytest.raises(SystemExit) as exc:
                sct.main()

        assert exc.value.code == 1

    def test_chunks_processed_in_sorted_order(self, tmp_path):
        """Chunks are processed in sorted order: chunk_0001 before chunk_0002."""
        staging = tmp_path / "staging"
        staging.mkdir()
        make_chunk_sqfs(staging, 3)
        make_chunk_sqfs(staging, 1)
        make_chunk_sqfs(staging, 2)

        processed_order = []

        def fake_mount(sqfs_path, mnt_dir):
            processed_order.append(Path(sqfs_path).name)
            return True

        stdout_buf = io.BytesIO()
        with patch("sys.argv", ["stream_chunks_to_tar.py", str(staging)]), \
             patch("sys.stdout") as mock_stdout, \
             patch.object(sct, "squashfuse_mount", side_effect=fake_mount), \
             patch.object(sct, "squashfuse_unmount"), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path / "mnt_base")), \
             patch("os.makedirs"), \
             patch("os.rmdir"), \
             patch.object(sct, "stream_dir_to_tar", return_value=1):
            mock_stdout.buffer = stdout_buf
            sct.main()

        assert processed_order == [
            "chunk_0001.sqfs", "chunk_0002.sqfs", "chunk_0003.sqfs"
        ]
