#!/usr/bin/env python3
"""
Stream all chunk sqfs files as a single tar archive to stdout, for piping to sqfstar.

Processes chunks ONE AT A TIME:
  squashfuse mount → add to tar stream → unmount → delete chunk → repeat

This keeps peak scratch usage at ~(remaining chunks + growing output) rather than
(all chunks + full output), which would exceed the scratch quota at 10M-image scale.

Exit codes:
  0 — all chunks streamed successfully
  1 — one or more chunks failed (corrupt, empty, or mount error) — sqfstar
      may still have produced a partial output; verify image count before use

Usage:
    python stream_chunks_to_tar.py <staging_base_dir> | sqfstar -comp zstd output.sqfs
"""

import atexit
import glob
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import argparse
from pathlib import Path


# ── Temp dir cleanup ──────────────────────────────────────────────────────────
# Registered once; also called on SIGTERM so SLURM job timeout leaves no dirs

_mount_base: str | None = None


def _cleanup_temp_dirs() -> None:
    """Unmount any still-mounted dirs and remove the temp base dir."""
    global _mount_base
    if _mount_base is None or not os.path.exists(_mount_base):
        return
    for sub in sorted(os.listdir(_mount_base)):
        full = os.path.join(_mount_base, sub)
        if os.path.isdir(full):
            subprocess.run(["fusermount", "-u", full], capture_output=True)
            try:
                os.rmdir(full)
            except OSError:
                pass
    try:
        shutil.rmtree(_mount_base, ignore_errors=True)
    except Exception:
        pass
    _mount_base = None


def _sigterm_handler(signum, frame):
    """On SLURM timeout (SIGTERM), clean up and exit 1."""
    print("[stream] SIGTERM received — cleaning up temp dirs and exiting",
          file=sys.stderr)
    _cleanup_temp_dirs()
    sys.exit(1)


atexit.register(_cleanup_temp_dirs)
signal.signal(signal.SIGTERM, _sigterm_handler)


# ── squashfuse helpers ────────────────────────────────────────────────────────

def squashfuse_mount(sqfs_path: str, mount_dir: str, retries: int = 1) -> bool:
    """Mount sqfs_path at mount_dir via squashfuse.

    Retries once on failure to handle transient FUSE errors.
    Returns True on success, False if all attempts fail.
    """
    for attempt in range(retries + 1):
        result = subprocess.run(
            ["squashfuse", sqfs_path, mount_dir],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return True
        if attempt < retries:
            print(
                f"[stream] squashfuse failed for {Path(sqfs_path).name} "
                f"(attempt {attempt + 1}/{retries + 1}), retrying in 5s...",
                file=sys.stderr,
            )
            time.sleep(5)

    print(
        f"[stream] ERROR: squashfuse failed for {sqfs_path}: "
        f"{result.stderr.strip()}",
        file=sys.stderr,
    )
    return False


def squashfuse_unmount(mount_dir: str, retries: int = 3) -> bool:
    """Unmount mount_dir via fusermount, with retries.

    Returns True on success. Logs a warning on failure but does not raise —
    the SLURM job will clean up the mount on node exit.
    """
    for attempt in range(retries):
        result = subprocess.run(
            ["fusermount", "-u", mount_dir],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return True
        if attempt < retries - 1:
            time.sleep(2)

    print(
        f"[stream] WARNING: fusermount -u {mount_dir} failed after {retries} attempts "
        f"— {result.stderr.strip()}. Mount will be cleaned up on node exit.",
        file=sys.stderr,
    )
    return False


# ── tar streaming ─────────────────────────────────────────────────────────────

def stream_dir_to_tar(tar: tarfile.TarFile, mount_dir: str) -> int:
    """Add all files from mount_dir into tar with paths relative to mount_dir.

    Returns number of files added. Directories are included in the tar but
    not counted. Returns 0 for empty mounts.
    """
    count = 0
    mount_path = Path(mount_dir)
    for entry in sorted(mount_path.rglob("*")):
        arcname = str(entry.relative_to(mount_path))
        if arcname == ".":
            continue
        info = tar.gettarinfo(str(entry), arcname=arcname)
        if entry.is_dir():
            tar.addfile(info)
        else:
            with open(entry, "rb") as f:
                tar.addfile(info, f)
            count += 1
    return count


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global _mount_base

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "staging_base",
        help=(
            "Directory containing chunk_*.sqfs files (searched recursively). "
            "Pass a task staging dir (e.g. bq_download_staging/task_0/) for "
            "per-task merge, or the base dir for a global merge."
        ),
    )
    parser.add_argument(
        "--delete-after-stream", action="store_true",
        help=(
            "Delete each chunk sqfs after it has been streamed. "
            "Saves scratch space but means a restart requires redownloading. "
            "Do NOT use this unless you are confident sqfstar has enough RAM."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List chunks that would be processed in sorted order, then exit.",
    )
    args = parser.parse_args()

    chunk_files = sorted(
        glob.glob(f"{args.staging_base}/**/chunk_*.sqfs", recursive=True)
    )
    total = len(chunk_files)

    if total == 0:
        print(
            f"[stream] ERROR: no chunk_*.sqfs files found under {args.staging_base}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[stream] Found {total} chunk sqfs files to stream", file=sys.stderr)

    if args.dry_run:
        for f in chunk_files:
            print(f)
        return

    if args.delete_after_stream:
        print(
            "[stream] WARNING: --delete-after-stream is active. Chunks will be "
            "deleted as they stream. Ensure sqfstar has sufficient RAM before "
            "proceeding — an OOM kill will cause data loss.",
            file=sys.stderr,
        )

    _mount_base = tempfile.mkdtemp(prefix="sqfs_stream_")
    total_images = 0
    errors = 0
    empty_chunks = 0

    try:
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
            for i, sqfs_file in enumerate(chunk_files, 1):
                mnt = os.path.join(_mount_base, f"mnt_{i}")
                os.makedirs(mnt, exist_ok=True)

                if not squashfuse_mount(sqfs_file, mnt):
                    errors += 1
                    try:
                        os.rmdir(mnt)
                    except OSError:
                        pass
                    continue

                count = stream_dir_to_tar(tar, mnt)

                squashfuse_unmount(mnt)
                try:
                    os.rmdir(mnt)
                except OSError:
                    pass

                if count == 0:
                    empty_chunks += 1
                    print(
                        f"[stream] WARNING: [{i}/{total}] {sqfs_file} — "
                        f"0 images found after mount. Chunk may be corrupt or "
                        f"download stage failed for these images.",
                        file=sys.stderr,
                    )
                else:
                    total_images += count

                deleted = ""
                if args.delete_after_stream:
                    os.unlink(sqfs_file)
                    deleted = " (deleted)"

                print(
                    f"[stream] [{i}/{total}] {Path(sqfs_file).name} "
                    f"→ {count} images{deleted}",
                    file=sys.stderr,
                )

    finally:
        _cleanup_temp_dirs()

    print(
        f"[stream] Done. total_images={total_images} "
        f"errors={errors} empty_chunks={empty_chunks}",
        file=sys.stderr,
    )

    if errors > 0 or empty_chunks > 0:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        _cleanup_temp_dirs()
        print(
            "[stream] FATAL: BrokenPipeError — the downstream process (sqfstar) "
            "died unexpectedly.\n"
            "  Common causes:\n"
            "    - sqfstar OOM killed (exit=137): increase --mem in job script\n"
            "    - sqfstar not found (exit=127): check module load and PATH\n"
            "    - sqfstar crashed on corrupt input: check chunk integrity\n"
            "  Chunk files are preserved (unless --delete-after-stream was used).",
            file=sys.stderr,
        )
        sys.exit(1)
