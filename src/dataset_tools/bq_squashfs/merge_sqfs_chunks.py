#!/usr/bin/env python3
"""
Merge per-chunk SquashFS files produced by download_images.py into a single
task-level SquashFS archive.

WHAT IT DOES
------------
download_images.py packs downloaded images into small chunk_NNNN.sqfs files
(one per 10,000 images) to keep inode usage under the cluster quota. This
script merges all those chunks for one task into a single task_N.sqfs archive
that the webdataset build step can use directly.

The merge is done by streaming: each chunk is mounted via squashfuse, its
files are written into a continuous tar stream on stdout, then unmounted. The
tar stream is consumed by sqfstar on the other end of the pipe to produce the
final sqfs. Only one chunk is mounted at a time — peak inode usage stays tiny
regardless of how many images are inside the chunks.

INPUT
-----
  <staging_dir>          Directory containing chunk_*.sqfs files, searched
                         recursively. Pass the staging dir for ONE task
                         (e.g. bq_download_staging/task_0/) to merge that
                         task's chunks into a single task_0.sqfs.

OUTPUT
------
  stdout                 A streaming tar archive consumed by sqfstar:
                           python merge_sqfs_chunks.py <staging_dir> \\
                             | sqfstar -comp zstd -b 131072 task_N.sqfs
  stderr                 Per-chunk progress: "[merge] [N/T] chunk_NNNN.sqfs → K images"
                         Summary: "[merge] Done. total_images=N errors=M empty_chunks=K"

DISK SPACE
----------
Chunks are NEVER deleted by this script. Deletion is the job script's
responsibility, and only after the output sqfs has been verified correct.
Peak disk usage during merge: all chunks + growing output sqfs (~2× space).
At production scale (~108 chunks × 10 GB = 1.1 TB chunks + 1.1 TB output),
this requires ~2.2 TB per task on scratch. The inode cost is negligible:
each chunk file is 1 inode regardless of how many images it contains.

EXIT CODES
----------
  0    All chunks streamed successfully. sqfstar output is complete.
  1    One or more chunks had errors (corrupt, empty, or mount failure).
       sqfstar may have produced a partial output — the job script must
       verify the image count before accepting the result. Chunk files are
       always preserved so the merge can be re-submitted after investigation.

TYPICAL USAGE
-------------
  # Merge task 3's chunks into task_3.sqfs:
  python merge_sqfs_chunks.py /scratch/$USER/staging/task_3 \\
      | sqfstar -comp zstd -Xcompression-level 3 -b 131072 -no-duplicates \\
            /scratch/$USER/task_3.sqfs

  # Dry-run — list chunks that would be processed:
  python merge_sqfs_chunks.py /scratch/$USER/staging/task_3 --dry-run

  # In a SLURM job (see job_bq_pack_per_task.sh):
  python merge_sqfs_chunks.py "${TASK_DIR}" \\
      | sqfstar -comp zstd -Xcompression-level 3 -b 131072 -no-duplicates \\
            "${OUTPUT_SQFS}"
  PIPE_STATUS=("${PIPESTATUS[@]}")   # capture both exits atomically
  MERGE_EXIT="${PIPE_STATUS[0]}"
  SQFSTAR_EXIT="${PIPE_STATUS[1]}"
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
# Registered with atexit so normal exit cleans up FUSE mounts.
# Also wired to SIGTERM so SLURM wall-time timeout leaves no stale mounts.
# NOTE: SIGKILL cannot be caught — atexit does not run on SIGKILL. This is
# mitigated by SLURM sending SIGTERM 30 s before SIGKILL, giving the handler
# time to unmount and clean up before the hard kill arrives.

_mount_base: str | None = None


def _cleanup_temp_dirs() -> None:
    """Unmount any still-mounted squashfuse dirs and remove the temp base dir."""
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
    """On SLURM timeout (SIGTERM), clean up mounts and exit 1."""
    print("[merge] SIGTERM received — cleaning up mounts and exiting",
          file=sys.stderr)
    _cleanup_temp_dirs()
    sys.exit(1)


atexit.register(_cleanup_temp_dirs)
signal.signal(signal.SIGTERM, _sigterm_handler)


# ── squashfuse helpers ────────────────────────────────────────────────────────

def squashfuse_mount(sqfs_path: str, mount_dir: str, retries: int = 1) -> bool:
    """Mount sqfs_path at mount_dir via squashfuse.

    Retries once on failure to handle transient FUSE errors (e.g. stale
    device entries). Returns True on success, False if all attempts fail.
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
                f"[merge] squashfuse failed for {Path(sqfs_path).name} "
                f"(attempt {attempt + 1}/{retries + 1}), retrying in 5s...",
                file=sys.stderr,
            )
            time.sleep(5)

    print(
        f"[merge] ERROR: squashfuse failed for {sqfs_path}: "
        f"{result.stderr.strip()}",
        file=sys.stderr,
    )
    return False


def squashfuse_unmount(mount_dir: str, retries: int = 3) -> bool:
    """Unmount mount_dir via fusermount -u, with retries on transient failures.

    Returns True on success. Logs a warning on persistent failure but does not
    raise — the SLURM node exit will clean up any remaining mounts.
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
        f"[merge] WARNING: fusermount -u {mount_dir} failed after {retries} "
        f"attempts — {result.stderr.strip()}. Mount will be cleaned up on node exit.",
        file=sys.stderr,
    )
    return False


# ── tar streaming ─────────────────────────────────────────────────────────────

def stream_dir_to_tar(tar: tarfile.TarFile, mount_dir: str) -> int:
    """Stream all files from mount_dir into tar, using paths relative to mount_dir.

    Directory entries are included in the tar (required by sqfstar) but are
    not counted in the return value. Returns 0 if the mounted sqfs is empty.
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
        "staging_dir",
        help=(
            "Directory containing chunk_*.sqfs files to merge, searched "
            "recursively. Typically the staging dir for one download task: "
            "bq_download_staging/task_N/. "
            "Chunks are processed in sorted order (chunk_0001, chunk_0002, …)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "List the chunk files that would be merged in processing order, "
            "then exit without streaming anything. Useful for verifying which "
            "chunks will be included before running the full merge."
        ),
    )
    args = parser.parse_args()

    chunk_files = sorted(
        glob.glob(f"{args.staging_dir}/**/chunk_*.sqfs", recursive=True)
    )
    total = len(chunk_files)

    if total == 0:
        print(
            f"[merge] ERROR: no chunk_*.sqfs files found under {args.staging_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[merge] Found {total} chunk sqfs file(s) to merge under {args.staging_dir}",
        file=sys.stderr,
    )

    if args.dry_run:
        for f in chunk_files:
            print(f)
        return

    _mount_base = tempfile.mkdtemp(prefix="sqfs_merge_")
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
                        f"[merge] WARNING: [{i}/{total}] {Path(sqfs_file).name} — "
                        f"0 images found after mount. The chunk may be corrupt or "
                        f"download_images.py may have failed for these images. "
                        f"Investigate before re-submitting.",
                        file=sys.stderr,
                    )
                else:
                    total_images += count

                print(
                    f"[merge] [{i}/{total}] {Path(sqfs_file).name} → {count} images",
                    file=sys.stderr,
                )

    finally:
        _cleanup_temp_dirs()

    print(
        f"[merge] Done. total_images={total_images} "
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
            "[merge] FATAL: BrokenPipeError — the downstream process (sqfstar) "
            "died before the merge completed.\n"
            "  Common causes:\n"
            "    - sqfstar OOM killed (exit=137): increase --mem in the job script\n"
            "    - sqfstar not found (exit=127): check 'module load' and PATH\n"
            "    - sqfstar crashed on corrupt input: check chunk integrity\n"
            "  Chunk files are preserved — fix the cause and re-submit the pack job.",
            file=sys.stderr,
        )
        sys.stderr.flush()
        # Redirect stdout to /dev/null before sys.exit so Python's shutdown
        # does not try to flush the broken pipe and produce exit code 120
        # instead of the expected 1.
        try:
            with open(os.devnull, "wb") as devnull:
                os.dup2(devnull.fileno(), sys.stdout.fileno())
        except Exception:
            pass
        sys.exit(1)
