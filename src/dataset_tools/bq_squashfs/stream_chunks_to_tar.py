#!/usr/bin/env python3
"""
Stream all chunk sqfs files as a single tar archive to stdout, for piping to sqfstar.

Processes chunks ONE AT A TIME:
  squashfuse mount → add to tar stream → unmount → delete chunk → repeat

This keeps peak scratch usage at ~(remaining chunks + growing output) rather than
(all chunks + full output), which would exceed the scratch quota at 10M-image scale.

Usage:
    python stream_chunks_to_tar.py <staging_base_dir> | sqfstar -comp zstd output.sqfs
"""

import os
import sys
import glob
import tarfile
import tempfile
import subprocess
import argparse
from pathlib import Path


def squashfuse_mount(sqfs_path: str, mount_dir: str) -> bool:
    result = subprocess.run(["squashfuse", sqfs_path, mount_dir],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[stream] ERROR: squashfuse failed for {sqfs_path}: {result.stderr.strip()}",
              file=sys.stderr)
        return False
    return True


def squashfuse_unmount(mount_dir: str) -> None:
    subprocess.run(["fusermount", "-u", mount_dir],
                   capture_output=True)


def stream_dir_to_tar(tar: tarfile.TarFile, mount_dir: str) -> int:
    """Add all files from mount_dir into tar with paths relative to mount_dir."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging_base",
                        help="Base dir containing task_N/ subdirs with chunk_*.sqfs files")
    parser.add_argument("--delete-after-stream", action="store_true",
                        help="Delete each chunk sqfs after it has been streamed (saves scratch space)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List chunks that would be processed, don't stream")
    args = parser.parse_args()

    chunk_files = sorted(glob.glob(f"{args.staging_base}/**/chunk_*.sqfs", recursive=True))
    total = len(chunk_files)

    if total == 0:
        print(f"[stream] ERROR: no chunk_*.sqfs files found under {args.staging_base}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[stream] Found {total} chunk sqfs files", file=sys.stderr)

    if args.dry_run:
        for f in chunk_files:
            print(f)
        return

    mount_base = tempfile.mkdtemp(prefix="sqfs_stream_")
    total_images = 0
    errors = 0

    # One tarfile object → one continuous stream → one EOF at the very end
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
        for i, sqfs_file in enumerate(chunk_files, 1):
            mnt = os.path.join(mount_base, f"mnt_{i}")
            os.makedirs(mnt, exist_ok=True)

            if not squashfuse_mount(sqfs_file, mnt):
                errors += 1
                os.rmdir(mnt)
                continue

            count = stream_dir_to_tar(tar, mnt)
            total_images += count

            squashfuse_unmount(mnt)
            os.rmdir(mnt)

            if args.delete_after_stream:
                os.unlink(sqfs_file)
                deleted = " (deleted)"
            else:
                deleted = ""

            print(f"[stream] [{i}/{total}] {sqfs_file} → {count} images{deleted}",
                  file=sys.stderr)

    os.rmdir(mount_base)
    print(f"[stream] Done. total_images={total_images} errors={errors}", file=sys.stderr)

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
