#!/usr/bin/env python3
"""Byte-level verification that a raw HDF5 family is fully + correctly on HF before NFS deletion.

For every LOCAL file under $DR/<family>_*_hdf5/, recompute SHA-256 + size and compare against what
HF recorded (lfs.sha256 + size) in BOTH the huzheyuan/ and CMU-AIRe/ public repos. HF/LFS verifies
the content hash server-side on upload, so a local recompute that equals HF's lfs.sha256 proves the
stored bytes are byte-identical. Prints per-problem detail and a hard VERDICT per repo + overall.

Usage:  HASHJOBS=32 ~/miniforge3/bin/python3 verify_raw_upload.py <family>
        e.g. real_hang | sim_double_insert
Exit code 0 == SAFE TO DELETE (every local file matched on every account); non-zero otherwise.
"""
import glob
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi

DR = os.environ.get("DR", "/data/group_data/rl/dexterous_robot_data")
ACCOUNTS = ["huzheyuan", "CMU-AIRe"]
TOKEN = open(os.path.expanduser("~/hf_key_2026.txt")).read().strip()
HASHJOBS = int(os.environ.get("HASHJOBS", "32"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path, size):
    """git blob SHA-1 = sha1(b"blob <size>\\0" + content). Matches HF's blob_id for NON-LFS files."""
    h = hashlib.sha1()
    h.update(b"blob " + str(size).encode() + b"\x00")
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    family = sys.argv[1]
    repo_name = f"{family}_hdf5"
    api = HfApi(token=TOKEN)

    # 1. enumerate every local file -> repo-relative path (<round_dir>/<subpath>)
    local = {}  # relpath -> abspath
    sizes = {}  # relpath -> size
    round_dirs = sorted(glob.glob(f"{DR}/{family}_*_hdf5"))
    for d in round_dirs:
        base = os.path.basename(d)
        for fp in glob.glob(f"{d}/**", recursive=True):
            if os.path.isfile(fp):
                rel = base + "/" + os.path.relpath(fp, d)
                local[rel] = fp
                sizes[rel] = os.path.getsize(fp)
    total_bytes = sum(sizes.values())
    print(f"[{family}] rounds={len(round_dirs)} local_files={len(local)} total_bytes={total_bytes:,} "
          f"({total_bytes/1e12:.3f} TB)")
    if not local:
        print(f"VERDICT {family}: NO LOCAL FILES FOUND -- abort")
        return 2

    # 2. recompute SHA-256 of every local file (reads every byte)
    print(f"  hashing {len(local)} files with {HASHJOBS} workers (reads every byte)...")
    local_sha = {}
    with ThreadPoolExecutor(max_workers=HASHJOBS) as ex:
        futs = {ex.submit(sha256, ab): rel for rel, ab in local.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            local_sha[futs[fut]] = fut.result()
            if i % 250 == 0 or i == len(local):
                print(f"    hashed {i}/{len(local)}")

    # 3. per account: fetch HF tree (path -> (size, sha256)) and compare
    overall_ok = True
    for acct in ACCOUNTS:
        repo = f"{acct}/{repo_name}"
        tree = {}
        try:
            for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True, expand=True):
                if getattr(f, "size", None) is None:
                    continue  # folder
                lfs = getattr(f, "lfs", None)
                sha = getattr(lfs, "sha256", None) if lfs else None
                tree[f.path] = (f.size, sha, getattr(f, "blob_id", None))
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {repo} ===\n  ERROR fetching tree: {type(e).__name__}: {e}")
            print(f"VERDICT {repo}: DO NOT DELETE (tree fetch failed)")
            overall_ok = False
            continue

        missing, size_mm, hash_mm, blob_mm = [], [], [], []
        for rel in local:
            if rel not in tree:
                missing.append(rel)
                continue
            hsz, hsha, hblob = tree[rel]
            if hsz != sizes[rel]:
                size_mm.append((rel, sizes[rel], hsz))
                continue
            if hsha is not None:
                # LFS file: compare content SHA-256 (HF verifies this server-side on upload).
                if hsha != local_sha[rel]:
                    hash_mm.append((rel, local_sha[rel], hsha))
            else:
                # non-LFS file (small, stored as a git blob): compare git-blob SHA-1 to HF blob_id.
                if hblob is None or git_blob_sha1(local[rel], sizes[rel]) != hblob:
                    blob_mm.append((rel, hblob))

        ok = not missing and not size_mm and not hash_mm and not blob_mm
        overall_ok = overall_ok and ok
        print(f"\n=== {repo} ===")
        print(f"  local_files={len(local)} hf_files={len(tree)} matched="
              f"{len(local) - len(missing) - len(size_mm) - len(hash_mm) - len(blob_mm)}")
        print(f"  missing={len(missing)} size_mismatch={len(size_mm)} "
              f"sha256_mismatch={len(hash_mm)} gitblob_mismatch={len(blob_mm)}", flush=True)
        for rel in missing[:20]:
            print(f"    MISSING    {rel}")
        for rel, ls, hs in size_mm[:20]:
            print(f"    SIZE-MM    {rel}  local={ls} hf={hs}")
        for rel, ls, hs in hash_mm[:20]:
            print(f"    SHA256-MM  {rel}  local={ls} hf={hs}")
        for rel, hb in blob_mm[:20]:
            print(f"    BLOB-MM    {rel}  hf_blob_id={hb}")
        print(f"VERDICT {repo}: {'SAFE' if ok else 'DO NOT DELETE'}", flush=True)

    print(f"\n########## OVERALL VERDICT {family}: "
          f"{'SAFE TO DELETE' if overall_ok else 'DO NOT DELETE'} ##########")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
