"""Download large Maxar scenes and annotations to a flat local directory.

The annotations should include subfolders containing:
- {X}-visual_labels.json
- {X}-visual.tif
- {x}-pan.tif
- {x}-ms.tif
"""

import argparse
import multiprocessing
import os
import shutil

import tqdm
from unidecode import unidecode
from upath import UPath


def retrieve_files(job: tuple[UPath, str]):
    """Retrieve files corresponding to the provided prefix.

    Args:
        job: a tuple (prefix, out_dir). prefix is a UPath like
            "gs://worldcover/x/y/z/abc-". out_dir is the local output directory.
    """
    prefix, out_dir = job

    # Get a flat name to use for these files in local output directory.
    label = unidecode(prefix.replace("/", "_"))

    wanted_suffixes = [
        "visual_labels.json",
        "visual.tif",
        "pan.tif",
        "ms.tif",
    ]
    for suffix in wanted_suffixes:
        src_path = prefix.parent / (prefix.name + suffix)
        dst_path = os.path.join(out_dir, label + suffix)

        with src_path.open("rb") as src:
            with open(dst_path + ".tmp", "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.rename(dst_path + ".tmp", dst_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", help="Source directory, can be prefixed with gs:// or s3://")
    parser.add_argument("--out_dir", help="Local output directory")
    parser.add_argument("--workers", help="Number of worker threads", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    src_dir = UPath(args.src_dir)
    jobs = []
    for fname in src_dir.glob("**/*visual_labels.json"):
        prefix = fname.parent / fname.name.split("visual_labels.json")[0]
        jobs.append((prefix, args.out_dir))

    p = multiprocessing.Pool(args.workers)
    outputs = p.imap_unordered(retrieve_files, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        continue
    p.close()
