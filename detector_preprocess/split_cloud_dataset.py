"""
Compute mapping from the downloaded scenes to train/val.
"""

import argparse
import hashlib
import json
import multiprocessing
import numpy as np
import os
import rasterio
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", help="Directory where the scenes have been downloaded")
parser.add_argument("--split_fname", help="Filename mapping each scene prefix to train or val")
parser.add_argument("--workers", help="Number of worker threads", type=int, default=32)
parser.add_argument("--mode", help="Mode to compute split, 'default' uses everything while 'nonempty' uses only scenes with at least one vessel label", type=str, default="default")
args = parser.parse_args()

signature = "-visual.tif"

def check_is_big(fname):
    with rasterio.open(fname) as src:
        array = src.read()
    num_positive_pixels = np.count_nonzero(array.max(axis=0) > 0)
    # We expect about 1 km x 1 km images for the small ones.
    # So threshold on double that @ 30 cm / pixel in the visual images.
    return num_positive_pixels > 2 * 1000 * 1000 / 0.3 / 0.3

def get_split(prefix):
    is_val = hashlib.sha256(prefix.encode()).hexdigest()[0] in ["0", "1"]

    # Assign all the old big scenes to train.
    if check_is_big(os.path.join(args.in_dir, prefix + signature)):
        is_val = False

    if args.mode == "nonempty":
        # Check that there is at least one label.
        with open(os.path.join(args.in_dir, prefix + "-visual_labels.json")) as f:
            data = json.load(f)
            if len(data["annotations"]) == 0:
                return (prefix, None)

    if is_val:
        return (prefix, "val")
    else:
        return (prefix, "train")

prefixes = []
for fname in os.listdir(args.in_dir):
    if not fname.endswith(signature):
        continue
    prefix = fname[:-len(signature)]
    prefixes.append(prefix)

split_data = {}
p = multiprocessing.Pool(64)
outputs = p.imap_unordered(get_split, prefixes)
for prefix, split in tqdm.tqdm(outputs, total=len(prefixes)):
    if split is None:
        continue
    split_data[prefix] = split
p.close()

with open(args.split_fname, "w") as f:
    json.dump(split_data, f)
