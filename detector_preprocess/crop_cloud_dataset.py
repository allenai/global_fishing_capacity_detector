"""
Divide scenes downloaded by download_cloud_dataset.py into small crops.
Also split into train/val.
"""

import argparse
import json
import math
import multiprocessing
import numpy as np
import os
import rasterio
import rasterio.warp
from rasterio.crs import CRS
import skimage.io
import skimage.transform
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", help="Directory where the scenes have been downloaded")
parser.add_argument("--out_dir", help="Directory to write train/val crops")
parser.add_argument("--split_fname", help="Filename mapping each scene prefix to train or val")
parser.add_argument("--workers", help="Number of worker threads", type=int, default=32)
parser.add_argument("--crop_size", help="Size of training example crops", type=int, default=1024)
args = parser.parse_args()

def get_center(segment):
    return ((segment[0][0]+segment[1][0])//2, (segment[0][1]+segment[1][1])//2)

def get_distance(point1, point2):
    d0 = point1[0] - point2[0]
    d1 = point1[1] - point2[1]
    return math.sqrt(d0*d0 + d1*d1)

def crop_data(job):
    in_dir, out_dir, prefix, split, crop_size = job

    # Get the "visual" image.
    raster = rasterio.open(os.path.join(in_dir, prefix + "-visual.tif"))
    image = raster.read()
    # CHW -> HWC
    image = image.transpose(1, 2, 0)

    # Get the pan + ms images too.
    # And resize them to be same size as the visual image.
    pan_raster = rasterio.open(os.path.join(in_dir, prefix + "-pan.tif"))
    ms_raster = rasterio.open(os.path.join(in_dir, prefix + "-ms.tif"))
    pan_image = pan_raster.read().transpose(1, 2, 0)
    ms_image = ms_raster.read().transpose(1, 2, 0)
    pan_image = skimage.transform.resize(pan_image, image.shape[0:2], order=1, preserve_range=True).astype(np.uint16)
    ms_image = skimage.transform.resize(ms_image, image.shape[0:2], order=1, preserve_range=True).astype(np.uint16)
    combined_image = np.concatenate([pan_image, ms_image], axis=2)

    with open(os.path.join(in_dir, prefix + "-visual_labels.json"), "r") as f:
        data = json.load(f)

    # Transform annotations into the coordinate system of the image.
    # Extract four key points: stern, bow, port, starboard.
    # The stern->bow and port->starboard are separate lines so we need to associate them.
    src_crs = CRS.from_epsg(4326)
    length_segments = []
    width_segments = []
    for ann in data["annotations"]:
        # Transform points.
        xs = [point["x"] for point in ann["polyline"]]
        ys = [point["y"] for point in ann["polyline"]]
        xs, ys = rasterio.warp.transform(src_crs, raster.crs, xs, ys)
        points = []
        for x, y in zip(xs, ys):
            row, col = raster.index(x, y)
            points.append((col, row))

        assert len(points) == 2

        assert len(ann["categories"]) == 1
        category = ann["categories"][0]["name"]
        if category == "VESSEL_LENGTH_AND_HEADING_BOW_TO_STERN_DIRECTION":
            length_segments.append(points)
        elif category == "VESSEL_BEAM_PORT_TO_STARBOARD_0":
            width_segments.append(points)
        else:
            raise Exception(f"unknown annotation category {category}")

    # Find best width segment for each length segment (closest centers).
    labels = []
    for length_segment in length_segments:
        length_center = get_center(length_segment)
        best_width_segment = None
        best_distance = None
        for width_segment in width_segments:
            width_center = get_center(width_segment)
            distance = get_distance(length_center, width_center)
            if best_distance is None or distance < best_distance:
                best_width_segment = width_segment
                best_distance = distance

        # Sanity check: the centers should be closer than the length of the vessel.
        # There are some labels that only have length or width. Skip them for now.
        # There's now even some images that only have a length label or a width label.
        if best_distance is None or best_distance > get_distance(length_segment[0], length_segment[1]):
            print(f"warning: bad length segment {length_segment} in {prefix} (best is {best_width_segment} at {best_distance})")
            continue

        # Write in stern, bow, port, starboard order.
        labels.append((
            length_segment[0],
            length_segment[1],
            best_width_segment[0],
            best_width_segment[1],
        ))

    # Now that the annotations are good, divide the image into crops.
    for col in range(0, image.shape[1], crop_size):
        for row in range(0, image.shape[0], crop_size):
            crop = image[row:row+crop_size, col:col+crop_size, :]
            if crop.max() == 0:
                continue

            cur_labels = []
            for label in labels:
                col1 = min(label[0][0], label[1][0])
                row1 = min(label[0][1], label[1][1])
                col2 = max(label[0][0], label[1][0])
                row2 = max(label[0][1], label[1][1])
                if col2 < col:
                    continue
                if row2 < row:
                    continue
                if col1 >= (col+crop_size):
                    continue
                if row1 >= (row+crop_size):
                    continue
                label = [
                    (point[0] - col, point[1] - row)
                    for point in label
                ]
                cur_labels.append(label)

            combined_crop = combined_image[row:row+crop_size, col:col+crop_size, :]

            out_prefix = os.path.join(out_dir, split, f"{prefix}_{col}_{row}")
            skimage.io.imsave(out_prefix+".png", crop, check_contrast=False)
            np.save(out_prefix+".npy", combined_crop)
            with open(out_prefix+".json", "w") as f:
                json.dump(cur_labels, f)

if __name__ == "__main__":
    os.makedirs(os.path.join(args.out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "val"), exist_ok=True)

    with open(args.split_fname) as f:
        split_data = json.load(f)

    jobs = []
    for prefix, split in split_data.items():
        jobs.append((args.in_dir, args.out_dir, prefix, split, args.crop_size))
    p = multiprocessing.Pool(args.workers)
    outputs = p.imap_unordered(crop_data, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        continue
    p.close()
