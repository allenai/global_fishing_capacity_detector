"""Produce visualization from collated vessel CSV.

Copy images in subfolders to one folder:

import os
import shutil
for dir_name in os.listdir("."):
    if dir_name == "combined":
        continue
    for fname in os.listdir(dir_name):
        label = fname.split(".tif.png")[0]
        dst_fname = os.path.join("combined", f"{label}_{dir_name}.png")
        if os.path.exists(dst_fname):
            continue
        shutil.copyfile(os.path.join(dir_name, fname), dst_fname)
"""

import argparse
import csv
import multiprocessing
import os

import numpy as np
from PIL import Image
import rasterio
import rasterio.features
import shapely
import skimage.morphology
import tqdm

Image.MAX_IMAGE_PIXELS = None


def process(job):
    args, image_fname, vessel_list = job

    dst_fname = os.path.join(args.out_dir, image_fname + ".jpg")
    if os.path.exists(dst_fname):
        return

    with rasterio.open(os.path.join(args.raw_dir, image_fname)) as src:
        image = src.read()
        shapes = [(polygon, 255) for _, polygon in vessel_list]
        mask = rasterio.features.rasterize(shapes, out_shape=(image.shape[1], image.shape[2]), dtype=np.uint8)
        mask = skimage.morphology.binary_dilation(mask > 0)
        vis_im = image.transpose(1, 2, 0)
        vis_im[mask] = [255, 255, 0]

        vis_im = Image.fromarray(vis_im)
        if args.autocrop:
            bbox = vis_im.getbbox()
            vis_im = vis_im.crop(bbox)

        vis_im.save(dst_fname)

        if not args.crop_dir:
            return

        vis_im = np.array(vis_im)
        padding = 32
        for vessel_idx, polygon in vessel_list:
            bounds = polygon.bounds
            bounds = [
                int(np.clip(bounds[0] - padding, 0, vis_im.shape[1])),
                int(np.clip(bounds[1] - padding, 0, vis_im.shape[0])),
                int(np.clip(bounds[2] + padding, 0, vis_im.shape[1])),
                int(np.clip(bounds[3] + padding, 0, vis_im.shape[0])),
            ]
            crop = vis_im[bounds[1]:bounds[3], bounds[0]:bounds[2], :]
            Image.fromarray(crop).save(os.path.join(args.crop_dir, f"{image_fname}_{vessel_idx}.jpg"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create visualization of vessel detections.",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        help="The directory containing the input GeoTIFFs.",
        required=True,
    )
    parser.add_argument(
        "--vessel_csv",
        type=str,
        help="The vessel CSV.",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="The directory to write the visualization PNGs.",
        required=True,
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=None,
        help="Only show vessel detections where score column is higher than this threshold.",
    )
    parser.add_argument(
        "--autocrop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-crop the images (remove nodata regions).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Only visualize images with this prefix (optional)",
    )
    parser.add_argument(
        "--crop_dir",
        type=str,
        default=None,
        help="Optional directory to write individual vessel crops",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.crop_dir is not None:
        os.makedirs(args.crop_dir, exist_ok=True)

    vessels_by_fname = {}
    with open(args.vessel_csv) as f:
        reader = csv.DictReader(f)
        for csv_row in reader:
            if args.score_thr and float(csv_row["score"]) < args.score_thr:
                continue

            image_fname = csv_row["fname"]
            if args.prefix and not image_fname.startswith(args.prefix):
                continue
            if image_fname not in vessels_by_fname:
                vessels_by_fname[image_fname] = []
            vessel_idx = int(csv_row["idx"])
            points = [
                (float(csv_row[f"pixel_x{i}"]), float(csv_row[f"pixel_y{i}"]))
                for i in range(1, 5)
            ]
            polygon = shapely.LineString(points + [points[0]])
            vessels_by_fname[image_fname].append((vessel_idx, polygon))

    p = multiprocessing.Pool(32)
    jobs = []
    for image_fname, vessel_list in vessels_by_fname.items():
        jobs.append((args, image_fname, vessel_list))
    outputs = p.imap_unordered(process, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
