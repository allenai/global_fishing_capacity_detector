"""Add land/water area to image CSV."""

import argparse
import csv
import multiprocessing
import os

import numpy as np
import rasterio
import skimage.transform
import tqdm
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

WATER_CLASS = 80
RESOLUTION = 5


def get_land_water_area(
    image_fname: str, image_dir: str, ds_path: UPath
) -> tuple[str, float, float]:
    """Computes the sq km of land and water in the image.

    Args:
        image_fname: the image filename.
        image_dir: the directory containing Maxar images.
        ds_path: the rslearn dataset containing WorldCover images.

    Returns:
        a tuple (image_fname, land sq km, water sq km)
    """
    with rasterio.open(os.path.join(image_dir, image_fname)) as raster:
        maxar_mask = (raster.read().max(axis=0) > 0).astype(np.uint8)

    worldcover_fname = (
        ds_path
        / "windows"
        / "default"
        / image_fname
        / "layers"
        / "worldcover"
        / "B1"
        / "geotiff.tif"
    )
    with worldcover_fname.open("rb") as f:
        with rasterio.open(f) as raster:
            worldcover_array = raster.read(1)

    maxar_mask = skimage.transform.resize(
        maxar_mask, worldcover_array.shape, preserve_range=True, order=0
    ).astype(np.uint8)
    worldcover_masked = worldcover_array * maxar_mask
    land_pixels = np.count_nonzero(
        (worldcover_masked > 0) & (worldcover_masked != WATER_CLASS)
    )
    water_pixels = np.count_nonzero(worldcover_masked == WATER_CLASS)
    sq_km_per_pixel = RESOLUTION * RESOLUTION / 1000 / 1000
    return (image_fname, land_pixels * sq_km_per_pixel, water_pixels * sq_km_per_pixel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute land and water area in image CSV.",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        help="Directory containing input image CSV(s)",
        required=True,
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Directory containing Maxar images",
        required=True,
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Path to dataset containing materialized WorldCover images",
        required=True,
    )
    parser.add_argument(
        "--out_fname",
        type=str,
        help="Output filename of concatenated image CSV with land and water area columns added.",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes to use",
        default=32,
    )
    args = parser.parse_args()

    columns = None
    csv_rows = []

    for fname in os.listdir(args.csv_dir):
        if not fname.endswith("_images.csv"):
            continue
        with open(os.path.join(args.csv_dir, fname)) as f:
            reader = csv.DictReader(f)
            for csv_row in reader:
                csv_rows.append(csv_row)
            if columns is None:
                columns = reader.fieldnames

    ds_path = UPath(args.ds_path)
    jobs = []
    for csv_row in csv_rows:
        jobs.append(
            dict(
                image_fname=csv_row["fname"],
                image_dir=args.image_dir,
                ds_path=ds_path,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, get_land_water_area, jobs)
    land_water_by_image = {}
    for image_id, land_area, water_area in tqdm.tqdm(outputs, total=len(jobs)):
        land_water_by_image[image_id] = (land_area, water_area)
    p.close()

    with open(args.out_fname, "w") as f:
        writer = csv.DictWriter(f, columns + ["land_sq_km", "water_sq_km"])
        writer.writeheader()
        for csv_row in csv_rows:
            land_area, water_area = land_water_by_image[csv_row["fname"]]
            csv_row["land_sq_km"] = land_area
            csv_row["water_sq_km"] = water_area
            writer.writerow(csv_row)
