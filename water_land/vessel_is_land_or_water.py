"""Add is_in_water attribute to vessel CSV.

The value should be "yes" if the vessel is in open water and "no" if it is on land or
moored.
"""

import argparse
import csv
import multiprocessing

import numpy as np
import rasterio
import rasterio.warp
import tqdm
from rasterio.crs import CRS
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

WATER_CLASS = 80
WGS84_CRS = CRS.from_epsg(4326)
RADIUS = 2


def clip(x: int, lo: int, hi: int) -> int:
    """Clip a value to a range (closed interval).

    Args:
        x: the value to clip.
        lo: minimum value in the range.
        hi: maximum value in the range.

    Returns:
        clipped value in [lo, hi].
    """
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def add_is_in_water_attribute(
    image_fname: str, csv_rows: list[dict[str, str]], ds_path: UPath
) -> list[dict[str, str]]:
    """Add is_in_water attribute to the provided CSV rows.

    The rows must share the same image fname, matching the one provided.

    Args:
        image_fname: the image filename.
        csv_rows: the vessel detection rows for which to add is_in_water attribute.
        ds_path: the rslearn dataset containing WorldCover images.

    Returns:
        updated rows with is_in_water attribute added.
    """
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
            for csv_row in csv_rows:
                # We need do some coordinate transforms to get WorldCover pixel for the
                # vessel:
                # (WGS84 vessel center) -> (projection coordinates) -> (pixel coordinates)
                xs = [float(csv_row[f"wgs84_x{idx}"]) for idx in range(1, 5)]
                ys = [float(csv_row[f"wgs84_y{idx}"]) for idx in range(1, 5)]
                wgs84_x = (min(xs) + max(xs)) / 2
                wgs84_y = (min(ys) + max(ys)) / 2
                result_xs, result_ys = rasterio.warp.transform(
                    WGS84_CRS, raster.crs, [wgs84_x], [wgs84_y]
                )
                proj_x = result_xs[0]
                proj_y = result_ys[0]
                pixel_row, pixel_col = raster.index(proj_x, proj_y)
                pixel_row = int(pixel_row)
                pixel_col = int(pixel_col)

                # Now we can check whether there is any non-water near the vessel.
                sx = clip(pixel_col - RADIUS, 0, worldcover_array.shape[1])
                sy = clip(pixel_row - RADIUS, 0, worldcover_array.shape[0])
                ex = clip(pixel_col + RADIUS + 1, 0, worldcover_array.shape[1])
                ey = clip(pixel_row + RADIUS + 1, 0, worldcover_array.shape[0])
                crop = worldcover_array[sy:ey, sx:ex]
                csv_row["is_in_water"] = np.count_nonzero(crop != WATER_CLASS) == 0

    return csv_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute is_in_water attribute for vessel CSV",
    )
    parser.add_argument(
        "--csv_fname",
        type=str,
        help="Filename of the input vessel CSV",
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
        help="Output filename of vessel CSV with is_in_water column added.",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes to use",
        default=32,
    )
    args = parser.parse_args()

    csv_rows_by_image = {}
    with open(args.csv_fname) as f:
        reader = csv.DictReader(f)
        for csv_row in reader:
            image_fname = csv_row["fname"]
            if image_fname not in csv_rows_by_image:
                csv_rows_by_image[image_fname] = []
            csv_rows_by_image[image_fname].append(csv_row)
        out_columns = reader.fieldnames + ["is_in_water"]

    ds_path = UPath(args.ds_path)
    jobs = []
    for image_fname, csv_rows in csv_rows_by_image.items():
        jobs.append(
            dict(
                image_fname=image_fname,
                csv_rows=csv_rows,
                ds_path=ds_path,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, add_is_in_water_attribute, jobs)
    out_csv_rows = []
    for output in tqdm.tqdm(outputs, total=len(jobs)):
        out_csv_rows.extend(output)
    p.close()

    with open(args.out_fname, "w") as f:
        writer = csv.DictWriter(f, out_columns)
        writer.writeheader()
        writer.writerows(out_csv_rows)
