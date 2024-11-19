"""Create a window corresponding to each Maxar image.

This is used for populating roughly corresponding Worldcover images.

Roughly corresponding because the resolution of the window is 5 m/pixel while the Maxar
images are much higher resolution.
"""

import argparse
import os

import rasterio
import shapely
from rasterio.crs import CRS
from rslearn.const import WGS84_EPSG
from rslearn.dataset import Window
from rslearn.utils.geometry import Projection, STGeometry
from upath import UPath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make windows to get WorldCover images",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Dataset path to use, make sure config.json is copied here",
        required=True,
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Directory of the processed Maxar images",
        required=True,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)

    for image_fname in os.listdir(args.image_dir):
        if not image_fname.endswith(".tif"):
            continue

        with rasterio.open(os.path.join(args.image_dir, image_fname)) as raster:
            projection = Projection(raster.crs, raster.transform.a, raster.transform.e)
            left = int(raster.transform.c / projection.x_resolution)
            top = int(raster.transform.f / projection.y_resolution)
            image_bounds = shapely.box(
                left, top, left + raster.width, top + raster.height
            )

        src_geom = STGeometry(projection, image_bounds, None)
        # We don't want to store WorldCover data at the high resolution of Maxar image.
        # So pick a reasonable resolution depending on the CRS (hopefully either UTM or
        # WGS84).
        if projection.crs == CRS.from_epsg(WGS84_EPSG):
            dst_projection = Projection(projection.crs, 0.00005, -0.00005)
        else:
            dst_projection = Projection(projection.crs, 5, -5)
        dst_geom = src_geom.to_projection(dst_projection)
        window_bounds = [int(value) for value in dst_geom.shp.bounds]

        group = "default"
        window_name = image_fname
        window_path = ds_path / "windows" / group / window_name
        Window(
            path=window_path,
            group=group,
            name=window_name,
            projection=dst_projection,
            bounds=window_bounds,
            time_range=None,
        ).save()
