"""Predict ship locations in a Maxar -visual.tif file."""

import argparse
import csv
import json
import os
from typing import Any, Optional

import numpy as np
import rasterio
import rasterio.features
import rasterio.io
import rasterio.warp
import shapely
import tqdm
import torch
from PIL import Image
from rasterio.crs import CRS

from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from mmrotate.core import obb2poly_np


class VesselDetection:
    """Represents one detection of a vessel in a GeoTIFF."""

    def __init__(
        self,
        fname: str,
        idx: int,
        pixel_coordinates: list[float, float, float, float, float, float, float, float],
        wgs84_coordinates: list[float, float, float, float, float, float, float, float],
        score: float,
    ):
        """Create a new VesselDetection.

        Args:
            fname: the image filename where this vessel was detected.
            idx: unique integer that identifies this vessel detection together with fname.
            pixel_coordinates: the polygon in pixel coordinates.
            wgs84_coordinates: the polygon in lon, lat coordinates.
            score: the confidence score of this detection.
        """
        self.fname = fname
        self.idx = idx
        self.pixel_coordinates = pixel_coordinates
        self.wgs84_coordinates = wgs84_coordinates
        self.score = score

    def to_str_dict(self) -> dict[str, str]:
        """Convert to dict for CSV writing."""
        return {
            "fname": self.fname,
            "idx": self.idx,
            "pixel_x1": str(self.pixel_coordinates[0]),
            "pixel_y1": str(self.pixel_coordinates[1]),
            "pixel_x2": str(self.pixel_coordinates[2]),
            "pixel_y2": str(self.pixel_coordinates[3]),
            "pixel_x3": str(self.pixel_coordinates[4]),
            "pixel_y3": str(self.pixel_coordinates[5]),
            "pixel_x4": str(self.pixel_coordinates[6]),
            "pixel_y4": str(self.pixel_coordinates[7]),
            "wgs84_x1": str(self.wgs84_coordinates[0]),
            "wgs84_y1": str(self.wgs84_coordinates[1]),
            "wgs84_x2": str(self.wgs84_coordinates[2]),
            "wgs84_y2": str(self.wgs84_coordinates[3]),
            "wgs84_x3": str(self.wgs84_coordinates[4]),
            "wgs84_y3": str(self.wgs84_coordinates[5]),
            "wgs84_x4": str(self.wgs84_coordinates[6]),
            "wgs84_y4": str(self.wgs84_coordinates[7]),
            "score": str(self.score),
        }

    def pixel_center(self) -> tuple[int, int]:
        xs = [self.pixel_coordinates[i] for i in range(0, 8, 2)]
        ys = [self.pixel_coordinates[i] for i in range(1, 8, 2)]
        cx = int(min(xs) + max(xs)) // 2
        cy = int(min(ys) + max(ys)) // 2
        return cx, cy

    def get_geojson(self) -> dict[str, Any]:
        """Returns GeoJSON geometry representing pixel coordinates of this vessel."""
        return {
            "type": "LineString",
            "coordinates": [
                [self.pixel_coordinates[0], self.pixel_coordinates[1]],
                [self.pixel_coordinates[2], self.pixel_coordinates[3]],
                [self.pixel_coordinates[4], self.pixel_coordinates[5]],
                [self.pixel_coordinates[6], self.pixel_coordinates[7]],
                [self.pixel_coordinates[0], self.pixel_coordinates[1]],
            ],
        }


class ImageMetadata:
    """Represents the metadata of an image to go in output CSV."""

    def __init__(
        self,
        fname: str,
        pixel_coordinates: list[float, float, float, float, float, float, float, float],
        wgs84_coordinates: list[float, float, float, float, float, float, float, float],
        timestamp: str,
    ):
        """Create a new ImageMetadata.

        Args:
            fname: the filename of this image.
            pixel_coordinates: the image bounds in pixel coordinates.
            wgs84_coordinates: the image bounds in lon, lat coordinates.
        """
        self.fname = fname
        self.pixel_coordinates = pixel_coordinates
        self.wgs84_coordinates = wgs84_coordinates
        self.timestamp = timestamp

    def to_str_dict(self) -> dict[str, str]:
        """Convert to dict for CSV writing."""
        return {
            "fname": self.fname,
            "pixel_x1": str(self.pixel_coordinates[0]),
            "pixel_y1": str(self.pixel_coordinates[1]),
            "pixel_x2": str(self.pixel_coordinates[2]),
            "pixel_y2": str(self.pixel_coordinates[3]),
            "pixel_x3": str(self.pixel_coordinates[4]),
            "pixel_y3": str(self.pixel_coordinates[5]),
            "pixel_x4": str(self.pixel_coordinates[6]),
            "pixel_y4": str(self.pixel_coordinates[7]),
            "wgs84_x1": str(self.wgs84_coordinates[0]),
            "wgs84_y1": str(self.wgs84_coordinates[1]),
            "wgs84_x2": str(self.wgs84_coordinates[2]),
            "wgs84_y2": str(self.wgs84_coordinates[3]),
            "wgs84_x3": str(self.wgs84_coordinates[4]),
            "wgs84_y3": str(self.wgs84_coordinates[5]),
            "wgs84_x4": str(self.wgs84_coordinates[6]),
            "wgs84_y4": str(self.wgs84_coordinates[7]),
            "timestamp": self.timestamp,
        }


def polygon_to_coords(
    polygon: shapely.Polygon,
    raster: rasterio.io.DatasetReader,
) -> tuple[
    list[float, float, float, float, float, float, float, float],
    list[float, float, float, float, float, float, float, float],
]:
    """Converts a polygon to pixel and WGS-84 coordinates.

    Args:
        polygon: the polygon in pixel coordinates with respect to the image.
        raster: the image.

    Returns:
        (pixel_coordinates, wgs84_coordinates)
    """
    dst_crs = CRS.from_epsg(4326)
    pixel_coordinates = []
    wgs84_coordinates = []
    for x, y in polygon.exterior.coords:
        pixel_coordinates.extend([x, y])
        # To projection coordinates.
        proj_x, proj_y = raster.xy(y, x)
        # To WGS84 coordinates.
        wgs84_xs, wgs84_ys = rasterio.warp.transform(raster.crs, dst_crs, [proj_x], [proj_y])
        wgs84_coordinates.extend([wgs84_xs[0], wgs84_ys[0]])
    return pixel_coordinates, wgs84_coordinates


def get_detections(
    fname: str,
    model: torch.nn.Module,
    raster: rasterio.io.DatasetReader,
    window_size: int,
    stride: int,
    margin: int,
    score_threshold: float,
    area_threshold: float,
    out_dir: Optional[str] = None,
) -> list[VesselDetection]:
    array = raster.read()
    row_offsets = (
        [0]
        + list(
            range(
                stride,
                array.shape[1] - window_size,
                stride,
            )
        )
        + [array.shape[1] - window_size]
    )
    col_offsets = (
        [0]
        + list(
            range(
                stride,
                array.shape[2] - window_size,
                stride,
            )
        )
        + [array.shape[2] - window_size]
    )

    polygons: list[tuple[shapely.Polygon, float]] = []

    for row_offset in row_offsets:
        for col_offset in col_offsets:
            patch = array[:, row_offset:row_offset+window_size, col_offset:col_offset+window_size]
            if patch.max() == 0:
                continue

            # IMPORTANT: mmcv's LoadImageFromFile will load in BGR order.
            # So here we need to put our RGB image into BGR order.
            # Otherwise we get a 10% recall hit.
            # (We also convert from CHW to HWC.)
            result = inference_detector(model, patch.transpose(1, 2, 0)[:, :, (2, 1, 0)])

            for box in result[0]:
                if box[5] < score_threshold:
                    continue

                # Check margin, but only if this crop doesn't sit on the edge of the
                # overall image.
                center = (box[0], box[1])
                if row_offset != 0 and row_offset != array.shape[1] - window_size:
                    if center[1] < margin or center[1] > window_size - margin:
                        continue
                if col_offset != 0 and col_offset != array.shape[2] - window_size:
                    if center[0] < margin or center[0] > window_size - margin:
                        continue

                # Convert [x_ctr, y_ctr, w, h, angle]
                # to [x0,y0,x1,y1,x2,y2,x3,y3].
                coords = obb2poly_np(box[None, :])[0, 0:8]
                coords = coords.reshape(-1, 2) + [col_offset, row_offset]
                polygon = shapely.Polygon(coords)
                polygons.append((polygon, float(box[5])))

    # Check intersect area.
    # We can have intersecting polygons due to the stride used above.
    # So here, if one polygon contains most of another, we remove the smaller polygon.
    bad_indices = set()
    for idx1, (poly1, _) in enumerate(polygons):
        if idx1 in bad_indices:
            continue
        for idx2, (poly2, _) in enumerate(polygons):
            if idx1 == idx2:
                continue
            if idx2 in bad_indices:
                continue
            if not poly1.intersects(poly2):
                continue
            intersect_area = poly1.intersection(poly2).area
            min_area = min(poly1.area, poly2.area)
            if intersect_area / min_area < area_threshold:
                continue

            if poly1.area < poly2.area:
                bad_indices.add(idx1)
                break
            else:
                bad_indices.add(idx2)
                continue

    # Save vessel detections to the output directory.
    # We save it in an rslearn dataset structure, with fake metadata.json.
    crop_size = 256
    padded_array = np.pad(array, [(0, 0), (crop_size//2, crop_size//2), (crop_size//2, crop_size//2)])
    for idx, (polygon, score) in enumerate(polygons):
        if idx in bad_indices:
            continue

        pixel_coordinates, wgs84_coordinates = polygon_to_coords(polygon, raster)

        detection = VesselDetection(
            fname=fname,
            idx=idx,
            pixel_coordinates=pixel_coordinates,
            wgs84_coordinates=wgs84_coordinates,
            score=score,
        )
        cx, cy = detection.pixel_center()
        if cx < 0 or cy < 0 or cx >= array.shape[2] or cy >= array.shape[1]:
            # Just skip predictions that are literally on the border.
            continue
        crop = padded_array[:, cy:cy+crop_size, cx:cx+crop_size]

        group = "output"
        window_id = f"{fname}_{idx}"
        window_dir = os.path.join(out_dir, "windows", group, window_id)

        output_layer_dir = os.path.join(window_dir, "layers", "maxar")
        output_fname = os.path.join(output_layer_dir, "R_G_B", "image.png")
        os.makedirs(os.path.dirname(output_fname), exist_ok=True)
        Image.fromarray(crop.transpose(1, 2, 0)).save(output_fname)
        with open(os.path.join(output_layer_dir, "completed"), "w") as f:
            pass

        with open(os.path.join(window_dir, "vessel.json"), "w") as f:
            json.dump(detection.to_str_dict(), f)

        with open(os.path.join(window_dir, "metadata.json"), "w") as f:
            fake_metadata = {
                "group": group,
                "name": window_id,
                "projection": {"crs": "EPSG:3857", "x_resolution": 1, "y_resolution": -1},
                "bounds": [0, 0, crop_size, crop_size],
                "time_range": ["2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00"],
                "options": {"split": "val"},
            }
            json.dump(fake_metadata, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply AI2/Minderoo vessel detection model on Maxar images",
    )
    parser.add_argument(
        "--fnames",
        type=str,
        help="Comma-separated list of -visual.tif GeoTIFF filenames to apply the model on",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=1024,
        help="Size of patches to run the model on",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Inference stride, typically half of crop size",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Score threshold",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=0,
        help="Margin on edge of crop to ignore detections",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="detector.pth",
        help="Model checkpoint filename.",
    )
    """
    Score threshold to precision, recall, and F1 score:
    minderoo_vessel@0.05_0.43665952500341887_0.8892499071667286 0.5857103549481794
    minderoo_vessel@0.1_0.5262457735158805_0.8813590790939473 0.659008086627564
    minderoo_vessel@0.2_0.623042580129273_0.8679910880059414 0.7253966406765197
    minderoo_vessel@0.3_0.6825044404973357_0.8561084292610471 0.7595124361719651
    minderoo_vessel@0.4_0.7291900561347233_0.8441329372447085 0.7824627828930385
    minderoo_vessel@0.5_0.7695614789337919_0.8308577794281471 0.7990358003749666
    minderoo_vessel@0.6_0.8041708588676484_0.8161901225399183 0.810135913384013
    minderoo_vessel@0.7_0.836780487804878_0.7962309691793539 0.8160022833222338
    minderoo_vessel@0.8_0.8738786279683377_0.7686594875603416 0.8178989479922952
    minderoo_vessel@0.9_0.9187469995199232_0.7106386929075381 0.8014028475711893
    minderoo_vessel@0.95_0.9452260620133861_0.6424062383958411 0.7649367158569613
    """
    parser.add_argument(
        "--area_threshold",
        type=float,
        default=0.5,
        help="Remove predictions that overlap by more than this threshold",
    )
    parser.add_argument(
        "--image_csv",
        type=str,
        help="Output CSV file for image metadata",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Directory to write the rslearn dataset.",
    )
    args = parser.parse_args()

    from .gfcd_train import get_cfg
    cfg = get_cfg()

    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.CLASSES = ["vessel"]

    device = "cuda:0"
    checkpoint = load_checkpoint(model, args.weight_path, map_location=device)
    model.cfg = cfg
    model.to(device)
    model.eval()

    all_image_metas = []

    for img_fname in tqdm.tqdm(args.fnames.split(",")):
        with rasterio.open(img_fname) as raster:
            get_detections(
                fname=os.path.basename(img_fname),
                model=model,
                raster=raster,
                window_size=args.window_size,
                stride=args.stride,
                margin=args.margin,
                score_threshold=args.score_threshold,
                area_threshold=args.area_threshold,
                out_dir=args.out_dir,
            )

            polygon = shapely.Polygon([
                [0, 0],
                [0, raster.width],
                [raster.height, raster.width],
                [raster.height, 0],
            ])
            pixel_coordinates, wgs84_coordinates = polygon_to_coords(polygon, raster)

            image_timestamp = "unknown"
            json_fname = img_fname.split(".tif")[0] + ".json"
            if os.path.exists(json_fname):
                with open(json_fname) as f:
                    location_data = json.load(f)
                if location_data["ordered_imagery_dates"] is not None:
                    image_timestamp = location_data["ordered_imagery_dates"][0]
                else:
                    image_timestamp = list(location_data["files"].values())[0]["collection_date"]

            image_meta = ImageMetadata(
                fname=os.path.basename(img_fname),
                pixel_coordinates=pixel_coordinates,
                wgs84_coordinates=wgs84_coordinates,
                timestamp=image_timestamp,
            )
            all_image_metas.append(image_meta)

    with open(args.image_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "fname",
            "timestamp",
            "pixel_x1",
            "pixel_y1",
            "pixel_x2",
            "pixel_y2",
            "pixel_x3",
            "pixel_y3",
            "pixel_x4",
            "pixel_y4",
            "wgs84_x1",
            "wgs84_y1",
            "wgs84_x2",
            "wgs84_y2",
            "wgs84_x3",
            "wgs84_y3",
            "wgs84_x4",
            "wgs84_y4",
        ])
        writer.writeheader()
        for image_meta in all_image_metas:
            writer.writerow(image_meta.to_str_dict())
