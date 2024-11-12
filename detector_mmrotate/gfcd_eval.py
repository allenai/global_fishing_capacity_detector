"""End-to-end evaluation on the CSVs from gfcd_collate.py."""

import argparse
import csv
import json
import os

import geopy.distance

class Vessel:
    def __init__(self, lons: list[float], lats: list[float]):
        # Compute center lon/lat.
        self.lon = (min(lons) + max(lons)) / 2
        self.lat = (min(lats) + max(lats)) / 2

        # Compute length based on longest side.
        max_side = 0
        for i in range(0, len(lons) - 1):
            p0 = (lats[i], lons[i])
            p1 = (lats[i+1], lons[i+1])
            max_side = max(max_side, geopy.distance.distance(p0, p1).m)
        self.length = max_side

    def distance(self, other: "Vessel") -> float:
        return geopy.distance.distance((self.lat, self.lon), (other.lat, other.lon)).m

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute metrics between collated CSVs and the annotations.",
    )
    parser.add_argument(
        "--image_csv",
        type=str,
        help="The image CSV.",
        required=True,
    )
    parser.add_argument(
        "--vessel_csv",
        type=str,
        help="The vessel CSV.",
        required=True,
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        help="Directory containing annotations (-visual_labels.json).",
        required=True,
    )
    parser.add_argument(
        "--dist_thr",
        type=float,
        default=1,
        help="Distance threshold as a multiple of vessel length.",
    )
    parser.add_argument(
        "--length_thr",
        type=float,
        default=None,
        help="Only match vessels that are similar length (default off). This is the length threshold as a factor of the ground truth length (e.g. 0.1 means between 0.9x and 1.1x of the gt length).",
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=None,
        help="Apply a threshold on the score column of predictions.",
    )
    args = parser.parse_args()

    # Read image_csv to get the list of image filenames.
    image_fnames = []
    with open(args.image_csv) as f:
        reader = csv.DictReader(f)
        for csv_row in reader:
            image_fnames.append(csv_row["fname"])

    # Get predictions.
    predictions = {}
    with open(args.vessel_csv) as f:
        reader = csv.DictReader(f)
        for csv_row in reader:
            if args.score_thr and float(csv_row["score"]) < args.score_thr:
                continue

            image_fname = csv_row["fname"]
            if image_fname not in predictions:
                predictions[image_fname] = []
            lons = [float(csv_row[f"wgs84_x{i}"]) for i in range(1, 5)]
            lats = [float(csv_row[f"wgs84_y{i}"]) for i in range(1, 5)]
            vessel = Vessel(lons, lats)
            predictions[image_fname].append(vessel)

    # Load ground truth.
    gt = {}
    for image_fname in image_fnames:
        prefix = image_fname.split("-visual.tif")[0]
        gt_fname = os.path.join(args.raw_dir, f"{prefix}-visual_labels.json")
        with open(gt_fname) as f:
            gt_data = json.load(f)
        for annot in gt_data["annotations"]:
            if annot["categories"][0]["name"] != "VESSEL_LENGTH_AND_HEADING_BOW_TO_STERN_DIRECTION":
                continue
            lons = [p["x"] for p in annot["polyline"]]
            lats = [p["y"] for p in annot["polyline"]]
            if image_fname not in gt:
                gt[image_fname] = []
            vessel = Vessel(lons, lats)
            gt[image_fname].append(vessel)

    tp = 0
    fp = 0
    fn = 0
    for image_fname in image_fnames:
        cur_gt = gt.get(image_fname, [])
        cur_pred = predictions.get(image_fname, [])
        matched_gt_indexes = set()
        # Match each prediction to closest gt.
        for pred_vessel in cur_pred:
            best_distance = None
            best_gt_idx = None
            for gt_idx, gt_vessel in enumerate(cur_gt):
                if gt_idx in matched_gt_indexes:
                    continue
                dist = pred_vessel.distance(gt_vessel)
                if dist > args.dist_thr * gt_vessel.length:
                    continue
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_gt_idx = gt_idx

            if best_gt_idx is None:
                fp += 1
                continue

            if args.length_thr:
                gt_vessel = cur_gt[best_gt_idx]
                too_small = pred_vessel.length < (1 - args.length_thr) * gt_vessel.length
                too_big = pred_vessel.length > (1 + args.length_thr) * gt_vessel.length
                if too_small or too_big:
                    fp += 1
                    continue

            tp += 1
            matched_gt_indexes.add(best_gt_idx)

        fn += len(cur_gt) - len(matched_gt_indexes)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(f"precision={precision}, recall={recall}")
