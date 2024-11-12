"""Collate the results from object detector + classification model into CSV.

These outputs are stored in an rslearn dataset. So here we just look through the
detections that were written and also have positive output from the classifier.

User can optionally provide a confidence threshold for the object detector score.
"""

import argparse
import csv
import json
import os

GROUP = "output"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collate outputs from detection + classification pipeline.",
    )
    parser.add_argument(
        "--ds_dir",
        type=str,
        help="The rslearn dataset directory.",
        required=True,
    )
    parser.add_argument(
        "--out_fname",
        type=str,
        help="The filename to write vessel CSV",
        required=True,
    )
    parser.add_argument(
        "--detect_threshold",
        type=float,
        default=None,
        help="Apply a confidence threshold on the object detector scores",
    )
    parser.add_argument(
        "--skip_classifier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore the classifier prediction.",
    )
    parser.add_argument(
        "--read_classifier_prob_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Read the classifier probability without applying the classifier.",
    )
    args = parser.parse_args()

    csv_rows = []
    group_dir = os.path.join(args.ds_dir, "windows", GROUP)
    if os.path.exists(group_dir):
        for window_id in os.listdir(group_dir):
            window_dir = os.path.join(args.ds_dir, "windows", GROUP, window_id)
            with open(os.path.join(window_dir, "vessel.json")) as f:
                vessel_data = json.load(f)

            if args.detect_threshold is not None and float(vessel_data["score"]) < args.detect_threshold:
                continue

            if not args.skip_classifier:
                output_fname = os.path.join(window_dir, "layers", "output", "data.geojson")
                with open(output_fname) as f:
                    cls_data = json.load(f)

                if args.read_classifier_prob_only:
                    vessel_data["score"] = str(cls_data["features"][0]["properties"]["prob"][0])

                elif cls_data["features"][0]["properties"]["label"] == "negative":
                    continue

            csv_rows.append(vessel_data)

    with open(args.out_fname, "w") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "fname",
            "idx",
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
            "score",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
