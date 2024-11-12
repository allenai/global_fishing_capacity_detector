import argparse
import json
import os
import shutil
import subprocess

def main(scene_fname: str, out_dir: str, worldcover_dir: str | None = None):
    assert not os.path.exists(out_dir), "output directory already exists"

    # Make a copy of the scene fname since a lot of scripts expect it to be in a
    # directory containing only the images we want to process.
    # This code can only process one image so it should be its own directory.
    image_dir = os.path.join(out_dir, "images")
    image_fname = os.path.join(image_dir, os.path.basename(scene_fname))
    os.makedirs(image_dir)
    shutil.copyfile(scene_fname, image_fname)

    # Apply the object detector.
    # It produces an rslearn dataset containing vessel detection crops that is suitable
    # for running the classifier in the next step.
    print("pipeline: apply detector")
    ds_path = os.path.join(out_dir, "rslearn_dataset")
    image_csv_fname = os.path.join(out_dir, "orig_images.csv")
    subprocess.check_call([
        "python",
        "-m",
        "detector_mmrotate.gfcd_predict",
        "--fnames",
        image_fname,
        "--out_dir",
        ds_path,
        "--image_csv",
        image_csv_fname,
        "--weight_path",
        "detector.pth",
    ])

    # So now we can run the classifier.
    # ds_path might not exist if there were zero vessel detections.
    if os.path.exists(ds_path):
        print("pipeline: apply classifier")
        shutil.copyfile("classify/config.json", os.path.join(ds_path, "config.json"))
        subprocess.check_call([
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "gfcd_classifier",
            "rslearn",
            "model",
            "predict",
            "--config",
            "classify/config.yaml",
            "--data.init_args.path",
            ds_path,
            "--ckpt_path",
            "classifier.ckpt",
        ])

    # Collect the vessel detections into CSV (only the ones where classifier thinks it
    # is a correct detection).
    print("pipeline: write vessel CSV")
    vessel_csv_fname = os.path.join(out_dir, "vessel.csv")
    subprocess.check_call([
        "python",
        "-m",
        "detector_mmrotate.gfcd_collate",
        "--ds_dir",
        ds_path,
        "--out_fname",
        vessel_csv_fname,
    ])

    # Create vessel crops and big visualization images.
    print("pipeline: generate visualizations")
    subprocess.check_call([
        "python",
        "-m",
        "detector_mmrotate.gfcd_vis",
        "--raw_dir",
        image_dir,
        "--vessel_csv",
        vessel_csv_fname,
        "--out_dir",
        os.path.join(out_dir, "vis"),
        "--crop_dir",
        os.path.join(out_dir, "crops"),
    ])

    if worldcover_dir is None:
        return
    if not os.path.exists(worldcover_dir):
        print(f"warning: ESA WorldCover data not found at {worldcover_dir}")
        return

    # Add info from ESA WorldCover.
    print("pipeline: get water/land attributes")
    worldcover_ds_path = os.path.join(out_dir, "worldcover")
    subprocess.check_call([
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "gfcd_classifier",
        "python",
        "water_land/create_windows.py",
        "--ds_path",
        worldcover_ds_path,
        "--image_dir",
        image_dir,
    ])
    # Write config that gets WorldCover GeoTIFFs, but need to fill in the source
    # directory placeholder for the LocalFiles data source.
    with open("water_land/config.json") as f:
        worldcover_ds_config = json.load(f)
    worldcover_ds_config["layers"]["worldcover"]["data_source"]["src_dir"] = worldcover_dir
    with open(os.path.join(worldcover_ds_path, "config.json"), "w") as f:
        json.dump(worldcover_ds_config, f)

    for action in ["prepare", "ingest", "materialize"]:
        print(action)
        subprocess.check_call([
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "gfcd_classifier",
            "rslearn",
            "dataset",
            action,
            "--root",
            worldcover_ds_path,
        ])
    subprocess.check_call([
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "gfcd_classifier",
        "python",
        "water_land/image_land_water_area.py",
        "--csv_dir",
        os.path.dirname(image_csv_fname),
        "--image_dir",
        image_dir,
        "--ds_path",
        worldcover_ds_path,
        "--out_fname",
        os.path.join(out_dir, "image.csv"),
    ])
    subprocess.check_call([
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "gfcd_classifier",
        "python",
        "water_land/vessel_is_land_or_water.py",
        "--csv_fname",
        vessel_csv_fname,
        "--out_fname",
        os.path.join(out_dir, "vessel_with_is_in_water.csv"),
        "--ds_path",
        worldcover_ds_path,
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prediction pipeline for detecting vessels in Maxar images",
    )
    parser.add_argument(
        "--scene_fname",
        type=str,
        help="Filename of 8-bit 3-band RGB Maxar visual-ready GeoTIFF",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory (must not exist yet)",
        required=True,
    )
    parser.add_argument(
        "--worldcover_dir",
        type=str,
        help="Directory containing ESA WorldCover map",
        default=None,
    )
    args = parser.parse_args()
    main(args.scene_fname, args.out_dir, args.worldcover_dir)
