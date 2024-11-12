import glob
import shutil
import subprocess
import sys

from prediction_pipeline import main as predict_pipeline

if __name__ == "__main__":
    action = sys.argv[1]

    if action == "predict":
        scene_fname = sys.argv[2]
        out_dir = sys.argv[3]
        predict_pipeline(scene_fname, out_dir, "/worldcover")

    if action == "train_detector":
        ds_dir = sys.argv[2]
        out_fname = sys.argv[3]
        subprocess.check_call([
            "python",
            "-um",
            "detector_mmrotate.gfcd_train",
            ds_dir,
        ])
        # After training, we copy latest.pth to detector.pth.
        # It is symlink but we want to copy the file.
        shutil.copyfile(
            "gfcd_detector_exp_dir/latest.pth",
            out_fname,
            follow_symlinks=True,
        )

    if action == "train_classifier":
        ds_dir = sys.argv[2]
        out_fname = sys.argv[3]
        subprocess.check_call([
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "gfcd_classifier",
            "python",
            "-um",
            "rslearn.main",
            "model",
            "fit",
            "--config",
            "classify/config.yaml",
            "--data.init_args.path",
            ds_dir,
        ])
        # Identify best checkpoint, it's the one with "epoch" in the name.
        fnames = glob.glob("lightning_logs/*/checkpoints/epoch*ckpt")
        assert len(fnames) == 1
        shutil.copyfile(fnames[0], out_fname)
