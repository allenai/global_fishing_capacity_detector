# Global Fishing Capacity - Vessel Detection Model

This is a computer vision pipeline for detecting vessels in high-resolution Maxar
satellite images.

The pipeline consists of two models:

1.  An object detector that detects rotated bounding boxes corresponding to vessels.
    It consists of a Swin-Transformer-Tiny backbone coupled with a RoI Trans head that
    predicts the rotated bounding boxes. To process a large scene, the detector is
    applied with a sliding window approach, where it processes 1024x1024 patches (with
    overlap between consecutive patches) at a time, and the detections are merged
    (while removing duplicates) across patches afterward.

2.  An image classification model that processes the initial set of vessels predicted by
    the object detector to improve precision. It inputs a 256x256 patch around each
    initial vessel prediction, and outputs a classification score. The model is a
    ViT-Large fine-tuned from the OpenAI CLIP pre-trained model. The final output is a
    subset of the initial detections, with likely false positives removed.

The following components are included:

-   `detector_preprocess`: scripts to convert the Maxar scenes and annotations to 1024x1024
    crops suitable for training the detector.
-   `detector_mmrotate`: adaptation of [mmrotate](https://github.com/open-mmlab/mmrotate).
    This folder includes the training and inference code for the object detector.
-   `classify`: code built on top of [rslearn](https://github.com/allenai/rslearn) for
    training and applying the classification model.
-   `water_land`: script to use ESA WorldCover (land cover map) to add the `land_sq_km`
    and `water_sq_km` attributes for images, and the `is_in_water` attribute for
    vessels.

The code and model checkpoints are released under Apache License 2.0.


## Inference

Clone the repository and download the object detector and classifier weights:

    git clone https://github.com/allenai/global_fishing_capacity_detector.git
    cd global_fishing_capacity_detector/

To run inference, you will need a three-band RGB 8-bit Maxar scene
(https://ard.maxar.com/docs/ard-order-delivery/image-files/visual-image/).
These images should be from Worldview 2, 3, or 4. Below we assume the file is called
`scene-visual.tif` in a directory `/path/to/image/`.

As an example, you can use Maxar's image of Beirut before the August 2020 explosion
from their open data program (cropped and warped to UTM projection):

    # See https://www.maxar.com/open-data/beirut-explosion
    wget https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/global_fishing_capacity_project/maxar-104001005BD06800-beirut-20240609-crop.tif -O /path/to/image/scene-visual.tif


### Usage with Docker Container (recommended)

Build the Docker container. The weights are stored in the Docker image, so this step
must be done after downloading the weights.

    docker build -t gfcd .

Run inference. The directory containing the image file (e.g. `/path/to/image/`) must be
mounted in the Docker container.

    docker run --gpus all --shm-size 16G --rm --mount type=bind,source=/path/to/image,target=/vol gfcd predict /vol/scene-visual.tif /vol/scene-visual-output/

Here, we mount the directory containing the image file to `/vol`, and then pass
`/vol/scene-visual.tif` as the path to the image file from inside the Docker container.

The outputs will then be in `/path/to/image/scene-visual-output/`.

To get the land_sq_km/water_sq_km/is_in_water attribute, download ESA WorldCover and
mount it in the Docker container:

    wget https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/global_fishing_capacity_project/esa_worldcover_10m_2021.tar
    tar xvf esa_worldcover_10m_2021.tar --directory /path/to/worldcover
    rm esa_worldcover_10m_2021.tar
    docker run --gpus all --shm-size 16G --rm --mount type=bind,source=/path/to/image,target=/vol --mount type=bind,source=/path/to/worldcover,target=/worldcover gfcd predict /vol/scene-visual.tif /vol/scene-visual-output/

On GCP using n1-highmem-8 with 1x NVIDIA V100 GPU, inference on the Beirut scene
(maxar-104001005BD06800-beirut-20240609-crop.tif, 17571x13252) takes 3.5 minutes. Using
ESA WorldCover takes another 20 minutes due to the need for scanning the WorldCover
GeoTIFFs to identify which ones are relevant.


### Non-Docker Usage

Download the detector and classifier weights:

    wget https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/global_fishing_capacity_project/detector.pth
    wget https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/global_fishing_capacity_project/classifier.ckpt

Setup object detector environment:

    conda create --name gfcd_detector python=3.10
    conda activate gfcd_detector
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install -U openmim numpy\<2.0
    mim install mmcv-full==1.7.2
    mim install mmdet\<3.0.0
    pip install -r requirements.txt
    pip install -v -e detector_mmrotate/
    conda deactivate

Setup classifier environment:

    conda create --name gfcd_classifier python=3.12
    conda activate gfcd_classifier
    pip install git+https://github.com/allenai/rslearn.git#egg=rslearn[extra]
    pip install -r requirements.txt
    conda deactivate

Run the pipeline (it will use the "gfcd_classifier" environment for certain steps):

    conda activate gfcd_detector
    python prediction_pipeline.py --scene /path/to/image/scene-visual.tif --out_dir /path/to/image/scene-visual-output/ --worldcover_dir /path/to/worldcover

The `--worldcover_dir` is optional.


### Output Format

The output directory will contain the following files/folders:

-   `image.csv`: CSV with one row describing the image.
-   `vessel_with_is_in_water.csv`: CSV with vessel detections.
-   `vis/`: folder containing a visualization of the input image with vessel detections
    overlayed. The visualization is only created if there is at least one vessel in the
    image.
-   `crops/`: crops of the input image centered at each detected vessel.

The `image.csv` contains the following columns:

-   `fname`: the image filename.
-   `pixel_x1`, `pixel_y1`, ..., `pixel_y4`: the pixel coordinates of the corners of
    the image.
-   `wgs84_x1`, `wgs84_y1`, ..., `wgs84_y4`: the longitude and latitude coordinates of
    the corners of the image.
-   `timestamp`: unused.
-   `water_sq_km`: the area of water in this image in square km.
-   `land_sq_km`: the area of land in this image in square km.

If running without ESA WorldCover data, then `orig_images.csv` is created instead
without the `water_sq_km` and `land_sq_km` columns.

The `vessel_with_is_in_water.csv` contains the following columns:

-   `fname`: the image filename.
-   `idx`: the unique vessel detection index.
-   `pixel_x1`, `pixel_y1`, ..., `pixel_y4`: the pixel coordinates of the corners of
    the vessel (relative to the topleft of the image).
-   `wgs84_x1`, `wgs84_y1`, ..., `wgs84_y4`: the longitude and latitude coordinates of
    the corners of the vessel.
-   `score`: the confidence score from the object detection model associated with this
    detection.
- `is_in_water`: "yes" if this vessel is in open water and "no" if it is on land or
    moored.

If running without ESA WorldCover data, then `vessel.csv` is created instead without
the `is_in_water` column.


## Limitations

Precision of the end-to-end pipeline (detector + classifier) is estimated at 93% and
recall at 74%. For comparison, inter-annotator agreement is approximately 95%. The
detector confidence threshold can be adjusted (e.g. change `--score_threshold` default
in `detector_mmrotate/gfcd_predict.py`) to change the precision-recall balance. The
models are evaluated on Maxar scenes from 2009 to 2021 in the Asia-Pacific region, and
performance elsewhere may be lower (especially performance of the classifier).


## Training

The training dataset includes proprietary Maxar images and thus cannot be released. The
training steps are still documented below but cannot be followed without the training
dataset.

Choose a local directory to store the datasets:

    export GFCD_DATA_DIR=/path/to/data/


### Usage with Docker Container

Download and extract the pre-processed object detector training dataset to
$GFCD_DATA_DIR/mmrotate and download and extract the classifier training dataset to
$GFCD_DATA_DIR/classifier_dataset/.

Train the object detector:

    # Download pre-trained model from which model will be fine-tuned.
    wget https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90/roi_trans_swin_tiny_fpn_1x_dota_le90-ddeee9ae.pth
    # Build (or rebuild) the Docker container with the pre-trained model weights in the container.
    docker build -t gfcd .
    # Run training.
    docker run --gpus all --shm-size 16G --rm --mount type=bind,source=$GFCD_DATA_DIR/mmrotate,target=/mmrotate --mount type=bind,source=.,target=/output gfcd train_detector /mmrotate /output/detector.pth

This will mount the training dataset to `/mmrotate` in the Docker container, while the
trained weights will be written to `detector.pth` in the current directory (mounted at
`/output`).

Train the classifier:

    docker run --gpus all --shm-size 16G --rm --mount type=bind,source=$GFCD_DATA_DIR/classifier_dataset,target=/classifier_dataset --mount type=bind,source=.,target=/output gfcd train_classifier /classifier_dataset /output/classifier.ckpt

Similarly, the training dataset is mounted in the container and in the end the
`classifier.ckpt` will be written in the current directory.

Now these weights can be used for inference.

The steps below document non-Docker usage.


### Object Detector Dataset Pre-processing

The pre-processing converts the large image scenes and annotations into 1024x1024
patches suitable for training the object detector, and also formats them to be
compatible with mmrotate.

Choose a local directory to store the datasets:

    export GFCD_DATA_DIR=/path/to/data/

First, download the scenes and convert into a flat directory structure:

    conda activate gfcd_detector
    python detector_preprocess/download_cloud_dataset.py --src_dir /path/to/raw/data/ --out_dir $GFCD_DATA_DIR/raw/

Replace `/path/to/raw/data/` with the path to the raw training dataset. If it is on GCS
or S3, you can use a prefix like `gcs://bucket-name/path/` or `s3://bucket-name/path/`.

Determine the train and val splits:

    python detector_preprocess/split_cloud_dataset.py --in_dir $GFCD_DATA_DIR/raw/ --split_fname $GFCD_DATA_DIR/split.json --mode nonempty

Crop it into 1024x1024 patches:

    python detector_preprocess/crop_cloud_dataset.py --in_dir $GFCD_DATA_DIR/raw/ --out_dir $GFCD_DATA_DIR/patches/ --split_fname $GFCD_DATA_DIR/split.json

Convert to mmrotate format:

    python detector_preprocess/convert_to_mmrotate.py --src $GFCD_DATA_DIR/patches/train/ --dst $GFCD_DATA_DIR/mmrotate/train/
    python detector_preprocess/convert_to_mmrotate.py --src $GFCD_DATA_DIR/patches/val/ --dst $GFCD_DATA_DIR/mmrotate/val/


### Object Detector Training

Ensure that the pre-processed object detector dataset is at $GFCD_DATA_DIR/mmrotate,
either by following the pre-processing steps above or by downloading and extracting
the pre-processed dataset directly.

Train the object detector:

    wget https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90/roi_trans_swin_tiny_fpn_1x_dota_le90-ddeee9ae.pth
    python -m detector_mmrotate.gfcd_train $GFCD_DATA_DIR/mmrotate

Evaluate the model:

    python -m detector_mmrotate.gfcd_infer --command score --data_root $GFCD_DATA_DIR/mmrotate

And copy the latest weights to `detector.pth`:

    cp gfcd_detector_exp_dir/latest.pth detector.pth


### Classifier Training

The steps below assume that the classifier training data has been downloaded and
extracted to $GFCD_DATA_DIR/classifier_dataset/.

Train the classifier:

    conda activate gfcd_classifier
    rslearn model fit --config classify/config.yaml --data.init_args.path $GFCD_DATA_DIR/classifier_dataset/

Identify the best checkpoint:

    cp lightning_logs/version_0/checkpoints/epoch*ckpt classifier.ckpt

You may need to change the `version_0` to the correct subfolder in `lightning_logs` if you ran `model fit` multiple times.

Evaluate it:

    rslearn model test --config classify/config.yaml --data.init_args.path $GFCD_DATA_DIR/classifier_dataset/ --ckpt_path classifier.ckpt
