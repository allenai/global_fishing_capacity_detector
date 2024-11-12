This helps get WorldCover images for two purposes.

1. Compute sq km of land and water in each Maxar scene.
2. Determine if each vessel detection is moored or on land, or in open water.

To use it, first set up rslearn dataset:

    mkdir /path/to/dataset
    cp water_land/config.json water_land/config.json
    python -m water_land.create_windows --ds_path /path/to/dataset/ --image_dir /path/to/images/
    rslearn dataset prepare --root /path/to/dataset/ --workers 32
    rslearn dataset ingest --root /path/to/dataset/ --workers 32 --no-use-initial-job --jobs-per-process 1
    rslearn dataset materialize --root /path/to/dataset/ --workers 32 --no-use-initial-job

Note: it depends on WorldCover images being present (see `config.json`).

Now assuming prediction has been run and there are one or a bunch of X_images.csv files
in some directory, then we can concatenate them and add land/water area:

    python -m water_land.image_land_water_area --csv_dir /path/to/csvs/ --image_dir /path/to/images/ --ds_path /path/to/dataset/ --out_fname /path/to/csvs/image_with_land_water_areas.csv
