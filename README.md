# Sentinel-2 Forest Browning Monitoring

Code for **Country-wide, high-resolution monitoring of forest browning with Sentinel-2**

Authors: Samantha Biegel, David Brüggemann, Francesco Grossi, Michele Volpi, Konrad Schindler, Benjamin Stocker


## Repository contents

This repository contains:
- the `forest_browning` Python package (`src/forest_browning`),
    - `src/forest_browning/train.py`: model training
    - `src/forest_browning/inference.py`: inference/anomaly outputs
    - `src/forest_browning/dataset.py`: dataset utilities
    - `src/forest_browning/mlp.py`: model architecture
    - `src/forest_browning/config.py`: constants
    - `src/forest_browning/data_processing/`: dataset creation and feature engineering
- analysis notebook (`notebooks/`),
    - `notebooks/plot_results.ipynb`: generates figures in `figs/`
- event polygons and summary CSVs (`data/`),
- model checkpoints (`checkpoints/`).

---

## Installation

```sh
git clone git@github.com:SamanthaBiegel/s2-forest-browning-monitoring.git
cd s2-forest-browning-monitoring

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install -r requirements.txt -e . --group dev

pre-commit install
pre-commit run --all-files
```

---

## Dataset creation

The processed dataset used by this project is created in the `data_processing/` folder.

### Run processing pipeline in order

```sh
python forest_browning/data_processing/1_extract_swisstopo_dataset.py
python forest_browning/data_processing/2_transpose_swisstopo_dataset.py
python forest_browning/data_processing/3_add_dates.py
python forest_browning/data_processing/4_load_dem_2m.py
python forest_browning/data_processing/5_dem_2m_zarr_to_geotiff.py
python forest_browning/data_processing/6_create_dem_features.py
python forest_browning/data_processing/7_add_vegetation_height.py
python forest_browning/data_processing/8_add_tree_species.py
python forest_browning/data_processing/9_add_habitat.py
python forest_browning/data_processing/10_add_missingness.py
python forest_browning/data_processing/11_add_forest_mix_rate.py
python forest_browning/data_processing/12_merge_features.py
```

### Output dataset

The final output dataset is a Zarr dataset with two versions:
- `ndvi_dataset_temporal.zarr`: includes the full NDVI time series with chunking along the temporal dimension (shape: `(num_forest_pixels, num_timesteps)`) as well as the feature arrays
- `ndvi_dataset_spatial.zarr`: includes the full NDVI time series with chunking along the spatial dimension (shape: `(num_timesteps, num_forest_pixels)`)

A forest mask is also generated as a NumPy array (`forest_mask.npy`), which is used to map between the 1D dataset and the original 2D spatial layout.

Adjust the paths in `src/forest_browning/config.py` to point to your local output locations for these datasets.

---

## Running this project

From `s2-forest-browning-monitoring`:
```sh
source .venv/bin/activate
```

First pre-shuffle the training dataset:

```sh
python -m forest_browning.shuffle_train_data --input_zarr /path/to/ndvi_dataset_temporal.zarr --output_zarr /path/to/ndvi_dataset_filtered_shuffled.zarr
```

Then run training to generate vegetation cycle parameters:

```sh
python -m forest_browning.train --data_path /path/to/ndvi_dataset_filtered_shuffled.zarr --output_dir /path/to/checkpoints
```

Finally, run inference to generate anomaly scores:

```sh
python -m forest_browning.inference --model_checkpoint checkpoints/encoder.pt --data_path /path/to/ndvi_dataset_temporal.zarr --output_dir /path/to/ndvi_dataset_temporal.zarr
```

To obtain anomaly scores in the spatial format for fast per-day retrieval, run:

```sh
python -m forest_browning.rechunk_output.py --source_zarr /path/to/ndvi_dataset_temporal.zarr --target_zarr /path/to/ndvi_dataset_spatial.zarr
```

For figure reproduction, run:

- `notebooks/plot_results.ipynb`

Figures are saved to `figs/`.

---

## Project structure

```text
s2-forest-browning-monitoring/
├── src/
│   └── forest_browning/
├── notebooks/
├── data/
│   └── event_polygons/
├── checkpoints/
├── figs/
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```
