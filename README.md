# Labelme2YOLO

**Labelme2YOLO** is a convenient tool designed to convert datasets annotated with [Labelme](https://github.com/wkentaro/labelme) into the YOLO format, streamlining the dataset preparation process for YOLO model training.  

## Why use Labelme2YOLO?

- Simplifies converting Labelme JSON annotations into YOLO-compatible text files.
- Automatically splits your dataset into train, validation, and test sets by configurable ratios.
- Supports filtering and grouping images by annotation flags.
- Generates dataset visualization and class distribution plots.
- Produces a ready-to-use dataset directory structure compatible with YOLO training frameworks.

## Prerequisites

Before using this tool, please label your images using [Labelme](https://github.com/wkentaro/labelme), a powerful open-source polygonal annotation tool in Python.

Once your dataset is ready, Labelme2YOLO will help prepare it for training with YOLO models.

For training your YOLO models, we recommend using the comprehensive [YOLO Training Framework](https://github.com/csyeungad/YOLO-Training-Framework) which supports streamlined workflows for training, validation, testing, and deployment.

## Labelme Annotation Configurations

When annotating images with Labelme, use the following **FLAGS** and **LABELS** conventions to facilitate correct dataset processing:

- **FLAGS:** `['PASS', 'REVIEW', 'REMOVE']`  
  Use flags to indicate the status or quality of annotations/images.
  - **PASS**: Indicates that there is no object present in the image. Useful for improving negative Sampling, and reduce false positive.
  - **REVIEW**: Marks images that require further review or label refinement after conversion; these images will be grouped separately.
  - **REMOVE**: Marks images of poor quality or those that do not belong to the dataset and should be excluded.

- **LABELS:** `['CLS_1', 'CLS_2', 'CLS_3', 'CLS_4', ...]`  
  Define your class labels consistently and clearly to ensure proper conversion and training.

## Usage

Configure your dataset and conversion settings in the `cfg.json` file:

```json
{
    "labeled_data_root": "<path_to_labelme_annotated_data>",
    "output_dir": "<output_directory_for_converted_dataset>",
    "split_ratio": [0.7, 0.15, 0.15],
    "single_cls_name": null,
    "include_pass_img": false
}
```

- `labeled_data_root`: Directory containing Labelme-labeled data (JSON + images).
- `output_dir`: Directory where the YOLO formatted dataset and visualizations will be saved.
- `split_ratio`: Dataset split ratios for train, validation, and test sets. Must sum to 1.
- `single_cls_name`: (Optional) If set, merges all classes into a single class with this name.
- `include_pass_img`: Whether to include images flagged as "PASS" in the dataset (default `false`).

## Output Dataset Structure

After conversion, your dataset will be organized in YOLO format under `<output_dir>` (e.g., `labeled_data_root_yolo_dataset`):

```
labeled_data_root_yolo_dataset/
├── images/
│   ├── train/
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── img_1.txt
│   │   ├── img_2.txt
│   ├── val/
│   └── test/
├── vis/
│   ├── train/
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   ├── val/
│   └── test/
├── dataset.yaml
└── Label_distribution_combined.png
```

- `images/` contains the raw images split into train/val/test folders.
- `labels/` contains the YOLO-format label text files corresponding to each image.
- `vis/` contains images with visualized bounding boxes for verification.
- `dataset.yaml` is a dataset configuration file compatible with YOLO training frameworks.
- `Label_distribution_combined.png` shows the class distribution across your dataset splits.


## Recommended Next Steps for Model Training

Once your dataset is prepared, you can train YOLO models using the [YOLO Training Framework](https://github.com/csyeungad/YOLO-Training-Framework). This framework provides:

- End-to-end training, validation, and testing pipelines.
- Flexible YAML-based configuration for dataset paths, model parameters, and hyperparameters.
- Support for model export and deployment.

Check out the repository for detailed instructions on setting up your training environment and running experiments.

## Acknowledgements

- [Labelme](https://github.com/wkentaro/labelme) for the excellent annotation tool.
- [YOLO Training Framework](https://github.com/csyeungad/YOLO-Training-Framework) for model training and deployment support.
