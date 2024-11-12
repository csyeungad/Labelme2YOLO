# Labelme2YOLO

This is a handy tool for converting data annotated with labelme to YOLO model training labeling methods.
Currently supported mode: detection

## Labelme config

Follow below flags and labels when labeling with Labelme.

    FLAGS: ['PASS', 'REVIEW', 'REMOVE']
    LABELS: ['CLS_1', 'CLS_2', 'CLS_3', 'CLS_4', ...]

## Usage

Modify the **cfg.json** for the data path and directory for grouping and visulization of results.

cfg.json

    {
        "labeled_data_root" : directory with data labeled with Labelme 
        "output_dir" : directory for data grouping of differents flags, labeled data, labeled data visualization and distributions
        "split_ratio" : ratio of train\val\test test e.g. [0.7, 0.15,0.15]
        "single_cls_name" : (str) the name of combined classes, default None to retain all multiple classes
        "include_pass_img" : (bool) To include pass images into datasets, default false
    }

## Output dataset converted

Data splited into train/val/test in yolo format is saved in **labeled_data_root_yolo_dataset**

Below shows the snapshot of the folder structure:

    |---labeled_data_root_yolo_dataset
        |---images                                  : raw images for training
        |   |---train
        |   |   |---img_1.jpg
        |   |   |---img_2.jpg
        |   |---val
        |   |---test
        |
        |---labels                                  : labeling file for each raw images
        |   |---train
        |   |   |---image_1.txt
        |   |   |---image_2.txt
        |   |---val
        |   |---test
        |---vis                                     : visuaization of labeled images
        |   |---train
        |   |   |---img_1.jpg
        |   |   |---img_2.jpg
        |   |---val
        |   |---test
        |
        |---dataset.yaml                            : dataset config for yolo training
        |---Label_distribution_combined.png         : class distribution for each set
