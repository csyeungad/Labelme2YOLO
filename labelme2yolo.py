import random
import os
import shutil
import cv2
import yaml
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import load_class_ids, mkdir
from tqdm import tqdm

"""Struture
root
    images
        train
            img_1
            img_2
        val
        test
    labels
        train
            img_1.txt
            img_2.txt
        val
        test
    vis
        train
        val
        test
"""

class Labelme2YOLO():
    """ 
    1. To convert label data from Labelme format to YOLO format.
    2. Split train/val/test data
    3. Visualization labeled data
    """
    def __init__(self, data_root: str, labelme_lbls: dict, imgs_shape_wh: dict, pass_imgs: list, split_ratio: list, single_cls_name: str = None, include_pass_imgs: bool = False):
        """
        Args:
            data_root (str): root of the dataset where
            labelme_lbls (dict): labels extracted from labelme format
            imgs_shape_wh (dict): image w h
            pass_imgs (list): image with pass as flag from labelme format
            split_ratio (list): train/val/test ratios
            class_ids (dict): class_id for each class
            single_cls_name (str): the name of combined classes, default None to retain all multiple classes
        """
        self.data_root = data_root
        self.save_dir = os.path.join(data_root + '_yolo_dataset')
        if single_cls_name:
            self.save_dir = self.save_dir + f'_single_cls_{single_cls_name}'
        self.include_pass_imgs : bool = include_pass_imgs
        if include_pass_imgs:
            self.save_dir = self.save_dir + '_include_pass'
        mkdir(self.save_dir)
        self.labelme_lbls: dict = labelme_lbls
        '''
        labelme_lbls = {
            label_img_path_1: [[label, xmin, ymin, xmax, ymax],...]
            label_img_path_2: [[label, xmin, ymin, xmax, ymax],...]
        }
        '''
        self.pass_imgs: list = pass_imgs
        self.class_ids = dict[str, int]
        self.single_cls_name : bool = single_cls_name
        if single_cls_name:
            self.class_ids = {f"{self.single_cls_name}": 0}
        else:
            class_ids_json = os.path.join(self.data_root, f'lbls_class_ids_{os.path.basename(self.data_root)}.json')
            if not os.path.exists(class_ids_json):
                raise f"lbls_class_ids_{os.path.basename(self.data_root)}.json does not exist!"
            self.class_ids : dict[str, int] = load_class_ids(class_ids_json)

        self.id_mapping = { id: cls for cls, id in self.class_ids.items()}
        self.split_ratio : list = split_ratio
        self.imgs_shape_wh: dict = imgs_shape_wh #k: img_path v: w,h
        self.yolo_lbls = {} 
        '''
        yolo_lbls = {
            label_img_path_1: [[cls_id, xc, yc, w, h],...]
            label_img_path_2: [[cls_id, xc, yc, w, h],...]
        } 
        '''
        self._convert_lbls()
        if include_pass_imgs:
            print(f"Include_pass_img: add {len(self.pass_imgs)} pass imgs")
            self.yolo_lbls.update({ img_path: [] for img_path in self.pass_imgs})
        self.split_data: dict = {}
        '''
        split_data = {
            train: {
                label_img_path_1 : [[cls_id, xc, yc, w, h],...]
            },
            val: {
            }
            test:{
            }
        }
        '''
        self._split_train_val_test(random_seed=0)
        self._generate_dataset()
        self._generate_yaml()
        self._show_cls_distribution()
        self._visualize_all_imgs()
    
    def __convert_xyxy_to_normalised_xywh(self, lbl: list, img_w: int, img_h: int) -> list:
        """ Convert [cls,x,y,x,y] to normalized [id,xc,yc,w,h]"""
        cls, x_min, y_min, x_max, y_max = lbl
        
        # Calculate center coordinates
        xc = (x_min + x_max) / 2 / img_w  # Normalize to [0, 1]
        yc = (y_min + y_max) / 2 / img_h  # Normalize to [0, 1]
        
        # Calculate width and height
        w = (x_max - x_min) / img_w  # Normalize to [0, 1]
        h = (y_max - y_min) / img_h  # Normalize to [0, 1]

        if self.single_cls_name:
            return [ int(0), xc, yc, w, h]
        
        return [self.class_ids[cls], xc, yc, w, h]

    def _convert_lbls(self):
        """ Convert Labelme lbl format to YOLO lbl format """
        yolo_lbls = {}
        for img_path, lbls in tqdm(self.labelme_lbls.items(), desc= 'Converting labels to YOLO format'):
            img_w, img_h = self.imgs_shape_wh[img_path]
            yolo_lbls[img_path] = [self.__convert_xyxy_to_normalised_xywh(lbl, img_w, img_h) for lbl in lbls]
            
        self.yolo_lbls = yolo_lbls

    def _split_train_val_test(self, random_seed:int = 0) -> list:
        random.seed(random_seed)
        ratios = self.split_ratio

        img_path_list = list(self.yolo_lbls.keys())
        random.shuffle(img_path_list)

        sizes = [int(len(img_path_list) * (ratio / sum(ratios))) for ratio in ratios] #convert ratio to number of images for each set
        sizes[-1] += len(img_path_list) - sum(sizes)
        
        # Slice the data into parts
        split_data = {}
        start = 0
        for set_type, size in zip(['train', 'val', 'test'],sizes):
            end = start + size
            data = { img: self.yolo_lbls[img] for img in img_path_list[start: end]}
            split_data[set_type] = data
            start = end
        
        self.split_data = split_data

    def _generate_dataset(self):
        """ Generate dataset for train/val/test """
        split_data = self.split_data
        save_dir = os.path.join(self.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        for set_type, data in split_data.items():
            
            save_img_dir = os.path.join(save_dir,'images', set_type)
            save_lbl_dir = os.path.join(save_dir, 'labels',set_type)
            os.makedirs(save_img_dir, exist_ok=True)
            os.makedirs(save_lbl_dir, exist_ok=True)

            for img_path, lbls in tqdm(data.items(), desc = f'[Generate_dataset] ({set_type})'):
                try:
                    txt_path = os.path.join(save_lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
                    with open(txt_path, 'w', newline= '\n') as f:
                        for lbl in lbls:
                            formatted_data = [str(lbl[0])] + [f"{x:.6g}" for x in lbl[1:]]
                            f.write(' '.join(formatted_data) + '\n')
                    shutil.copy2(img_path, os.path.join(save_img_dir, os.path.basename(img_path)))
                except Exception as e:
                    print(e)

        print(f"[Generate_dataset]: Successfully generated dataset in path: {save_dir}")

    def _generate_yaml(self):
        """ Save dataset yaml required in YOLO training """
        '''
        path: ../datasets/xxx
        train: images/train
        val: images/val
        test: images/test

        names:
            0: cls_0
            1: cls_1

        '''
        names = {id:cls for cls, id in self.class_ids.items()}
        content = {
            'path': f'../datasets/{os.path.basename(self.save_dir)}',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': names
        }
        save_path = os.path.join(self.save_dir, f'dataset.yaml')
        # Write the distribution to a YAML file
        with open(save_path, 'w') as file:
            yaml.dump(content, file)
        print(f"[Generate_yaml]: Successfully generated yaml in path: {save_path}")

    def _get_cls_distribution(self) -> dict:
        from collections import Counter
        split_data_distribution = {}
        for set_type, data in self.split_data.items():
            _distribution = Counter(lbl[0] for lbls in data.values() for lbl in lbls ) + Counter('PASS'  for lbls in data.values() if len(lbls) == 0)
            split_data_distribution[set_type] = dict(_distribution)
        return split_data_distribution
        
    def _show_cls_distribution(self):
        split_data_distribution: dict = self._get_cls_distribution()

        # Prepare for plotting
        num_colors = len(self.class_ids)
        colors = plt.cm.viridis(np.linspace(0, 3, num_colors))

        # Set up the figure and axes
        plt.figure(figsize=(12, 6))
        
        # Get the x locations for the labels
        labels = list(self.id_mapping)  # Assuming class_ids contains the labels
        x = np.arange(len(labels))  # The label locations

        # Width of the bars
        width = 0.2  # Adjust the width as needed

        for i, set_type in enumerate(['train', 'val', 'test']):
            set_dist = split_data_distribution[set_type]
            heights = [set_dist.get(label, 0) for label in labels]  # Get heights for each label, default to 0
            
            # Create bars for each set type
            plt.bar(x + i * width, heights, width, label=set_type, color=colors[i % num_colors])

            # Annotate the count for each bar
            fontsize = 8
            for j, height in enumerate(heights):
                plt.text(x[j] + i * width, height, height, ha='center', va='bottom', fontsize=fontsize)

        for i, set_type in enumerate(['train', 'val', 'test']):
            set_dist = split_data_distribution[set_type]
            lbls_count = sum([ v for k,v in set_dist.items() if k != 'PASS' ])
            print(f"\tTotal #{set_type} images: {len(self.split_data[set_type])} with {lbls_count} labels.")
            plt.text(0.7, 1.09 - i*0.03, f"Total #{set_type} images: {len(self.split_data[set_type])} with {lbls_count} labels.", 
                ha='left', va='baseline', fontsize=fontsize, transform=plt.gca().transAxes)
            
        # Add labels and title
        plt.ylabel('# Labels', fontsize=fontsize)
        plt.xlabel('Label Class', fontsize=fontsize)
        plt.title('Label Labels Distribution', fontsize=fontsize)
        plt.xticks(x + width, list(self.class_ids), fontsize=6)  # Set x-ticks to the center of the grouped bars
        plt.legend()  # Add a legend for the set types

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'Label_distribution_combined.png'))

        print(f"Label objects distribution saved in {os.path.join(self.save_dir, 'Label_distribution_combined.png')}")

    def __convert_normalised_xywh_to_xyxy(self, lbl, img_w, img_h):
        cls = self.id_mapping[lbl[0]]
        xc, yc, nw, nh = map(float, lbl[1:])
        x1 = int(xc * img_w - nw * img_w / 2)
        y1 = int(yc * img_h - nh * img_h / 2)
        x2 = int(xc * img_w + nw * img_w / 2)
        y2 = int(yc * img_h + nh * img_h / 2)
        return [cls, x1, y1, x2, y2]
    
    def __put_gt(self, img , lbl):
        """ Put [cls, x1, y1, x2, y2] lbl on img """
        #print(f"\txyxy:{[cls, x1, y1, x2, y2]}")
        cls, x1, y1, x2, y2 = lbl
        w, h = x2 - x1, y2 - y1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cls_position = (x1 + 5, y1 + 15)
        cv2.putText(img, cls, cls_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        size_position = (x1 + 5, y1 + 35)
        cv2.putText(img, f"{w} X {h}", size_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        area_position = (x1 + 5, y1 + 55)
        cv2.putText(img, f"{(w * h):.2f}", area_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        ratio_position = (x1 + 5, y1 + 75)
        cv2.putText(img, f"{(w / h):.2f}", ratio_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        return img
        
    def _visualize_all_imgs(self):
        
        for set_type, data in self.split_data.items():
            save_path = os.path.join(self.save_dir, 'vis', set_type)
            os.makedirs(save_path, exist_ok=True)

            for img_path, lbls in tqdm(data.items(), desc= f'Visualizing {set_type} image:'):
                if len(lbls) == 0:
                    shutil.copy2(img_path, os.path.join(save_path, os.path.basename(img_path)))
                    continue
                img = cv2.imread(img_path)
                lbls = self.yolo_lbls[img_path]
                img_h, img_w, _ = img.shape  # (H,W,D)

                for lbl in lbls:
                    self.__put_gt(img, self.__convert_normalised_xywh_to_xyxy(lbl, img_w, img_h))

                # Convert BGR to RGB for plotting
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)),img_rgb)



