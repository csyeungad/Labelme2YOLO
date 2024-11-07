import os
import json
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import SUPPORT_FORMAT, save_class_ids, mkdir

class Labelme:
    """Load annotated data in labelme format. label distribution saves to data root"""
    def __init__(self, data_root: str) -> None:
        self.data_root = data_root
        self.imgs_shape_wh = {} #stored image w and h
        self.class_ids = {}
        self.labeled_imgs = {}
        self.pass_imgs = []
        self.remove_imgs = []
        self.review_imgs = []
        self.label_distribution = {}
        self._load_detection_anno()
        save_class_ids(self.data_root, self.class_ids)
        self.id_mapping = { id: cls for cls, id in self.class_ids.items()}
        self._get_distribution()
        self.box_area_dist = {}
        self.box_ratio_dist = {}
        self._compute_boxes_distribution()
        self.show_distribution()
        
    def _load_detection_anno(self) -> None:
        """Load object detection rectangle labels and class_ids\n
        labeled_imgs = {\n
            label_img_path_1: [[cls_1, xmin, ymin, xmax, ymax],...]\n
            label_img_path_2: [[cls_2, xmin, ymin, xmax, ymax],...]\n
        }\n

        pass_imgs = [pass_img_path_1, pass_img_path_1 ,....]\n
        remove_imgs = [remove_img_path_1, remove_img_path_1 ,....]\n
        
        """
        class_list = set()
        for cur_root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith(SUPPORT_FORMAT):
                    file_path = os.path.join(cur_root, file)
                    json_path = file_path.replace(os.path.splitext(file)[1], ".json")
                    if os.path.exists(json_path):
                        lbls = []
                        with open(json_path,'r') as f:
                                data = json.load(f)
                        if "REMOVE" in data['flags'] and data['flags']['REMOVE']:
                            self.remove_imgs.append(file_path)
                            continue
                        if 'REVIEW' in data['flags'] and data['flags']['REVIEW']:
                            self.review_imgs.append(file_path)
                            continue
                        if 'PASS' in data['flags'] and data['flags']['PASS']:
                            self.pass_imgs.append(file_path)
                            continue       
                        for item in data["shapes"]:
                            if item["shape_type"].lower() == "rectangle":
                                xmin = int(min(item["points"][0][0], item["points"][1][0]))
                                ymin = int(min(item["points"][0][1], item["points"][1][1]))
                                xmax = int(max(item["points"][0][0], item["points"][1][0]))
                                ymax = int(max(item["points"][0][1], item["points"][1][1]))
                                lbls.append([item["label"], xmin, ymin, xmax, ymax])
                                class_list.add(item["label"])
                                
                        if len(lbls) >0:
                            self.labeled_imgs[file_path] = lbls

                        self.imgs_shape_wh[file_path] = [data["imageWidth"],data['imageHeight']]       
        self.class_ids = { cls: int(i) for i, cls in enumerate(sorted(class_list))}

    def _save_class_ids(self):
        with open(os.path.join(self.data_root, f'yolo_class_ids_{os.path.basename(self.data_root)}.json'), 'w') as f:
            json.dump(self.class_ids, f)
        print(f"Saved yolo_class_ids_{os.path.basename(self.data_root)}.json'")
    
    def _get_distribution(self):
        from collections import Counter
        """Get distribution of label: count of label."""
        # Use Counter to count occurrences of class labels
        self.label_distribution = dict(Counter(lbl[0] for lbls in self.labeled_imgs.values() for lbl in lbls))

    def show_distribution(self):
        
        num_colors = len(self.class_ids.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.label_distribution.keys(), self.label_distribution.values(), data = self.label_distribution, color = colors)

        # Annotate the count for each bar
        fontsize= 8
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, yval, 
                    ha='center', va='bottom', fontsize=fontsize)

        plt.ylabel('#lables', fontsize=fontsize)
        plt.xlabel('label class', fontsize=fontsize)
        plt.title(f"label lables distribution: ", fontsize=fontsize)
        plt.xticks(fontsize=6)  # Set font size for x-axis tick labels
        plt.savefig(os.path.join(self.data_root, 'Label_distribution.png'))
        
        print(f"Total #label images: {len(self.labeled_imgs)} with {sum(list(self.label_distribution.values()))} lbls.")
        print(f"Total #pass images: {len(self.pass_imgs)}")
        print(f"Total #review images: {len(self.review_imgs)}")
        print(f"Total #remove images: {len(self.remove_imgs)}")
        print(f"Total #images: {len(self.labeled_imgs) +len(self.pass_imgs) + len(self.review_imgs) + len(self.remove_imgs) }")
        print(f"label objects distribution in {os.path.join(self.data_root, 'Label_distribution.png')}")

    def _compute_boxes_distribution(self):
        box_area_dist = { cls: [] for cls in self.class_ids.keys()}
        box_ratio_dist = { cls: [] for cls in self.class_ids.keys()}

        for img_path in self.labeled_imgs.keys():
            lbls = self.labeled_imgs[img_path]
            for lbl in lbls:
                class_lbl = lbl[0]
                w , h = int(lbl[3] - lbl[1]) , int(lbl[4] - lbl[2])
                box_area_dist[class_lbl].append( w*h)
                box_ratio_dist[class_lbl].append( round(w/h, 3))
        self.box_area_dist = box_area_dist
        self.box_ratio_dist = box_ratio_dist

class LabelmeVisualizer:
    def __init__(self, save_dir: str, box_area_dist: dict, box_ratio_dist: dict):
        self.save_dir = os.path.join(save_dir, 'LabelmeVisualization')
        mkdir(self.save_dir)
        self.box_area_dist = box_area_dist
        self.box_ratio_dist = box_ratio_dist

    @staticmethod
    def _categorize_metric(metric, num_bins=5, bin_range=None):
        """
        Categorizes a metric into different bins.

        Args:
            metric (list): List of metric values.
            num_bins (int): Number of bins to categorize the metric.
            bin_range (tuple): Optional range for the bins.

        Returns:
            tuple: (bin_counts, bin_edges) where bin_counts is a list of counts in each bin
                   and bin_edges is the edges of the bins.
        """
        metric = np.array(metric)
        if bin_range:
            bin_edges = np.linspace(bin_range[0], bin_range[1], num_bins + 1)
        else:
            bin_edges = np.linspace(np.min(metric), np.max(metric), num_bins + 1)

        # Digitize metric to get corresponding bin indices
        bin_indices = np.digitize(metric, bin_edges)
        # Count the number of metrics in each bin
        bin_counts = np.bincount(bin_indices, minlength=len(bin_edges))

        return bin_counts, bin_edges

    @staticmethod
    def _plot_histogram(save_dir, data, bin_edges, xlabel, title, xticks_step=None):
        """
        Plots a histogram of the given data and shows the count on each bar.

        Args:
            save_dir (str): Directory to save the plot.
            data (list): Data to plot.
            bin_edges (ndarray): Edges of the bins.
            xlabel (str): Label for the x-axis.
            title (str): Title of the plot.
            xticks_step (float): Step size for x-ticks.
        """
        plt.figure(figsize=(12, 6))
        counts, _, patches = plt.hist(data, bins=bin_edges, edgecolor='black', alpha=0.7)
        plt.xlabel(xlabel)

        if xticks_step:
            plt.xticks(np.arange(min(bin_edges), max(bin_edges) + 1, xticks_step))
        
        plt.ylabel('Count')
        plt.title(title)
        plt.grid(axis='y')

        # Annotate counts on the histogram bars
        for count, patch in zip(counts, patches):
            if count > 0:  # Only annotate if count is greater than zero
                height = patch.get_height()
                plt.text(patch.get_x() + patch.get_width() / 2, height, int(count), 
                         ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{title.replace(" ", "_")}.png')
        plt.savefig(save_path)
        #print(f"Histogram saved in {save_path}")
        plt.close() 

    def show_bbox_area_distribution(self, num_bins: int, bin_range: list, xticks_step: float, cls=None):
        """
        Visualizes the label bounding box area distribution for all or a specific class.

        Args:
            num_bins (int): Number of bins.
            bin_range (list): Range for the bins ([min, max]).
            xticks_step (float): Step size for x-ticks.
            cls (str or None): Specific class to visualize, or None for all classes.
        """
        if cls:
            areas = np.sqrt(self.box_area_dist.get(cls, []))
        else:
            areas = np.sqrt(np.concatenate(list(self.box_area_dist.values())))

        if len(areas) == 0:
            print("No data available for bounding box areas.")
            return

        bin_counts, bin_edges = self._categorize_metric(areas, num_bins, bin_range)
        self._plot_histogram(self.save_dir, areas, bin_edges, xlabel='Bounding Box sqrt(Area)', 
                             title=f'Distribution of Bbox sqrt(Area) {cls or "All Classes"}', 
                             xticks_step=xticks_step)

    def show_box_ratio_distribution(self, num_bins: int, bin_range: list, xticks_step: float, cls=None):
        """
        Visualizes the label bounding box ratio distribution for all or a specific class.

        Args:
            num_bins (int): Number of bins.
            bin_range (list): Range for the bins ([min, max]).
            xticks_step (float): Step size for x-ticks.
            cls (str or None): Specific class to visualize, or None for all classes.
        """
        if cls:
            ratios = self.box_ratio_dist.get(cls, [])
        else:
            ratios = np.concatenate(list(self.box_ratio_dist.values()))

        if len(ratios) == 0:
            print("No data available for bounding box ratios.")
            return

        bin_counts, bin_edges = self._categorize_metric(ratios, num_bins, bin_range)
        self._plot_histogram(self.save_dir, ratios, bin_edges, xlabel='Bounding Box Ratio', 
                             title=f'Distribution of Bbox Ratio {cls or "All Classes"}', 
                             xticks_step=xticks_step)