from labelme import Labelme, LabelmeVisualizer
from labelme2yolo import Labelme2YOLO
from utils import utils, vis_helper


if __name__ == "__main__":

    cfg = utils.load_json('cfg.json')
    print(cfg)
    
    #Load annotation in the data root, save class_ids pair & data distribution
    #TODO: support loading data in any folder structure, now only support single layer 
    labelme_data = Labelme(data_root= cfg.labeled_data_root)

    #Visualise bounding box area & ratio distribution
    visualizer = LabelmeVisualizer(
        save_dir = cfg.labeled_data_root,
        box_area_dist= labelme_data.box_area_dist,
        box_ratio_dist= labelme_data.box_ratio_dist
    )

    #For all cls
    visualizer.show_bbox_area_distribution(num_bins= 100, bin_range= [0,1500], xticks_step= 100)
    visualizer.show_box_ratio_distribution(num_bins= 100, bin_range= [0,10], xticks_step= 0.5)
    #For individual cls
    for cls in labelme_data.class_ids.keys():
        visualizer.show_bbox_area_distribution(num_bins= 100, bin_range= [0,1500], xticks_step= 100, cls = cls)
        visualizer.show_box_ratio_distribution(num_bins= 100, bin_range= [0,10], xticks_step= 0.5, cls = cls)

    #Group defect / pass / remove / review images 
    utils.group_images(
        dest_path= cfg.grouping_dest,
        pass_imgs= labelme_data.pass_imgs,
        review_imgs= labelme_data.review_imgs,
        defect_labels= labelme_data.defect_labels,
        remove_imgs= labelme_data.remove_imgs
    )

    # Visualize bounding boxs and cls
    vis_helper.vis_defect_images(
        dest_path= cfg.label_vis_dest,
        defect_labels= labelme_data.defect_labels)

    #Label Conversion to YOLO data labeling format, split dataset into train/val/test, visulize the labeled data
    #TODO: support evenly splitting different defect classes to train/val/test
    #TODO: support include splitting pass data into train/val/test
    conversion = Labelme2YOLO(
        data_root = cfg.labeled_data_root,
        labelme_lbls = labelme_data.defect_labels,
        imgs_shape_wh= labelme_data.imgs_shape_wh,
        pass_imgs= labelme_data.pass_imgs,
        split_ratio= [0.7, 0.15, 0.15]
    )

    
