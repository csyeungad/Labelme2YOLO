import os
import cv2
from .utils import mkdir
SUPPORT_FORMAT = (".jpg", ".jpeg", ".jpe", ".jp2", ".png", ".webp", ".bmp", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".sr", ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic")

def annotate_image_idxyxy(image_path, labels):
    """
    Annotate an image with bounding boxes and labels.

    Args:
        image_path (str): Path to the input image.
        labels (list): List of labels containing class names and bounding box coordinates.
                       Each label is in the format: [class_name, x1, y1, x2, y2].
        dest_path (str): Path to save the annotated image.
    """
    image = cv2.imread(image_path)
    for label in labels: #[['INCOMPLETE_FILL', 138, 238, 849, 1034],...]
        class_name = label[0]
        x1, y1, x2, y2 = map(int, label[1:])
        w, h = x2 - x1, y2-y1
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cls_position = (x1 + 5, y1 + 15)
        cv2.putText(image, class_name, cls_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        size_position = (x1 + 5, y1 + 35)
        cv2.putText(image, f"{w} X {h}", size_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        area_position = (x1 + 5, y1 + 55)
        cv2.putText(image, f"{(w * h):.2f}", area_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        ratio_position = (x1 + 5, y1 + 75)
        cv2.putText(image, f"{(w / h):.2f}", ratio_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    return image

def vis_defect_images(dest_path, defect_imgs: dict) -> None:
    """Draw bounding boxes on defect images, grouping by first occurence of cls"""
    mkdir(dest_path)
    mkdir(os.path.join(dest_path, "DEFECT")) #"VI-MIS-002_20240601_WPPD21171000_01_328_54_VOID_TOP_INCOMPLETE_FILL.jpg" , [['INCOMPLETE_FILL', 138, 238, 849, 1034]]
    for defect_img, lbls in defect_imgs.items(): 
        main_lbl = lbls[0][0] #TODO: define label priority based on importance
        mkdir(os.path.join(dest_path, "DEFECT", main_lbl))
        name = os.path.basename(defect_img)
        anno_image = annotate_image_idxyxy(image_path = defect_img, labels= lbls)
        cv2.imwrite(os.path.join(dest_path, "DEFECT", main_lbl, name), anno_image)