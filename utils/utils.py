import os
import shutil
import cv2
import json
SUPPORT_FORMAT = (".jpg", ".jpeg", ".jpe", ".jp2", ".png", ".webp", ".bmp", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".sr", ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic")

def listdir(path):
    return [ dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]

def listImg(path):
    return [ file for file in os.listdir(path) if file.endswith(SUPPORT_FORMAT)]

def listImg_dir(path):
    return [ os.path.join(path, file) for file in os.listdir(path) if file.endswith(SUPPORT_FORMAT)]

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def save_class_ids(save_dir: str, class_ids: dict[str, int]):
    with open(os.path.join(save_dir, f'yolo_class_ids_{os.path.basename(save_dir)}.json'), 'w') as f:
        json.dump(class_ids, f)
    print(f"Saved yolo_class_ids_{os.path.basename(save_dir)}.json'")

def load_class_ids(json_path: str) -> dict[str, int]:
    with open(json_path, 'r') as f:
        class_ids = json.load(f)  
    return class_ids

def save_yolo_json(save_path: str, data: dict):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_yolo_json(json_path:str) -> dict:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def group_images(dest_path, pass_imgs: list, review_imgs: list, defect_labels: dict, remove_imgs: list) -> None:
    """ Group the labeled data into different folders"""
    mkdir(dest_path)
    mkdir(os.path.join(dest_path, "PASS")) #"VI-MIS-002_20240601_WPPD21171000_01_328_54_VOID_TOP_INCOMPLETE_FILL.jpg"
    for pass_img in pass_imgs: 
        name = os.path.basename(pass_img)
        json_path = pass_img.replace(".jpg", ".json")
        json_name = name.replace(".jpg", ".json")
        try:
            shutil.copy2(pass_img, os.path.join(dest_path, "PASS", name))
            shutil.copy2(json_path, os.path.join(dest_path, "PASS",json_name))
        except Exception as e:
            print(e)

    mkdir(os.path.join(dest_path, "REVIEW")) #"VI-MIS-002_20240601_WPPD21171000_01_328_54_VOID_TOP_INCOMPLETE_FILL.jpg"
    for review_img in review_imgs:
        name = os.path.basename(review_img)
        json_path = review_img.replace(".jpg", ".json")
        json_name = name.replace(".jpg", ".json")
        try:
            shutil.copy2(review_img, os.path.join(dest_path, "REVIEW", name))
            shutil.copy2(json_path, os.path.join(dest_path, "REVIEW",json_name))
        except Exception as e:
            print(e)

    mkdir(os.path.join(dest_path, "DEFECT")) #"VI-MIS-002_20240601_WPPD21171000_01_328_54_VOID_TOP_INCOMPLETE_FILL.jpg" , [['INCOMPLETE_FILL', 138, 238, 849, 1034]]
    ## Use first label as the main label for image grouping
    for defect_img, lbls in defect_labels.items(): 
        main_lbl = lbls[0][0]
        mkdir(os.path.join(dest_path, "DEFECT", main_lbl))
        name = os.path.basename(defect_img)
        json_path = defect_img.replace(".jpg", ".json")
        json_name = name.replace(".jpg", ".json")
        try:
            shutil.copy2(defect_img, os.path.join(dest_path, "DEFECT", main_lbl, name))
            shutil.copy2(json_path, os.path.join(dest_path, "DEFECT",main_lbl ,json_name))
        except Exception as e:
            print(e)
    
    mkdir(os.path.join(dest_path, "REMOVE")) #"VI-MIS-002_20240601_WPPD21171000_01_328_54_VOID_TOP_INCOMPLETE_FILL.jpg"
    for remove_img in remove_imgs:
        name = os.path.basename(remove_img)
        json_path = remove_img.replace(".jpg", ".json")
        json_name = name.replace(".jpg", ".json")
        try:
            shutil.copy2(remove_img, os.path.join(dest_path, "REMOVE", name))
            shutil.copy2(json_path, os.path.join(dest_path, "REMOVE",json_name))
        except Exception as e:
            print(e)
