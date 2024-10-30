Label data using open source Labelme with below configs:
FLAGS: ['PASS', 'REVIEW', 'REMOVE']
LABELS: ['CLS_1', 'CLS_2', 'CLS_3', 'CLS_4', ...]

cfg.json
{
    "labeled_data_root" : directory with data labeled with Labelme,
    "grouping_dest" : directory for data grouping of labeled data ,
    "label_vis_dest" directory for data visualization of labeled data : 
}
