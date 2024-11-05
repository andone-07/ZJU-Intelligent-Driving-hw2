import json
import os

# 设置路径
coco_annotations_file = 'D://Desktop//ultralytics-main//dataset//SSLAD-2D//annotations//instance_train.json'  # COCO格式的annotations文件路径
output_dir = 'D://Desktop//ultralytics-main//dataset//SSLAD-2D//labels//train'  # YOLO格式标签输出目录
images_dir = 'D://Desktop//ultralytics-main//dataset//SSLAD-2D//images//train'  # 图像目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取COCO格式的annotations文件
with open(coco_annotations_file, 'r') as f:
    coco_data = json.load(f)

# 提取类别信息
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

# 遍历所有图像和对应的annotations
for img in coco_data['images']:
    img_id = img['id']
    img_width = img['width']
    img_height = img['height']
    
    # 创建YOLO格式的标签文件
    label_file = os.path.join(output_dir, f"{img['file_name'].split('.')[0]}.txt")
    
    with open(label_file, 'w') as lbl_f:
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == img_id:
                category_id = annotation['category_id']
                bbox = annotation['bbox']
                
                # 计算YOLO格式的坐标
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = min(bbox[2] / img_width, 1.0)
                height = min(bbox[3] / img_height, 1.0)
                
                # 写入YOLO格式
                lbl_f.write(f"{category_id - 1} {x_center} {y_center} {width} {height}\n")  # 类别ID需要减去1

print("转换完成！")
