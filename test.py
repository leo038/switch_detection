import os

from ultralytics import YOLO

#
# Load a model
model = YOLO("./weights/switch-seg.pt")

test_data_dir = "./datasets/switch/segment/test/images"
result_save_dir = "./results/seg"
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)

test_img_list = os.listdir(test_data_dir)
for img_name in test_img_list:
    img_file = os.path.join(test_data_dir, img_name)
    results = model(img_file)

    save_name = os.path.join(result_save_dir, img_name)
    results[0].save(save_name)

