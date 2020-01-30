import dlib
import random
import os
#from matplotlib import pyplot as plt 
from hideface import tools, utils

img_input_dir = "data/WIDER_train/images/51--Dresses/"
truth_file = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
recognizer_dict = {'hog':dlib.get_frontal_face_detector()} 
num_imgs = 10
print("Using " + str(num_imgs) + " Images")
img_paths = random.sample([img_input_dir+img_name for img_name in os.listdir(img_input_dir)],num_imgs)
image_labels_list = [utils.ImageLabels(img_path).add_all_labels(truth_file, recognizer_dict) for img_path in img_paths]
#image_labels_list = [image_labels.add_all_labels(truth_file, recognizer_dict) for image_labels in image_labels_list]
print(image_labels_list[0])
for image_labels in image_labels_list:
    utils.draw_labeled_image(image_labels, 'data/boxed_output')

