import dlib
#import sys
#import re
import random
import os
#from skimage import io 
#from matplotlib import pyplot as plt 
from hideface import tools, utils

img_input_dir = "data/WIDER_train/images/51--Dresses/"
truth_file = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
recognizer_dict = {'hog':dlib.get_frontal_face_detector()} 
num_imgs = 5
img_names = random.sample([img_name for img_name in os.listdir(img_input_dir)],num_imgs)
image_labels_list = [utils.ImageLabels(img_input_dir + img_name) for img_name in img_names]
print(image_labels_list[0])
image_labels_list = [image_labels.add_truth_labels(truth_file) for image_labels in image_labels_list]
print(image_labels_list[0])
image_labels_list = [image_labels.add_recognizer_labels(recognizer_dict) for image_labels in image_labels_list]
print(image_labels_list[0])
for image_labels in image_labels_list:
    utils.draw_labeled_image(image_labels, 'data/boxed_output')

