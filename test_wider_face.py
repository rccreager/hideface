import dlib
#import sys
#import re
#import random
#import os
#from skimage import io 
#from matplotlib import pyplot as plt 
from hideface import tools, utils

img_input_dir = "data/WIDER_train/images/51--Dresses/"
truth_file = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
recognizer_dict = {'hog':dlib.get_frontal_face_detector()} 
num_imgs = 5
results = utils.get_labels(img_input_dir, num_imgs, truth_file, recognizer_dict)
print(results[0])
for result in results:
    utils.create_labeled_image(result, 'data/boxed_output')

