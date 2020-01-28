# this is an explansion of a fork of Adam Geitgey's HoG script
# https://gist.github.com/ageitgey/1c1cb1c60ace321868f7410d48c228e1

import sys
import dlib
import re
import random
from os import listdir
from os.path import join
from skimage import io 
from matplotlib import pyplot as plt 
sys.path.insert(0, '/home/ubuntu/face-detection-adversarial-attack/Insight_Project_Framework')
from Insight_Project_Framework import tools

file_path = "/wider-face/WIDER_train/images/51--Dresses/"
files = [join(file_path, file_name) for file_name in listdir(file_path)] 
#files = random.sample(files,10)

iou = [] # use this to store IoU for all found faces

for file_name in files:
    
    file_num = re.findall(r"[0-9]+_[0-9]+\.jpg", file_name)[0][:-4]
    print('Processing image: ' + str(file_num))
    image = io.imread(file_name)

    face_detector = dlib.get_frontal_face_detector()

    true_box_list = tools.get_ground_truth_boxes(file_num, '/wider-face/wider_face_split/wider_face_train_bbx_gt.txt')
    if (len(true_box_list) != 1): continue
    if (true_box_list[0].quality != tools.TruthBoxQuality(0,0,0,0,0,0)): continue
    
    found_box_list = tools.get_found_boxes(image, face_detector)
    best_matches = tools.get_matches_to_truth(true_box_list, found_box_list)
    for match in best_matches:
        iou.append(match.true_box.iou(match.found_box))
    tools.draw_boxes(image, true_box_list, found_box_list)
    io.imsave('data/boxed_output/boxed_'+file_num+'.jpg', image)

plt.hist(iou, bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) 
plt.title("IoU Histogram") 
plt.savefig("data/histos/iou.png")    
