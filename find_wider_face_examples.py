import dlib
import random
import os
import re
from hideface import tools, utils

img_input_dir = "data/WIDER_train/images/51--Dresses/"
truth_file = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
recognizer_dict = {'hog':dlib.get_frontal_face_detector()} 
#num_imgs = len(os.listdir(img_input_dir))
#num_imgs = 50
#img_paths = random.sample([img_input_dir+img_name for img_name in os.listdir(img_input_dir)],num_imgs)
img_paths = [img_input_dir+img_name for img_name in os.listdir(img_input_dir)]
counter = 0
for recognizer_name, _ in recognizer_dict.items():
    for img_path in img_paths:
        img_num = re.findall(r"[0-9]+_[0-9]+\.jpg", str(img_path))[0][:-4]
        image_labels = utils.ImageLabels(img_path).add_all_labels(truth_file, recognizer_dict)
        
        print("Testing Image: " + img_num)
        if (len(image_labels.true_box_list) > 1): 
            print("Found >1 true faces, skipping")
            continue
        #if (image_labels.true_box_list[0].quality != tools.TruthBoxQuality(0,0,0,0,0,0)):
        #    print("True Face Quality too poor (" + str(image_labels.true_box_list[0].quality) + "), skipping")
        #    continue 
        #if (len(image_labels.found_box_dict[recognizer_name]) == 0):
        #    print("Could not find face prior to noise application, skipping")
        #    continue
        if (image_labels.true_box_list[0].quality == tools.TruthBoxQuality(0,0,0,0,0,0)):
            utils.draw_labeled_image(image_labels, 'data/examples_000000')
        if (image_labels.true_box_list[0].quality == tools.TruthBoxQuality(1,0,0,0,0,0)):
            utils.draw_labeled_image(image_labels, 'data/examples_100000')
        if (image_labels.true_box_list[0].quality == tools.TruthBoxQuality(2,0,0,0,0,0)):
            utils.draw_labeled_image(image_labels, 'data/examples_200000')
        if (image_labels.true_box_list[0].quality == tools.TruthBoxQuality(0,1,0,0,0,0)):
            utils.draw_labeled_image(image_labels, 'data/examples_010000')
        if (image_labels.true_box_list[0].quality == tools.TruthBoxQuality(0,0,1,0,0,0)):
            utils.draw_labeled_image(image_labels, 'data/examples_001000')
        if (image_labels.true_box_list[0].quality == tools.TruthBoxQuality(0,0,0,1,0,0)):
            utils.draw_labeled_image(image_labels, 'data/examples_000100')
        if (image_labels.true_box_list[0].quality == tools.TruthBoxQuality(0,0,0,0,1,0)):
            utils.draw_labeled_image(image_labels, 'data/examples_000010')
        if (image_labels.true_box_list[0].quality == tools.TruthBoxQuality(0,0,0,0,2,0)):
            utils.draw_labeled_image(image_labels, 'data/examples_000020')
        if (image_labels.true_box_list[0].quality == tools.TruthBoxQuality(0,0,0,0,0,1)):
            utils.draw_labeled_image(image_labels, 'data/examples_000001')

        #if (len(image_labels.found_box_dict[recognizer_name]) > len(image_labels.true_box_list)):
        #    print("False Positive")
        #    utils.draw_labeled_image(image_labels, 'data/examples_fp')
        #if (len(image_labels.true_box_list) > len(image_labels.found_box_dict[recognizer_name])):
        #    print("False Negative")
        #    utils.draw_labeled_image(image_labels, 'data/examples_fn')
        
        #match_found_to_truth = tools.get_matches(image_labels.true_box_list, image_labels.found_box_dict[recognizer_name]) 
        #match_truth_to_found = tools.get_matches(image_labels.found_box_dict[recognizer_name], image_labels.true_box_list)
