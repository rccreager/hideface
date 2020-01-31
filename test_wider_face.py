import dlib
import random
import os
import re
from hideface import tools, utils

img_input_dir = "data/WIDER_train/images/51--Dresses/"
truth_file = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
recognizer_dict = {'hog':dlib.get_frontal_face_detector()} 
#num_imgs = len(os.listdir(img_input_dir))
#img_paths = random.sample([img_input_dir+img_name for img_name in os.listdir(img_input_dir)],num_imgs)
img_paths = [img_input_dir+img_name for img_name in os.listdir(img_input_dir)]
max_good_image_count = 999

good_image_count = 0
fail_count = 0
for recognizer_name, _ in recognizer_dict.items():
    for img_path in img_paths:
        if (good_image_count > max_good_image_count): 
            print("Found " + str(max_good_image_count) + " images")
            print("Failed to beat " + str(fail_count) + " images")
            break
        img_num = re.findall(r"[0-9]+_[0-9]+\.jpg", str(img_path))[0][:-4]
        image_labels = utils.ImageLabels(img_path).add_all_labels(truth_file, recognizer_dict)
        
        print("Testing Image: " + img_num)
        if (len(image_labels.true_box_list) > 1): 
            print("Found >1 true faces, skipping")
            continue
        if (image_labels.true_box_list[0].quality != tools.TruthBoxQuality(0,0,0,0,0,0)):
            print("True Face Quality too poor (" + str(image_labels.true_box_list[0].quality) + "), skipping")
            continue 
        if (len(image_labels.found_box_dict[recognizer_name]) == 0):
            print("Could not find face prior to noise application, skipping")
            continue
        
        epsilon = 16
        noisy_image_labels = utils.draw_and_return_noisy_image_labels(img_path, epsilon, 'data/noisy_outputs').add_all_labels(truth_file, recognizer_dict)
        while (epsilon < 240):
            if (len(noisy_image_labels.found_box_dict[recognizer_name]) == 0):
                print("No boxes found with epsilon: " + str(epsilon))
                utils.draw_labeled_image(image_labels, 'data/noisy_compare')
                utils.draw_labeled_image(noisy_image_labels, 'data/noisy_compare')
                good_image_count += 1
                break
            if (len(noisy_image_labels.found_box_dict[recognizer_name]) == 1):
                if (image_labels.true_box_list[0].iou(noisy_image_labels.found_box_dict[recognizer_name][0]) < 0.2):
                    print("Box IoU with truth < 0.2 with epsilon: " + str(epsilon))
                    utils.draw_labeled_image(image_labels, 'data/noisy_compare')
                    utils.draw_labeled_image(noisy_image_labels, 'data/noisy_compare')
                    good_image_count += 1
                    break
            epsilon += 16
            #print('Trying noise epsilon: ' + str(epsilon))
            noisy_image_labels = utils.draw_and_return_noisy_image_labels(img_path, epsilon, 'data/noisy_outputs').add_all_labels(truth_file, recognizer_dict)
        else:
            print("Noise attack failed")
            fail_count += 1
