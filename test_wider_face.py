import dlib
import random
import os
import re
import sys
from hideface import tools, imagelabels, attacks

img_input_dir = 'data/WIDER_train/images/51--Dresses/'
truth_file = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
detector_dict = {'hog':dlib.get_frontal_face_detector()} 
#num_imgs = len(os.listdir(img_input_dir))
num_imgs = 20
img_paths = random.sample([img_input_dir+img_name for img_name in os.listdir(img_input_dir)],num_imgs)
output_dir = 'data/test_noise'
epsilon_start_value = 16
max_epsilon_value = 240
epsilon_delta = 16
iou_cutoff_value = 0.2

if (os.path.exists(output_dir) and len(os.listdir(output_dir)) != 0):
    user_input = input('Output directory not empty: ' + output_dir + '\nProceed? [y/N]')
    if (user_input != 'y'): sys.exit('Exiting')

fail_count = 0
for detector_name, _ in detector_dict.items():
    for img_path in img_paths:
        img_num = re.findall(r'[0-9]+_[0-9]+\.jpg', str(img_path))[0][:-4]
        image_labels = imagelabels.ImageLabels(img_path).add_all_labels(truth_file, detector_dict)
        print('Testing Image: ' + img_num)
        if (len(image_labels.true_box_list) > 1): print('Found >1 true faces, skipping'); continue
        if (image_labels.true_box_list[0].quality != tools.TruthBoxQuality(0,0,0,0,0,0)): print('True Face Quality too poor (' + str(image_labels.true_box_list[0].quality) + '), skipping'); continue 
        if (len(image_labels.found_box_dict[detector_name]) == 0): print('Could not find face prior to noise application, skipping'); continue
        
        epsilon = epsilon_start_value
        while (epsilon < max_epsilon_value):
            attacked_img_path, noise_img_path = attacks.create_noisy_image(img_path, epsilon, output_dir)
            noisy_image_labels = imagelabels.ImageLabels(attacked_img_path).add_detector_labels(detector_dict)
            if (len(noisy_image_labels.found_box_dict[detector_name]) == 0):
                print('No boxes found with epsilon: ' + str(epsilon))
                noisy_image_labels.draw_images(output_dir,'noisy_eps'+str(epsilon))
                image_labels.draw_images(output_dir)
                break
            if (len(noisy_image_labels.found_box_dict[detector_name]) == 1):
                if (image_labels.true_box_list[0].iou(noisy_image_labels.found_box_dict[detector_name][0]) < iou_cutoff_value):
                    print('Truth-Found IoU < '+str(iou_cutoff_value)+' with epsilon: ' + str(epsilon))
                    noisy_image_labels.draw_images(output_dir,'noisy_eps'+str(epsilon))
                    image_labels.draw_images(output_dir)
                    break
            #attack failed with given epsilon value, so delete old image and increase epsilon
            os.remove(noisy_image_labels.img_path)
            os.remove(noise_img_path)
            epsilon += epsilon_delta
        else: print('Noise attack failed'); fail_count += 1
print('Failed to beat ' + str(fail_count) + ' images with max epsilon: ' + str(max_epsilon_value))
