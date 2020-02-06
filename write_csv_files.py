import dlib
import random
import os
import re
import sys
import csv
import matplotlib as plt
import numpy as np
from hideface import tools, imagelabels, attacks

img_input_dir = 'data/WIDER_train/images/51--Dresses/'
truth_file = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
detector_dict = {'hog':dlib.get_frontal_face_detector()}
num_imgs = len(os.listdir(img_input_dir))
#num_imgs = 10
if (num_imgs < len(os.listdir(img_input_dir))):
    img_paths = random.sample([img_input_dir+img_name for img_name in os.listdir(img_input_dir)],num_imgs) 
else: 
    img_paths = [img_input_dir+img_name for img_name in os.listdir(img_input_dir)] 
output_dir = 'data/51_multnoise'
epsilon_start_value = 16
max_epsilon_value = 240
epsilon_delta = 16
iou_cutoff_value = 0.2
performance_list_csv = 'performance_list.csv'
tunnel_csv = 'tunnel.csv'

if (os.path.exists(performance_list_csv)): sys.exit('Output file ' + performance_list_csv + ' already exists, exiting') 
if (os.path.exists(tunnel_csv)): sys.exit('Output file ' + tunnel_csv + ' already exists, exiting') 
if (os.path.exists(output_dir) and len(os.listdir(output_dir)) != 0):
    user_input = input('Output directory not empty: ' + output_dir + '\nProceed? [y/N]')
    if (user_input != 'y'): sys.exit('Exiting')

performance_list = [] 
tunnel_dict = {'total_img_count':num_imgs,'multiple_faces_count':0, 'bad_quality_count':0, 'no_found_faces_count':0, 'fail_count':0,'successful_attack_no_faces_count':0, 'successful_attack_small_iou_count':0}

for detector_name, _ in detector_dict.items():
    count = 0
    for img_path in img_paths:
        img_num = re.findall(r'[0-9]+_[0-9]+\.jpg', str(img_path))[0][:-4]
        image_labels = imagelabels.ImageLabels(img_path).add_all_labels(truth_file, detector_dict)
        count += 1 
        print('Testing Image: ' + img_num + ' (' + str(count) + '/' + str(num_imgs) + ')')
        if (len(image_labels.true_box_list) > 1): 
            print('Found >1 true faces, skipping'); tunnel_dict['multiple_faces_count'] += 1; continue
        if (image_labels.true_box_list[0].quality != tools.TruthBoxQuality(0,0,0,0,0,0)): 
            print('True Face Quality too poor (' + str(image_labels.true_box_list[0].quality) + '), skipping'); tunnel_dict['bad_quality_count'] += 1; continue
        if (len(image_labels.found_box_dict[detector_name]) == 0): 
            print('Could not find face prior to noise application, skipping'); tunnel_dict['no_found_faces_count'] += 1; continue

        truth_iou_no_noise = image_labels.true_box_list[0].iou(image_labels.found_box_dict[detector_name][0])

        epsilon = epsilon_start_value
        while (epsilon < max_epsilon_value):
            attacked_img_path,noise_img_path = attacks.create_noisy_image(img_path, epsilon, output_dir, use_mult_noise=True)
            noisy_image_labels = imagelabels.ImageLabels(attacked_img_path).add_detector_labels(detector_dict)
            truth_iou_noise = 0 if (len(noisy_image_labels.found_box_dict[detector_name]) == 0) else image_labels.true_box_list[0].iou(noisy_image_labels.found_box_dict[detector_name][0])
            if (len(noisy_image_labels.found_box_dict[detector_name]) == 0):
                print('No boxes found with epsilon: ' + str(epsilon))
                #noisy_image_labels.draw_images(output_dir,'noisy_eps'+str(epsilon))
                image_labels.draw_images(output_dir)
                tunnel_dict['successful_attack_no_faces_count'] += 1
                performance_list.append({'img_num':img_num,'epsilon':epsilon,'img_shape':list(image_labels.img_shape),  
                    'true_box_shape':[image_labels.true_box_list[0].height,image_labels.true_box_list[0].width],'truth_iou_no_noise':truth_iou_no_noise,'truth_iou_noise':truth_iou_noise})
                break
            if (len(noisy_image_labels.found_box_dict[detector_name]) == 1 and truth_iou_noise < iou_cutoff_value):
                    print('Truth-Found IoU < '+str(iou_cutoff_value)+' with epsilon: ' + str(epsilon))
                    #noisy_image_labels.draw_images(output_dir,'noisy_eps'+str(epsilon))
                    image_labels.draw_images(output_dir)
                    tunnel_dict['successful_attack_small_iou_count'] += 1
                    performance_list.append({'img_num':img_num,'epsilon':epsilon,'img_shape':list(image_labels.img_shape), 
                        'true_box_shape':[image_labels.true_box_list[0].height,image_labels.true_box_list[0].width],'truth_iou_no_noise':truth_iou_no_noise,'truth_iou_noise':truth_iou_noise}) 
                    break
            #if attack failed with given epsilon value, delete old image and increase epsilon
            os.remove(noisy_image_labels.img_path)
            os.remove(noise_img_path)
            epsilon += epsilon_delta
        else: 
            performance_list.append({'img_num':img_num,'epsilon':-1,'img_shape':list(image_labels.img_shape),
                        'true_box_shape':[image_labels.true_box_list[0].height,image_labels.true_box_list[0].width],'truth_iou_no_noise':truth_iou_no_noise,'truth_iou_noise':-1})
            print('Noise attack failed'); tunnel_dict['fail_count'] += 1
print(performance_list)
with open(output_dir + '/' + performance_list_csv, 'w', encoding='utf8', newline='') as output_file:
    fc = csv.DictWriter(output_file,fieldnames=performance_list[0].keys())
    fc.writeheader()
    fc.writerows(performance_list)
print(tunnel_dict)
with open(output_dir + '/' + tunnel_csv, 'w', encoding='utf8', newline='') as output_file:
    fc = csv.DictWriter(output_file,fieldnames=tunnel_dict.keys())
    fc.writeheader()
    fc.writerow(tunnel_dict)
