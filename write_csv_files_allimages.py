import dlib
import random
import os
import re
import sys
import csv
import matplotlib as plt
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim
from hideface import tools, imagelabels, attacks

truth_file = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
detector_dict = {'hog':dlib.get_frontal_face_detector()}
num_imgs = 3000
img_input_dir = '/s3mnt/WIDER_train/images/'
img_paths = []
for img_dir in os.scandir(img_input_dir): 
    if (len(img_paths) > num_imgs): break
    for img in os.scandir(img_dir.path):
        print(img.path)
        img_paths.append(img.path)
        if (len(img_paths) > num_imgs): break
#print(img_paths)

#img_input_dir = '/s3mnt/WIDER_train/images/10--People_Marching/'
#if (num_imgs < len(os.listdir(img_input_dir))):
#    img_paths = random.sample([img_input_dir+img_name for img_name in os.listdir(img_input_dir)],num_imgs) 
#else: 
#    img_paths = [img_input_dir+img_name for img_name in os.listdir(img_input_dir)] 
output_dir = 'data/all_imgs_addnoise'
performance_list_csv = 'performance_list.csv'
tunnel_csv = 'tunnel.csv'
use_mult_noise = False
epsilon_start_value = 16
max_epsilon_value = 240
epsilon_delta = 16
iou_cutoff_value = 0.2

if (os.path.exists(performance_list_csv)): sys.exit('Output file ' + performance_list_csv + ' already exists, exiting') 
if (os.path.exists(tunnel_csv)): sys.exit('Output file ' + tunnel_csv + ' already exists, exiting') 
if (os.path.exists(output_dir) and len(os.listdir(output_dir)) != 0):
    user_input = input('Output directory not empty: ' + output_dir + '\nProceed? [y/N]')
    if (user_input != 'y'): sys.exit('Exiting')

performance_list = [] 
tunnel_dict = {'total_img_count':0,'bad_image':0,'multiple_faces_count':0, 'zero_faces_count':0, 'bad_quality_count':0, 'no_found_faces_count':0, 'fail_count':0,'successful_attack_no_faces_count':0, 'successful_attack_small_iou_count':0}

for detector_name, _ in detector_dict.items():
    count = 0
    for img_path in img_paths:
        img_num = re.findall(r'[0-9]+_[0-9]+\.jpg', str(img_path))[0][:-4]
        tunnel_dict['total_img_count'] += 1
        count += 1 
        print('Testing Image: ' + img_num + '\t(' + str(count) + '/' + str(num_imgs) + ')')
        try:
            image_labels = imagelabels.ImageLabels(img_path) 
        except IOError:
            print('File Path does not exist or cannot be found: ' + str(img_path)); tunnel_dict['bad_image'] += 1; continue
        except:
            print('Problem during initialization of ImageLabels (probably a corrupted image): ' + str(img_path)); tunnel_dict['bad_image'] += 1; continue 
        image_labels.add_all_labels(truth_file, detector_dict)
        if (len(image_labels.true_box_list) > 1): print('Found >1 true faces, skipping'); tunnel_dict['multiple_faces_count'] += 1; continue
        if (len(image_labels.true_box_list) == 0): print('Found 0 true faces, skipping'); tunnel_dict['zero_faces_count'] += 1; continue
        if (image_labels.true_box_list[0].quality != tools.TruthBoxQuality(0,0,0,0,0,0)): 
            print('True Face Quality too poor (' + str(image_labels.true_box_list[0].quality) + '), skipping'); tunnel_dict['bad_quality_count'] += 1; continue
        if (len(image_labels.found_box_dict[detector_name]) == 0): 
            print('Could not find face prior to noise application, skipping'); tunnel_dict['no_found_faces_count'] += 1; continue
        truth_iou_no_noise = image_labels.true_box_list[0].iou(image_labels.found_box_dict[detector_name][0])
        if (truth_iou_no_noise < iou_cutoff_value):
            print('Found a face, but match to truth poor (IoU < ' + str(iou_cutoff_value)); tunnel_dict['no_found_faces_count'] += 1; continue
        image_labels.draw_images(output_dir)

        epsilon = epsilon_start_value
        while (epsilon < max_epsilon_value):
            attacked_img_path,noise_img_path = attacks.create_noisy_image(img_path, epsilon, output_dir, use_mult_noise=use_mult_noise)
            noisy_image_labels = imagelabels.ImageLabels(attacked_img_path).add_detector_labels(detector_dict)
            truth_iou_noise = 0 if (len(noisy_image_labels.found_box_dict[detector_name]) == 0) else image_labels.true_box_list[0].iou(noisy_image_labels.found_box_dict[detector_name][0])
            img = io.imread(image_labels.img_path)
            noise_img = io.imread(noisy_image_labels.img_path)
            ssim_val = ssim_noise = ssim(img, noise_img, data_range=noise_img.max() - noise_img.min(), multichannel=True)
            if (len(noisy_image_labels.found_box_dict[detector_name]) == 0):
                print('No boxes found with epsilon: ' + str(epsilon))
                tunnel_dict['successful_attack_no_faces_count'] += 1
                performance_list.append({'img_num':img_num,'epsilon':epsilon,'img_size':image_labels.img_shape[0]*image_labels.img_shape[1],      
                    'true_box_size':image_labels.true_box_list[0].height*image_labels.true_box_list[0].width,'truth_iou_no_noise':truth_iou_no_noise,'truth_iou_noise':truth_iou_noise,'ssim':ssim_val}) 
                break
            if (len(noisy_image_labels.found_box_dict[detector_name]) == 1 and truth_iou_noise < iou_cutoff_value):
                    print('Truth-Found IoU < '+str(iou_cutoff_value)+' with epsilon: ' + str(epsilon))
                    tunnel_dict['successful_attack_small_iou_count'] += 1
                    performance_list.append({'img_num':img_num,'epsilon':epsilon,'img_size':image_labels.img_shape[0]*image_labels.img_shape[1], 
                        'true_box_size':image_labels.true_box_list[0].height*image_labels.true_box_list[0].width,'truth_iou_no_noise':truth_iou_no_noise,'truth_iou_noise':truth_iou_noise,'ssim':ssim_val}) 
                    break
            #if attack failed with given epsilon value, delete old image and increase epsilon
            os.remove(noisy_image_labels.img_path)
            os.remove(noise_img_path)
            epsilon += epsilon_delta
        else: 
            performance_list.append({'img_num':img_num,'epsilon':-1,'img_size':image_labels.img_shape[0]*image_labels.img_shape[1],
                'true_box_size':image_labels.true_box_list[0].height*image_labels.true_box_list[0].width,'truth_iou_no_noise':truth_iou_no_noise,'truth_iou_noise':-1,'ssim':-1})
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
