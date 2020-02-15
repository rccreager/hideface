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
from hideface import tools, imagelabels, attacks, utils
from PIL import Image


if __name__ == "__main__":
    truth_file = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
    detector_dict = {'hog':dlib.get_frontal_face_detector()} 
    num_imgs = 10000
    img_input_dir = '/s3mnt/WIDER_train/images/'
    img_paths = utils.get_img_paths(img_input_dir, num_imgs)
    output_dir = 'data/10k_attacks_12Feb'
    attack_record_filename = 'performance_list.csv'
    result_counter_filename = 'tunnel.csv'
    use_mult_noise = False #use multiplicative noise, where large pixel value means more noise
    n_attack_tests = 10
    attack_success_freq_threshold = 0.8
    epsilon_start_value = 16
    max_epsilon_value = 240
    epsilon_delta = 16
    iou_cutoff_value = 0.2
    performance_list = []
    #a dictionary for storing the counts of how often each type of image error/result occurs
    tunnel_dict = {
        'total_img_count':0,
        'bad_image':0,
        'multiple_faces_count':0, 
        'zero_faces_count':0, 
        'bad_quality_count':0, 
        'no_found_faces_count':0, 
        'fail_count':0,
        'successful_attack':0}
    utils.test_paths(attack_record_filename, result_counter_filename, output_dir)
    for detector_name, _ in detector_dict.items():
        count = 0
        for img_path in img_paths:
            img_num = re.findall(r'[0-9]+_[0-9]+\.jpg', str(img_path))[0][:-4]
            tunnel_dict['total_img_count'] += 1
            count += 1 
            if (count > num_imgs): break
            print('Testing Image: ' + img_num + '\t(' + str(count) + '/' + str(num_imgs) + ')')
            try: 
                image_labels, truth_iou_no_noise = utils.test_image(
                    img_path, 
                    truth_file, 
                    detector_dict,
                    detector_name,
                    iou_cutoff_value, 
                    tunnel_dict, 
                    output_dir)
            except ValueError as e: 
                print(e)
                continue
            box_height = image_labels.true_box_list[0].height
            box_width = image_labels.true_box_list[0].width
            performance_dict = {
                'img_num':img_num,
                'epsilon':-1,
                'img_size':image_labels.img_shape[0]*image_labels.img_shape[1],
                'true_box_size':box_height * box_width,
                'truth_iou_no_noise':truth_iou_no_noise,
                'truth_iou_noise_avg':-1,
                'ssim_avg':-1}
            epsilon = epsilon_start_value
            while (epsilon < max_epsilon_value):
                try:
                    image_noise_pairs = attacks.create_noisy_face(
                            image_labels.found_box_dict[detector_name][0], 
                            img_path, 
                            epsilon, 
                            n_attack_tests, 
                            output_dir, 
                            use_mult_noise=use_mult_noise)
                except ValueError as e: 
                    print(e)
                    continue
                attack_success_count = 0
                attack_fail_count = 0
                truth_iou_noise_avg = 0
                ssim_avg = 0
                for img_pair in image_noise_pairs:
                    attacked_img = img_pair.img
                    noise_img = img_pair.noise
                    found_boxes = tools.get_found_boxes(attacked_img, detector_dict[detector_name]) 
                    base_img = np.array(Image.open(image_labels.img_path))
                    ssim_avg += ssim(base_img,
                            attacked_img,
                            data_range = attacked_img.max() - attacked_img.min(),
                            multichannel=True)
                    if (len(found_boxes) == 0):
                        truth_iou_noise = 0
                    else:
                        true_box = image_labels.true_box_list[0]
                        found_box = found_boxes[0]
                        truth_iou_noise = true_box.iou(found_box)
                        truth_iou_noise_avg += truth_iou_noise
                    #if you find no faces or the truth-found face IoU is small, the test attack succeeded
                    if (len(found_boxes) == 0 
                            or (len(found_boxes) == 1 and truth_iou_noise < iou_cutoff_value)):
                        attack_success_count += 1 
                    attack_fail_count += 1
                #if attack succeed in enough tests, write files as output and report success
                if (attack_fail_count == 0 
                    or attack_success_count / attack_fail_count >= attack_success_freq_threshold):
                    print('Attack succeeded at least ' + str(100*attack_success_freq_threshold) 
                        + '% of the time with epsilon: ' + str(epsilon))
                    tunnel_dict['successful_attack'] += 1
                    performance_dict['epsilon'] = epsilon
                    performance_dict['truth_iou_noise_avg'] = truth_iou_noise_avg / n_attack_tests
                    performance_dict['ssim_avg'] = ssim_avg / n_attack_tests
                    utils.draw_img_noise_pair_arrays(image_noise_pairs, epsilon, img_num, output_dir)
                    break
                #if attack failed with given epsilon value, increase epsilon
                if (attack_success_count / attack_fail_count < attack_success_freq_threshold):
                    print(str(attack_success_count) + '/' + str(attack_fail_count) 
                            + ' attacks succeeded with epsilon=' + str(epsilon)
                            + '. This is below threshold (' + str(attack_success_freq_threshold) 
                            + '). Continuing...')
                    epsilon += epsilon_delta
            #if epsilon is larger than max_epsilon_value, the attack has failed 
            else: 
                print('Noise attack failed')
                performance_dict['epsilon'] = -1
                performance_dict['truth_iou_noise_avg'] = -1
                performance_dict['ssim_avg'] = -1
                tunnel_dict['fail_count'] += 1
            performance_list.append(performance_dict)
    utils.write_csv(performance_list, attack_record_filename, output_dir)
    utils.write_csv(tunnel_dict, result_counter_filename, output_dir)
