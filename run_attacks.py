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


def get_ssim(image_labels, attacked_image_labels):
    """
    Given an unattacked and an attacked ImageLabels objects, calculate the SSIM score between them
    
    Args:
        image_labels: the ImageLabels object for the unattacked image
        attacked_imaged_labels: the ImageLabels object for the attacked image
    Returns:
        ssim_val: the SSIM value calculated between the two images
    """
    img = io.imread(image_labels.img_path)
    attack_img = io.imread(attacked_image_labels.img_path)
    ssim_val = ssim(
            img, 
            attack_img, 
            data_range = attack_img.max() - attack_img.min(), 
            multichannel=True)
    return ssim_val


def write_csv(rows, output_filename, output_dir):
    """
    Given a list of dictionary entries, write the list as a CSV file

    Args:
        rows: a dictionary or a list of dictionary entries
        output_filename: the name for the output file
        output_dir: the path to the directory to write the output file
    """
    if (len(rows) != 0):
        file_path = os.path.join(output_dir, output_filename)
        with open(file_path, 'w', encoding='utf8', newline='') as output_file:
            if (isinstance(rows, dict)): rows = [rows]
            fc = csv.DictWriter(output_file,fieldnames=rows[0].keys())
            fc.writeheader()
            fc.writerows(rows)
    else: 
        print('write_csv \'rows\' argument has no entries; cannot write CSV file: ' 
            + str(os.path.join(output_dir, attack_record_filename)))


def get_img_paths(img_input_dir, num_imgs):
    """
    A function for getting a list of images from a directory

    Args:
        img_input_dir: the directory containing images or containing directories containing images
        num_imgs: the desired number of images to return
    Returns:
        img_paths: a list of paths to input images. Should have len == num_imgs
    """
    img_paths = []
    for img_dir in os.scandir(img_input_dir):
        if os.path.isfile(img_dir): img_paths.append(img_dir.path)
        if (len(img_paths) >= num_imgs): break
        if os.path.isdir(img_dir.path):
            for img in os.scandir(img_dir.path):
                img_paths.append(img.path)
                if (len(img_paths) >= num_imgs): break
    return img_paths

def test_paths(attack_record_filename, result_counter_filename, output_dir):
    """
    Check whether the provided output filenames and directory exist and are empty

    Args:
        attack_record_filename: path for output file for writing CSV record of image attacks
        result_counter_filename: path for output file for writing CSV record of results counters
        output_dir: the directory for writing all output images and CSV files
    """
    if (os.path.exists(attack_record_filename)): 
        sys.exit('Output file ' + attack_record_filename + ' already exists, exiting')
    if (os.path.exists(result_counter_filename)): 
        sys.exit('Output file ' + result_counter_filename + ' already exists, exiting')
    if (os.path.exists(output_dir) and len(os.listdir(output_dir)) != 0):
        user_input = input('Output directory not empty: ' + output_dir + '\nProceed? [y/N]')
        if (user_input != 'y'): sys.exit('Exiting')


def test_image(img_path, truth_file, detector_dict, iou_cutoff_value, tunnel_dict, output_dir):
    """
    Run quality tests/checks on an input image and draw the image with true and found face boxes

    Args:
        img_path: the full path to the desired input image
        truth_file: the path to the Wider-Face truth file
        detector_dict: a dictionary of detector_name:face_detector pairs
        iou_cutoff_value: the IoU value below which two face boxes are considered a mismatch
        tunnel_dict: a dictionary for counting how many images pass/fail a given quality requirement
        output_dir: a path to a directory for writing the final image
    Returns:
        image_labels: an ImageLabels object with truth and found boxes already calculated
        truth_iou_no_noise: the IoU between the found face and truth face before applying any noise
    """
    
    def quality_error(error_text, counter_name, tunnel_dict):
        tunnel_dict[counter_name] += 1
        raise ValueError(error_text)
    
    try: 
        image_labels = imagelabels.ImageLabels(img_path)
    except IOError: 
        error_str = 'File Path does not exist or cannot be found: ' + str(img_path)
        quality_error(error_str, 'bad_image', tunnel_dict)
    except: 
        error_str = 'Problem during init of ImageLabels (corrupted image?): ' + str(img_path)
        quality_error(error_str, 'bad_image', tunnel_dict)
    image_labels.add_all_labels(truth_file, detector_dict)
    if (len(image_labels.true_box_list) > 1): 
        quality_error('Found >1 true faces, skipping','multiple_faces_count',tunnel_dict)
    if (len(image_labels.true_box_list) == 0): 
        quality_error('Found 0 true faces, skipping','zero_faces_count',tunnel_dict)
    if (image_labels.true_box_list[0].quality != tools.TruthBoxQuality(0,0,0,0,0,0)):
        error_str = ('True Face Quality too poor (' 
                + str(image_labels.true_box_list[0].quality) + '), skipping')
        quality_error(error_str, 'bad_quality_count', tunnel_dict)
    if (len(image_labels.found_box_dict[detector_name]) == 0):
        error_str = 'Could not find face prior to noise application, skipping'
        quality_error(error_str,'no_found_faces_count',tunnel_dict)
    true_box = image_labels.true_box_list[0]
    truth_iou_no_noise = true_box.iou(image_labels.found_box_dict[detector_name][0])
    if (truth_iou_no_noise < iou_cutoff_value):
        error_str = 'Found a face, but match to truth poor (IoU < ' + str(iou_cutoff_value)
        quality_error(error_str, 'no_found_faces_count', tunnel_dict)
    image_labels.draw_images(output_dir)
    return image_labels, truth_iou_no_noise


if __name__ == "__main__":
    truth_file = 'data/wider_face_split/wider_face_train_bbx_gt.txt'
    detector_dict = {'hog':dlib.get_frontal_face_detector()} 
    num_imgs = 10
    img_input_dir = '/s3mnt/WIDER_train/images/51--Dresses/'
    img_paths = get_img_paths(img_input_dir, num_imgs)
    output_dir = 'data/testing'
    attack_record_filename = 'performance_list.csv'
    result_counter_filename = 'tunnel.csv'
    use_mult_noise = False #use multiplicative noise, where large pixel value means more noise
    apply_noise_to_face = True
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
    test_paths(attack_record_filename, result_counter_filename, output_dir)
    for detector_name, _ in detector_dict.items():
        count = 0
        for img_path in img_paths:
            img_num = re.findall(r'[0-9]+_[0-9]+\.jpg', str(img_path))[0][:-4]
            tunnel_dict['total_img_count'] += 1
            count += 1 
            if (count > num_imgs): break
            print('Testing Image: ' + img_num + '\t(' + str(count) + '/' + str(num_imgs) + ')')
            try: 
                image_labels, truth_iou_no_noise = test_image(
                    img_path, 
                    truth_file, 
                    detector_dict, 
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
                'truth_iou_noise':-1,
                'ssim':-1}
            epsilon = epsilon_start_value
            while (epsilon < max_epsilon_value):
                try:
                    if (apply_noise_to_face):
                        attacked_img_path, noise_img_path = attacks.create_noisy_face(
                                image_labels.found_box_dict[detector_name][0], 
                                img_path, 
                                epsilon, 
                                output_dir, 
                                use_mult_noise=use_mult_noise)
                    else:
                        attacked_img_path, noise_img_path = attacks.create_noisy_image(
                            img_path, 
                            epsilon, 
                            output_dir, 
                            use_mult_noise=use_mult_noise)
                except ValueError as e: 
                    print(e)
                    continue
                noisy_image_labels = imagelabels.ImageLabels(attacked_img_path)
                noisy_image_labels = noisy_image_labels.add_detector_labels(detector_dict)
                if (len(noisy_image_labels.found_box_dict[detector_name]) == 0):
                    truth_iou_noise = 0
                else:
                    true_box = image_labels.true_box_list[0]
                    found_box = noisy_image_labels.found_box_dict[detector_name][0]
                    truth_iou_noise = true_box.iou(found_box)
                ssim_val = get_ssim(image_labels, noisy_image_labels) 
                performance_dict['epsilon'] = epsilon
                performance_dict['truth_iou_noise'] = truth_iou_noise
                performance_dict['ssim'] = ssim_val
                #if you find no faces or the truth-found face IoU is small, the attack has succeeded
                if (len(noisy_image_labels.found_box_dict[detector_name]) == 0 
                        or (len(noisy_image_labels.found_box_dict[detector_name]) == 1 
                            and truth_iou_noise < iou_cutoff_value)):
                    if (len(noisy_image_labels.found_box_dict[detector_name]) == 0): 
                        print('No boxes found with epsilon: ' + str(epsilon))
                    if (len(noisy_image_labels.found_box_dict[detector_name]) == 1 
                            and truth_iou_noise < iou_cutoff_value): 
                        print('Truth-Found IoU < ' 
                                + str(iou_cutoff_value) + ' with epsilon: ' + str(epsilon))
                    tunnel_dict['successful_attack'] += 1
                    break
                #if attack failed with given epsilon value, delete old image and increase epsilon
                os.remove(noisy_image_labels.img_path)
                os.remove(noise_img_path)
                epsilon += epsilon_delta
            #if epsilon is larger than max_epsilon_value, the attack has failed 
            else: 
                print('Noise attack failed')
                performance_dict['epsilon'] = -1
                performance_dict['truth_iou_noise'] = -1
                performance_dict['ssim'] = -1
                tunnel_dict['fail_count'] += 1
            performance_list.append(performance_dict)
    write_csv(performance_list, attack_record_filename, output_dir)
    write_csv(tunnel_dict, result_counter_filename, output_dir)
