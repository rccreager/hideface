import os
import sys
import csv
from skimage import io
from skimage.metrics import structural_similarity as ssim
from hideface import tools, imagelabels, attacks
from PIL import Image


def draw_img_noise_pair_arrays(image_noise_pairs, epsilon, img_num, output_dir, use_mult_noise=False):
    """
    Given a list of ImgNoisePair objects, epsilon value, img_num, and output_dir, draw the img
    and noise and save them to the output_dir with appropriate file names

    Args:
        image_noise_pairs: list of ImgNoisePair objects
        epsilon: the noise epsilon value used
        img_num: a string giving the wider-face file number
        output_dir: directory to write the output files
        use_mult_noise: if True, use multiplicative noise file naming. Otherwise use additive name
    """
    for test_num, img_pair in enumerate(image_noise_pairs):
        attacked_img = img_pair.img
        noise_img = img_pair.noise
        noise_str = 'face_unlabeled_multnoise' if (use_mult_noise) else 'face_unlabeled_addnoise'
        file_suffix = '_eps' + str(epsilon) + '_testnum' + str(test_num) + '_' + img_num +'.jpg'
        image_file_name = 'face_unlabeled_' + noise_str + file_suffix
        image_file_path = os.path.join(output_dir,image_file_name)
        im_pix = Image.fromarray(attacked_img)
        im_pix.save(image_file_path)
        noise_file_name = noise_str + file_suffix
        noise_file_path = os.path.join(output_dir,noise_file_name)
        im_noise = Image.fromarray(noise_img)
        im_noise.save(noise_file_path)


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

def test_image(img_path, truth_file, detector_dict, detector_name, iou_cutoff_value, tunnel_dict, output_dir):
    """
    Run quality tests/checks on an input image and draw the image with true and found face boxes

    Args:
        img_path: the full path to the desired input image
        truth_file: the path to the Wider-Face truth file
        detector_dict: a dictionary of detector_name:face_detector pairs
        detector_name: the specific detector you're studying from detector_dict
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
    detector_dict_small = {detector_name: detector_dict[detector_name]}
    image_labels.add_all_labels(truth_file, detector_dict_small)
    if (len(image_labels.true_box_list) > 1):
        quality_error('Found >1 true faces, skipping','multiple_faces_count',tunnel_dict)
    if (len(image_labels.true_box_list) == 0):
        quality_error('Found 0 true faces, skipping','zero_faces_count',tunnel_dict)
    if (image_labels.true_box_list[0].quality != tools.TruthBoxQuality(0,0,0,0,0,0)):
        error_str = ('True Face Quality too poor '
                + str(image_labels.true_box_list[0].quality) + ', skipping')
        quality_error(error_str, 'bad_quality_count', tunnel_dict)
    if (len(image_labels.found_box_dict[detector_name]) == 0):
        error_str = 'Could not find face prior to noise application, skipping'
        quality_error(error_str,'no_found_faces_count',tunnel_dict)
    true_box = image_labels.true_box_list[0]
    truth_iou_no_noise = true_box.iou(image_labels.found_box_dict[detector_name][0])
    if (truth_iou_no_noise < iou_cutoff_value):
        error_str = 'Found a face, but match to truth poor (IoU < ' + str(iou_cutoff_value) + ')'
        quality_error(error_str, 'no_found_faces_count', tunnel_dict)
    image_labels.draw_images(output_dir)
    return image_labels, truth_iou_no_noise


