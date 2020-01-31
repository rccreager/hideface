import re
import os
#import tensorflow as tf
import numpy as np
from skimage import io
from hideface import tools

class ImageLabels:
    """
    A class for storing true FaceBox and found FaceBox information for a given image 

    Attributes:
        img_path: the path to the image
        true_box_list: a list of ground truth FaceBox objects
        found_box_dict: a dictionary of recognizer names and found box lists prior to attack 
        noise_epsilon: epsilon value for noise applied to the image 
     
    Methods:
        add_recognizer_labels: given a recognizer_dict, set the found_box_dict and return
        add_truth_labels: given a truth file, set the truth_box_list and return
        add_all_labels: given a truth file and recognizer_dict, set relevant attributes and return
    """
    def __init__(self, img_path, true_box_list=[], found_box_dict={}, noise_epsilon=0):
        self.img_path = img_path
        self.true_box_list = true_box_list
        self.found_box_dict = found_box_dict
        self.noise_epsilon = noise_epsilon
    def __str__(self):
        return "(Img Path: \n{} \nTruth Box List: \n{} \nFound Box Dict: \n{} \nNoise Epsilon: \n{})".format(self.img_path, self.true_box_list, self.found_box_dict, self.noise_epsilon)
    def __repr__(self):
        return str(self)
    def add_recognizer_labels(self,recognizer_dict):
        """
        For each dictionary entry {recognizer_name:recognizer}, find the FaceBox list for that recognizer and modify the class attribute
        
        Args:
            recognizer_dict: a dictionary pairing a facial recognizer name to the recognizer
        Returns: 
            self
        """
        found_box_dict = {}
        for recognizer_name, recognizer in recognizer_dict.items():
            found_box_list = tools.get_found_boxes(self.img_path,recognizer)
            found_box_dict.update( {recognizer_name : found_box_list} )
        self.found_box_dict = found_box_dict
        return self
    def add_truth_labels(self,truth_file):
        """
        Given a WiderFace truth file, find the ground truth FaceBox for the image and modify the class attribute
        
        Args:
            truth_file: the path to a WiderFace truth file
        Returns:
            self    
        """
        true_box_list=[]
        img_num = re.findall(r"[0-9]+_[0-9]+\.jpg", str(self.img_path))[0][:-4] 
        if(truth_file != None): true_box_list = tools.get_ground_truth_boxes(img_num,truth_file)
        self.true_box_list = true_box_list
        return self
    def add_all_labels(self,truth_file,recognizer_dict):
        """
        Apply recognizer and truth labels in one step
        
        Args:
            truth_file: the path to a WiderFace truth file
            recognizer_dict: a dictionary pairing a facial recognizer name to the recognizer
        Returns: 
            self
        """
        self.add_recognizer_labels(recognizer_dict)
        self.add_truth_labels(truth_file)
        return self

def draw_and_return_noisy_image_labels(original_img_path, max_noise_epsilon, output_img_dir):
    """
    Given a path to an image, this function creates a new image with normally-distributed noise
    randomly applied and returns an ImageLabels object for the newly created image

    Args:
        original_img_path: the path of the image you want to attack
        max_noise_epsilon: an integer giving the maximum variation (out of 255)
        output_img_dir: the directory where you'd like to store the attacked image
    Returns:
        an ImageLabels object for the noisy file without any FaceBox labels set
    """
    eps = max_noise_epsilon / 255.0
    file_num = re.findall(r"[0-9]+_[0-9]+\.jpg", str(original_img_path))[0][:-4]
    #print("Making Noisy (max_eps: "+str(max_noise_epsilon)+") Image: " + file_num)
    image = io.imread(original_img_path)
    pixels = np.copy(np.asarray(image).astype(np.float32))
    pixels /= 255.0
    noisy_pixels = pixels + eps * np.sign(np.random.normal(loc=0.0, scale=1.0, size=pixels.shape))
    noisy_pixels = np.clip(noisy_pixels, 0.0, 1.0)
    noisy_pixels *= 255
    noisy_pixels = np.asarray(noisy_pixels).astype(np.uint8)
    file_name = 'noisy_maxeps'+str(max_noise_epsilon)+'_'+file_num+'.jpg'
    noisy_file_path = os.path.join(output_img_dir,file_name)
    io.imsave(noisy_file_path,noisy_pixels)
    return ImageLabels(noisy_file_path,noise_epsilon=max_noise_epsilon)

def draw_labeled_image(image_labels, output_dir):
    """
    For a given ImageLabels object, write a copy of the image with FaceBoxes drawn to output_dir

    Args:
        image_labels: an ImageLabel object which we want to visualize
        output_dir: a string giving the path to the desired output directory
    """
    image = io.imread(image_labels.img_path)
    tools.draw_boxes(image,image_labels.true_box_list, (0,0,255))
    file_num = re.findall(r"[0-9]+_[0-9]+\.jpg", image_labels.img_path)[0][:-4]
    if (len(image_labels.found_box_dict) == 0): 
        name_str = 'no_recognizer'
        if (len(image_labels.true_box_list) == 0): name_str = name_str + '_no_truth'
        io.imsave(output_dir + '/boxed_'+name_str+'_eps'+str(image_labels.noise_epsilon)+'_'+file_num+'.jpg', image)
    for recognizer_name, found_box_list in image_labels.found_box_dict.items():
        tools.draw_boxes(image, found_box_list, (255,0,0))
        io.imsave(output_dir + '/boxed_'+recognizer_name+'_eps'+str(image_labels.noise_epsilon)+'_'+file_num+'.jpg', image)

