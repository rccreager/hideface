import os
import re
import random
import numpy as np
from skimage import io
from hideface import tools

class Result:
    """
    A class for storing the results of labeling and attacking images 

    Attributes:
        original_img_name: the name of the original image 
        true_box_list: a list of ground truth FaceBox objects (can be empty)
        original_found_box_dict: a dictionary of recognizer names and found box lists prior to attack 
        attacked_found_box_dict: a nested dictionary of found boxes for each recognizer after each attack
    """
    def __init__(self, original_img_name, true_box_list, original_found_box_dict={}, attacked_found_box_dict={}):
        self.original_img_name = original_img_name
        self.true_box_list = true_box_list
        self.original_found_box_dict = original_found_box_dict
        self.attacked_found_box_dict = attacked_found_box_dict 
    def __str__(self):
        return "(WF Img Name: \n{} \nTruth Box List: \n{} \nOriginal Found Box Dict: \n{} \nAttacked Found Box Dict: \n{})".format(self.original_img_name, self.true_box_list, self.original_found_box_dict, self.attacked_found_box_dict)
    def __repr__(self):
        return str(self)

def create_labeled_image(result, output_dir):
    """
    Create labeled images for images in a given directory
    
    Attributes:
        result: a Result object for a given image
        output_dir: a string giving the path to the desired output directory
    Returns:
        None. Just saves image outputs to output_dir 
    """
    image = io.imread(result.original_img_name)
    tools.draw_boxes(image,result.true_box_list, (0,0,255))
    if (len(result.original_found_box_dict) == 0): 
        io.imsave(output_dir + '/boxed_truth_'+file_num+'.jpg', image)
    for recognizer_name, found_box_list in result.original_found_box_dict.items():
        tools.draw_boxes(image, found_box_list, (255,0,0))
        file_num = re.findall(r"[0-9]+_[0-9]+\.jpg", result.original_img_name)[0][:-4]
        io.imsave(output_dir + '/boxed_'+recognizer_name+'_'+file_num+'.jpg', image)

def get_labels(img_input_dir, num_imgs=None, truth_file=None, recognizer_dict={}, csv_output=None):
    """
    Given a directory of images and (possibly) a truth file and list of recognizers, return
    a numpy array of Results objects

    Args:
        img_input_dir: path to the image directory
        num_imgs: optionally set the number of images to check
        truth_file: optional path to the truth file
        recognizer_dict: optionally specify dictionary of recognizer names and recognizers
        csv_output: optionally specify a path to write a CSV output file
    Returns:
        results: a list of Result objects
    """
    img_names = [img_input_dir + img_name for img_name in os.listdir(img_input_dir)]    
    if (num_imgs != None): 
        print('Number of Images to Label: ' + str(num_imgs))
        img_names = random.sample(img_names,num_imgs)
    results = np.empty(shape=(0))
    for img_name in img_names:
        print('Processing image: ' + img_name)
        found_box_dict = {}
        true_box_list=[]
        if(truth_file != None): true_box_list = tools.get_ground_truth_boxes(img_name,truth_file)
        for recognizer_name, recognizer in recognizer_dict.items(): 
            found_box_list = tools.get_found_boxes(img_name,recognizer)
            found_box_dict = {recognizer_name: found_box_list}
        result = Result(img_name,np.asarray(true_box_list),found_box_dict)
        results = np.append(results,result)
    if (csv_output != None): numpy.savetxt(csv_output, results, delimiter=",")
    return results
