import re
import os
import sys
import copy
from skimage import io
from PIL import Image
import numpy as np
from hideface import tools

class ImageLabels:
    """
    A class for storing true FaceBox and found FaceBox information for a given image 

    Attributes:
        img_path: the path to the image
        img_shape: a tuple giving the image shape (height,width,channels)
        true_box_list: a list of ground truth FaceBox objects
        found_box_dict: a dictionary of detector names and found box lists prior to attack 
        **kwargs: arguments for describing the attack applied (ex: noise_epsilon)
        drawn_images: a list of file paths for images drawn using draw() method
    Methods:
        add_detector_labels: given a detector_dict, set the found_box_dict and return
        add_truth_labels: given a truth file, set the truth_box_list and return
        add_all_labels: given a truth file and detector_dict, set relevant attributes and return
        draw_images: drawn image(s) with available boxes and set drawn_images attribute
        delete_drawn_images: delete all images in drawn_images list and set list to empty
    """
    def __init__(self, img_path, true_box_list=[], found_box_dict={}, **kwargs):
        self.img_path = img_path
        if (not os.path.isfile(self.img_path)): 
            print('Attempting to create ImageLabels object for bad image path: ' 
                    + str(self.img_path))
            raise IOError
        try:
            image = Image.open(img_path)
            image = image.convert("RGB")
            image.verify() # verify that it is, in fact an image
        except:
            print('Input Image Read Error (ImageLabels init): ' + img_path)  
            raise
            #raise ValueError 
        image = np.array(image)
        self.img_shape = image.shape
        self.true_box_list = true_box_list
        self.found_box_dict = found_box_dict
        self.__dict__.update(kwargs)
        self.drawn_images = [] 
    
    def __str__(self):
        main_str = '(Img Path: \n{} \nTruth Box List: \n{} \nFound Box Dict: \n{}\n'.format(
            self.img_path, 
            self.true_box_list, 
            self.found_box_dict)
        kwargs_str = 'Attack Information:'
        for key, item in self.kwargs.items(): kwargs_str += '\n{}: {}'.format(key, item)
        kwargs_str += ')'
        return main_str + kwargs_str
    
    def __repr__(self):
        return str(self)
    
    def add_detector_labels(self,detector_dict):
        """
        For each dictionary entry, find the FaceBox list for that detector and modify attribute
        
        Args:
            detector_dict: a dictionary pairing a facial detector name to the detector
        Returns: 
            self
        """
        if (not os.path.isfile(self.img_path)): 
            sys.exit('Attempted to add detector labels to bad image: ' + self.img_path)
        found_box_dict = {}
        for detector_name, detector in detector_dict.items():
            found_box_list = tools.get_found_boxes(self.img_path,detector)
            found_box_dict.update( {detector_name : found_box_list} )
        self.found_box_dict = found_box_dict
        return self
    
    def add_truth_labels(self,truth_file):
        """
        Given a WiderFace truth file, find the ground truth FaceBox(es) and modify attribute
        
        Args:
            truth_file: the path to a WiderFace truth file
        Returns:
            self    
        """
        if (not os.path.isfile(self.img_path)): 
            sys.exit('Attempted to add truth labels to bad image: ' + self.img_path)
        true_box_list=[]
        img_num = re.findall(r'[0-9]+_[0-9]+\.jpg', str(self.img_path))[0][:-4] 
        if(truth_file != None): true_box_list = tools.get_ground_truth_boxes(img_num,truth_file)
        self.true_box_list = true_box_list
        return self
    
    def add_all_labels(self,truth_file,detector_dict):
        """
        Apply detector and truth labels in one step
        
        Args:
            truth_file: the path to a WiderFace truth file
            detector_dict: a dictionary pairing a facial detector name to the detector
        Returns: 
            self
        """
        self.add_detector_labels(detector_dict)
        self.add_truth_labels(truth_file)
        return self
    
    def draw_images(self, output_dir, name_str=''):
        """
        Draw image for each set of detector labels and append drawn filename to self.drawn_images
        
        Args:
            output_dir: a string giving the path to the desired output directory
            name_str: an optional string describing the image 
        """
        file_num = re.findall(r'[0-9]+_[0-9]+\.jpg', self.img_path)[0][:-4]
        if (not os.path.exists(output_dir)): os.makedirs(output_dir)
        try:
            image = Image.open(self.img_path).convert("RGB")
            image.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e: 
            print('Input Image Read Error (draw_images): ' + self.img_path)   
        image = copy.deepcopy(np.array(image))
        tools.draw_boxes(image,self.true_box_list, (0,0,255))
        if (len(self.found_box_dict) == 0):
            holder_str = 'no_detector'
            if (len(self.true_box_list) == 0): holder_str = holder_str + '_no_truth'
            if (name_str != ''): holder_str = holder_str + '_' + name_str + '_'
            filename = output_dir + '/image_'+holder_str+file_num+'.jpg'
            io.imsave(filename, image)
            self.drawn_images.append(filename)
        for detector_name, found_box_list in self.found_box_dict.items():
            image_copy = copy.deepcopy(image)
            tools.draw_boxes(image_copy, found_box_list, (255,0,0))
            holder_str = ''
            if (name_str != ''): holder_str = holder_str + '_' + name_str
            filename = output_dir + '/image_'+ detector_name + holder_str + '_' + file_num+'.jpg'
            io.imsave(filename, image_copy)
            self.drawn_images.append(filename) 
    
    def delete_drawn_images(self):
        """
        Delete all drawn images and set self.drawn_images to []
        """
        for image in self.draw_images:
            os.remove(image)
        self.draw_images = []    
