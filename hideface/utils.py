import re
from skimage import io
from hideface import tools

class ImageLabels:
    """
    A class for storing true FaceBox and found FaceBox information for a given image 

    Attributes:
        img_path: the path to the image
        true_box_list: a list of ground truth FaceBox objects
        found_box_dict: a dictionary of recognizer names and found box lists prior to attack 
        attack_name: if an attack has been applied, a string naming the attack
     
    Methods:
        add_recognizer_labels: given a recognizer_dict, set the found_box_dict and return
        add_truth_labels: given a truth file, set the truth_box_list and return
        add_all_labels: given a truth file and recognizer_dict, set relevant attributes and return
    """
    def __init__(self, img_path, true_box_list=[], found_box_dict={}, attack_name=None):
        self.img_path = img_path
        self.true_box_list = true_box_list
        self.found_box_dict = found_box_dict
        self.attack_name = attack_name
    def __str__(self):
        return "(Img Path: \n{} \nTruth Box List: \n{} \nFound Box Dict: \n{} \nAttack Applied: \n{})".format(self.img_path, self.true_box_list, self.found_box_dict, self.attack_name)
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
        if(truth_file != None): true_box_list = tools.get_ground_truth_boxes(self.img_path,truth_file)
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

def draw_and_return_attacked_image_labels(original_img_path, attack_name, output_img_dir):
    """
    Given a path to an image, this function creates a new image with the attack applied
    and returns an ImageLabels object for the newly created image

    Args:
        original_img_path: the path of the image you want to attack
        attack_name: a string to identify the attack type to apply
        output_img_dir: the directory where you'd like to store the attacked image
    Returns:

    """
    print("hello")

def draw_labeled_image(image_labels, output_dir):
    """
    For a given ImageLabels object, write a copy of the image with FaceBoxes drawn to output_dir

    Args:
        image_labels: an ImageLabel object which we want to visualize
        output_dir: a string giving the path to the desired output directory
    """
    image = io.imread(image_labels.img_path)
    tools.draw_boxes(image,image_labels.true_box_list, (0,0,255))
    if (len(image_labels.found_box_dict) == 0): 
        name_str = 'no_recognizer'
        if (len(image_labels.true_box_list) == 0): name_str = name_str + '_no_truth'
        io.imsave(output_dir + '/boxed_'+name_str+'_'+file_num+'.jpg', image)
    for recognizer_name, found_box_list in image_labels.found_box_dict.items():
        tools.draw_boxes(image, found_box_list, (255,0,0))
        file_num = re.findall(r"[0-9]+_[0-9]+\.jpg", image_labels.img_path)[0][:-4]
        io.imsave(output_dir + '/boxed_'+recognizer_name+'_'+file_num+'.jpg', image)

