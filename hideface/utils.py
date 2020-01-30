import re
from skimage import io
from hideface import tools

class ImageLabels:
    """
    A class for storing true FaceBox and found FaceBox information for a given image 

    Attributes:
        img_name: the path to the image
        true_box_list: a list of ground truth FaceBox objects
        found_box_dict: a dictionary of recognizer names and found box lists prior to attack 
        attack_name: if an attack has been applied, a string naming the attack
    """
    def __init__(self, img_name, true_box_list=[], found_box_dict={}, attack_name=None):
        self.img_name = img_name
        self.true_box_list = true_box_list
        self.found_box_dict = found_box_dict
        self.attack_name = attack_name
    def __str__(self):
        return "(Img Name: \n{} \nTruth Box List: \n{} \nFound Box Dict: \n{} \nAttack Applied: \n{})".format(self.img_name, self.true_box_list, self.found_box_dict, self.attack_name)
    def __repr__(self):
        return str(self)
    def add_recognizer_labels(self,recognizer_dict):
        """
        For each dictionary entry {recognizer_name:recognizer}, find the FaceBox list for that recognizer and modify the class attribute
        """
        print("adding facial recognition for image: " + self.img_name)
        found_box_dict = {}
        for recognizer_name, recognizer in recognizer_dict.items():
            found_box_list = tools.get_found_boxes(self.img_name,recognizer)
            found_box_dict.update( {recognizer_name : found_box_list} )
        self.found_box_dict = found_box_dict
        return self
    def add_truth_labels(self,truth_file):
        """
        Given a WiderFace truth file, find the ground truth FaceBox for the image and modify the class attribute
        """
        print("adding truth labels for image: " + self.img_name)
        true_box_list=[]
        if(truth_file != None): true_box_list = tools.get_ground_truth_boxes(self.img_name,truth_file)
        self.true_box_list = true_box_list
        return self

#def draw_and_return_attacked_image_labels(original_img_name, )

def draw_labeled_image(image_labels, output_dir):
    """
    For a given ImageLabels object, write a copy of the image with FaceBoxes drawn to output_dir

    Attributes:
        image_labels: an ImageLabel object which we want to visualize
        output_dir: a string giving the path to the desired output directory
    Returns:
        None. Just saves image outputs to output_dir 
    """
    image = io.imread(image_labels.img_name)
    tools.draw_boxes(image,image_labels.true_box_list, (0,0,255))
    if (len(image_labels.found_box_dict) == 0): 
        io.imsave(output_dir + '/boxed_no_recognizer_'+file_num+'.jpg', image)
    for recognizer_name, found_box_list in image_labels.found_box_dict.items():
        tools.draw_boxes(image, found_box_list, (255,0,0))
        file_num = re.findall(r"[0-9]+_[0-9]+\.jpg", image_labels.img_name)[0][:-4]
        io.imsave(output_dir + '/boxed_'+recognizer_name+'_'+file_num+'.jpg', image)

