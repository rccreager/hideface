import re

class FaceBox:
    """
    A class for storing bounding boxes of faces.

    Attributes:
        x1: x position of upper-left box corner in pixels 
        y1: y position of upper-left box corner in pixels
        width: box width in pixels
        height: box height in pixels
    
    Methods:
        area: returns the area of the bounding box in pixels as a float
        iou: returns the intersection over union between two FaceBox objects
    """
    def __init__(self, x1=0, y1=0, width=0, height=0):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.width = int(width)
        self.height = int(height)
    def __str__(self):
        return "(x1:{} y1:{} Width:{} Height:{})".format(self.x1, self.y1, self.width, self.height) 
    def __repr__(self):
        return str(self)
    def area(self):
        """Area of FaceBox as a float"""
        return float(self.width*self.height)
    def iou(self, box):
        """Intersection over union of two FaceBox objects as a float"""
        intersect = FaceBox( max(0,self.x1,box.x1), max(0,self.y1,box.y1), 
                min(self.x1+self.width,box.x1+box.width)-max(0,self.x1,box.x1), 
                min(self.y1+self.height,box.y1+box.height)-max(0,self.y1,box.y1)  )
        return intersect.area() / (self.area() + box.area() - intersect.area())

class FaceBoxMatch:
    """
    A class for storing a match between a ground truth FaceBox and a found FaceBox

    Attributes:
        true_box: the ground truth FaceBox matched to the found_box
        found_box: the found FaceBox matched to the true_box
    """
    def __init__(self, true_box=FaceBox(), found_box=FaceBox()):
        self.true_box = true_box
        self.found_box = found_box
    def __str__(self):
        return "[True Box:{} Found Box:{} IoU:{}]".format(self.true_box, self.found_box, self.true_box.iou(self.found_box))
    def __repr__(self):
        return str(self)

def get_ground_truth(image_number="", path_to_test_file=""):
    """
    Get a list of ground truth FaceBox objects

    Attributes:
        image_number: a string indicating which WiderFace image to locate
        path_to_test_file: a string giving the path to the ground truth WiderFace boxes
    
    Returns a list of FaceBox objects
    """
    box_list = []
    with open(path_to_test_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for i in range(0, len(lines)):
            line = lines[i]
            if (image_number+'.jpg') in line:
                num_faces = int(lines[i+1])
                for j in range(2,num_faces+2):
                    box_vals = re.findall(r"^[0-9]+ [0-9]+ [0-9]+ [0-9]+", lines[i+j])[0].split()
                    box = FaceBox(box_vals[0], box_vals[1], box_vals[2], box_vals[3])
                    box_list.append(box)
    return box_list                   

def get_best_matches(true_box_list=[], found_box_list=[]):
    """
    Compare true FaceBox objects to found FaceBox objects and return a list of best matches
    If a true box doesn't have a match, it is matched to a null FaceBox

    Attributes:
        true_box_list: a list of FaceBox objects for ground truth
        found_box_list: a list of FaceBox objects found via face detection

    Returns a list of FaceBoxMatch objects
    """
    null_box = FaceBox()
    best_matches = [FaceBoxMatch(true_box, null_box) for true_box in true_box_list]
    for box_pair in best_matches:
        for found_box in found_box_list:
            old_iou = box_pair.true_box.iou(box_pair.found_box)
            new_iou = box_pair.true_box.iou(found_box)
            if (new_iou > old_iou):
                box_pair.found_box = found_box
    return best_matches
