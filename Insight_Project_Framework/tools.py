import re

class TruthBoxQuality:
    """
    A class for storing the face image quality attributes for ground truth

    Attributes:
        blur: the bluriness of the face. 0=clear, 1=light, 2=heavy
        expression: the facial expression. 0=normal, 1=exaggerated
        illumination: the illumination of the face. 0=normal, 1=extreme
        invalid: the validity of the image. 0=valid, 1=invalid
        occlusion: how much the face is block.0=none, 1=partial, 2=heavy
        pose: the angle of the face. 0=normal, 1=abnormal

    Methods:    
        No methods so far
    """
    def __init__(self, blur, expression, illumination, invalid, occlusion, pose):
        self.blur = int(blur)
        self.expression = int(expression)
        self.illumination = int(illumination)
        self.occlusion = int(occlusion)
        self.pose = int(pose)
        self.invalid = int(invalid)
    def __str__(self):
        return "(Blur: {} Expression: {} Illumination: {} Invalid: {} Occlusion: {} Pose: {})".format(self.blur, self.expression, self.illumination, self.invalid, self.occlusion, self.pose)
    def __repr__(self):
        return str(self)

class FaceBox:
    """
    A class for storing bounding boxes of faces.

    Attributes:
        x1: x position of upper-left box corner in pixels 
        y1: y position of upper-left box corner in pixels
        width: box width in pixels
        height: box height in pixels
        quality: if ground truth, this is a TruthBoxQuality object; for found boxes, it's None

    Methods:
        area: returns the area of the bounding box in pixels as a float
        iou: returns the intersection over union between two FaceBox objects
    """
    def __init__(self, x1=0, y1=0, width=0, height=0, quality=None):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.width = int(width)
        self.height = int(height)
        self.quality = quality
    def __str__(self):
        return "(x1:{} y1:{} Width:{} Height:{} Quality: {})".format(self.x1, self.y1, self.width, self.height, self.quality) 
    def __repr__(self):
        return str(self)
    def area(self):
        """Area of FaceBox as a float"""
        return float(self.width*self.height)
    def iou(self, box):
        """Intersection over union of two FaceBox objects as a float"""
        if (self.width == 0 or self.height == 0 or box.width == 0 or box.height == 0): return 0
        intersect = FaceBox( max(0,self.x1,box.x1), max(0,self.y1,box.y1), 
                max(0,min(self.x1+self.width,box.x1+box.width)-max(0,self.x1,box.x1)), 
                max(0,min(self.y1+self.height,box.y1+box.height)-max(0,self.y1,box.y1))  )
        return max(0,intersect.area() / (self.area() + box.area() - intersect.area()))

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
    
    Returns a list of FaceBox objects with TruthBoxQuality properly set 
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
                    box_vals = re.findall(r"^[0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-2] [0,1] [0,1] [0,1] [0-2] [0,1]", lines[i+j])[0].split()
                    #make sure to skip invalid truth boxes
                    if (box_vals[7] == 1): next
                    box = FaceBox(box_vals[0], box_vals[1], box_vals[2], box_vals[3], TruthBoxQuality(box_vals[4], box_vals[5], box_vals[6], box_vals[7], box_vals[8], box_vals[9]))
                    box_list.append(box)
    return box_list                   

def get_matches_to_truth(true_box_list=[], found_box_list=[]):
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
