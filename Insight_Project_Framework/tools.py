import re

class FaceBox:
    def __init__(self, x1, y1, width, height):
        # (x1, y1) is the point in the upper left hand corner
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.width = float(width)
        self.height = float(height)
    def __str__(self):
        return "(x1:{} y1:{} Width:{} Height:{})".format(self.x1, self.y1, self.width, self.height) 
    def __repr__(self):
        return str(self)
    def area(self):
        return float(self.width*self.height)
    def iou(self, box):
        intersect = FaceBox( max(0,self.x1,box.x1), max(0,self.y1,box.y1), 
                min(self.x1+self.width,box.x1+box.width)-max(0,self.x1,box.x1), 
                min(self.y1+self.height,box.y1+box.height)-max(0,self.y1,box.y1)  )
        return intersect.area() / float(self.area() + box.area() - intersect.area())

class FaceBoxMatch:
    def __init__(self, true_box, found_box):
        self.true_box = true_box
        self.found_box = found_box
    def __str__(self):
        return "[True Box:{} Found Box:{} IoU:{}]".format(self.true_box, self.found_box, self.true_box.iou(self.found_box))
    def __repr__(self):
        return str(self)

def get_ground_truth(image_number, path_to_test_file):
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

def get_best_matches(true_box_list, found_box_list):
    null_box = FaceBox(0,0,0,0)
    best_matches = [FaceBoxMatch(true_box, null_box) for true_box in true_box_list]
    for box_pair in best_matches:
        for found_box in found_box_list:
            old_iou = box_pair.true_box.iou(box_pair.found_box)
            new_iou = box_pair.true_box.iou(found_box)
            if (new_iou > old_iou):
                box_pair.found_box = found_box
    return best_matches
