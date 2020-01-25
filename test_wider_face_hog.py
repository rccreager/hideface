# this is an explansion of a fork of Adam Geitgey's HoG script
# https://gist.github.com/ageitgey/1c1cb1c60ace321868f7410d48c228e1

import sys
import dlib
import re
import random
from os import listdir
from skimage import io
from skimage.draw import polygon_perimeter
sys.path.insert(0, '/home/ubuntu/face-detection-adversarial-attack/Insight_Project_Framework')
from Insight_Project_Framework import tools

file_path = "/wider-face/WIDER_train/images/51--Dresses/"
files = listdir(file_path)
#files = random.sample(listdir(file_path),10)

for file_name in files:
    
    file_num = re.findall(r"[0-9]+_[0-9]+\.jpg", file_name)[0][:-4]
    print(file_num)
    true_box_list = tools.get_ground_truth(file_num, '/wider-face/wider_face_split/wider_face_train_bbx_gt.txt')
    true_faces = [dlib.rectangle(left=box.x1, top=box.y1, right=box.x1+box.width, bottom=box.y1+box.height) for box in true_box_list]
    
    image = io.imread(file_path + file_name)
    face_detector = dlib.get_frontal_face_detector()
    found_faces = face_detector(image, 1)
    found_box_list = []
    for face in found_faces:
        width = face.right()-face.left()
        height = face.bottom()-face.top()
        if (face.right() > image.shape[1]): width = image.shape[1] - face.left()
        if (face.bottom() > image.shape[0]): height = image.shape[0] - face.bottom()
        found_box_list.append(tools.FaceBox(face.left(), face.top(), width, height))
    
    best_matches = tools.get_matches_to_truth(true_box_list, found_box_list)
    print(*best_matches,sep="\n")
   
    for found_face in found_box_list: 
        rr,cc = polygon_perimeter([found_face.y1, found_face.y1, found_face.y1+found_face.height-1, found_face.y1+found_face.height-1],
                                 [found_face.x1, found_face.x1+found_face.width-1, found_face.x1+found_face.width-1, found_face.x1]) 
        image[rr, cc] = (255, 0, 0)
    for true_face in true_box_list:
        rr,cc = polygon_perimeter([true_face.y1, true_face.y1, true_face.y1+true_face.height-1, true_face.y1+true_face.height-1],
                                 [true_face.x1, true_face.x1+true_face.width-1, true_face.x1+true_face.width-1, true_face.x1])
        image[rr, cc] = (0, 0, 255)
    #io.imsave('data/boxed_output/boxed_'+file_num+'.jpg', image)



#win = dlib.image_window()
#win.set_image(image)
#for found_face_rect in found_faces:
#    # Draw a box around each face we found
#    win.add_overlay(found_face_rect, dlib.rgb_pixel(0,0,255))
#for true_face_rect in true_faces:
#    win.add_overlay(true_face_rect, dlib.rgb_pixel(255,0,0))
## Wait until the user hits <enter> to close the window	 
#dlib.hit_enter_to_continue()
