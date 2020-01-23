#this is a copy of Adam Geitgey's HoG script
#https://gist.github.com/ageitgey/1c1cb1c60ace321868f7410d48c228e1
#I am using it for testing purposes!

import sys
import dlib
from skimage import io
sys.path.insert(0, '/home/ubuntu/face-detection-adversarial-attack/Insight_Project_Framework')
from Insight_Project_Framework import tools

file_name = "51_Dresses_wearingdress_51_53.jpg"
file_path = "/wider-face/WIDER_train/images/51--Dresses/"

true_box_list = tools.get_ground_truth('51_53', '/wider-face/wider_face_split/wider_face_train_bbx_gt.txt')
#print(true_box_list)

image = io.imread(file_path + file_name)
face_detector = dlib.get_frontal_face_detector()
detected_faces = face_detector(image, 1)
found_box_list = [tools.FaceBox(face.left(), face.top(), face.right()-face.left(), face.bottom()-face.top()) for face in detected_faces]
#print(found_box_list)

best_matches = tools.get_best_matches(true_box_list, found_box_list)
print(*best_matches,sep="\n")

#null_box = tools.FaceBox(0,0,0,0)
#best_matches = [[found_box,null_box] for found_box in found_box_list]
#
#for box_pair in best_matches:
#    for true_box in true_box_list:
#        old_iou = box_pair[0].iou(box_pair[1])
#        new_iou = box_pair[0].iou(true_box)
#        if (new_iou > old_iou):
#            box_pair[1] = true_box
#print(best_matches)

win = dlib.image_window()
win.set_image(image)
# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):
    # Draw a box around each face we found
    win.add_overlay(face_rect)
# Wait until the user hits <enter> to close the window	 
dlib.hit_enter_to_continue()
