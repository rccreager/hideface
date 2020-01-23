#this is a copy of Adam Geitgey's HoG script
#https://gist.github.com/ageitgey/1c1cb1c60ace321868f7410d48c228e1
#I am using it for testing purposes!

import sys
import dlib
from skimage import io

sys.path.insert(0, '/home/ubuntu/face-detection-adversarial-attack/Insight_Project_Framework')

from Insight_Project_Framework import tools

file_name = "51_Dresses_wearingdress_51_53.jpg"
file_path = "/widerface/WIDER_train/images/51--Dresses/"

tools.get_ground_truth('51_53', '/widerface/wider_face_split/wider_face_train_bbx_gt.txt')


image = io.imread(file_path + file_name)
face_detector = dlib.get_frontal_face_detector()
win = dlib.image_window()
detected_faces = face_detector(image, 1)
print("I found {} faces in the file {}".format(len(detected_faces), file_name))
win.set_image(image)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

	# Detected faces are returned as an object with the coordinates 
	# of the top, left, right and bottom edges
        width = face_rect.right() - face_rect.left()
        height = face_rect.bottom() - face_rect.top()
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {} Width: {} Height: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom(), width, height))
        # Draw a box around each face we found
        win.add_overlay(face_rect)
	        
# Wait until the user hits <enter> to close the window	 

dlib.hit_enter_to_continue()
