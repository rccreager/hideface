import sys
import dlib
from skimage import io

file_name = "/s3mnt/wider_face/51--Dresses/51_Dresses_wearingdress_51_1002.jpg"
image = io.imread(file_name)
face_detector = dlib.get_frontal_face_detector()
win = dlib.image_window()
