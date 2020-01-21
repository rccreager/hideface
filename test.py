import sys
import dlib
from skimage import io

file_name = "/s3mnt/lfw/George_W_Bush/George_W_Bush_0001.jpg"
image = io.imread(file_name)
face_detector = dlib.get_frontal_face_detector()
win = dlib.image_window()
