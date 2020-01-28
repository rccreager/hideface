import os
from skimage import io
from cleverhans.attacks import FastGradientMethod
import cleverhans
import cv2

face_cascade = cv2.CascadeClassifier('/home/ubuntu/face-detection-adversarial-attack/data/haar_xml/haarcascade_frontalface_default.xml')
print(type(face_cascade))
#fgsm = FastGradientMethod(face_cascade)



file_path = "/wider-face/WIDER_train/images/51--Dresses/51_Dresses_wearingdress_51_10.jpg"
img = cv2.imread(file_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)


#adv = fgsm.generate(image, clip_min=-1., clip_max=1.)

