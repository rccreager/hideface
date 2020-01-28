import os
from skimage import io
from cleverhans.attacks import FastGradientMethod
import cleverhans
import tensorflow as tf

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

file_path = "/s3mnt/WIDER_train/images/51--Dresses/51_Dresses_wearingdress_51_10.jpg"
img = io.imread(file_path)



#img = cv2.imread(file_path)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#faces = face_cascade.detectMultiScale(gray, 1.3, 5)


#adv = fgsm.generate(image, clip_min=-1., clip_max=1.)

