import re
import os
import sys
import numpy as np
from skimage import io
from hideface import tools, imagelabels

def create_noisy_image(original_img_path, noise_epsilon, output_dir):
    """
    Given a path to an image, this function creates a new image with normally-distributed noise randomly applied

    Args:
        original_img_path: the path of the image you want to attack
        noise_epsilon: an integer giving the maximum variation (out of 255)
        output_dir: the directory where you'd like to store the attacked image
    Returns:
        noisy_file_path: the path to the noisy image you just created 
    """
    if (not os.path.isfile(original_img_path)): sys.exit("Bad input image path: " + original_img_path)
    if (not os.path.exists(output_dir)): os.makedirs(output_dir)
    eps = noise_epsilon / 255.0
    image = io.imread(original_img_path)
    pixels = np.copy(np.asarray(image).astype(np.float32))
    pixels /= 255.0
    noisy_pixels = pixels + eps * np.sign(np.random.normal(loc=0.0, scale=1.0, size=pixels.shape))
    noisy_pixels = np.clip(noisy_pixels, 0.0, 1.0)
    noisy_pixels *= 255
    noisy_pixels = np.asarray(noisy_pixels).astype(np.uint8)
    file_num = re.findall(r'[0-9]+_[0-9]+\.jpg', original_img_path)[0][:-4]
    file_name = 'unlabeled_noisy_eps'+str(noise_epsilon)+'_'+file_num+'.jpg'
    noisy_file_path = os.path.join(output_dir,file_name)
    io.imsave(noisy_file_path,noisy_pixels)
    return noisy_file_path

