import re
import os
import sys
import copy
import numpy as np
from skimage import io,util
from PIL import Image
from hideface import tools, imagelabels


def create_noisy_image(original_img_path, noise_epsilon, output_dir, use_mult_noise=False):
    """
    Given a path to an image, create a new image with normally-distributed noise randomly applied

    Args:
        original_img_path: the path of the image you want to attack
        noise_epsilon: an integer giving the maximum variation (out of 255)
        output_dir: the directory where you'd like to store the attacked image
        use_mult_noise: a bool indicating whether to use multiplicative noise
    Returns:
        image_file_path: the path to the output image you just created 
        noise_file_path: the path to the noise image you created
    """
    if (not os.path.isfile(original_img_path)): 
        sys.exit('Bad input image path: ' + original_img_path)
    if (not os.path.exists(output_dir)): os.makedirs(output_dir)
    try:
        image = Image.open(original_img_path).convert("RGB")
        image.verify() # verify that it is, in fact an image
    except (IOError, SyntaxError) as e:
        print('Input Image Read Error (create_noisy_image): ' + original_img_path)
        return '',''
    image = copy.deepcopy(np.array(image))
    pixels = np.asarray(image).astype(np.float32)
    pixels /= 255.0
    noise = (noise_epsilon/255.0) * np.random.normal(loc=0.0, scale=1.0, size=pixels.shape)
    image_pixels = pixels * (1.0 + noise) if (use_mult_noise) else pixels + noise
    image_pixels = np.clip(image_pixels, 0.0, 1.0)
    image_pixels *= 255
    image_pixels = np.asarray(image_pixels).astype(np.uint8)
    noise = np.clip(noise, 0.0, 1.0)
    noise *= 255
    noise = np.asarray(noise).astype(np.uint8)
    file_num = re.findall(r'[0-9]+_[0-9]+\.jpg', original_img_path)[0][:-4]
    noise_str = 'multnoise' if (use_mult_noise) else 'addnoise'
    image_file_name = 'unlabeled_'+noise_str+'_eps'+str(noise_epsilon)+'_'+file_num+'.jpg'
    image_file_path = os.path.join(output_dir,image_file_name)
    io.imsave(image_file_path,image_pixels)
    noise_file_name = noise_str+ '_'+str(noise_epsilon)+'_'+file_num+'.jpg'
    noise_file_path = os.path.join(output_dir,noise_file_name)
    io.imsave(noise_file_path,noise)
    return image_file_path, noise_file_path

