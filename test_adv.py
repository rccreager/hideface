###
# Copied from Cleverhans 
# Testing
###

import os
from skimage import io
import tensorflow as tf

def load_images(input_dir, batch_shape):
    """Read jpg images from input directory in batches.
    Args:
        input_dir: input directory
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
        filenames: list of file names (without path) of each image
            Length of this list could be less than batch_size 
            In this case only first few images of the result are elements of the minibatch
        images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.jpg')):
        with tf.gfile.Open(filepath) as f:
            images[idx, :, :, :] = io.imread(f).astype(np.float) / 255.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx   > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.
  Args:
      images: array of minibatches of images
      filenames: list of filenames without path
          If number of file names in this list less than number of images in
          the minibatch then only first len(filenames) images will be saved
      output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
      with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
          io.imsave(f, images[i, :, :, :])

def main():
    """Run the noise attack"""
    max_epsilon = 16
    eps = max_epsilon / 255.0
    batch_size = 16
    image_width = 600
    image_height = 600
    input_dir = 'data/WIDER_train/images/51--Dresses'
    output_dir = 'data/noise_output'
    batch_shape = [batch_size, image_height, image_width, 3]

    with tf.Graph().as_default():
        x_input = tf.keras.Input(shape=batch_shape,dtype=tf.float32)
        noisy_images = x_input + eps * tf.sign(tf.random.normal(batch_shape))
        x_output = tf.clip_by_value(noisy_images, 0.0, 1.0)

        with tf.Session() as sess:
            for filenames, images in load_images(input_dir, batch_shape):
                out_images = sess.run(x_output, feed_dict={x_input: images})
                save_images(out_images, filenames, output_dir)


if __name__ == '__main__':
    main()



