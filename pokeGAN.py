# Generating new Dicom Images

import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
from utils import *

slim = tf.contrib.slim

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 64
EPOCH = 5000
version = 'newDicom'
newDicom_path = './' + version

def lrelu(x, n, leak=0.2):
    return tf.maximum(x, leak*x, name=n)

def process_data():
    current_dir = os.getcwd()
    dicom_dir = os.path.join(current_dir, 'data')
    images = []
    for each in os.listdir(dicom_dir):
        images.append(os.path.join(dicom_dir, each))
    all_images = tf.convert_to_tensor(images, dtype = tf.string)

    images_queue = tf.train.slice_input_producer([all_images])

    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)

    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT, WIDTH, CHANNEL])

    image = tf.casr(image, tf.float32)
    image = image / 255.0

    images_batch = tf.train.shuffle_batch([image], batch_size = BATCH_SIZE, num_threads = 4, capacity = 200 + 3 * BATCH_SIZE, min_after_dequeue = 200)
    num_images = len(images)
    return images_batch, num_images
    
