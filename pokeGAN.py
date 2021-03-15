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

def generator(input, random_dim, is_train, reuse = False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 #Channel numbers
    s4 = 4
    output_dim = CHANNEL
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape = [random_dim, s4 * s4 * c4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape = [c4 * s4 * s4], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')

        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='flat_conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*256
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size = [5, 5], strides=[2, 2], padding="SAME", kernel_initializer = tf.truncated_normal_initializer(stddev=0.02), name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        #16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size = [5, 5], strides=[2, 2], padding="SAME", kernel_initializer = tf.truncated_normal_initializer(stddev=0.02), name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        #32*32*64
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size = [5, 5], strides=[2, 2], padding="SAME", kernel_initializer = tf.truncated_normal_initializer(stddev=0.02), name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn3, name='act4')
        #64*64*32
        conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size = [5, 5], strides=[2, 2], padding="SAME", kernel_initializer = tf.truncated_normal_initializer(stddev=0.02), name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn3, name='act5')
        #128*128*3
        conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size = [5, 5], strides=[2, 2], padding="SAME", kernel_initializer = tf.truncated_normal_initializer(stddev=0.02), name='conv6')
        #bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn6')
        act6 = tf.nn.tanh(conv6, name='act6')
        return act6
