import PIL
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception

import numpy as np

from object_vectors import *


########################
# set these variables: #
########################
models_dir = 'models'
height = 299
width = 299
channels = 3


def calc_vectors(sess, image_path, pool=pool1):
    """
    Return vector representations for a directory filled with images

    sess: a TF session object passed in from the enclosing scope
    image_path: a directory filled with images
    """
    return forward_net_pass(sess, read_and_resize(image_path), pool)


def setup_model():
    # Create graph
    X = tf.placeholder(tf.float32, shape=[None, height, width, channels])

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        net, end_points = inception.inception_v3(
            X, num_classes=1001, is_training=False)

    saver = tf.train.Saver()

    return saver


def forward_net_pass(sess, image_list, pool=pool1):
    """
    Pass a list of raw image data through the network, return activations

    sess: TF session object loaded in the enclosing scope
    image_list: list of tuples of (filename, resized image data)
    returns: final pooling layer activations.
    """
    # we'll fill this list with tuples of (filename, feature vector)
    vectors = []

    # TODO: have this be parameterized by model, and pull earlier layers too
    # this is the graph node with the tensor we're interested in
    repr_tensor = sess.graph.get_tensor_by_name(pool)

    for x in image_list:
        vectors.append(
            (x[0], np.squeeze(sess.run(repr_tensor, {"Placeholder:0": x[1]}))))
    return vectors


def read_and_resize(image_path):
    """
    Collect all the jpeg files in a directory, read and resize them, then return
    the raw data in a list of tuples of (filename, raw_data)

    image_path: directory filled with jpeg images
    """
    # parse all the jpeg files in the 'image_path' folder into
    files = [os.path.join(image_path, x) for x in os.listdir(
        image_path) if (x.endswith(".jpg") or x.endswith(".jpeg"))]
    return [(x, load_resized_image(x)) for x in files]


def load_resized_image(path):
    """
    path: full path to image file
    returns: resize raw data for image
    """
    img = PIL.Image.open(path)
    newImg = img.resize((299, 299), PIL.Image.BILINEAR).convert("RGB")
    data = np.array(newImg.getdata())
    return 2 * (data.reshape((1, newImg.size[0], newImg.size[1], 3)).astype(np.float32) / 255) - 1
