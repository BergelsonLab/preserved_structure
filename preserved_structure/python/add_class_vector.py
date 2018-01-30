import os
import json
import sys
import calc_vector
import argparse
import csv
import numpy as np
import tensorflow as tf

from object_vectors import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# this is where the class specific vector dictionaries are stored
vectors_dir = "seedlings_vectors"

network_layer = pool1


asterisk = "\n{}\n".format("*" * 50)

# main function for the script


def main(class_name=None, image_dir=None, class_map_file=None):
    """
    If class_map_file is supplied, then we recalculate all the class dictionaries in that map.
    Otherwise, we expect a single path to a folder filled with images, and a single class 
    name.

    class_name: name of the class (for images in image_path)
    image_dir: path to directory filled with images of a given class
    class_map_file: path to a csv file containing class name/image directory pairs
    """
    if image_dir and class_name:
        class_map = {class_name: image_dir}
    if class_map_file:
        class_map = load_class_map(class_map_file)

    print "\nBuilding vector dictionaries for: \n"
    saver = calc_vector.setup_model()
    with tf.Session() as sess:
        # TODO: have this be parameterized by model
        saver.restore(sess, os.path.join(models_dir, "inception_v3.ckpt"))
        for class_name, img_dir in class_map.items():
            print "\t{}:  {}".format(class_name, img_dir)
            vectors = calc_vector.calc_vectors(
                sess, img_dir, pool=network_layer)
            vector_path, dict_path = get_path(class_name)
            save_vector(vectors, vector_path, dict_path, class_name)


def load_class_map(class_map):
    """
    class_map: a path to a csv file containing the category_name to image_directory pairs
    returns: a dict of those pairs
    """
    categ_dict = {}
    with open(class_map, "rU") as input:
        reader = csv.reader(input)
        for row in reader:
            categ_dict[row[0]] = row[1]
    return categ_dict

# save diciontary and vector to vector folder


def save_vector(vectors, vector_path, dict_path, class_name):
    vec_dict = {}
    if os.path.exists(dict_path):
        print "{0} Existing dictionary found for: {1}{0}".format(asterisk, class_name)
        with open(dict_path) as reader:
            vec_dict = json.load(reader)

    if os.path.exists(vector_path):
        print "{0} Existing vector array found for: {1}{0}".format(asterisk, class_name)
        arr = np.load(vector_path)
        row, col = arr.shape
        for v in vectors:
            vec_dict[str(row)] = v[0]
            arr = np.append(arr, [v[1]], axis=0)
            row += 1
    else:
        arr = np.zeros([0, 2048])
        row = 0
        for v in vectors:
            vec_dict[str(row)] = v[0]
            arr = np.append(arr, [v[1]], axis=0)
            row += 1

    # save array and dictionary
    np.save(vector_path, arr)
    print "{0} {1} vectors added to: {2}{0}".format(asterisk, repr(len(vectors)), vector_path)

    with open(dict_path, 'w') as writer:
        json.dump(vec_dict, writer)


def get_path(class_name):
    if not os.path.exists(vectors_dir):
        os.makedirs(vectors_dir)

    vector_path = os.path.join(vectors_dir, "{}.npy".format(class_name))
    dict_path = os.path.join(vectors_dir, "{}.json".format(class_name))

    return vector_path, dict_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate a dictionary of vectors for a given category")
    parser.add_argument("--class_map", nargs=1,
                        help="a csv file containing a list of class_name to image_directory pairs", metavar="CLASS_MAP.csv")
    parser.add_argument("--single_class", nargs=2, help="a single CLASS_NAME and IMAGE_DIR pair, in that order",
                        action="append", metavar=("CLASS_NAME", "IMAGE_DIR"))
    args = parser.parse_args()

    if args.class_map:
        main(None, None, args.class_map[0])
    elif args.single_class:
        main(class_name=args.single_class[0][0],
             image_dir=args.single_class[0][1],
             class_map_file=None)
