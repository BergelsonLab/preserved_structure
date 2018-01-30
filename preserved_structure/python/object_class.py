import os
import json
import numpy as np
from scipy.spatial.distance import cosine


from object_vectors import *


def class_variance(class_name):
    """
    Calculate the variance in the distribution of vector
    for a given class

    :param class_name: name of object class
    :return: class variance
    """
    vecs = lookup_class(class_name)
    return np.var(vecs)


def confusability(class_name):
    """
    Calculate the confusability score for a given category of
    object

    :param class_name: name of object class
    :return: confusability score
    """
    target_vecs = lookup_class(class_name)
    # summed_target =np.sum(target_vecs, axis=0)
    # summed_target = np.mean(target_vecs, axis=0)
    summed_target = find_central_vector(class_name)[0]
    # other_sums = [np.sum(lookup_class(x), axis=0)
    #                 for x in seedlings_categories
    #                     if x != class_name]
    # other_sums = [np.mean(lookup_class(x), axis=0)
    #               for x in seedlings_categories
    #               if x != class_name]
    other_sums = [find_central_vector(x)[0]
                  for x in seedlings_categories
                  if x != class_name]
    confuse_score = sum([cosine(summed_target, x) for x in other_sums])
    return confuse_score


def lookup_class(class_name):
    """
    Look up the path to the .npy file which stores the
    list of vectors for a given object class

    :param class_name: name of object class
    :return:
    """
    vector_path = os.path.join(os.path.dirname(__file__),
                               vectors_dir,
                               "{}.npy".format(class_name))
    std_v = np.load(vector_path)
    std_v = np.vsplit(std_v, std_v.shape[0])
    return std_v


def find_central_vector(class_name):
    vecs = lookup_class(class_name)
    minimal = (None, 100000000)
    for i in range(len(vecs)):
        s = 0
        for j in range(len(vecs)):
            if j != i:
                s += cosine(vecs[i], vecs[j])
        if s < minimal[1]:
            minimal = (vecs[i], s)
    return minimal


def wordvecs(words, word_dict, np_vectors):
    with open(word_dict, "rU") as input:
        vocab = json.load(input)
    vectors = np.load(np_vectors)
    wordvecs = []
    for word in words:
        vec = vectors[vocab[word], :]
        wordvecs.append((word, vec))
    return wordvecs


def wordvec_confusability(vecs):
    results = []
    for vec in vecs:
        confuse_score = 0
        for other in vecs:
            if vec[0] != other[0]:
                confuse_score += cosine(vec[1], other[1])
        results.append((vec[0], confuse_score))
    return results


# confusability("apple")
