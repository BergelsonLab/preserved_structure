import numpy as np
import tensorflow as tf

import calc_vector


from object_vectors import *

pool = "InceptionV3/Predictions/Softmax:0"


labelfile = "inference_model/imagenet_slim_labels.txt"
labels = []
with open(labelfile, "rU") as input:
    for line in input:
        labels.append(line.replace("\n", ""))


def run_inference(images, num_top_predictions):
    saver = calc_vector.setup_model()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(models_dir, "inception_v3.ckpt"))
        repr_tensor = sess.graph.get_tensor_by_name(pool)

        for x in images:
            print("\n\n{}: \n".format(os.path.basename(x)))
            x = calc_vector.load_resized_image(x)

            predictions = np.squeeze(sess.run(repr_tensor, {"Placeholder:0": x}))

            top_k = predictions.argsort()[-num_top_predictions:][::-1]

            for node_id in top_k:
                human_string = labels[node_id]
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))