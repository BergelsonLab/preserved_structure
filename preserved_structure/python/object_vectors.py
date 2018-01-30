import os

# this is where we'll dump the centrality scores
output_dir = "output"
# this is where the class specific vector dictionaries are stored
vectors_dir = "seedlings_vectors"
# this is where the TensorFlow checkpoints are
models_dir = "models_local"
# the seedlings image directories
seedlings_stimuli = "seedlings_stimuli"

seedlings_categories = filter(lambda x: (
    not x.startswith(".")), os.listdir(seedlings_stimuli))

pool1 = "InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0"
pool2 = "InceptionV3/InceptionV3/MaxPool_5a_3x3:0"
