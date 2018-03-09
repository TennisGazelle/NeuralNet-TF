# for python 2 and 3 support
from __future__ import division, print_function, unicode_literals
# pretty figures
import matplotlib
import matplotlib.pyplot as plt
# math
import numpy as np
import tensorflow as tf
# default data
from tensorflow.examples.tutorials.mnist import input_data

epsilon = 1e-9
init_sigma = 0.1
batch_size = None


def preview_mnist(mnist, n_samples = 5):
    plt.figure(figsize=(n_samples*2, 3))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i+1)
        sample_image = mnist.train.images[i].reshape(28, 28)
        plt.imshow(sample_image, cmap="binary")
        plt.axis("on")

    print(mnist.train.labels[:n_samples])

    plt.show()


def squash(vector, axis=-1, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm  = tf.reduce_sum(tf.square(vector), axis=axis, keepdims=True)
        safe_norm     = tf.sqrt(squared_norm + epsilon)  # adding small number to avoid nan issues
        squash_factor = squared_norm / (1.0 + squared_norm)
        unit_vector   = vector / safe_norm
        return squash_factor * unit_vector


def safe_norm(vector, axis=-1, keepdims=False, name=None):
    with tf.name_scope(name=name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(vector), axis=axis, keepdims=keepdims)
        return tf.sqrt(squared_norm + epsilon)

if __name__ == '__main__':
    # reset tf (to not have to restart the kernel?)
    tf.reset_default_graph()

    np.random.seed(42)
    tf.set_random_seed(42)

    mnist = input_data.read_data_sets("data/mnist/")
    # preview_mnist(mnist, 5)

    ### SET UP THE INPUT LAYER ###
    # 28 * 28 images with a single channel (grayscale)
    X = tf.placeholder(dtype=tf.float32, shape=[batch_size, 28, 28, 1], name="X")

    ### SET UP THE PRIMARY CAPS LAYER ###
    primary_caps_n_map = 32
    primary_caps_n_caps = primary_caps_n_map * 6 * 6
    primary_caps_n_dims = 8
    conv1_params = {
        "filters": 256,
        "kernel_size": 9,
        "strides": 1,
        "padding": "valid",  # this is actually no padding
        "activation": tf.nn.relu
    }
    conv2_params = {
        "filters": primary_caps_n_map * primary_caps_n_dims, # 256 conv. filters as well
        "kernel_size": 9,
        "strides": 2,  # to help get down to 6 by 6 outputs
        "padding": "valid",
        "activation": tf.nn.relu
    }
    conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)      # 28x28 ==> 20x20
    print(conv1)
    conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)  # 20x20 ==> 12x12 (with stride of two, it's 6x6)
    print(conv2)

    ### RESHAPE THE OUTPUT TO LONG ARRAY OF VECTORS ###
    caps1_raw = tf.reshape(conv2, shape=[-1, primary_caps_n_caps, primary_caps_n_dims], name="caps1_raw")
    print(caps1_raw)
    caps1_output = squash(caps1_raw, name="caps1_output")
    print(caps1_output)

    ### SET UP THE DIGIT CAPS LAYER ###
    # 2D matrix of 2D Matrices (each W_ij for all 1152 vectors, for all 10 digit caps)
    # Outer matrix dims: 1152 x 10
    # each matrix sub-dims: 8 x 16
    digit_caps_n_caps = 10
    digit_caps_n_dims = 16
    W_init = tf.random_normal(
        shape=[1, primary_caps_n_caps, digit_caps_n_caps, digit_caps_n_dims, primary_caps_n_dims],
        stddev=init_sigma,
        dtype=tf.float32,
        name="W_init"
    )
    W = tf.Variable(W_init, name="W")
    print(W)
    # now create each one for each batch
    W_tiled = tf.tile(W, [tf.shape(X)[0], 1, 1, 1, 1], name="W_tiled")
    print(W_tiled)

    # vector array needs to be repeated along the ten capsule digits (and then depth wise for batching)
    caps1_output_expanded = tf.expand_dims(caps1_output, axis=-1, name="caps1_output_expanded")     # batching
    caps1_output_tile     = tf.expand_dims(caps1_output_expanded, axis=2, name="caps1_output_tile") # setting up
    caps1_output_tiled    = tf.tile(caps1_output_tile, [1, 1, digit_caps_n_caps, 1, 1], name="caps1_output_tiled")
    print(caps1_output_tiled)

    #u_hat = w * u
    caps2_prediction = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_prediction")
    print(caps2_prediction)

    ### ROUTING BY AGREEMENT ### MAIN ALGORITHM IN PAPER ###
    b_weights = tf.zeros(shape=[tf.shape(X)[0], primary_caps_n_caps, digit_caps_n_caps, 1, 1], dtype=np.float32, name="b_raw_weights")
    # these weights have extra dimensions so that it will match with the caps2_prediction sizes
    # this way, these will use TF's built in BROADCASTING function

    # round 1 (we will loop this later)
    # start loop here
    c_weights = tf.nn.softmax(b_weights, axis=2, name="c_weights")
    # s = sum( c_ij*u_hat_ji ) across i
    all_predictions = tf.multiply(c_weights, caps2_prediction, name="all_predictions")
    s = tf.reduce_sum(all_predictions, axis=1, keepdims=True, name="s")
    # v = squish
    digit_caps_output = squash(s, axis=-2, name="digit_caps_output")
    print(digit_caps_output)

    # update the b_weights
    digit_caps_output_tiled = tf.tile(digit_caps_output, [1, primary_caps_n_caps, 1, 1, 1], name="digit_caps_output_tiled")
    print(digit_caps_output_tiled)
    scalar_products = tf.matmul(caps2_prediction, digit_caps_output_tiled, transpose_a=True, name="agreement")
    b_weights = tf.add(b_weights, scalar_products, name="b_weights")
    # end loop here

    ### GET CLASS GUESS ### ESTIMATED CLASS PROBABIILITIES (LENGTH) ###
    y_probability = safe_norm(vector=digit_caps_output, axis=-2, name="y_probability")
    print(y_probability)  # to double check the dimensions

    y_argmax = tf.argmax(y_probability, axis=2, name="y_probability")
    print(y_argmax)       # to double check the argmax
    # let's get rid of dimensions that have size 1 (since we have the index of the vector that we like
    y_pred = tf.squeeze(y_argmax, axis=[1,2], name"y_pred")

    # placeholder for the labels
    y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

    # since y will contain the digit classes (from 0 to 9), to get T_k for every class instance...
    T = tf.one_hot(y, depth=digit_caps_n_caps, name="T")

    

