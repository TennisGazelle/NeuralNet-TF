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

def make_model():
    # reset tf (to not have to restart the kernel?)
    tf.reset_default_graph()

    np.random.seed(42)
    tf.set_random_seed(42)

    ### SET UP THE INPUT LAYER ###
    # 28 * 28 images with a single channel (grayscale)
    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name="X")

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
    y_pred = tf.squeeze(y_argmax, axis=[1, 2], name="y_pred")

    # placeholder for the labels
    y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

    # since y will contain the digit classes (from 0 to 9), to get T_k for every class instance...
    T = tf.one_hot(y, depth=digit_caps_n_caps, name="T")

    digit_caps_output_norm = safe_norm(digit_caps_output, axis=-2, keepdims=True, name="digit_caps_output_norm")
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    # Note: on following lines for the T_k values, m_* is a double which gets upgraded to a tensor.
    # Therefore, the output of m_plus - digit_caps_output_norm (or the other one) returns a tensor at the end
    # How it maps it is still to be understood by TensorFlow...

    # t_k == 1
    class_present_raw = tf.square(tf.maximum(0.0, m_plus - digit_caps_output_norm), name="class_present_raw_error")
    class_present_error = tf.reshape(class_present_raw, shape=(-1, 10), name="class_present_error")
    # t_k = 0
    class_absent_raw = tf.square(tf.maximum(0.0, digit_caps_output_norm - m_minus), name="class_absent_raw_error")
    class_absent_error = tf.reshape(class_absent_raw, shape=(-1, 10), name="class_absent_error")
    # L = sum of all errors
    L = tf.add(T*class_present_error, (1.0 - T) * lambda_ * class_absent_error, name="L")
    print(L)
    # final margin loss for each instance
    margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
    print(margin_loss)


    ### ADDING MASK for RECONSTRUCTION ### TO LATER FEED IT TO THE DECODER ###
    # tell tensorflow to mask the output labels, based on whether or not the predicted output was true
    mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")
    reconstruction_targets = tf.cond(mask_with_labels,
                                     lambda: y,
                                     lambda: y_pred,
                                     name="reconstruction_targets")

    # let's create the mask
    reconstruction_mask = tf.one_hot(reconstruction_targets, depth=digit_caps_n_caps, name="reconstruction_mask")
    print(reconstruction_mask)
    # let's reshape it so that it can be multiplied (for the masking) against the digit caps output
    reconstruction_mask_reshaped = tf.reshape(reconstruction_mask, shape=[-1, 1, digit_caps_n_caps, 1, 1], name="reconstruction_mask_reshaped")
    print(reconstruction_mask_reshaped)
    # apply the mask
    digit_caps_output_masked = tf.multiply(digit_caps_output, reconstruction_mask_reshaped, name="digit_caps_output_masked")
    print(digit_caps_output_masked)
    # one last reshape to flatten the input
    decoder_input = tf.reshape(digit_caps_output_masked, shape=[-1, digit_caps_n_caps * digit_caps_n_dims], name="decoder_input")

    ### DECODER ###
    # two fully connected (ReLU) layers, then one dense (sigmoid) layer
    n_hidden1 = 512
    n_hidden2 = 1024
    n_output = 28 * 28

    with tf.name_scope("decoder"):
        hidden_1 = tf.layers.dense(decoder_input, n_hidden1, activation=tf.nn.relu,    name="hidden_1")
        hidden_2 = tf.layers.dense(hidden_1,      n_hidden2, activation=tf.nn.relu,    name="hidden_2")
        decoder_output = tf.layers.dense(hidden_2, n_output, activation=tf.nn.sigmoid, name="decoder_output")

    # this is the reconstruction loss which is just the squared differences between the image and the truth
    x_flat = tf.reshape(X, shape=[-1, n_output], name="x_flat")
    squared_diff = tf.square(x_flat - decoder_output, name="squared_diff")
    reconstruction_loss = tf.reduce_mean(squared_diff, name="reconstruction_loss")
    print(reconstruction_loss)

    # make the total loss to minimize
    # end of the model
    alpha = 0.0005
    total_loss = tf.add(margin_loss, alpha * reconstruction_loss, name="total_loss")
    print(total_loss)

    ### SAVE ACCURACY ### TRAIN ### ADD SAVER ###
    num_correct = tf.equal(y, y_pred, name="num_correct")
    accuracy = tf.reduce_mean(tf.cast(num_correct, tf.float32), name="accuracy")
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(total_loss, name="training_op")
    init = tf.global_variables_initializer()
    saver = tf.train.Saver

    return total_loss, accuracy, training_op, init, saver


if __name__ == '__main__':
    mnist = input_data.read_data_sets("data/mnist/")
    # preview_mnist(mnist, 5)
    total_loss, accuracy, training_op, init, saver = make_model()

    n_epochs = 10
    batch_size = 50
    restore_checkpoint = True

    n_iterations_per_epoch = mnist.train.num_examples // batch_size  # '//' is floordiv()
    n_iterations_validation = mnist.validation.num_examples // batch_size
    best_loss_val = np.infty
    checkpoint_path = "./capsnet_checkpoint"

    with tf.Session() as sess:
        if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
            saver.restore(sess, checkpoint_path)
        else:
            init.run()

        for epoch in range(n_epochs):
            for iter in range(1, n_iterations_per_epoch + 1):
                X_batch, Y_batch = mnist.train.next_batch(batch_size)
                # run the training operations and measure the loss  for this
                _, loss_train = sess.run(
                    [training_op, total_loss],
                    feed_dict={
                        X: X_batch.reshape([-1, 28, 28, 1]),
                        Y: Y_batch,
                        mask_with_labels: True
                    }
                )
                print ("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                    iter, n_iterations_per_epoch,
                    iter * 100 / n_iterations_per_epoch,
                    loss_train
                ), end="")

