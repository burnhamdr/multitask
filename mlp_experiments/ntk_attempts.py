import tensorflow as tf
import numpy as np
import h5py
import os
from tensorflow.examples.tutorials.mnist import input_data

# Force CPU (if needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 1) Check TF version
print("TensorFlow version:", tf.__version__)
assert tf.__version__.startswith('1.13'), "This script requires TensorFlow 1.13"

# 2) Load initial weights & biases from your .h5
h5_file = '../../models/mlp_models/paper_unif/model_weights.h5'
with h5py.File(os.path.join(os.getcwd(), h5_file), 'r') as f:
    W1_init = np.array(f['dense_14']['dense_14']['kernel:0'])
    b1_init = np.array(f['bias_only_layer_14']['bias_only_layer_14']['bias:0'])
    W2_init = np.array(f['dense_15']['dense_15']['kernel:0'])
    b2_init = np.array(f['bias_only_layer_15']['bias_only_layer_15']['bias:0'])

# Dimensions
d, H = W1_init.shape  # d=784, H=30000
H, C = W1_init.shape[1], W2_init.shape[1]

# 3) Load GD‐trained biases (optional, for comparison)
biases = np.load(os.path.join(os.getcwd(), '../../models/mlp_models/paper_unif/Mnist_biases.npy'),
                 allow_pickle=True)
b1_trained, b2_trained = biases

# 4) MNIST subset
mnist   = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train, y_train = mnist.train.images[:1000], mnist.train.labels[:1000]
x_test,  y_test  = mnist.test.images[:200],   mnist.test.labels[:200]
N = x_train.shape[0]

# Evaluate updated network
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, d])
net = tf.layers.dense(X, units=H, activation=tf.nn.relu,
                      kernel_initializer=tf.constant_initializer(W1_init),
                      bias_initializer=tf.constant_initializer(b1_trained))
logits = tf.layers.dense(net, units=C,
                         kernel_initializer=tf.constant_initializer(W2_init),
                         bias_initializer=tf.constant_initializer(b2_trained))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_pred = sess.run(logits, feed_dict={X: x_test})
    acc_ntk_params = (y_pred.argmax(axis=1) == y_test.argmax(axis=1)).mean()
print("Test accuracy trained biases:", acc_ntk_params)

# --- NTK computation, bias-only --------------------------------------
def compute_bias_only_ntk(X1, X2, W1, W2):
    # Masks for ReLU activity
    M1 = (X1.dot(W1) > 0).astype(np.float32)
    M2 = (X2.dot(W1) > 0).astype(np.float32)

    # J_b1 per‐sample: (N, H, C) → flatten to (N, H*C)
    J1_b1 = np.einsum('ij,jk->ijk', M1, W2).reshape(X1.shape[0], -1)
    J2_b1 = np.einsum('ij,jk->ijk', M2, W2).reshape(X2.shape[0], -1)

    # J_b2 per‐sample: derivative of each logit wrt its bias
    J1_b2 = np.ones((X1.shape[0], W2.shape[1]), dtype=np.float32)
    J2_b2 = np.ones((X2.shape[0], W2.shape[1]), dtype=np.float32)

    J1 = np.hstack([J1_b1, J1_b2])
    J2 = np.hstack([J2_b1, J2_b2])
    return J1.dot(J2.T)

# Build NTK matrices once
theta_train      = compute_bias_only_ntk(x_train, x_train, W1_init, W2_init)
theta_test_train = compute_bias_only_ntk(x_test,  x_train, W1_init, W2_init)

# Regularization
lambda_reg = 1e-6

# --- 1) Kernel‐regression on labels ----------------------------------
alpha_kr = np.linalg.solve(theta_train + lambda_reg * np.eye(N), y_train)
y_pred_kr = theta_test_train.dot(alpha_kr)
acc_kernel = (y_pred_kr.argmax(axis=1) == y_test.argmax(axis=1)).mean()
print("NTK kernel‐regression test accuracy:", acc_kernel)

# --- 2) One‐shot NTK bias update ------------------------------------
# Compute initial network outputs f0 on training set
tf.reset_default_graph()
X0 = tf.placeholder(tf.float32, [None, d])
net0 = tf.layers.dense(X0, units=H, activation=tf.nn.relu,
                       kernel_initializer=tf.constant_initializer(W1_init),
                       bias_initializer=tf.constant_initializer(b1_init))
f0   = tf.layers.dense(net0, units=C,
                       kernel_initializer=tf.constant_initializer(W2_init),
                       bias_initializer=tf.constant_initializer(b2_init))
with tf.Session() as sess0:
    sess0.run(tf.global_variables_initializer())
    f0_train = sess0.run(f0, feed_dict={X0: x_train})

residual = y_train - f0_train
alpha_ntk = np.linalg.solve(theta_train + lambda_reg * np.eye(N), residual)

# Compute Jacobians for biases
def compute_bias_jacobian(X, W1, W2):
    N, D = X.shape
    H, C = W1.shape[1], W2.shape[1]
    mask = (X.dot(W1) > 0).astype(np.float32)
    J_b1 = np.einsum('ij,jk->ijk', mask, W2).reshape(N, H*C)
    J_b2 = np.ones((N, C), dtype=np.float32)
    return np.hstack([J_b1, J_b2])

J_train = compute_bias_jacobian(x_train, W1_init, W2_init)
J1_b1 = J_train[:, :H*C].reshape(N, H, C)
J1_b2 = J_train[:, H*C:]

# Aggregate parameter updates
delta_b1 = np.zeros(H, dtype=np.float32)
delta_b2 = np.zeros(C, dtype=np.float32)
for k in range(C):
    delta_b1 += J1_b1[:, :, k].T.dot(alpha_ntk[:, k])
    delta_b2[k] = J1_b2[:, k].dot(alpha_ntk[:, k])

# Apply learning rate
eta = 0.01
b1_ntk = b1_init + eta * delta_b1
b2_ntk = b2_init + eta * delta_b2

# Evaluate updated network
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, d])
net = tf.layers.dense(X, units=H, activation=tf.nn.relu,
                      kernel_initializer=tf.constant_initializer(W1_init),
                      bias_initializer=tf.constant_initializer(b1_ntk))
logits = tf.layers.dense(net, units=C,
                         kernel_initializer=tf.constant_initializer(W2_init),
                         bias_initializer=tf.constant_initializer(b2_ntk))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_pred = sess.run(logits, feed_dict={X: x_test})
    acc_ntk_params = (y_pred.argmax(axis=1) == y_test.argmax(axis=1)).mean()
print("Test accuracy after one‐shot NTK bias update:", acc_ntk_params)
