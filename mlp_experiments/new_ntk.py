# Dataset imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
import h5py
import psutil
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Disable TensorFlow warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# record script start time
START_TIME = time.time()

# Reproducibility
SEED = 10
tf.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Control dataset download: pre-cache above, then rely on local cache
DO_DOWNLOAD = True

USE_FULL_KERNEL = True  # if True, use explicit kernel solve, otherwise use CG
# Select NTK mode: 'bias' or 'full'
NTK_MODE = 'bias'  # change to 'full' to run the full-NTK solution

# Device detection (TF 1.x style)
config = tf.ConfigProto()
if tf.test.is_gpu_available():
    print("GPU available")
    config.gpu_options.allow_growth = True
else:
    print("Using CPU")

print(f"TensorFlow version: {tf.__version__}")

# === Memory stats ===
try:
    vm = psutil.virtual_memory()
    print(f"System RAM: total={vm.total/1e9:.2f} GB, available={vm.available/1e9:.2f} GB")
except Exception as e:
    print(f"Could not fetch system RAM info: {e}")

# === Configuration ===
N_CLASSES   = 10
N_PER_CLASS = 60   # samples per class
N_TEST      = 5000  # number of test samples per dataset
INPUT1      = 28    # reduced dimension
INPUT_DIM   = INPUT1 * INPUT1  # reduced images
N_HIDDEN    = 30000  # fixed number of hidden units

# List of datasets to load
DATASETS = [
    'mnist',
    'fashion-mnist',
    # 'kmnist',  # May not be available in older sklearn/openml
]
LoadDatasets = [1, 0]  # Flags: 1 to include, 0 to skip

print(f"Using {N_HIDDEN} hidden units")
print(f"Ratio of hidden units to N_CLASSES*N_PER_CLASS: {N_HIDDEN / (N_CLASSES * N_PER_CLASS):.2f}")


def load_and_preprocess_data(dataset_name):
    """Load and preprocess dataset using sklearn/openml or TensorFlow MNIST"""
    print(f"Loading {dataset_name}...")
    
    if dataset_name == 'mnist':
        # Use TensorFlow's MNIST loader
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)  # Changed to one_hot=False
        
        # Get all training and test data
        x_train_full = mnist.train.images  # Shape: (55000, 784)
        y_train_full = mnist.train.labels.astype(np.int32)  # Shape: (55000,)
        x_test_full = mnist.test.images    # Shape: (10000, 784)
        y_test_full = mnist.test.labels.astype(np.int32)    # Shape: (10000,)
        
        # Combine for consistent processing
        X = np.concatenate([x_train_full, x_test_full], axis=0)
        y = np.concatenate([y_train_full, y_test_full], axis=0)
        
        # Data is already normalized to [0,1] and flattened
        # Reshape to 28x28 for resizing
        X = X.reshape(-1, 28, 28)
        
    elif dataset_name == 'fashion-mnist':
        # Use sklearn/openml
        data = fetch_openml('Fashion-MNIST', version=1, cache=True)
        X, y = data.data.astype(np.float32), data.target.astype(np.int32)
        
        # Normalize to [0, 1]
        X = X / 255.0
        
        # Reshape to 28x28
        X = X.reshape(-1, 28, 28)
        
    elif dataset_name == 'kmnist':
        # Use sklearn/openml
        data = fetch_openml('Kuzushiji-MNIST', version=1, cache=True)
        X, y = data.data.astype(np.float32), data.target.astype(np.int32)
        
        # Normalize to [0, 1]
        X = X / 255.0
        
        # Reshape to 28x28
        X = X.reshape(-1, 28, 28)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Simple resize using numpy (bilinear-like interpolation)
    def resize_images(images, new_size):
        old_size = images.shape[1]
        factor = old_size / new_size
        resized = np.zeros((images.shape[0], new_size, new_size), dtype=np.float32)
        for i in range(new_size):
            for j in range(new_size):
                old_i = int(i * factor)
                old_j = int(j * factor)
                # Ensure indices are within bounds
                old_i = min(old_i, old_size - 1)
                old_j = min(old_j, old_size - 1)
                resized[:, i, j] = images[:, old_i, old_j]
        return resized
    
    # Resize to INPUT1 x INPUT1 and flatten
    X_resized = resize_images(X, INPUT1)
    X_flat = X_resized.reshape(-1, INPUT_DIM)
    
    print(f"Loaded {dataset_name}: {X_flat.shape[0]} samples, resized to {INPUT1}x{INPUT1}")
    
    return X_flat, y


def subsample_data(X, y, n_per_class=N_PER_CLASS, n_classes=N_CLASSES, seed=SEED):
    """Subsample n_per_class samples from each class"""
    np.random.seed(seed)
    indices = []
    
    for c in range(n_classes):
        class_indices = np.where(y == c)[0]
        if len(class_indices) >= n_per_class:
            selected = np.random.choice(class_indices, n_per_class, replace=False)
            indices.extend(selected)
    
    indices = np.array(indices)
    return X[indices], y[indices]


def subsample_test_data(X, y, n_test=N_TEST, seed=SEED):
    """Subsample test data"""
    if n_test is None or len(X) <= n_test:
        return X, y
    
    np.random.seed(seed)
    indices = np.random.choice(len(X), n_test, replace=False)
    return X[indices], y[indices]


# === Conjugate Gradient Solver ===
def cg_solve_np(A_func, b, tol=1e-6, maxiter=500):
    """Conjugate gradient solver using numpy"""
    x = np.zeros_like(b)
    r = b - A_func(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    
    for _ in range(maxiter):
        Ap = A_func(p)
        alpha = rs_old / (np.dot(p, Ap) + 1e-12)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    
    return x


class MLPModel:
    """MLP model for TensorFlow 1.x"""
    
    def __init__(self, sess, params=None):
        self.sess = sess
        self.build_model(params)
        self.sess.run(tf.global_variables_initializer())
    
    def build_model(self, params):
        # Placeholders
        self.x_ph = tf.placeholder(tf.float32, [None, INPUT_DIM], name='x_input')
        self.y_ph = tf.placeholder(tf.int32, [None], name='y_input')
        
        if params!=None:
            # Parameters
            self.W1 = tf.Variable(
                params[0],
                name='W1'
            )
            self.b1 = tf.Variable(params[1], name='b1')
            self.W2 = tf.Variable(
                params[2],
                name='W2'
            )
            self.b2 = tf.Variable(params[3], name='b2')
        else:
            # Parameters
            self.W1 = tf.Variable(
                tf.random_normal([INPUT_DIM, N_HIDDEN], stddev=np.sqrt(1.0/INPUT_DIM)),
                name='W1'
            )
            self.b1 = tf.Variable(tf.zeros([N_HIDDEN]), name='b1')
            self.W2 = tf.Variable(
                tf.random_normal([N_HIDDEN, N_CLASSES], stddev=np.sqrt(1.0/N_HIDDEN)),
                name='W2'
            )
            self.b2 = tf.Variable(tf.zeros([N_CLASSES]), name='b2')
        
        # Forward pass
        self.h = tf.nn.relu(tf.matmul(self.x_ph, self.W1) + self.b1)
        self.logits = tf.matmul(self.h, self.W2) + self.b2
        
        # Predictions
        self.predictions = tf.argmax(self.logits, axis=1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predictions, tf.cast(self.y_ph, tf.int64)), tf.float32)
        )
    
    def forward(self, x_data):
        """Forward pass returning logits and hidden activations"""
        logits, h = self.sess.run(
            [self.logits, self.h],
            feed_dict={self.x_ph: x_data}
        )
        return logits, h
    
    def get_accuracy(self, x_data, y_data):
        """Get accuracy on given data"""
        return self.sess.run(
            self.accuracy,
            feed_dict={self.x_ph: x_data, self.y_ph: y_data}
        )
    
    def get_parameters(self):
        """Get current parameter values"""
        return self.sess.run([self.W1, self.b1, self.W2, self.b2])

    def update_parameters(self, delta_params, mode='bias'):
        """Actually update the model parameters"""
        if mode == 'bias':
            # delta_params = [delta_b1, delta_b2]
            delta_b1 = delta_params[:N_HIDDEN]
            delta_b2 = delta_params[N_HIDDEN:]
            
            # Create update operations
            update_b1 = tf.assign_add(self.b1, delta_b1)
            update_b2 = tf.assign_add(self.b2, delta_b2)
            
            self.sess.run([update_b1, update_b2])
            
        elif mode == 'full':
            # delta_params = [delta_W1, delta_b1, delta_W2, delta_b2] (flattened)
            param_idx = 0
            
            # Reshape deltas back to parameter shapes
            delta_W1 = delta_params[param_idx:param_idx + N_HIDDEN * INPUT_DIM].reshape(INPUT_DIM, N_HIDDEN)
            param_idx += N_HIDDEN * INPUT_DIM
            
            delta_b1 = delta_params[param_idx:param_idx + N_HIDDEN]
            param_idx += N_HIDDEN
            
            delta_W2 = delta_params[param_idx:param_idx + N_HIDDEN * N_CLASSES].reshape(N_HIDDEN, N_CLASSES)
            param_idx += N_HIDDEN * N_CLASSES
            
            delta_b2 = delta_params[param_idx:param_idx + N_CLASSES]
            
            # Create update operations
            update_W1 = tf.assign_add(self.W1, delta_W1)
            update_b1 = tf.assign_add(self.b1, delta_b1)
            update_W2 = tf.assign_add(self.W2, delta_W2)
            update_b2 = tf.assign_add(self.b2, delta_b2)
            
            self.sess.run([update_W1, update_b1, update_W2, update_b2])


def bias_only_ntk_update_with_actual(x_train, y_train, x_test, y_test, model, epsilon=1e-6):
    """
    Bias-only NTK update with both linear approximation AND actual parameter updates
    Returns (train_acc_lin, test_acc_lin, train_acc_actual, test_acc_actual, delta_b)
    """
    # ... [Keep all the existing NTK computation code] ...
    # Forward on train
    out_init, h_init = model.forward(x_train)
    
    # Build residual and Jacobian
    P = x_train.shape[0]
    C = N_CLASSES
    
    # Targets: +1 for correct, -1 for others
    y_target = -np.ones((P, C), dtype=np.float32)
    y_target[np.arange(P), y_train] = 1.0
    res0 = (y_target - out_init).reshape(-1)
    
    # Get current parameters
    W1, b1, W2, b2 = model.get_parameters()
    
    # Build Jacobian for bias terms
    mask = (h_init > 0).astype(np.float32)
    
    Jb1 = np.zeros((P * C, N_HIDDEN), dtype=np.float32)
    for p in range(P):
        for c in range(C):
            Jb1[p * C + c, :] = mask[p, :] * W2[:, c]
    
    Jb2 = np.tile(np.eye(C, dtype=np.float32), (P, 1))
    J_bias = np.concatenate([Jb1, Jb2], axis=1)
    
    # Solve for alpha
    if USE_FULL_KERNEL:
        K = J_bias @ J_bias.T
        K += epsilon * np.eye(K.shape[0])
        alpha = np.linalg.solve(K, res0)
    else:
        alpha = cg_solve_np(lambda v: J_bias @ (J_bias.T @ v) + epsilon * v, res0)
    
    # Compute delta_b and linearized accuracies
    delta_b = J_bias.T @ alpha
    out_lin = (out_init.reshape(-1) + (J_bias @ delta_b)).reshape(P, C)
    train_acc_lin = np.mean(out_lin.argmax(axis=1) == y_train)
    
    # Test set linearized
    out_test_init, h_test_init = model.forward(x_test)
    P_test = x_test.shape[0]
    
    mask_test = (h_test_init > 0).astype(np.float32)
    
    Jb1_t = np.zeros((P_test * C, N_HIDDEN), dtype=np.float32)
    for p in range(P_test):
        for c in range(C):
            Jb1_t[p * C + c, :] = mask_test[p, :] * W2[:, c]
    
    Jb2_t = np.tile(np.eye(C, dtype=np.float32), (P_test, 1))
    Jb_test = np.concatenate([Jb1_t, Jb2_t], axis=1)
    
    out_test_lin = (out_test_init.reshape(-1) + (Jb_test @ delta_b)).reshape(P_test, C)
    test_acc_lin = np.mean(out_test_lin.argmax(axis=1) == y_test)
    
    # NOW: Actually update the model parameters
    model.update_parameters(delta_b, mode='bias')
    
    # Get actual accuracies with updated model
    train_acc_actual = model.get_accuracy(x_train, y_train)
    test_acc_actual = model.get_accuracy(x_test, y_test)
    
    return train_acc_lin, test_acc_lin, train_acc_actual, test_acc_actual, delta_b


def full_ntk_update(x_train, y_train, x_test, y_test, model, epsilon=1e-6):
    """
    Full NTK update on all parameters
    Returns (train_acc_lin, test_acc_lin, delta_theta)
    """
    # Forward on train
    out_init, h_init = model.forward(x_train)
    
    P = x_train.shape[0]
    C = N_CLASSES
    
    # Build +1/-1 targets
    y_target = -np.ones((P, C), dtype=np.float32)
    y_target[np.arange(P), y_train] = 1.0
    res0 = (y_target - out_init).reshape(-1)
    
    # Get current parameters
    W1, b1, W2, b2 = model.get_parameters()
    
    # Precompute mask
    mask = (h_init > 0).astype(np.float32)
    
    # Build full Jacobian
    total_params = N_HIDDEN * INPUT_DIM + N_HIDDEN + N_HIDDEN * N_CLASSES + N_CLASSES
    J_full = np.zeros((P * C, total_params), dtype=np.float32)
    
    param_idx = 0
    
    # dOut/dW1
    for p in range(P):
        for c in range(C):
            row_idx = p * C + c
            for h_idx in range(N_HIDDEN):
                if mask[p, h_idx] > 0:  # ReLU is active
                    for i_idx in range(INPUT_DIM):
                        col_idx = param_idx + h_idx * INPUT_DIM + i_idx
                        J_full[row_idx, col_idx] = W2[c, h_idx] * x_train[p, i_idx]
    param_idx += N_HIDDEN * INPUT_DIM
    
    # dOut/db1
    for p in range(P):
        for c in range(C):
            row_idx = p * C + c
            for h_idx in range(N_HIDDEN):
                if mask[p, h_idx] > 0:  # ReLU is active
                    col_idx = param_idx + h_idx
                    J_full[row_idx, col_idx] = W2[c, h_idx]
    param_idx += N_HIDDEN
    
    # dOut/dW2
    for p in range(P):
        for c in range(C):
            row_idx = p * C + c
            for c2 in range(N_CLASSES):
                for h_idx in range(N_HIDDEN):
                    col_idx = param_idx + c2 * N_HIDDEN + h_idx
                    if c == c2:
                        J_full[row_idx, col_idx] = h_init[p, h_idx]
    param_idx += N_HIDDEN * N_CLASSES
    
    # dOut/db2
    for p in range(P):
        for c in range(C):
            row_idx = p * C + c
            col_idx = param_idx + c
            J_full[row_idx, col_idx] = 1.0
    
    # Solve for alpha
    if USE_FULL_KERNEL:
        K = J_full @ J_full.T
        K += epsilon * np.eye(K.shape[0])
        alpha = np.linalg.solve(K, res0)
    else:
        alpha = cg_solve_np(lambda v: J_full @ (J_full.T @ v) + epsilon * v, res0)
    
    # Parameter update vector
    delta_theta = J_full.T @ alpha
    
    # Linearized train output
    lin_flat = out_init.reshape(-1) + (J_full @ delta_theta)
    out_lin = lin_flat.reshape(P, C)
    train_acc_lin = np.mean(out_lin.argmax(axis=1) == y_train)
    
    # Test set (similar process)
    out_test_init, h_test_init = model.forward(x_test)
    P_test = x_test.shape[0]
    mask_test = (h_test_init > 0).astype(np.float32)
    
    # Build test Jacobian (similar to training)
    J_full_test = np.zeros((P_test * C, total_params), dtype=np.float32)
    
    param_idx = 0
    
    # dOut/dW1 for test
    for p in range(P_test):
        for c in range(C):
            row_idx = p * C + c
            for h_idx in range(N_HIDDEN):
                if mask_test[p, h_idx] > 0:
                    for i_idx in range(INPUT_DIM):
                        col_idx = param_idx + h_idx * INPUT_DIM + i_idx
                        J_full_test[row_idx, col_idx] = W2[c, h_idx] * x_test[p, i_idx]
    param_idx += N_HIDDEN * INPUT_DIM
    
    # dOut/db1 for test
    for p in range(P_test):
        for c in range(C):
            row_idx = p * C + c
            for h_idx in range(N_HIDDEN):
                if mask_test[p, h_idx] > 0:
                    col_idx = param_idx + h_idx
                    J_full_test[row_idx, col_idx] = W2[c, h_idx]
    param_idx += N_HIDDEN
    
    # dOut/dW2 for test
    for p in range(P_test):
        for c in range(C):
            row_idx = p * C + c
            for c2 in range(N_CLASSES):
                for h_idx in range(N_HIDDEN):
                    col_idx = param_idx + c2 * N_HIDDEN + h_idx
                    if c == c2:
                        J_full_test[row_idx, col_idx] = h_test_init[p, h_idx]
    param_idx += N_HIDDEN * N_CLASSES
    
    # dOut/db2 for test
    for p in range(P_test):
        for c in range(C):
            row_idx = p * C + c
            col_idx = param_idx + c
            J_full_test[row_idx, col_idx] = 1.0
    
    lin_flat_test = out_test_init.reshape(-1) + (J_full_test @ delta_theta)
    out_test_lin = lin_flat_test.reshape(P_test, C)
    test_acc_lin = np.mean(out_test_lin.argmax(axis=1) == y_test)
    
    return train_acc_lin, test_acc_lin, delta_theta


# === Multi-Dataset Loading & Subsampling ===
train_sets, test_sets, names = [], [], []

for flag, dataset_name in zip(LoadDatasets, DATASETS):
    if not flag:
        continue
    
    try:
        # Load full dataset
        X, y = load_and_preprocess_data(dataset_name)
        
        # Split into train/test (MNIST has predefined split, we'll use sklearn)
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        
        # Subsample training data
        X_train_sub, y_train_sub = subsample_data(X_train_full, y_train_full)
        
        # Subsample test data
        X_test_sub, y_test_sub = subsample_test_data(X_test_full, y_test_full)
        
        train_sets.append((X_train_sub, y_train_sub))
        test_sets.append((X_test_sub, y_test_sub))
        names.append(dataset_name)
        
        print(f"Loaded {dataset_name}: train {X_train_sub.shape}, test {X_test_sub.shape}")
        
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        continue

from mlp_aux import preprocessing,get_data

train_datasets=[
    'Mnist',
    'Ethiopic',
    'Vai',
    'Osmanya',
    'NKo',
    'Fashion',
    'Kmnist'
]

with tf.Session(config=config) as sess:

    for dataset in train_datasets:
        
        train_imgs, train_labels, val_imgs, val_labels = get_data(dataset)

        # Flatten the images if they're not already flattened
        if len(train_imgs.shape) == 3:  # If shape is (N, 28, 28)
            train_imgs = train_imgs.reshape(train_imgs.shape[0], -1)  # Flatten to (N, 784)
        if len(val_imgs.shape) == 3:  # If shape is (N, 28, 28)
            val_imgs = val_imgs.reshape(val_imgs.shape[0], -1)  # Flatten to (N, 784)

        train_imgs, train_labels = subsample_data(train_imgs, train_labels)
        # Subsample test data
        val_imgs, val_labels = subsample_test_data(val_imgs, val_labels)
        val_imgs=val_imgs/255

        print(f"###Train shape: {train_imgs.shape}")
        print(f"###Val shape: {val_imgs.shape}")

        model = MLPModel(sess, params=None)
        
        # Initial accuracies
        acc_tr_init = model.get_accuracy(train_imgs, train_labels)
        acc_te_init = model.get_accuracy(val_imgs, val_labels)
        print(f"{dataset} [Init]: train acc {acc_tr_init*100:.2f}%, test acc {acc_te_init*100:.2f}%")

        if NTK_MODE == 'bias':
            # NTK update for this dataset with actual parameter updates
            train_lin, test_lin, train_actual, test_actual, delta = bias_only_ntk_update_with_actual(
                train_imgs, train_labels, val_imgs, val_labels, model
            )
            
            print(f"{dataset} [Bias-NTK Linear]: train acc {train_lin*100:.2f}%, test acc {test_lin*100:.2f}%")
            print(f"{dataset} [Bias-NTK Actual]: train acc {train_actual*100:.2f}%, test acc {test_actual*100:.2f}%")
            print(f"{dataset} [Difference]: train {abs(train_lin-train_actual)*100:.2f}pp, test {abs(test_lin-test_actual)*100:.2f}pp")

with tf.Session(config=config) as sess:
    for (x_tr, y_tr), (x_te, y_te), name in zip(train_sets, test_sets, names):
        
        print(f"\n--- Processing {name} ---")
        print(x_tr[0,:])
        print(x_te[0,:])


        # Create fresh model for this dataset
        model = MLPModel(sess, params=None)
        
        # Initial accuracies
        acc_tr_init = model.get_accuracy(x_tr, y_tr)
        acc_te_init = model.get_accuracy(x_te, y_te)
        
        print(f"{name} [Init]: train acc {acc_tr_init*100:.2f}%, test acc {acc_te_init*100:.2f}%")
        
        if NTK_MODE == 'bias':
            # NTK update for this dataset with actual parameter updates
            train_lin, test_lin, train_actual, test_actual, delta = bias_only_ntk_update_with_actual(
                x_tr, y_tr, x_te, y_te, model
            )
            
            print(f"{name} [Bias-NTK Linear]: train acc {train_lin*100:.2f}%, test acc {test_lin*100:.2f}%")
            print(f"{name} [Bias-NTK Actual]: train acc {train_actual*100:.2f}%, test acc {test_actual*100:.2f}%")
            print(f"{name} [Difference]: train {abs(train_lin-train_actual)*100:.2f}pp, test {abs(test_lin-test_actual)*100:.2f}pp")
        
        # You could do similar for full NTK mode...