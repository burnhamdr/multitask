import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_aux import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, KMeans
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Parameters
all_tasks = ['odd', 'firsthalf', 'prime', 'notodd', 'notfirsthalf']
batch_size = 64
learning_rate = 0.01
num_neurons_1 = 15000
num_neurons_2 = 2500
num_neurons_3 = 2500
train_tasks = ['odd', 'firsthalf', 'prime', 'notodd', 'notfirsthalf']
model_dir='./../../models/mlp_models/bah_prova_auto'

# Load data
train_set, _ = datasets.mnist.load_data()
train_imgs, train_labels = train_set

# Split into training and validation sets
validation_split = 0.2
split_index = int(len(train_imgs) * (1 - validation_split))
train_imgs, val_imgs = train_imgs[:split_index], train_imgs[split_index:]
train_labels, val_labels = train_labels[:split_index], train_labels[split_index:]

# Custom Bias-Only Layer
class BiasOnlyLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, initializer='zeros', **kwargs):
        super(BiasOnlyLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.initializer = initializer

    def build(self, input_shape):
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer=self.initializer,
            trainable=True
        )
        super(BiasOnlyLayer, self).build(input_shape)

    def call(self, inputs):
        output = inputs + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

# Model Definition
uniform_initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
normal_initializer = tf.keras.initializers.RandomNormal()

model = models.Sequential([
    layers.InputLayer(input_shape=(784+len(all_tasks),)),
    layers.Dense(512, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(256, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(10, use_bias=True, trainable=True, activation='relu'),
    layers.Dense(1, use_bias=True, trainable=True, activation='sigmoid')
])

"""
model = models.Sequential([
    layers.InputLayer(input_shape=(784 + len(all_tasks),)),
    layers.Dense(num_neurons_1, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    BiasOnlyLayer(num_neurons_1, activation='relu'),
    layers.Dense(num_neurons_2, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    BiasOnlyLayer(num_neurons_2, activation='relu'),
    layers.Dense(num_neurons_3, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    BiasOnlyLayer(num_neurons_3, activation='relu'),
    layers.Dense(1, use_bias=False, trainable=False, kernel_initializer=normal_initializer),
    BiasOnlyLayer(1, activation='sigmoid')
])
"""

model.load_weights(os.path.join(model_dir, 'model_weights.h5'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Task-specific activation extraction
tv_matrix = []

for task in all_tasks:
    print(f'Doing task {task}')
    curr_val_imgs, curr_val_labels = preprocessing(imgs=val_imgs, labels=val_labels,
                                                    train_tasks=[task], all_tasks=all_tasks)

    bias_only_layers = [layer for layer in model.layers if isinstance(layer, layers.Dense)]
    bias_only_outputs = [layer.output for layer in bias_only_layers]
    activation_model = Model(inputs=model.input, outputs=bias_only_outputs)
    activations = activation_model.predict(curr_val_imgs, verbose=0)

    activations = activations[:-1]  # Remove output layer activations

    flattened_activations = []
    for layer_acts in activations:
        num_samples, num_neurons = layer_acts.shape
        for neuron_idx in range(num_neurons):
            neuron_activation = layer_acts[:, neuron_idx]
            flattened_activations.append(neuron_activation)

    flattened_activations_np = np.stack(flattened_activations)
    print(flattened_activations_np.shape)
    flattened_activations_np = flattened_activations_np.var(axis=1)
    flattened_activations_np = (flattened_activations_np - np.min(flattened_activations_np)) / \
                                (np.max(flattened_activations_np) - np.min(flattened_activations_np))
    tv_matrix.append(flattened_activations_np)

tv_matrix = np.stack(tv_matrix, axis=1)  # shape: (n_neurons, n_tasks)
ind_active = np.where(tv_matrix.sum(axis=1) > 1e-3)[0]
tv_matrix  = tv_matrix[ind_active, :]
print(tv_matrix.shape)

# Clustering
n_clusters_range = range(2, 30)
scores = []
labels_list = []

for n_cluster in n_clusters_range:
    print(f'Trying with {n_cluster} clusters')
    clustering = KMeans(n_clusters=n_cluster, algorithm='elkan', n_init=20, random_state=0)
    clustering.fit(tv_matrix)
    labels = clustering.labels_
    score = metrics.silhouette_score(tv_matrix, labels)
    scores.append(score)
    labels_list.append(labels)

scores = np.array(scores)
print(scores)
lambda_ = 0.1
pen_scores = scores - np.log(list(n_clusters_range)) * lambda_
print(pen_scores)
index=4#np.argmax(pen_scores)
n_clusters = list(n_clusters_range)[index]

# Final clustering
best_labels = labels_list[index]
sorted_indices = np.argsort(best_labels,)
sorted_tv_matrix = tv_matrix[sorted_indices, :]

# Plotting
plt.figure(figsize=(12, 6))
plt.imshow(sorted_tv_matrix.T, cmap='hot', interpolation='none', aspect='auto', vmax=1, vmin=0)

# Cluster boundaries
cluster_boundaries = np.where(np.diff(np.sort(best_labels)) != 0)[0] + 1
for boundary in cluster_boundaries:
    plt.axvline(x=boundary, color='white', linestyle='--', linewidth=1)
    continue

plt.xticks([])
plt.yticks(ticks=range(len(all_tasks)), labels=all_tasks)
plt.colorbar(label='Task Variance (Normalized)')
plt.title(f'Clustered Neuron Task Variance (k={n_clusters})')
plt.xlabel('Neurons (Sorted by Cluster)')
plt.ylabel('Tasks')
plt.tight_layout()
plt.savefig('clustered_tv_matrix.png')
plt.show()
