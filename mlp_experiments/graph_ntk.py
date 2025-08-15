import numpy as np
import os

# Configuration (assuming these are defined elsewhere or passed in)
INPUT_FILE = 'neural_activations_3d.npy'
DATASET_NAMES = [
    'Mnist',
    'Ethiopic',
    'Vai',
    'Osmanya',
    'NKo',
    'Fashion',
    'Kmnist'
]

def load_and_process_activations(filename):
    """
    Load activations and calculate normalized variances

    Args:
        filename: Path to the .npy file containing activations

    Returns:
        normalized_variances: Array of shape (7, 30000) with normalized variances
        raw_variances: Array of shape (7, 30000) with raw variances
    """
    print(f"Loading activations from {filename}...")

    # Load the 3D activations array
    activations = np.load(filename)
    print(f"Loaded activations with shape: {activations.shape}")

    # Verify expected shape
    n_datasets, n_samples, n_hidden = activations.shape
    print(f"  Datasets: {n_datasets}")
    print(f"  Samples per dataset: {n_samples}")
    print(f"  Hidden units: {n_hidden}")

    # Calculate variance across the samples dimension (axis=1)
    print("Calculating variances across samples...")
    raw_variances = np.var(activations, axis=1)
    print(f"Raw variances shape: {raw_variances.shape}")
    ind_active = np.where(raw_variances.sum(axis=0) > 1e-3)[0]
    raw_variances  = raw_variances[:, ind_active]
    print(f"Raw non zero variances shape: {raw_variances.shape}")

    # Normalize across the datasets dimension (axis=0) to sum to 1
    print("Normalizing variances across datasets...")

    # Add small epsilon to avoid division by zero
    epsilon = 1e-12
    variance_sums = np.sum(raw_variances, axis=0, keepdims=True) + epsilon
    normalized_variances = raw_variances / variance_sums

    # Verify normalization (should sum to 1 across datasets for each hidden unit)
    normalization_check = np.sum(normalized_variances, axis=0)
    print(f"Normalization check - min: {normalization_check.min():.6f}, max: {normalization_check.max():.6f}")
    print(f"Should be close to 1.0")

    return normalized_variances, raw_variances

norm_tv, raw_tv=load_and_process_activations('neural_activations_3d.npy')

tv_matrix=norm_tv.T

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import metrics

# Clustering
n_clusters_range = range(2, 4)
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

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
kelly_colors = \
[np.array([ 0.94901961,  0.95294118,  0.95686275]),
 np.array([ 0.13333333,  0.13333333,  0.13333333]),
 np.array([ 0.95294118,  0.76470588,  0.        ]),
 np.array([ 0.52941176,  0.3372549 ,  0.57254902]),
 np.array([ 0.95294118,  0.51764706,  0.        ]),
 np.array([ 0.63137255,  0.79215686,  0.94509804]),
 np.array([ 0.74509804,  0.        ,  0.19607843]),
 np.array([ 0.76078431,  0.69803922,  0.50196078]),
 np.array([ 0.51764706,  0.51764706,  0.50980392]),
 np.array([ 0.        ,  0.53333333,  0.3372549 ]),
 np.array([ 0.90196078,  0.56078431,  0.6745098 ]),
 np.array([ 0.        ,  0.40392157,  0.64705882]),
 np.array([ 0.97647059,  0.57647059,  0.4745098 ]),
 np.array([ 0.37647059,  0.30588235,  0.59215686]),
 np.array([ 0.96470588,  0.65098039,  0.        ]),
 np.array([ 0.70196078,  0.26666667,  0.42352941]),
 np.array([ 0.8627451 ,  0.82745098,  0.        ]),
 np.array([ 0.53333333,  0.17647059,  0.09019608]),
 np.array([ 0.55294118,  0.71372549,  0.        ]),
 np.array([ 0.39607843,  0.27058824,  0.13333333]),
 np.array([ 0.88627451,  0.34509804,  0.13333333]),
 np.array([ 0.16862745,  0.23921569,  0.14901961])]

# Figure 3.28

index=np.argmax(scores)
n_clusters = list(n_clusters_range)[index]

# Final clustering
best_labels = labels_list[index]
sorted_indices = np.argsort(best_labels,)
sorted_labels = best_labels[sorted_indices]
sorted_tv_matrix = tv_matrix[sorted_indices, :]

# Plotting
fig = plt.figure(figsize=(15,5.5))

rect = [0.3, 0.2, 0.5, 0.7]
rect_cb = [0.82, 0.2, 0.03, 0.7]
rect_color = [0.3, 0.15, 0.5, 0.05]
cluster_labels=range(index+3)

ax = fig.add_axes(rect)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_xlabel('Clusters',labelpad=20)
ax.set_yticks(range(7))
ax.set_yticklabels(DATASET_NAMES)
ax.set_title('Task variance and clustering plot for a \nmodel using ntk calculated biases\n tested on 7 classification tasks',fontsize=10)

hmap=ax.imshow(sorted_tv_matrix.T, cmap='hot', interpolation='none', aspect='auto', vmax=1, vmin=0)

# Cluster boundaries
cluster_boundaries = np.where(np.diff(np.sort(best_labels)) != 0)[0] + 1
for boundary in cluster_boundaries:
    ax.axvline(x=boundary, color='white', linestyle='--', linewidth=2)
    continue

# After plotting hmap with imshow:
x_start, x_end = ax.get_xlim()  # Get actual axis limits used by imshow
neuron_count = len(sorted_labels)

bar_ax = fig.add_axes(rect_color)

prev = 0
for il, l in enumerate(cluster_labels):
    count = np.sum(sorted_labels == l-1)
    
    # Map neuron index to image axis (same range as ax)
    ind_l = [
        x_start + (prev / neuron_count) * (x_end - x_start),
        x_start + ((prev + count) / neuron_count) * (x_end - x_start)
    ]

    bar_ax.plot(ind_l, [0, 0], linewidth=4, solid_capstyle='butt',
                color=kelly_colors[il % len(kelly_colors)])
    bar_ax.text(np.mean(ind_l), -0.5, str(l), fontsize=6,
                ha='center', va='top', color=kelly_colors[il % len(kelly_colors)])
    
    prev += count

bar_ax.set_xlim([x_start, x_end])  # Match heatmap width
bar_ax.set_ylim([-1, 1])
bar_ax.axis('off')

plt.savefig('ntk_graph.png')