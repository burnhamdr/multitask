"""
Clustering analysis
Analyze how units are involved in various tasks
"""

from __future__ import division

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import tensorflow as tf

from task import rule_name
from network import Model
import tools

# Colors used for clusters
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

from matplotlib import colors as mcolors
from skimage.color import rgb2lab
from sklearn.metrics import pairwise_distances

def get_maximally_contrastive_colors(N, light_threshold=95, dark_threshold=5):
    # Retrieve CSS4 colors and filter out those that are too light or too dark
    css4_colors = list(mcolors.CSS4_COLORS.items())
    css4_rgb_values = np.array([mcolors.to_rgb(color[1]) for color in css4_colors])
    css4_lab_values = rgb2lab(css4_rgb_values.reshape(1, -1, 3)).reshape(-1, 3)
    
    # Filter colors: exclude colors that are too close to white or black in Lab space
    filtered_indices = [
        i for i, lab in enumerate(css4_lab_values)
        if dark_threshold < lab[0] < light_threshold
    ]
    # Check if there are enough colors left after filtering
    if N > len(filtered_indices) or N <= 0:
        raise ValueError(f"N should be between 1 and {len(filtered_indices)} after filtering")

    # Filtered colors and lab values
    filtered_colors = [css4_colors[i] for i in filtered_indices]
    filtered_lab_values = css4_lab_values[filtered_indices]

    # Step 1: Choose the first color arbitrarily and select maximally distinct colors
    selected_indices = [0]
    remaining_indices = set(range(1, len(filtered_lab_values)))
    
    # Select N colors that are maximally distinct from each other
    for _ in range(N - 1):
        last_color = filtered_lab_values[selected_indices[-1]].reshape(1, -1)
        distances = pairwise_distances(last_color, filtered_lab_values[list(remaining_indices)])
        next_index = list(remaining_indices)[np.argmax(distances)]
        selected_indices.append(next_index)
        remaining_indices.remove(next_index)
    
    # Step 2: Reorder the selected colors for maximum contrast
    ordered_indices = [selected_indices[0]]
    remaining_indices = set(selected_indices[1:])
    
    while remaining_indices:
        last_color = filtered_lab_values[ordered_indices[-1]].reshape(1, -1)
        distances = pairwise_distances(last_color, filtered_lab_values[list(remaining_indices)])
        next_index = list(remaining_indices)[np.argmax(distances)]
        ordered_indices.append(next_index)
        remaining_indices.remove(next_index)
    
    # Retrieve the ordered color names and hex codes
    ordered_colors = [filtered_colors[i] for i in ordered_indices]
    ordered_rgb_values = [np.array(mcolors.to_rgb(hex_code)) for _, hex_code in ordered_colors]

    return ordered_rgb_values

kelly_expanded = [
np.array([0.9411764705882353, 0.9725490196078431, 1.0]), # aliceblue (#F0F8FF)
np.array([0.0, 1.0, 0.0]), # lime (#00FF00)
np.array([0.0, 0.0, 0.803921568627451]), # mediumblue (#0000CD)
np.array([0.4980392156862745, 1.0, 0.0]), # chartreuse (#7FFF00)
np.array([1.0, 0.0, 1.0]), # fuchsia (#FF00FF)
np.array([0.6784313725490196, 1.0, 0.1843137254901961]), # greenyellow (#ADFF2F)
np.array([0.5803921568627451, 0.0, 0.8274509803921568]), # darkviolet (#9400D3)
np.array([0.5411764705882353, 0.16862745098039217, 0.8862745098039215]), # blueviolet (#8A2BE2)
np.array([0.0, 0.0, 0.5450980392156862]), # darkblue (#00008B)
np.array([0.19607843137254902, 0.803921568627451, 0.19607843137254902]), # limegreen (#32CD32)
np.array([0.6, 0.19607843137254902, 0.8]), # darkorchid (#9932CC)
np.array([0.0, 0.9803921568627451, 0.6039215686274509]), # mediumspringgreen (#00FA9A)
np.array([0.0, 0.0, 0.5019607843137255]), # navy (#000080)
np.array([1.0, 0.8431372549019608, 0.0]), # gold (#FFD700)
np.array([0.29411764705882354, 0.0, 0.5098039215686274]), # indigo (#4B0082)
np.array([0.6039215686274509, 0.803921568627451, 0.19607843137254902]), # yellowgreen (#9ACD32)
np.array([0.4823529411764706, 0.40784313725490196, 0.9333333333333333]), # mediumslateblue (#7B68EE)
np.array([0.0, 0.5019607843137255, 0.0]), # green (#008000)
np.array([0.13333333333333333, 0.5450980392156862, 0.13333333333333333]), # forestgreen (#228B22)
np.array([1.0, 0.0784313725490196, 0.5764705882352941]), # deeppink (#FF1493)
np.array([0.596078431372549, 0.984313725490196, 0.596078431372549]), # palegreen (#98FB98)
np.array([0.5019607843137255, 0.0, 0.5019607843137255]), # purple (#800080)
np.array([1.0, 0.0, 0.0]), # red (#FF0000)
np.array([0.0, 1.0, 1.0]), # aqua (#00FFFF)
np.array([1.0, 0.27058823529411763, 0.0]), # orangered (#FF4500)
np.array([0.8627450980392157, 0.0784313725490196, 0.23529411764705882]), # crimson (#DC143C)
np.array([0.25098039215686274, 0.8784313725490196, 0.8156862745098039]), # turquoise (#40E0D0)
np.array([0.7803921568627451, 0.08235294117647059, 0.5215686274509804]), # mediumvioletred (#C71585)
np.array([0.0, 0.39215686274509803, 0.0]), # darkgreen (#006400)
np.array([0.9333333333333333, 0.5098039215686274, 0.9333333333333333]), # violet (#EE82EE)
np.array([0.4196078431372549, 0.5568627450980392, 0.13725490196078433]), # olivedrab (#6B8E23)
np.array([0.2549019607843137, 0.4117647058823529, 0.8823529411764706]), # royalblue (#4169E1)
np.array([1.0, 0.6470588235294118, 0.0]), # orange (#FFA500)
np.array([0.11764705882352941, 0.5647058823529412, 1.0]), # dodgerblue (#1E90FF)
np.array([1.0, 0.5490196078431373, 0.0]), # darkorange (#FF8C00)
np.array([0.8549019607843137, 0.6470588235294118, 0.12549019607843137]), # goldenrod (#DAA520)
np.array([0.41568627450980394, 0.35294117647058826, 0.803921568627451]), # slateblue (#6A5ACD)
np.array([0.5019607843137255, 0.5019607843137255, 0.0]), # olive (#808000)
np.array([0.4, 0.2, 0.6]), # rebeccapurple (#663399)
np.array([0.9411764705882353, 0.9019607843137255, 0.5490196078431373]), # khaki (#F0E68C)
np.array([0.5764705882352941, 0.4392156862745098, 0.8588235294117647]), # mediumpurple (#9370DB)
np.array([0.7215686274509804, 0.5254901960784314, 0.043137254901960784]), # darkgoldenrod (#B8860B)
np.array([0.39215686274509803, 0.5843137254901961, 0.9294117647058824]), # cornflowerblue (#6495ED)
np.array([0.8235294117647058, 0.4117647058823529, 0.11764705882352941]), # chocolate (#D2691E)
np.array([0.0, 0.7490196078431373, 1.0]), # deepskyblue (#00BFFF)
np.array([1.0, 0.38823529411764707, 0.2784313725490196]), # tomato (#FF6347)
np.array([0.0, 0.807843137254902, 0.8196078431372549]), # darkturquoise (#00CED1)
np.array([0.5450980392156862, 0.0, 0.0]), # darkred (#8B0000)
np.array([0.6980392156862745, 0.13333333333333333, 0.13333333333333333]), # firebrick (#B22222)
np.array([0.23529411764705882, 0.7019607843137254, 0.44313725490196076]), # mediumseagreen (#3CB371)
np.array([0.8549019607843137, 0.4392156862745098, 0.8392156862745098]), # orchid (#DA70D6)
np.array([0.1803921568627451, 0.5450980392156862, 0.3411764705882353]), # seagreen (#2E8B57)
np.array([1.0, 0.4117647058823529, 0.7058823529411765]), # hotpink (#FF69B4)
np.array([0.4, 0.803921568627451, 0.6666666666666666]), # mediumaquamarine (#66CDAA)
np.array([0.5019607843137255, 0.0, 0.0]), # maroon (#800000)
np.array([0.12549019607843137, 0.6980392156862745, 0.6666666666666666]), # lightseagreen (#20B2AA)
np.array([1.0, 0.4980392156862745, 0.3137254901960784]), # coral (#FF7F50)
np.array([0.2823529411764706, 0.23921568627450981, 0.5450980392156862]), # darkslateblue (#483D8B)
np.array([0.9333333333333333, 0.9098039215686274, 0.6666666666666666]), # palegoldenrod (#EEE8AA)
np.array([0.0, 0.0, 0.0]), # black (#000000)
np.array([1.0, 1.0, 0.8784313725490196]), # lightyellow (#FFFFE0)
np.array([0.6470588235294118, 0.16470588235294117, 0.16470588235294117]), # brown (#A52A2A)
np.array([0.6862745098039216, 0.9333333333333333, 0.9333333333333333]), # paleturquoise (#AFEEEE)
np.array([0.5450980392156862, 0.27058823529411763, 0.07450980392156863]), # saddlebrown (#8B4513)
np.array([0.5294117647058824, 0.807843137254902, 0.9803921568627451]), # lightskyblue (#87CEFA)
np.array([0.803921568627451, 0.5215686274509804, 0.24705882352941178]), # peru (#CD853F)
np.array([0.27450980392156865, 0.5098039215686274, 0.7058823529411765]), # steelblue (#4682B4)
np.array([0.9568627450980393, 0.6431372549019608, 0.3764705882352941]), # sandybrown (#F4A460)
np.array([0.0, 0.5450980392156862, 0.5450980392156862]), # darkcyan (#008B8B)
np.array([0.9803921568627451, 0.5019607843137255, 0.4470588235294118]), # salmon (#FA8072)
np.array([0.0, 0.5019607843137255, 0.5019607843137255]), # teal (#008080)
np.array([0.803921568627451, 0.3607843137254902, 0.3607843137254902]), # indianred (#CD5C5C)
np.array([0.5294117647058824, 0.807843137254902, 0.9215686274509803]), # skyblue (#87CEEB)
np.array([0.6274509803921569, 0.3215686274509804, 0.17647058823529413]), # sienna (#A0522D)
np.array([0.8784313725490196, 1.0, 1.0]), # lightcyan (#E0FFFF)
np.array([0.8588235294117647, 0.4392156862745098, 0.5764705882352941]), # palevioletred (#DB7093)
np.array([0.3333333333333333, 0.4196078431372549, 0.1843137254901961]), # darkolivegreen (#556B2F)
np.array([0.8666666666666667, 0.6274509803921569, 0.8666666666666667]), # plum (#DDA0DD)
np.array([0.7411764705882353, 0.7176470588235294, 0.4196078431372549]), # darkkhaki (#BDB76B)
np.array([0.1843137254901961, 0.30980392156862746, 0.30980392156862746]), # darkslategray (#2F4F4F)
np.array([1.0, 0.6274509803921569, 0.47843137254901963]), # lightsalmon (#FFA07A)
np.array([0.1843137254901961, 0.30980392156862746, 0.30980392156862746]), # darkslategrey (#2F4F4F)
np.array([1.0, 0.9803921568627451, 0.803921568627451]), # lemonchiffon (#FFFACD)
np.array([0.9411764705882353, 0.5019607843137255, 0.5019607843137255]), # lightcoral (#F08080)
np.array([0.37254901960784315, 0.6196078431372549, 0.6274509803921569]), # cadetblue (#5F9EA0)
np.array([0.9137254901960784, 0.5882352941176471, 0.47843137254901963]), # darksalmon (#E9967A)
np.array([0.6901960784313725, 0.8784313725490196, 0.9019607843137255]), # powderblue (#B0E0E6)
np.array([0.4392156862745098, 0.5019607843137255, 0.5647058823529412]), # slategray (#708090)
np.array([1.0, 0.8705882352941177, 0.6784313725490196]), # navajowhite (#FFDEAD)
np.array([1.0, 0.8941176470588236, 0.7098039215686275]), # moccasin (#FFE4B5)
np.array([0.9607843137254902, 0.8705882352941177, 0.7019607843137254]), # wheat (#F5DEB3)
np.array([0.5019607843137255, 0.5019607843137255, 0.5019607843137255]), # gray (#808080)
np.array([0.9411764705882353, 1.0, 0.9411764705882353]), # honeydew (#F0FFF0)
np.array([0.7372549019607844, 0.5607843137254902, 0.5607843137254902]), # rosybrown (#BC8F8F)
np.array([0.5607843137254902, 0.7372549019607844, 0.5607843137254902]), # darkseagreen (#8FBC8F)
np.array([1.0, 0.7137254901960784, 0.7568627450980392]), # lightpink (#FFB6C1)
np.array([0.6784313725490196, 0.8470588235294118, 0.9019607843137255]), # lightblue (#ADD8E6)
np.array([0.8705882352941177, 0.7215686274509804, 0.5294117647058824]), # burlywood (#DEB887)
np.array([0.6901960784313725, 0.7686274509803922, 0.8705882352941177]), # lightsteelblue (#B0C4DE)
np.array([0.8235294117647058, 0.7058823529411765, 0.5490196078431373]), # tan (#D2B48C)
np.array([0.9019607843137255, 0.9019607843137255, 0.9803921568627451]), # lavender (#E6E6FA)
np.array([1.0, 0.8549019607843137, 0.7254901960784313]), # peachpuff (#FFDAB9)
np.array([0.8470588235294118, 0.7490196078431373, 0.8470588235294118]), # thistle (#D8BFD8)
np.array([1.0, 0.8941176470588236, 0.7686274509803922]), # bisque (#FFE4C4)
np.array([0.6627450980392157, 0.6627450980392157, 0.6627450980392157]), # darkgray (#A9A9A9)
np.array([1.0, 0.7529411764705882, 0.796078431372549]), # pink (#FFC0CB)
np.array([0.9411764705882353, 1.0, 1.0]), # azure (#F0FFFF)
np.array([0.6627450980392157, 0.6627450980392157, 0.6627450980392157]), # darkgrey (#A9A9A9)
np.array([1.0, 0.9215686274509803, 0.803921568627451]), # blanchedalmond (#FFEBCD)
np.array([0.7529411764705882, 0.7529411764705882, 0.7529411764705882]), # silver (#C0C0C0)
np.array([1.0, 0.9372549019607843, 0.8352941176470589]), # papayawhip (#FFEFD5)
np.array([0.9607843137254902, 0.9607843137254902, 0.8627450980392157]), # beige (#F5F5DC)
np.array([0.8274509803921568, 0.8274509803921568, 0.8274509803921568]), # lightgrey (#D3D3D3)
np.array([1.0, 0.9411764705882353, 0.9607843137254902]), # lavenderblush (#FFF0F5)
np.array([0.9803921568627451, 0.9411764705882353, 0.9019607843137255]), # linen (#FAF0E6)
np.array([1.0, 0.8941176470588236, 0.8823529411764706]), # mistyrose (#FFE4E1)
np.array([1.0, 0.9607843137254902, 0.9333333333333333]), # seashell (#FFF5EE)

]
save = True


class Analysis(object):
    def __init__(self, model_dir, data_type, normalization_method='max',predef_num_clusters=None):
        #check if model_dir is a list of directories
        if isinstance(model_dir, list):
            h_var_all_ = []
            keys = []
            for md in model_dir:
                 # If not computed, use variance.py
                fname = os.path.join(md, 'variance_' + data_type + '.pkl')
                res = tools.load_pickle(fname)
                h_var_all_temp = res['h_var_all']# n_units x n_rules/n_epochs
                keys_temp  = res['keys']
                h_var_all_.append(h_var_all_temp)
                keys.append(keys_temp)
            h_var_all_ = np.concatenate(h_var_all_, axis=1)
            self.keys = keys
            self.model_dir = model_dir[0]
            self.hp = tools.load_hp(self.model_dir)
            self.concat_dirs = True
            #unpack all the keys
            self.rules = [subkey for key in keys for subkey in key]
        else:
            hp = tools.load_hp(model_dir)
            # If not computed, use variance.py
            fname = os.path.join(model_dir, 'variance_' + data_type + '.pkl')
            res = tools.load_pickle(fname)
            h_var_all_ = res['h_var_all']# n_units x n_rules/n_epochs
            self.keys  = res['keys']
            self.model_dir = model_dir
            self.hp = hp
            self.rules = hp['rules']
            self.concat_dirs = False

        # First only get active units. Total variance across tasks larger than 1e-3
        # ind_active = np.where(h_var_all_.sum(axis=1) > 1e-2)[0]
        ind_active = np.where(h_var_all_.sum(axis=1) > 1e-3)[0]
        h_var_all  = h_var_all_[ind_active, :]

        # Normalize by the total variance across tasks
        if normalization_method == 'sum':
            h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T
        elif normalization_method == 'max':
            h_normvar_all = (h_var_all.T/np.max(h_var_all, axis=1)).T
        elif normalization_method == 'none':
            h_normvar_all = h_var_all
        else:
            raise NotImplementedError()

        ################################## Clustering ################################
        from sklearn import metrics
        X = h_normvar_all

        # Clustering
        from sklearn.cluster import AgglomerativeClustering, KMeans

        # Choose number of clusters that maximize silhouette score
        n_clusters = range(2, 30)
        scores = list()
        labels_list = list()
        for n_cluster in n_clusters:
            # clustering = AgglomerativeClustering(n_cluster, affinity='cosine', linkage='average')
            clustering = KMeans(n_cluster, algorithm='full', n_init=20, random_state=0)
            clustering.fit(X) # n_samples, n_features = n_units, n_rules/n_epochs
            labels = clustering.labels_ # cluster labels

            score = metrics.silhouette_score(X, labels)

            scores.append(score)
            labels_list.append(labels)

        scores = np.array(scores)
        #penalize by number of clusters
        lambda_ = 0.1
        pen_scores = scores - np.log(n_clusters)*lambda_

        # Heuristic elbow method
        # Choose the number of cluster when Silhouette score first falls
        # Choose the number of cluster when Silhouette score is maximum
        if data_type == 'rule':
            #i = np.where((scores[1:]-scores[:-1])<0)[0][0]
            if not predef_num_clusters:
                i = np.argmax(scores)
            else:
                i = predef_num_clusters-2
            #i = np.argmax(pen_scores)
        else:
            # The more rigorous method doesn't work well in this case
            i=np.argmax(scores)
            #i = n_clusters.index(10)

        labels = labels_list[i]
        n_cluster = n_clusters[i]
        print('Choosing {:d} clusters'.format(n_cluster))

        # Sort clusters by its task preference (important for consistency across nets)
        if data_type == 'rule':
            label_prefs = [np.argmax(h_normvar_all[labels==l].sum(axis=0)) for l in set(labels)]
        elif data_type == 'epoch':
            label_prefs = [self.keys[0][np.argmax(h_normvar_all[labels==l].sum(axis=0))][0] for l in set(labels)]

        ind_label_sort = np.argsort(label_prefs)
        label_prefs = np.array(label_prefs)[ind_label_sort]
        # Relabel
        labels2 = np.zeros_like(labels)
        for i, ind in enumerate(ind_label_sort):
            labels2[labels==ind] = i
        labels = labels2

        # # Sort data by labels and by input connectivity
        # model = Model(save_name)
        # hp = model.hp
        # with tf.Session() as sess:
        #     model.restore(sess)
        #     var_list = sess.run(model.var_list)
        #
        # # Get connectivity
        # w_out = var_list[0].T
        # b_out = var_list[1]
        # w_in  = var_list[2][:n_input, :].T
        # w_rec = var_list[2][n_input:, :].T
        # b_rec = var_list[3]
        #
        # # nx, nh, ny = hp['shape']
        # nr = hp['n_eachring']
        #
        # sort_by = 'w_in'
        # if sort_by == 'w_in':
        #     w_in_mod1 = w_in[ind_active, :][:, 1:nr+1]
        #     w_in_mod2 = w_in[ind_active, :][:, nr+1:2*nr+1]
        #     w_in_modboth = w_in_mod1 + w_in_mod2
        #     w_prefs = np.argmax(w_in_modboth, axis=1)
        # elif sort_by == 'w_out':
        #     w_prefs = np.argmax(w_out[1:, ind_active], axis=0)
        #
        # ind_sort        = np.lexsort((w_prefs, labels)) # sort by labels then by prefs

        ind_sort = np.argsort(labels)

        labels          = labels[ind_sort]
        self.h_normvar_all   = h_normvar_all[ind_sort, :]
        self.ind_active      = ind_active[ind_sort]

        self.n_clusters = n_clusters
        self.scores = scores
        self.pen_scores = pen_scores
        self.n_cluster = n_cluster

        self.h_var_all = h_var_all
        self.normalization_method = normalization_method
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.data_type = data_type

    def plot_cluster_score(self, save_name=None, save=False):
        """Plot the score by the number of clusters."""
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_axes([0.3, 0.3, 0.55, 0.55])
        ax.plot(self.n_clusters, self.scores, 'o-', ms=3, color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        """
        #create twin axis
        ax2 = ax.twinx()
        ax2.plot(self.n_clusters, self.pen_scores, 'o-', ms=3, color='tab:red')
        ax2.set_ylabel('Penalized score', fontsize=7)
        #color the twin axis labels and ticks and axis
        ax2.tick_params(axis='y', labelcolor='tab:red')
        """
        ax.set_xlabel('Number of clusters', fontsize=7)
        ax.set_ylabel('Silhouette score', fontsize=7)
        ax.set_title('Chosen number of clusters: {:d}'.format(self.n_cluster),
                     fontsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        #manually construct the legend
        handles = [Line2D([0], [0], color='tab:blue', lw=1, label='Silhouette score')]#,
                    #Line2D([0], [0], color='tab:red', lw=1, label='Penalized score')]
        ax.legend(handles=handles, loc='upper right', fontsize=7)
        # ax.set_ylim([0, 0.32])
        if save:
            fig_name = 'cluster_score'
            if save_name is None:
                save_name = self.hp['activation']
            fig_name = fig_name + save_name
            plt.savefig('./../figure/'+fig_name+'.pdf', transparent=True)
        plt.show()

    def plot_variance(self, save_name=None):
        labels = self.labels
        ######################### Plotting Variance ###################################
        # Plot Normalized Variance
        if self.data_type == 'rule':
            figsize = (7.5,4.)
            #figsize = (4.5,5.5)
            #figsize = (3.5,2.5)
            rect = [0.25, 0.2, 0.6, 0.7]
            rect_color = [0.25, 0.15, 0.6, 0.05]
            rect_cb = [0.87, 0.2, 0.03, 0.7]
            tick_names = [rule_name[r] for r in self.rules]
            fs = 6
            labelpad = 22
        elif self.data_type == 'epoch':
            figsize = (4.,5.5)
            rect = [0.25, 0.1, 0.6, 0.85]
            rect_color = [0.25, 0.05, 0.6, 0.05]
            rect_cb = [0.87, 0.1, 0.03, 0.85]
            tick_names = [rule_name[key[0]]+' '+key[1] for key in self.keys[0]]
            fs = 5
            labelpad = 20
        else:
            raise ValueError

        h_plot  = self.h_normvar_all.T
        vmin, vmax = 0, 1
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        im = ax.imshow(h_plot, cmap='hot',
                       aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

        plt.yticks(range(len(tick_names)), tick_names,
                   rotation=0, va='center', fontsize=fs)
        plt.xticks([])
        plt.title('Units', fontsize=7, y=0.99)
        plt.xlabel('Clusters', fontsize=7, labelpad=labelpad)
        ax.tick_params('both', length=0)
        for loc in ['bottom','top','left','right']:
            ax.spines[loc].set_visible(False)
        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(im, cax=ax, ticks=[vmin,vmax])
        cb.outline.set_linewidth(0.5)
        if self.normalization_method == 'sum':
            clabel = 'Normalized Task Variance'
        elif self.normalization_method == 'max':
            clabel = 'Normalized Task Variance'
        elif self.normalization_method == 'none':
            clabel = 'Variance'

        cb.set_label(clabel, fontsize=7, labelpad=0)
        plt.tick_params(axis='both', which='major', labelsize=7)
        

        # Plot color bars indicating clustering
        if True:
            if len(self.unique_labels) > len(kelly_colors):
                colors = kelly_expanded#get_maximally_contrastive_colors(len(self.unique_labels)+1, light_threshold=95, dark_threshold=5)
            else:
                colors = kelly_colors

            ax = fig.add_axes(rect_color)
            for il, l in enumerate(self.unique_labels):
                ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
                ax.plot(ind_l, [0,0], linewidth=4, solid_capstyle='butt',
                        color=colors[il+1])
                ax.text(np.mean(ind_l), -0.5, str(il+1), fontsize=6,
                        ha='center', va='top', color=colors[il+1])
            ax.set_xlim([0, len(labels)])
            ax.set_ylim([-1, 1])
            ax.axis('off')

        if save:
            fig_name = ('feature_map_by' + self.data_type +
                        '_norm' + self.normalization_method)
            if save_name is not None:
                fig_name = fig_name + save_name

            #join with the model directory
            fig_path = os.path.join(self.model_dir, 'figure/'+fig_name+'.pdf')
            #check if the figure directory exists, if not, create it
            if not os.path.exists(os.path.join(self.model_dir, 'figure')):
                os.makedirs(os.path.join(self.model_dir, 'figure'))
            plt.savefig(fig_path, transparent=True)

        plt.show()

    def plot_compositional_variance(self, save_name=None):
        labels = self.labels
        ######################### Plotting Variance ###################################
        # Plot Normalized Variance
        if self.data_type == 'rule':
            figsize = (7.5,3.5) # (width,heigth)
            #figsize = (4.5,5.5)
            #figsize = (3.5,2.5)
            rect = [0.25, 0.2, 0.6, 0.7]
            rect_color = [0.25, 0.15, 0.6, 0.05]
            rect_cb = [0.87, 0.2, 0.03, 0.7]
            tick_names = [int(i*10)/10 for i in np.linspace(0,1,11).tolist()]
            #tick_names = [abs(rule_name[r]) for r in self.rules]
            fs = 6
            labelpad = 22
        elif self.data_type == 'epoch':
            figsize = (3.5,4.5)
            rect = [0.25, 0.1, 0.6, 0.85]
            rect_color = [0.25, 0.05, 0.6, 0.05]
            rect_cb = [0.87, 0.1, 0.03, 0.85]
            tick_names = [rule_name[key[0]]+' '+key[1] for key in self.keys]
            fs = 5
            labelpad = 20
        else:
            raise ValueError

        h_plot  = self.h_normvar_all.T
    
        vmin, vmax = 0, 1
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        im = ax.imshow(h_plot, cmap='hot',
                       aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

        plt.yticks(range(len(tick_names)), tick_names,
                   rotation=0, va='center', fontsize=fs)
        plt.xticks([])
        plt.title('Units', fontsize=7, y=0.99)
        plt.xlabel('Clusters', fontsize=7, labelpad=labelpad)
        plt.ylabel('Gamma (= a)', fontsize=7)
        ax.tick_params('both', length=0)
        for loc in ['bottom','top','left','right']:
            ax.spines[loc].set_visible(False)
        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(im, cax=ax, ticks=[vmin,vmax])
        cb.outline.set_linewidth(0.5)
        if self.normalization_method == 'sum':
            clabel = 'Normalized Task Variance'
        elif self.normalization_method == 'max':
            clabel = 'Normalized Task Variance'
        elif self.normalization_method == 'none':
            clabel = 'Variance'

        cb.set_label(clabel, fontsize=7, labelpad=0)
        plt.tick_params(axis='both', which='major', labelsize=7)
        

        # Plot color bars indicating clustering
        if True:
            if len(self.unique_labels) > len(kelly_colors):
                colors = kelly_expanded#get_maximally_contrastive_colors(len(self.unique_labels)+1, light_threshold=95, dark_threshold=5)
            else:
                colors = kelly_colors

            ax = fig.add_axes(rect_color)
            for il, l in enumerate(self.unique_labels):
                ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
                ax.plot(ind_l, [0,0], linewidth=4, solid_capstyle='butt',
                        color=colors[il+1])
                ax.text(np.mean(ind_l), -0.5, str(il+1), fontsize=6,
                        ha='center', va='top', color=colors[il+1])
            ax.set_xlim([0, len(labels)])
            ax.set_ylim([-1, 1])
            ax.axis('off')

        if save:
            fig_name = ('feature_map_by' + self.data_type +
                        '_norm' + self.normalization_method)
            if save_name is not None:
                fig_name = fig_name + save_name

            #join with the model directory
            fig_path = os.path.join(self.model_dir, 'figure/'+fig_name+'.pdf')
            #check if the figure directory exists, if not, create it
            if not os.path.exists(os.path.join(self.model_dir, 'figure')):
                os.makedirs(os.path.join(self.model_dir, 'figure'))
            plt.savefig(fig_path, transparent=True)

        plt.show()


    def plot_similarity_matrix(self):
        labels = self.labels
        ######################### Plotting Similarity Matrix ##########################

        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(self.h_normvar_all) # TODO: check
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
        im = ax.imshow(similarity, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        ax.axis('off')

        ax = fig.add_axes([0.87, 0.25, 0.03, 0.6])
        cb = plt.colorbar(im, cax=ax, ticks=[0,1])
        cb.outline.set_linewidth(0.5)
        cb.set_label('Similarity', fontsize=7, labelpad=0)
        plt.tick_params(axis='both', which='major', labelsize=7)

        ax1 = fig.add_axes([0.25, 0.85, 0.6, 0.05])
        ax2 = fig.add_axes([0.2, 0.25, 0.05, 0.6])
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
            ax1.plot(ind_l, [0,0], linewidth=2, solid_capstyle='butt',
                    color=kelly_colors[il+1])
            ax2.plot([0,0], len(labels)-ind_l, linewidth=2, solid_capstyle='butt',
                    color=kelly_colors[il+1])
        ax1.set_xlim([0, len(labels)])
        ax2.set_ylim([0, len(labels)])
        ax1.axis('off')
        ax2.axis('off')
        if save:
            plt.savefig('figure/feature_similarity_by'+self.data_type+'.pdf', transparent=True)
        plt.show()

    def plot_2Dvisualization(self, method='tSNE'):
        labels = self.labels
        ######################## Plotting 2-D visualization of variance map ###########
        from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
        from sklearn.decomposition import PCA

        # model = LocallyLinearEmbedding()
        if method == 'PCA':
            model = PCA(n_components=2, whiten=False)
        elif method == 'MDS':
            model = MDS(metric=True, n_components=2, n_init=10, max_iter=1000)
        elif method == 'tSNE':
            model = TSNE(n_components=2, random_state=0, init='pca',
                         verbose=1, method='exact',
                         learning_rate=100, perplexity=30)
        else:
            raise NotImplementedError

        Y = model.fit_transform(self.h_normvar_all)

        if len(self.unique_labels) > len(kelly_colors):
            colors = kelly_expanded#get_maximally_contrastive_colors(len(self.unique_labels)+1, light_threshold=95, dark_threshold=5)
        else:
            colors = kelly_colors

        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels==l)[0]
            ax.scatter(Y[ind_l, 0], Y[ind_l, 1], color=colors[il+1], s=10)
        ax.axis('off')
        plt.title(method, fontsize=7)

        #check if the figure directory exists in the model directory
        if not os.path.exists(os.path.join(self.model_dir, 'figure')):
            os.makedirs(os.path.join(self.model_dir, 'figure'))
        #combine figname with the model directory
        figname = 'figure/taskvar_visual_by'+method+self.data_type+'.pdf'
        figname = os.path.join(self.model_dir, figname)

        if save:
            plt.savefig(figname, transparent=True)
        plt.show()

        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.scatter(Y[:,0], Y[:,1], color='black')
        ax.axis('off')

    def plot_example_unit(self):
        ######################## Plotting Variance for example unit ###################
        if self.data_type == 'rule':
            tick_names = [rule_name[r] for r in self.rules]

            ind = 2 # example unit
            fig = plt.figure(figsize=(1.2,1.0))
            ax = fig.add_axes([0.4,0.4,0.5,0.45])
            ax.plot(range(self.h_var_all.shape[1]), self.h_var_all[ind, :], 'o-', color='black', lw=1, ms=2)
            plt.xticks(range(len(tick_names)), [tick_names[0]] + ['.']*(len(tick_names)-2) + [tick_names[-1]],
                       rotation=90, fontsize=6, horizontalalignment='center')
            plt.xlabel('Task', fontsize=7, labelpad=-10)
            plt.ylabel('Task Variance', fontsize=7)
            plt.title('Unit {:d}'.format(self.ind_active[ind]), fontsize=7, y=0.85)
            plt.locator_params(axis='y', nbins=3)
            ax.tick_params(axis='both', which='major', labelsize=6, length=2)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if save:
                plt.savefig('./../figure/exampleunit_variance.pdf', transparent=True)
            plt.show()

            from analysis.standard_analysis import pretty_singleneuron_plot
            # Plot single example neuron in time
            pretty_singleneuron_plot(self.model_dir, ['fdgo'], [self.ind_active[ind]],
                                     epoch=None, save=save, ylabel_firstonly=True)
            
    def plot_connectivity_byclusters(self):
        """Plot connectivity of the model"""

        ind_active = self.ind_active

        # Sort data by labels and by input connectivity
        model = Model(self.model_dir)
        hp = model.hp
        with tf.Session() as sess:
            model.restore()
            w_in = sess.run(model.w_in).T
            w_rec = sess.run(model.w_rec).T
            w_out = sess.run(model.w_out).T
            b_rec = sess.run(model.b_rec)
            b_out = sess.run(model.b_out)

        w_rec = w_rec[ind_active, :][:, ind_active]
        w_in = w_in[ind_active, :]
        w_out = w_out[:, ind_active]
        b_rec = b_rec[ind_active]

        # nx, nh, ny = hp['shape']
        nr = hp['n_eachring']

        sort_by = 'w_in'
        if sort_by == 'w_in':
            w_in_mod1 = w_in[:, 1:nr+1]
            w_in_mod2 = w_in[:, nr+1:2*nr+1]
            w_in_modboth = w_in_mod1 + w_in_mod2
            w_prefs = np.argmax(w_in_modboth, axis=1)
        elif sort_by == 'w_out':
            w_prefs = np.argmax(w_out[1:], axis=0)

        # sort by labels then by prefs
        ind_sort = np.lexsort((w_prefs, self.labels))

        ######################### Plotting Connectivity ###############################
        nx = self.hp['n_input']
        ny = self.hp['n_output']
        nh = len(self.ind_active)
        nr = self.hp['n_eachring']
        nrule = len(self.hp['rules'])

        # Plot active units
        _w_rec  = w_rec[ind_sort,:][:,ind_sort]
        _w_in   = w_in[ind_sort,:]
        _w_out  = w_out[:,ind_sort]
        _b_rec  = b_rec[ind_sort, np.newaxis]
        _b_out  = b_out[:, np.newaxis]
        labels  = self.labels[ind_sort]

        l = 0.3
        l0 = (1-1.5*l)/nh

        plot_infos = [(_w_rec              , [l               ,l          ,nh*l0    ,nh*l0]),
                      (_w_in[:,[0]]        , [l-(nx+15)*l0    ,l          ,1*l0     ,nh*l0]), # Fixation input
                      (_w_in[:,1:nr+1]     , [l-(nx+11)*l0    ,l          ,nr*l0    ,nh*l0]), # Mod 1 stimulus
                      (_w_in[:,nr+1:2*nr+1], [l-(nx-nr+8)*l0  ,l          ,nr*l0    ,nh*l0]), # Mod 2 stimulus
                      (_w_in[:,2*nr+1:]    , [l-(nx-2*nr+5)*l0,l          ,nrule*l0 ,nh*l0]), # Rule inputs
                      (_w_out[[0],:]       , [l               ,l-(4)*l0   ,nh*l0    ,1*l0]),
                      (_w_out[1:,:]        , [l               ,l-(ny+6)*l0,nh*l0    ,(ny-1)*l0]),
                      (_b_rec              , [l+(nh+6)*l0     ,l          ,l0       ,nh*l0]),
                      (_b_out              , [l+(nh+6)*l0     ,l-(ny+6)*l0,l0       ,ny*l0])]

        # cmap = sns.diverging_palette(220, 10, sep=80, as_cmap=True)
        cmap = 'coolwarm'
        fig = plt.figure(figsize=(6, 6))
        for plot_info in plot_infos:
            ax = fig.add_axes(plot_info[1])
            vmin, vmid, vmax = np.percentile(plot_info[0].flatten(), [5,50,95])
            _ = ax.imshow(plot_info[0], interpolation='nearest', cmap=cmap, aspect='auto',
                          vmin=vmid-(vmax-vmin)/2, vmax=vmid+(vmax-vmin)/2)
            ax.axis('off')

        ax1 = fig.add_axes([l     , l+nh*l0, nh*l0, 6*l0])
        ax2 = fig.add_axes([l-6*l0, l      , 6*l0 , nh*l0])
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
            ax1.plot(ind_l, [0,0], linewidth=2, solid_capstyle='butt',
                    color=kelly_colors[il+1])
            ax2.plot([0,0], len(labels)-ind_l, linewidth=2, solid_capstyle='butt',
                    color=kelly_colors[il+1])
        ax1.set_xlim([0, len(labels)])
        ax2.set_ylim([0, len(labels)])
        ax1.axis('off')
        ax2.axis('off')
        if save:
            plt.savefig('./../figure/connectivity_by'+self.data_type+'.pdf', transparent=True)
        plt.show()

    def lesions(self):
        labels = self.labels

        from network import get_perf
        from task import generate_trials

        # The first will be the intact network
        lesion_units_list = [None]
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels == l)[0]
            # In original indices
            lesion_units_list += [self.ind_active[ind_l]]

        perfs_store_list = list()
        perfs_changes = list()
        cost_store_list = list()
        cost_changes = list()

        for i, lesion_units in enumerate(lesion_units_list):
            model = Model(self.model_dir)
            hp = model.hp
            with tf.Session() as sess:
                model.restore()
                model.lesion_units(sess, lesion_units)

                perfs_store = list()
                cost_store = list()
                
                print('SELF.RULES')
                print(self.rules)

                for rule in self.rules:
                    n_rep = 16
                    batch_size_test = 256
                    batch_size_test_rep = int(batch_size_test / n_rep)
                    clsq_tmp = list()
                    perf_tmp = list()
                    for i_rep in range(n_rep):
                        trial = generate_trials(rule, hp, 'random',
                                                batch_size=batch_size_test_rep)
                        feed_dict = tools.gen_feed_dict(model, trial, hp)
                        y_hat_test, c_lsq = sess.run(
                            [model.y_hat, model.cost_lsq], feed_dict=feed_dict)

                        # Cost is first summed over time, and averaged across batch and units
                        # We did the averaging over time through c_mask

                        # IMPORTANT CHANGES: take overall mean
                        perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
                        clsq_tmp.append(c_lsq)
                        perf_tmp.append(perf_test)

                    perfs_store.append(np.mean(perf_tmp))
                    cost_store.append(np.mean(clsq_tmp))

            perfs_store = np.array(perfs_store)
            cost_store = np.array(cost_store)

            perfs_store_list.append(perfs_store)
            cost_store_list.append(cost_store)

            if i > 0:
                perfs_changes.append(perfs_store - perfs_store_list[0])
                cost_changes.append(cost_store - cost_store_list[0])

        perfs_changes = np.array(perfs_changes)
        cost_changes = np.array(cost_changes)

        return perfs_changes, cost_changes

    def plot_lesions(self):
        """Lesion individual cluster and show performance."""

        perfs_changes, cost_changes = self.lesions()

        cb_labels = ['Performance change after lesioning',
                     'Cost change after lesioning']
        vmins = [-0.5, -0.5]
        vmaxs = [+0.5, +0.5]
        ticks = [[-0.5,0.5], [-0.5, 0.5]]
        changes_plot = [perfs_changes, cost_changes]

        fs = 6
        figsize = (3,4)#(2.5,2.5)
        rect = [0.3, 0.2, 0.5, 0.7]
        rect_cb = [0.82, 0.2, 0.03, 0.7]
        rect_color = [0.3, 0.15, 0.5, 0.05]
        for i in range(2):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes(rect)
            im = ax.imshow(changes_plot[i].T, cmap='coolwarm', aspect='auto',
                           interpolation='nearest', vmin=vmins[i], vmax=vmaxs[i])

            tick_names = [rule_name[r] for r in self.rules]
            _ = plt.yticks(range(len(tick_names)), tick_names,
                       rotation=0, va='center', fontsize=fs)
            plt.xticks([])
            plt.xlabel('Clusters', fontsize=7, labelpad=25)
            ax.tick_params('both', length=0)
            for loc in ['bottom','top','left','right']:
                ax.spines[loc].set_visible(False)

            ax = fig.add_axes(rect_cb)
            cb = plt.colorbar(im, cax=ax, ticks=ticks[i])
            cb.outline.set_linewidth(0.5)
            cb.set_label(cb_labels[i], fontsize=7, labelpad=-10)
            plt.tick_params(axis='both', which='major', labelsize=7)

            ax = fig.add_axes(rect_color)
            for il, l in enumerate(self.unique_labels):
                ax.plot([il, il+1], [0,0], linewidth=4, solid_capstyle='butt',
                        color=kelly_colors[il+1])
                ax.text(np.mean(il+0.5), -0.5, str(il+1), fontsize=6,
                        ha='center', va='top', color=kelly_colors[il+1])
            ax.set_xlim([0, len(self.unique_labels)])
            ax.set_ylim([-1, 1])
            ax.axis('off')

            if save:
                plt.savefig('./../figure/lesion_cluster_by'+self.data_type+'_{:d}.pdf'.format(i), transparent=True)

if __name__ == '__main__':
    root_dir = './data/train_all'
    model_dir = root_dir + '/1'
    # CA = Analysis(model_dir, data_type='rule')
