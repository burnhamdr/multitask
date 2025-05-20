import train
import tools
from analysis import performance
from analysis import standard_analysis
from analysis import clustering
from analysis import variance
from analysis import taskset
from analysis import varyhp
from analysis import data_analysis
from analysis import contextdm_analysis
from analysis import posttrain_analysis
from network import Model
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train.train(model_dir='./../models/newsinglering_bias_fdgo_fdanti_delaygo_256_01/new01', 
                hp={"n_rnn": 1024,
                    "target_perf": 0.85,
                    "learning_rate": 0.01,
                    "batch_size_train": 64,
                    "batch_size_test": 512,
                    "in_type": "normal",
                    "rnn_type": "LeakyRNN",
                    "use_separate_input": False, 
                    "loss_type": "lsq", 
                    "optimizer": "adam", 
                    "activation": "softplus",
                    "tau": 100, "dt": 20, "alpha": 0.2, "sigma_rec": 0.05, "sigma_x": 0.01, "w_rec_init": "randgauss", "w_in_init": "randgauss", "w_out_init": "glorot_uniform", "b_rec_init": "uniform", "b_out_init": "zeros", "l1_h": 0, "l2_h": 0, "l1_weight": 0, "l2_weight": 0, "l2_weight_init": 0, "p_weight_train": None, "n_eachring": 32, "num_ring": 2, "n_rule": 22, "rule_start": 65, "n_input": 87, "n_output": 33, "ruleset": "all", "save_name": "test", "c_intsyn": 0, "ksi_intsyn": 0, "rule_strength": 1.0, "no_rule": False, "seed": 0, "rule_trains": ["fdgo", "fdanti", "delaygo"], "rules": ["fdgo", "fdanti", "delaygo"], "rule_probs": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]},
                pretrained_dir='./../models/newsinglering_bias_fdgo_fdanti_delaygo_256_01/zeros10',
                apply_pretrained_params='all',
                ruleset='all',
                rule_trains=['fdgo', 'fdanti', 'delaygo'],
                trainables='all_bias')