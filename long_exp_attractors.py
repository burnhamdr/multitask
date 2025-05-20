import numpy as np
import train
import os
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# STILL MISSING
# SOMEWHAT: dm1, dm2, contextdm1, contextdm2, delaydm1, 
# NOT AT ALL: delaydm2, multidm, contextdelaydm1, contextdelaydm2, multidelaydm

train.train(model_dir='./../models/newsinglering_bias_fdgo_fdanti_delaygo_1024_01_6', 
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
                    "w_rec_init": 'newsinglering',
                    "num_attractors": 15,
                    "w_in_init": "randgauss",
                    "w_out_init": "randgauss",
                    "b_rec_init": "zeros",
                    "b_out_init": "zeros",
                    "rule_strength": 1.0,
                    "tau": 100, "dt": 20, "alpha": 0.2, "sigma_rec": 0.05, "sigma_x": 0.01, "l1_h": 0, "l2_h": 0, "l1_weight": 0, "l2_weight": 0, "l2_weight_init": 0, "p_weight_train": None, "n_eachring": 32, "num_ring": 2, "n_rule": 22, "rule_start": 65, "n_input": 87, "n_output": 33, "ruleset": "all", "save_name": "test", "c_intsyn": 0, "ksi_intsyn": 0, "no_rule": False, "seed": 0, "rule_trains": ["fdgo", "fdanti", "delaygo"], "rules": ["fdgo", "fdanti", "delaygo"], "rule_probs": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]},
                ruleset='all',
                max_steps=1e8,
                rule_trains = ['fdgo','fdanti','delaygo'],
                trainables='all_bias')
