import numpy as np
import train
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import network
from tensorflow.python.ops import init_ops

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

in_init='randgauss'
rec_init='newsinglering'
out_init='randgauss'
n_input=87
n_hidden=256
n_out=33
b_inits=['zeros']*20#+['randgauss']*5+['uniform']*5
names=[f'zeros{i}' for i in range(20)]#+[f'randgauss{i+6}' for i in range(5)]+[f'uniform{i+6}' for i in range(5)]
rand=np.random.RandomState(0)

from multiprocessing.pool import ThreadPool

def run_train_parallel(b_init,name,seed):

    train.train(model_dir=f'./../models/newsinglering_bias_fdgo_fdanti_delaygo_{n_hidden}_01/{name}', 
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
                seed=seed+1,
                ruleset='all',
                max_steps=1e8,
                rule_trains = ['fdgo','fdanti','delaygo'],
                trainables='all_bias')
    
    return

pool = ThreadPool()
results = pool.starmap(run_train_parallel, zip(b_inits,names,range(len(names))))