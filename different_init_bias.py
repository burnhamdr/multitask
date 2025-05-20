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

#initialize input weights
if in_init=='randgauss':
    in_weights = (rand.randn(n_input, n_hidden) /
            np.sqrt(n_input) * 1.)
elif in_init=='ones':
    in_weights = (np.ones((n_input, n_hidden)) * 1.)

# initialize recurrent weights
if rec_init == 'singlering':
    rec_weights = network.create_recurrent_weights_ring_attractor(n_hidden, 0.5, 1)
elif rec_init == 'lowranknoise':
    rec_weights = network.init_bernoulli_lowrank_plus_noise(n_hidden, 10, 0.5)
elif rec_init == 'newsinglering':
    rec_weights = network.new_create_recurrent_weights_ring_attractor(n_hidden, 2.1, 1.6, 2.)
elif rec_init == 'newdoublering':
    rec_weights = network.new_create_recurrent_weights_two_ring_attractor(n_hidden, 2.1, 1.6, 2.)

# initialize output weights
if out_init=='glorot_uniform':
    out_weights=tf.get_variable('weights',[n_hidden, n_out],dtype=tf.float32,initializer=init_ops.glorot_uniform_initializer(dtype=tf.float32))
elif out_init=='randgauss':
    out_weights=tf.get_variable('weights',[n_hidden, n_out],dtype=tf.float32,initializer=init_ops.random_normal_initializer(dtype=tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out_weights = sess.run(out_weights)

print("################### IN WEIGHTS ###################")
print(in_weights)
print(in_weights.shape)
print(type(in_weights))
print("################### REC WEIGHTS ###################")
print(rec_weights)
print(rec_weights.shape)
print(type(rec_weights))
print("################### OUT WEIGHTS ###################")
print(out_weights)
print(out_weights.shape)
print(type(out_weights))

from multiprocessing.pool import ThreadPool

def run_train_parallel(b_init,name,seed):

    train.train(model_dir=f'./../models/newsingle_bias_fdgo_fdanti_delaygo_{n_hidden}_01/{name}', 
                hp={"n_rnn": n_hidden,
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
                    "w_rec_init": rec_weights,
                    "num_attractors": 15,
                    "w_in_init": in_weights,
                    "w_out_init": out_weights,
                    "b_rec_init": b_init,
                    "b_out_init": b_init,
                    "rule_strength": 1.0,
                    "tau": 100, "dt": 20, "alpha": 0.2, "sigma_rec": 0.05, "sigma_x": 0.01, "l1_h": 0, "l2_h": 0, "l1_weight": 0, "l2_weight": 0, "l2_weight_init": 0, "p_weight_train": None, "n_eachring": 32, "num_ring": 2, "n_rule": 22, "rule_start": 65, "n_input": 87, "n_output": 33, "ruleset": "all", "save_name": "test", "c_intsyn": 0, "ksi_intsyn": 0, "no_rule": False, "seed": 0, "rule_trains": ["fdgo", "fdanti", "delaygo"], "rules": ["fdgo", "fdanti", "delaygo"], "rule_probs": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]},
                seed=seed+1,
                ruleset='all',
                rule_trains = ['dm1'],
                max_steps=1e8,
                trainables='all_bias')

    return

pool = ThreadPool()
results = pool.starmap(run_train_parallel, zip(b_inits,names,range(len(names))))