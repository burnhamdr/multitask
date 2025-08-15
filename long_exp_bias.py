import train
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#print(tf.test.gpu_device_name())
#print(tf.test.is_gpu_available())
#with tf.device('/device:CPU:0'):

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

#import keras.backend as K
#cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
#K.set_session(K.tf.Session(config=cfg))

#with tf.Graph().as_default():

#with sess.as_default():
if True:

    train.train(model_dir='./../models/bias_dmsgo_1024_01', 
                hp={"n_rnn": 1024,
                    "target_perf": 0.97,
                    "learning_rate": 0.01,
                    "batch_size_train": 64,
                    "batch_size_test": 512,
                    "in_type": "normal",
                    "rnn_type": "LeakyRNN",
                    "use_separate_input": False, 
                    "loss_type": "lsq",
                    "optimizer": "adam", 
                    "activation": "softplus",
                    "w_rec_init": "randgauss",
                    "w_in_init": "randgauss",
                    "w_out_init": "randgauss",
                    "b_rec_init": "zeros",
                    "b_out_init": "zeros",
                    "rule_strength": 1.0,
                    "tau": 100, "dt": 20, "alpha": 0.2, "sigma_rec": 0.05, "sigma_x": 0.01, "l1_h": 0, "l2_h": 0, "l1_weight": 0, "l2_weight": 0, "l2_weight_init": 0, "p_weight_train": None, "n_eachring": 32, "num_ring": 2, "n_rule": 22, "rule_start": 65, "n_input": 87, "n_output": 33, "ruleset": "all", "save_name": "test", "c_intsyn": 0, "ksi_intsyn": 0, "no_rule": False, "seed": 0, "rule_trains": ["fdgo", "fdanti", "delaygo"], "rules": ["fdgo", "fdanti", "delaygo"], "rule_probs": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]},
                ruleset='all',
                rule_trains = ['dmsgo'],#'fdgo',contextdm1,dm1,dmsgo
                trainables='all_bias')
