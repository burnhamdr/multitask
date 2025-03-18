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

train.train(model_dir='compositional_delayanti', 
                hp={'learning_rate': 0.001, 
                    'n_rnn': 1024,
                    'w_rec_init': 'randgauss',
                    'b_rec_init': 'uniform',
                    'rule_strength': 0.,
                    'no_rule': False,
                    'target_perf': 0.90,
                    'activation': 'softplus',
                    'alpha': 0.2,
                    'use_separate_input':False},
                ruleset='all',
                rule_trains=['delayanti'])