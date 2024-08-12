import train

train.train(model_dir='retrain_rnnbias_outputbias_contextdm2', 
            hp={'learning_rate': 0.001, 
                'n_rnn': 1024,#1024, 16384,8192
                'b_rec_init': 'uniform',
                'w_rec_init': 'randgauss',#'randortho'
                'rule_strength': 0.0,
                'no_rule': True,
                'target_perf':0.98,
                'activation': 'softplus',
                'alpha':0.2},#'relu'
            ruleset='all',
            rule_trains = ['contextdm2'],#'fdgo',contextdm1,dm1,dmsgo,delaydm1,dmcgo,reactgo,multidm
            pretrained_dir = 'train_all_params_contextdelaydm1',
            apply_pretrained_params = ['rnn/leaky_rnn_cell/kernel:0','output/weights:0','output/biases:0'],#initialize ins and rec weights
            trainables='all_bias'
)