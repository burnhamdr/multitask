import train

# train.train(model_dir='retrain_rnnbias_outputweights_fdgo', 
#             hp={'learning_rate': 0.001, 
#                 'n_rnn': 1024,#1024, 16384,8192
#                 'b_rec_init': 'uniform',
#                 'w_rec_init': 'randgauss',#'randortho'
#                 'rule_strength': 0.0,
#                 'no_rule': True,
#                 'target_perf':0.98,
#                 'activation': 'softplus',
#                 'alpha':0.2},#'relu'
#             ruleset='all',
#             rule_trains = ['fdgo'],#'fdgo',contextdm1,dm1,dmsgo,delaydm1,dmcgo,reactgo,multidm
#             pretrained_dir = 'train_all_params_contextdelaydm1',
#             apply_pretrained_params = ['rnn/leaky_rnn_cell/kernel:0','output/weights:0','output/biases:0'],#initialize ins and rec weights
#             trainables='rnn_bias_and_output_weights'#'all_bias'
# )

train.train(model_dir='train_all_params_reactgo',
            hp={'learning_rate': 0.001, 
                'n_rnn': 1024,#512, 16384,8192,1024
                'w_rec_init': 'randgauss',#'randortho'
                'b_rec_init': 'uniform',
                'rule_strength': 0.0,
                'no_rule': True,
                'target_perf':1.0,
                'activation': 'softplus',
                'alpha':0.2},
            ruleset='all',
            rule_trains = ['reactgo'])#,trainables='bias')