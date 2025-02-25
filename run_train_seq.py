import train
import os

#train on gpu 1
rules = ['multidm', 'delaydm1', 'delaydm2', 'contextdelaydm1', 
        'contextdelaydm2', 'dmsgo', 
        'dmsnogo', 'dmcgo', 'dmcnogo']
# rules = ['reactgo', 'delaygo', 'fdgo', 
#         'reactanti', 'delayanti', 'fdanti', 
#         'dm1', 'dm2', 'contextdm1', 'contextdm2']

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

for rule in rules:
    # Call the training function from the train module
    train.train(model_dir=f'retrain_reactgo_rnnbias_outputbias_{rule}', 
                hp={'learning_rate': 0.001, 
                    'n_rnn': 1024,
                    'b_rec_init': 'uniform',
                    'w_rec_init': 'randgauss',
                    'rule_strength': 0.0,
                    'no_rule': True,
                    'target_perf': 0.90,
                    'activation': 'softplus',
                    'alpha': 0.2},
                ruleset='all',
                rule_trains=[rule], # Current rule to train
                pretrained_dir='train_all_params_reactgo',
                apply_pretrained_params=['rnn/leaky_rnn_cell/kernel:0','output/weights:0','output/biases:0'],
                trainables='all_bias')