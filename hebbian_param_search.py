import train
from train import Model
import tensorflow as tf
import matplotlib.pyplot as plt
from analysis import taskset
import tools
import numpy as np 
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

etas=[10**(-i) for i in range(5,16)]
alphas=np.linspace(0,0.9,10)[::-1]

for eta in etas:
    for alpha in alphas:

        # Parameters
        model_dir='./../models/hebbian_bias_dm1_256_01_2'

        hebbian_update_steps=100

        hp={"n_rnn": 256,
                "target_perf": 0.95,
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
                "tau": 100, "dt": 20, "alpha": 0.2, "sigma_rec": 0.05, "sigma_x": 0.01, "l1_h": 0, "l2_h": 0, "l1_weight": 0, "l2_weight": 0, "l2_weight_init": 0, "p_weight_train": None, "n_eachring": 32, "num_ring": 2, "n_rule": 22, "rule_start": 65, "n_input": 87, "n_output": 33, "ruleset": "all", "save_name": "test", "c_intsyn": 0, "ksi_intsyn": 0, "no_rule": False, "seed": 0, "rule_trains": ["fdgo", "fdanti", "delaygo"], "rules": ["fdgo", "fdanti", "delaygo"], "rule_probs": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]}

        # Creation of the model (trains for 1 step just to create the model)
        train.train(model_dir=model_dir, 
            hp=hp,
            ruleset='all',
            max_steps=1,
            rule_trains = ['dm1'],
            trainables='all_bias',
            display_hp=False)

        # Hessian pretraining
        model=Model(model_dir,display_info=False)

        with tf.Session() as sess:
            model.restore()

            # Get initial weights
            initial_in_weights=model.w_in.eval()
            initial_in_weights=initial_in_weights.reshape((initial_in_weights.shape[0]*initial_in_weights.shape[1]))

            mean_before=np.mean(initial_in_weights)
            var_before=np.var(initial_in_weights)

            # Make histogram of them
            fig,ax=plt.subplots()
            ax.hist(initial_in_weights)
            fig.savefig('prima.png')

            for i in tqdm(range(hebbian_update_steps), desc='Running pretraining'):

                # Get current weights
                in_weights_old=model.w_in.eval()

                # Generate trial and pass it through the network
                curr_task=np.random.choice(model.hp['rule_trains'])
                trial=taskset.generate_trials(curr_task,model.hp,mode='random',batch_size=1)
                feed_dict = tools.gen_feed_dict(model, trial, model.hp)
                phi = sess.run(model.h, feed_dict=feed_dict) # Shape (timesteps, batchsize, numrecneurons)
                s=trial.x   # Shape (timesteps, batchsize, numinputs)

                # Reshape to remove batchsize dimension and average over timesteps
                phi_collapsed=np.mean(phi.reshape((phi.shape[0],phi.shape[2])),axis=0)
                s_collapsed=np.mean(s.reshape((s.shape[0],s.shape[2])),axis=0)

                # Calculate new weights
                in_weights_new=(1-alpha)*in_weights_old + eta*np.outer(s_collapsed,phi_collapsed)

                # Normalize to avoid overflow
                #norm = np.linalg.norm(in_weights_new)
                #in_weights_new = in_weights_new / norm

                # Assign updated weights
                sess.run(model.w_in.assign(in_weights_new))
            
            # Extract input weights post pre-training
            final_in_weights=model.w_in.eval()
            final_in_weights=final_in_weights.reshape((final_in_weights.shape[0]*final_in_weights.shape[1]))

            
            print('Alpha: ', alpha)
            print('Eta: ', eta)

            print('Mean before: ', mean_before)
            print('Var before: ', var_before)

            print('Mean after: ', np.mean(final_in_weights))
            print('Var after: ', np.var(final_in_weights))
