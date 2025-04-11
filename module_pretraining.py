import task
from task import generate_trials 
from train import get_default_hp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

def get_stacked_io(hp,leave_out=[]):

    all_tasks=['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm']#,
              #'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo', 'random', 'random_mod']

    stacked_inputs=[]
    stacked_outputs=[]

    trials={}

    for task in all_tasks:

        if task not in leave_out:

            trials[task]=generate_trials(task,hp,'random',batch_size=10)
            print('####################')
            print(task)
            print(trials[task].__str__())
            print('####################')

    max_time=0

    for task in trials:

        max_time=max(max_time,trials[task].x.shape[0])

    print(max_time)

    for task in trials:

        trial=trials[task]
        inputs=trial.x
        outputs=trial.y

        task_length=trials[task].x.shape[0]
        missing_timesteps=max_time-task_length

        indexes_to_copy=np.random.choice(a=np.arange(task_length),size=missing_timesteps)

        inputs=np.insert(inputs,indexes_to_copy,inputs[indexes_to_copy-1,:,:],axis=0)
        outputs=np.insert(outputs,indexes_to_copy,outputs[indexes_to_copy-1,:,:],axis=0)

        inputs_shape=inputs.shape
        outputs_shape=outputs.shape

        inputs_transformed = np.reshape(inputs,(inputs_shape[0]*inputs_shape[1],inputs_shape[2]))
        outputs_transformed = np.reshape(outputs,(outputs_shape[0]*outputs_shape[1],outputs_shape[2]))

        inputs_transformed = StandardScaler().fit_transform(inputs_transformed)
        outputs_transformed = StandardScaler().fit_transform(outputs_transformed)


    



hp=get_default_hp('all')
hp['rule_trains'] = task.rules_dict['all']
hp['seed'] = 0
hp['rng'] = np.random.RandomState(0)
get_stacked_io(hp)




