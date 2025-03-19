import numpy as np
from analysis import taskset
import os


model_dir='./../models/fdanti_delaygo_fdgo_delaydm1_dm1_delaydm2_dm2_contextdelaydm1_contextdm1_contextdelaydm2_dm2_multidelaydm_multidm_1024'
replace_rule= ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo', 'random', 'random_mod']

out_of_diag_perc=0.0005
grid=np.linspace(0,1,11)

performances=np.zeros(shape=(11,11,11,11,11,11))

for i,a in enumerate(grid):
    for j,b in enumerate(grid):
        for k,c in enumerate(grid):
            for l,d in enumerate(grid):
                for m,e in enumerate(grid):
                    for n,f in enumerate(grid):
            
                        tot=(int((a+b+c+d+e+f)*10))/10

                        if tot < 0.8 or tot > 1.2:
                            if np.random.rand() < 1-out_of_diag_perc:
                                continue    

                        print(a,b,c,d,e,f)

                        rule_strength= a*np.array([-1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])+\
                                        b*np.array([0,0,0,1,0,0,-1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])+\
                                        c*np.array([0,0,0,1,0,0,0,-1,0,0,0,0,1,0,0,0,0,0,0,0,0,0])+\
                                        d*np.array([0,0,0,1,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,0,0,0,0])+\
                                        e*np.array([0,0,0,1,0,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,0,0,0])+\
                                        f*np.array([0,0,0,1,0,0,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,0,0])

                        perf, _ = taskset.run_network_replacerule(model_dir, 'delayanti', replace_rule, rule_strength)
                        performances[i,j,k,l,m,n]=perf

np.save(os.path.join(model_dir,'comp_rule_linear_comb_3.npy'),performances)
        
