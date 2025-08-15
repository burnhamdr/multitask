import matplotlib.pyplot as plt
import numpy as np

kelly_colors = \
[np.array([ 0.94901961,  0.95294118,  0.95686275]),
 np.array([ 0.13333333,  0.13333333,  0.13333333]),
 np.array([ 0.95294118,  0.76470588,  0.        ]),
 np.array([ 0.52941176,  0.3372549 ,  0.57254902]),
 np.array([ 0.95294118,  0.51764706,  0.        ]),
 np.array([ 0.63137255,  0.79215686,  0.94509804]),
 np.array([ 0.74509804,  0.        ,  0.19607843]),
 np.array([ 0.76078431,  0.69803922,  0.50196078]),
 np.array([ 0.51764706,  0.51764706,  0.50980392]),
 np.array([ 0.        ,  0.53333333,  0.3372549 ]),
 np.array([ 0.90196078,  0.56078431,  0.6745098 ]),
 np.array([ 0.        ,  0.40392157,  0.64705882]),
 np.array([ 0.97647059,  0.57647059,  0.4745098 ]),
 np.array([ 0.37647059,  0.30588235,  0.59215686]),
 np.array([ 0.96470588,  0.65098039,  0.        ]),
 np.array([ 0.70196078,  0.26666667,  0.42352941]),
 np.array([ 0.8627451 ,  0.82745098,  0.        ]),
 np.array([ 0.53333333,  0.17647059,  0.09019608]),
 np.array([ 0.55294118,  0.71372549,  0.        ]),
 np.array([ 0.39607843,  0.27058824,  0.13333333]),
 np.array([ 0.88627451,  0.34509804,  0.13333333]),
 np.array([ 0.16862745,  0.23921569,  0.14901961])]

with open('./perf_profiles2.txt','r') as file:
    for i,line in enumerate(file):
        perf_prof=line[:-1].split(',')
        col=kelly_colors[i+2]
        plt.scatter([len(perf_prof)], [0.35], c=[col], clip_on=False)
        plt.plot(range(len(perf_prof)),[float(x) for x in perf_prof],label=f'Run n.{i+1}',c=col)

with open('./perf_profiles_pretrained2.txt') as file:
    for line in file:
        perf_prof=line[:-1].split(',')
        plt.scatter([len(perf_prof)], [0.35], c=['black'], clip_on=False)
        plt.plot(range(len(perf_prof)),[float(x) for x in perf_prof],label='Pretrained', c='black')

plt.title('Comparison in learning time between\n random and pretrained models')
plt.xlabel('Number of processed inputs')
plt.ylabel('Performance')
plt.ylim(0.35,1)
plt.axhline(0.6,label='Baseline',linestyle='--',c='red')
plt.axhline(0.9,label='Target\nperformance',linestyle='--',c='green')
plt.legend(ncol=3,fontsize=8,bbox_to_anchor=(0.35, 0.05, 0.5, 0.5))
plt.savefig('prova2.png')