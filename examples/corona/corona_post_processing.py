import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import os
import fabsim3_cmd_api as fab
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
from scipy import stats

# author: Wouter Edeling
__license__ = "LGPL"

#home directory of user
home = os.path.expanduser('~')

def sobol_table(results, param_names, **kwargs):
    
    if 'qoi_cols' in kwargs:
        qoi_cols = kwargs['qoi_cols']
    else:
        qoi_cols = results['sobols'].keys()

    for qoi in qoi_cols:
        sobol_idx = results['sobols'][qoi]
        print('=======================')
        print('Sobol indices', qoi)
        for key in sobol_idx.keys():
            name = 'S('
            l = 0
            for idx in key[0:-1]:
                name += param_names[l] + ', '
                l += 1
            name += param_names[key[l]] + ')'
            print(name, '=' , '%.4f' % sobol_idx[key][0])        
        
#Reload the campaign
my_campaign = uq.Campaign(state_file = "./campaign_state.json", work_dir = '/tmp')

print('========================================================')
print('Reloaded campaign', my_campaign.campaign_dir.split('/')[-1])
print('========================================================')

#get sampler and output columns from my_campaign object
my_sampler = my_campaign.get_active_sampler()
output_columns = my_campaign._active_app_decoder.output_columns

#fetch the results from the (remote) host via FabSim3
fab.get_uq_samples(my_campaign.campaign_dir, machine='localhost')

#collate output
my_campaign.collate()


# Post-processing analysis
sc_analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
my_campaign.apply_analysis(sc_analysis)
results = my_campaign.get_last_analysis()

mu = results['statistical_moments']['V100']['mean']
std = results['statistical_moments']['V100']['std']

fig = plt.figure()
ax = fig.add_subplot(111, xlabel = 'time', ylabel='number of infected')
ax.plot(mu)
ax.plot(mu + std, '--r')
ax.plot(mu - std, '--r')
plt.tight_layout()
plt.show()