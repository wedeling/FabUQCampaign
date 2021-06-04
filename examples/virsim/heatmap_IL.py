"""
@author: Federica Gugole

__license__= "LGPL"
"""

import numpy as np
import easyvvuq as uq
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = 8,6

"""
*****************
* VVUQ ANALYSES *
*****************
"""

# home directory of this file    
HOME = os.path.abspath(os.path.dirname(__file__))

# Reload the campaign
workdir = '/export/scratch1/federica/VirsimCampaigns'
campaign = uq.Campaign(state_file = "campaign_state_IL_nobio_MC2k.json", work_dir = workdir)
print('========================================================')
print('Reloaded campaign', campaign.campaign_dir.split('/')[-1])
print('========================================================')

# get sampler from my_campaign object
sampler = campaign._active_sampler

# collate output
campaign.collate()

# get full dataset of data
data = campaign.get_collation_result()
#print(data.columns)

# get analysis object
output_columns = campaign._active_app_decoder.output_columns
qmc_analysis = uq.analysis.QMCAnalysis(sampler=sampler, qoi_cols=output_columns)

###
IC_capacity = 109
n_runs = 2000
n_params = 5

lockdown_effect, tmp1, tmp2 = qmc_analysis._separate_output_values(data['lockdown_effect',0], n_params, n_runs)

uptake, tmp1, tmp2 = qmc_analysis._separate_output_values(data['uptake',0], n_params, n_runs)

IC_prev_avg_max, tmp1, tmp2 = qmc_analysis._separate_output_values(data['IC_prev_avg_max',0], n_params, n_runs)

IC_ex_max, tmp1, tmp2 = qmc_analysis._separate_output_values(data['IC_ex_max',0], n_params, n_runs)

# Plot

f = plt.figure('heatmap',figsize=[12,6])
ax_p = f.add_subplot(121, xlabel='Relative level of transmission \n due to lockdown', ylabel='Uptake by the population')
im_p = ax_p.scatter(x=lockdown_effect[np.where(IC_prev_avg_max <= IC_capacity)], y=uptake[np.where(IC_prev_avg_max <= IC_capacity)], \
	c='black')
im_p = ax_p.scatter(x=lockdown_effect[np.where(IC_prev_avg_max > IC_capacity)], y=uptake[np.where(IC_prev_avg_max > IC_capacity)], \
	c=IC_prev_avg_max[np.where(IC_prev_avg_max > IC_capacity)], cmap='plasma')
cbar_p = f.colorbar(im_p, ax=ax_p)
cbar_p.set_ticks([200, 400, 600, 800])
cbar_p.set_ticklabels(['200', '400', '600', '800'])

ax_p.set_xticks([0.1, 0.2, 0.3, 0.4])
ax_p.set_yticks([0.6, 0.8, 1.0])

ax_e = f.add_subplot(122, xlabel='Relative level of transmission \n due to lockdown')
im_e = ax_e.scatter(x=lockdown_effect[np.where(IC_ex_max == 0)], y=uptake[np.where(IC_ex_max == 0)], \
	c='black')
im_e = ax_e.scatter(x=lockdown_effect[np.where(IC_ex_max > 0)], y=uptake[np.where(IC_ex_max > 0)], \
	c=IC_ex_max[np.where(IC_ex_max > 0)], cmap='plasma')
cbar_e = f.colorbar(im_e, ax=ax_e)
cbar_e.set_ticks([1e4, 2e4, 3e4, 4e4])
cbar_e.set_ticklabels(['10000', '20000', '30000', '40000'])

ax_e.set_xticks([0.25, 0.35, 0.45])
ax_e.set_yticks([0.6, 0.8, 1.0])

plt.tight_layout()
f.savefig('figures/Fig6_heatmap_IL.eps')

plt.show()

### END OF CODE ###

