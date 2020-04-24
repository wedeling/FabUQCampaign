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
 
#post processing of UQ samples executed via FabSim. All samples must have been completed
#before this subroutine is executed. Use 'fabsim <machine_name> job_stat' to check their status

#home dir of this file    
HOME = os.path.abspath(os.path.dirname(__file__))
# work_dir = home + "/VECMA/Campaigns/"
work_dir = '/home/wouter/VECMA/Campaigns'

#Reload the campaign
my_campaign = uq.Campaign(state_file = 'campaign_state.json', work_dir = work_dir)

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
results['n_samples'] = sc_analysis._number_of_samples

mu_S = results['statistical_moments']['S']['mean']
std_S = results['statistical_moments']['S']['std']
mu_I = results['statistical_moments']['I']['mean']
std_I = results['statistical_moments']['I']['std']
mu_R = results['statistical_moments']['R']['mean']
std_R = results['statistical_moments']['R']['std']
mu_H = results['statistical_moments']['H']['mean']
std_H = results['statistical_moments']['H']['std']
mu_D = results['statistical_moments']['D']['mean']
std_D = results['statistical_moments']['D']['std']

#Plot mean and stddev of S, I and R in one plot.
T = mu_S.size
time = range(T)
fig = plt.figure('SIR') 

ax = fig.add_subplot(111, xlabel='time')
ax.plot(time, mu_S, 'b', label=r'$\bar{S}\pm\sigma_s$')
ax.plot(time, mu_S + std_S, '--b', time, mu_S - std_S, '--b')

ax.plot(time, mu_I, 'r', label=r'$\bar{I}\pm\sigma_I$')
ax.plot(time, mu_I + std_I, '--r', time, mu_I - std_I, '--r')

ax.plot(time, mu_R, 'g', label=r'$\bar{R}\pm\sigma_R$')
ax.plot(time, mu_R + std_R, '--g', time, mu_R - std_R, '--g')

leg = plt.legend()
leg.set_draggable(True)
plt.tight_layout()

#Plot mean and stddev of H and D in one plot.
fig = plt.figure('HD') 
ax = fig.add_subplot(111, xlabel='time')
ax.plot(time, mu_H, 'b', label=r'$\bar{H}\pm\sigma_H$')
ax.plot(time, mu_H + std_H, '--b', time, mu_H - std_H, '--b')

ax.plot(time, mu_D, 'r', label=r'$\bar{D}\pm\sigma_D$')
ax.plot(time, mu_D + std_D, '--r', time, mu_D - std_D, '--r')

leg = plt.legend()
leg.set_draggable(True)
plt.tight_layout()

#################################
# Use SC expansion as surrogate #
#################################

#number of MC samples
n_mc = 500

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='time')
    
#get the input distributions
theta = my_sampler.vary.get_values()
xi = np.zeros([n_mc, 5])
idx = 0

#draw random sampler from the input distributions
for theta_i in theta:
    xi[:, idx] = theta_i.sample(n_mc)
    idx += 1
    
#evaluate the surrogate at the random values
Q = 'S'
surr = np.zeros([n_mc, T])
for i in range(n_mc):
    surr[i] = sc_analysis.surrogate(Q, xi[i])
    
#make a list of actual samples
samples = sc_analysis.get_sample_array('S')

ax.plot(time, surr.T, 'g', label=r'surrogate S')
ax.plot(time, samples.T, 'r+', alpha = 0.3, label='code sample S')
#remove duplicate lagends
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.tight_layout()

######################
# plot Sobol indices #
######################

#first order Sobol indices and parameter names
sobols = results['sobols_first']
params = list(my_sampler.vary.get_keys())

#there is very litte variation in the first points (D = approx 0), leads
#to unstable results, do not plot these points
skip = 20

fig = plt.figure('Sobols_SIR', figsize=[12, 4])
ax_S = fig.add_subplot(131, xlabel='time', title = 'S')
ax_I = fig.add_subplot(132, xlabel='time', title = 'I')
ax_R = fig.add_subplot(133, xlabel='time', title = 'R')

for param in params: 
    ax_S.plot(time[skip:], sobols['S'][param][skip:], label=param)
    ax_I.plot(time[skip:], sobols['I'][param][skip:])
    ax_R.plot(time[skip:], sobols['R'][param][skip:])

leg = plt.legend()
leg.set_draggable(True)
plt.tight_layout()

#######################

fig = plt.figure('Sobols_HD', figsize=[8, 4])
ax_H = fig.add_subplot(121, xlabel='time', title = 'H')
ax_D = fig.add_subplot(122, xlabel='time', title = 'D')

for param in params: 
    ax_H.plot(time[skip:], sobols['H'][param][skip:], label=param)
    ax_D.plot(time[skip:], sobols['D'][param][skip:])

leg = plt.legend()
leg.set_draggable(True)
plt.tight_layout()

plt.show()