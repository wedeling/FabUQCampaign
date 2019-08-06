import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import os
import fabsim3_cmd_api as fab
import pandas as pd
import vvp
from scipy import stats

# author: Wouter Edeling
__license__ = "LGPL"

#home directory of user
home = os.path.expanduser('~')

##subroutine which runs the gets the EasyVVUQ samples via FabSim's ensemble2campaign 
#def get_UQ_results(campaign_dir, machine = 'localhost'):
#    sim_ID = campaign_dir.split('/')[-1]
#    os.system("fabsim " + machine + " get_uq_samples:" + sim_ID + ",campaign_dir=" + campaign_dir)

#should be part of EasyVVUQ SCSampler
def load_uq_csv_output(run_dir, **kwargs):
    qoi = kwargs['qoi']
    df = pd.read_csv(run_dir + '/output.csv', names=['E_mean', 'Z_mean', 'E_std', 'Z_std'])
    return np.float(df[qoi].values[1])

def store_uq_results(campaign_dir, results):
    df = pd.DataFrame.from_dict(results)
    df.to_pickle(campaign_dir + '/results.pickle')

def load_uq_results(campaign_dir, **kwargs):

    df = pd.read_pickle(campaign_dir + '/results.pickle')
    return df

def plot_convergence(scores, **kwargs):
    """
    plot the convegence of the statistical moments
    """
    means = []
    stds = []    
    
    for score in scores:
        means.append(score['statistical_moments'][kwargs['qoi']]['mean'][0])
        stds.append(score['statistical_moments'][kwargs['qoi']]['std'][0])

    fig = plt.figure(figsize=[8,4])
    ax1 = fig.add_subplot(121)
    ax1.plot(means, '-bo')
    ax2 = fig.add_subplot(122)
    ax2.plot(stds, '-bo')
    
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    
def get_kde(X, Npoints = 100):

    kernel = stats.gaussian_kde(X)
    x = np.linspace(np.min(X), np.max(X), Npoints)
    pde = kernel.evaluate(x)
    return x, pde
    
def sobol_table(results, param_names, **kwargs):
    
    if 'qoi_cols' in kwargs:
        qoi_cols = kwargs['qoi_cols']
    else:
        qoi_cols = results['sobol_indices'].keys()

    for qoi in qoi_cols:
        sobol_idx = results['sobol_indices'][qoi]
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
        
#post processing of UQ samples executed via FabSim. All samples must have been completed
#before this subroutine is executed. Use 'fabsim <machine_name> job_stat' to check their status
def post_proc(state_file, work_dir):
    
    #Reload the campaign
    my_campaign = uq.Campaign(state_file = state_file, work_dir = work_dir)

    print('========================================================')
    print('Reloaded campaign', my_campaign.campaign_dir.split('/')[-1])
    print('========================================================')
    
    #get sampler and output columns from my_campaign object
    my_sampler = my_campaign._active_sampler
    output_columns = my_campaign._active_app_decoder.output_columns
    
    #fetch the results from the (remote) host via FabSim3
    #get_UQ_results(my_campaign.campaign_dir, machine='eagle_vecma')
    fab.get_uq_samples(my_campaign.campaign_dir, machine='eagle_vecma')

    #collate output
    my_campaign.collate()

    # Post-processing analysis
    sc_analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
    my_campaign.apply_analysis(sc_analysis)
    results = my_campaign.get_last_analysis()
    results['n_samples'] = sc_analysis._number_of_samples
    
    #store data
    store_uq_results(my_campaign.campaign_dir, results)

    return results, sc_analysis, my_sampler, my_campaign

if __name__ == "__main__":
    
    #home dir of this file    
    HOME = os.path.abspath(os.path.dirname(__file__))

    work_dir = home + "/VECMA/Campaigns/"

    results, sc_analysis, my_sampler, my_campaign = post_proc(state_file="campaign_state_p6.json", work_dir = work_dir)
    mu_E = results['statistical_moments']['E_mean']['mean']
    std_E = results['statistical_moments']['E_mean']['std']
    mu_Z = results['statistical_moments']['Z_mean']['mean']
    std_Z = results['statistical_moments']['Z_mean']['std']

    print('========================================================')
    print('Mean E =', mu_E)
    print('Std E =', std_E)
    print('Mean Z =', mu_Z)
    print('Std E =', std_Z)
    print('========================================================')
    print('Sobol indices E:')
    print(results['sobol_indices']['E_mean'])
    print(results['sobol_indices']['E_std'])
    print('Sobol indices Z:')
    print(results['sobol_indices']['Z_mean'])
    print(results['sobol_indices']['Z_std'])
    print('========================================================')
#     
#
#
#    my_campaign_p1 = uq.Campaign(state_file="campaign_state_p1.json", work_dir = work_dir)
#    my_campaign_p2 = uq.Campaign(state_file="campaign_state_p2.json", work_dir = work_dir)
#    my_campaign_p3 = uq.Campaign(state_file="campaign_state_p3.json", work_dir = work_dir)
#    my_campaign_p4 = uq.Campaign(state_file="campaign_state_p4.json", work_dir = work_dir)
#    my_campaign_p5 = uq.Campaign(state_file="campaign_state_p5.json", work_dir = work_dir)
#    my_campaign_p6 = uq.Campaign(state_file="campaign_state_p6.json", work_dir = work_dir)
#    
#    #make a histrogram from the samples for each EasyVVUQ campaign
#    vvp.ensemble_vvp([my_campaign_p4.campaign_dir + '/runs', 
#                      my_campaign_p5.campaign_dir + '/runs',
#                      my_campaign_p6.campaign_dir + '/runs'], 
#                      load_uq_csv_output, plt.hist)
# 
#    sample_dirs = [my_campaign_p1.campaign_dir,
#                   my_campaign_p2.campaign_dir,
#                   my_campaign_p3.campaign_dir, 
#                   my_campaign_p4.campaign_dir, 
#                   my_campaign_p5.campaign_dir,
#                   my_campaign_p6.campaign_dir]
#    
#    #print the results for all EasyVVUQ campaigns found in the work directory
#    vvp.ensemble_vvp(work_dir, load_uq_results, plot_convergence, qoi='Z_mean',
#                     sample_dirs=sample_dirs)
#
#    items = ['Run_' + str(i) for i in range(1, 50)]
#    vvp.ensemble_vvp(my_campaign_p6.campaign_dir + '/runs', load_uq_csv_output, plot_samples_2D, qoi='E_mean', items=items)
#
    #################################
    # Use SC expansion as surrogate #
    #################################
    
    #number of MC samples
    n_mc = 50000
    
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel=r'$E$', yticks = [])
        
    
    #get the input distributions
    theta = my_sampler.vary.get_values()
    xi = np.zeros([n_mc, 2])
    idx = 0
    
    #draw random sampler from the input distributions
    for theta_i in theta:
        xi[:, idx] = theta_i.sample(n_mc)
        idx += 1
        
    #evaluate the surrogate at the random values
    Q = 'E_mean'
    qoi = np.zeros(n_mc)
    for i in range(n_mc):
        qoi[i] = sc_analysis.surrogate(Q, xi[i])
        
    #plot histogram of surrogate samples
    x, kde = get_kde(qoi)
    plt.plot(x, kde, label=r'$\mathrm{surrogate\;KDE}$')

    #make a list of actual samples
    samples = []
    for i in range(sc_analysis._number_of_samples):
        samples.append(sc_analysis.samples[Q][i].values)
    
    plt.plot(samples, np.zeros(sc_analysis._number_of_samples), 'ro', label=r'$\mathrm{code\;samples}$')
    
    leg = plt.legend(loc=0)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    leg.set_draggable(True)
    
    plt.tight_layout()
    
plt.show()