import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import os

# author: Wouter Edeling
__license__ = "LGPL"

#home directory of user
home = os.path.expanduser('~')

#subroutine which runs the gets the EasyVVUQ samples via FabSim's ensemble2campaign 
def get_UQ_results(campaign_dir, machine = 'localhost'):
    sim_ID = campaign_dir.split('/')[-1]
    os.system("fabsim " + machine + " get_uq_samples:" + sim_ID + ",campaign_dir=" + campaign_dir)

#post processing of UQ samples executed via FabSim. All samples must have been completed
#before this subroutine is executed. Use 'fabsim <machine_name> job_stat' to check their status
def post_proc(tmpdir):
    
    #Reload the campaign
    my_campaign = uq.Campaign(state_file="campaign_state_p4.json", work_dir=tmpdir)

    print('========================================================')
    print('Reloaded campaign', my_campaign.campaign_dir.split('/')[-1])
    print('========================================================')
    
    #get sampler and output columns from my_campaign object
    my_sampler = my_campaign._active_sampler
    output_columns = my_campaign._active_app_decoder.output_columns
    
    #fetch the results from the (remote) host via FabSim3
    get_UQ_results(my_campaign.campaign_dir, machine='eagle_vecma')

    #collate output
    my_campaign.collate()

    # Post-processing analysis
    sc_analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
    my_campaign.apply_analysis(sc_analysis)
    results = my_campaign.get_last_analysis()

    return results, sc_analysis, my_sampler

if __name__ == "__main__":
    
    #home dir of this file    
    HOME = os.path.abspath(os.path.dirname(__file__))

    #test_sc("/tmp/")

    results, sc_analysis, my_sampler = post_proc(home + "/VECMA/Campaigns/")
    mu_E = results['statistical_moments']['E_mean']['mean']
    std_E = results['statistical_moments']['E_mean']['std']
    mu_Z = results['statistical_moments']['Z_mean']['mean']
    std_Z = results['statistical_moments']['Z_mean']['std']

    print('========================================================')
    print('Sobol indices E:')
    print(results['sobol_indices']['E_mean'])
    print(results['sobol_indices']['E_std'])
    print('Sobol indices Z:')
    print(results['sobol_indices']['Z_mean'])
    print(results['sobol_indices']['Z_std'])
    print('========================================================')
        
    #################################
    # Use SC expansion as surrogate #
    #################################
    
    #number of MC samples
    n_mc = 5000
    
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
    plt.hist(qoi, 20)

    #make a list of actual samples
    samples = []
    for i in range(sc_analysis._number_of_samples):
        samples.append(sc_analysis.samples[Q][i].values)
    
    plt.plot(samples, np.zeros(sc_analysis._number_of_samples), 'ro')
    
plt.show()