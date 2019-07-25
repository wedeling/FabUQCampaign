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
    my_campaign = uq.Campaign(state_file="campaign_state.json", work_dir=tmpdir)
    print('Reloaded campaign', my_campaign.campaign_dir.split('/')[-1])
    
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

    return results, my_campaign

if __name__ == "__main__":
    
    #home dir of this file    
    HOME = os.path.abspath(os.path.dirname(__file__))

    #test_sc("/tmp/")

    results, my_campaign = post_proc("/tmp/")
    mu = results['statistical_moments']['E']['mean']
    std = results['statistical_moments']['E']['std']

    print('=================================================')    
    print('Sobol indices E:')
    print(results['sobol_indices']['E'])
    print('Sobol indices Z:')
    print(results['sobol_indices']['Z'])
    print('=================================================')