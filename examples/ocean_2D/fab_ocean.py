import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import os

# author: Wouter Edeling
__license__ = "LGPL"

####################
# HARD-CODED INPUT #
####################
Fab_home = '~/CWI/VECMA/FabSim3'    #specify the home dir of FabSim3

#home directory of user
home = os.path.expanduser('~')

#subroutine which runs the EasyVVUQ ensemble with FabSim's campaign2ensemble 
#Assumes standard directory ~/FabSim3/results contains the results
#If not, specify fab_results as well
def run_FabUQ_ensemble(campaign_dir, fab_results = home + '/FabSim3/results'):
    
    #sets the sim_ID to the random EasyVVUQ id, e.g. sim_ID = 'EasyVVUQ_Campaign_pk9yoovr'
    sim_ID = campaign_dir.split('/')[-1]
    
    #the 2 commandline instructions needed to run the ensemble
    cmd1 = "cd " + Fab_home + " && fab localhost campaign2ensemble:" + \
            sim_ID + ",campaign_dir=" + campaign_dir
    cmd2 = "cd " + Fab_home + " && fab localhost uq_ensemble:" + sim_ID
    
    print(cmd1)
    print(cmd2)
 
    #execute the ensemble
    os.system(cmd1)
    os.system(cmd2)
    
    #loop through all result dirs to find result dir of sim_ID
    dirs = os.listdir(fab_results)
    for dir_i in dirs:
        if sim_ID in dir_i:
            break
    
    #where FabSim stored the results   
    result_dirs = fab_results + '/' + dir_i + '/RUNS/Run_* '
    #where EasyVVUQ expects the results
    dest_dirs = campaign_dir + '/runs'
    print('Copying results from', result_dirs, 'to', dest_dirs)
    
    #copy results back
    os.system('cp -r ' + result_dirs + dest_dirs)

def test_sc(tmpdir):
    
    # Set up a fresh campaign called "sc"
    my_campaign = uq.Campaign(name='sc', work_dir=tmpdir)

    # Define parameter space
    params = {
        "decay_time_nu": {
            "type": "real",
            "min": "0.0",
            "max": "1000.0",
            "default": "5.0"},
        "decay_time_mu": {
            "type": "real",
            "min": "0.0",
            "max": "1000.0",
            "default": "90.0"},
        "out_file": {
            "type": "str",
            "default": "output.csv"}}

    output_filename = params["out_file"]["default"]
    output_columns = ["E"]

    # Create an encoder, decoder and collation element for PCE test app
    encoder = uq.encoders.GenericEncoder(
        template_fname='./sc/ocean.template',
        delimiter='$',
        target_filename='ocean_in.json')
    decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                    output_columns=output_columns,
                                    header=0)
    collation = uq.collate.AggregateSamples(average=False)

    # Add the SC app (automatically set as current app)
    my_campaign.add_app(name="sc",
                        params=params,
                        encoder=encoder,
                        decoder=decoder,
                        collation=collation
                        )

    # Create the sampler
    vary = {
        "decay_time_nu": cp.Normal(5.0, 0.1),
        "decay_time_mu": cp.Normal(90.0, 1.0)
    }

    my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=2)

    # Associate the sampler with the campaign
    my_campaign.set_sampler(my_sampler)

    # Will draw all (of the finite set of samples)
    my_campaign.draw_samples()

    my_campaign.populate_runs_dir()
 
    #Run execution using Fabsim (on the localhost)
    run_FabUQ_ensemble(my_campaign.campaign_dir)
    
#   #Use this instead to run the samples using EasyVVUQ on the localhost
#    my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
#        "./sc/ocean.py ocean_in.json"))

    my_campaign.collate()

    # Post-processing analysis
    sc_analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)

    my_campaign.apply_analysis(sc_analysis)

    results = my_campaign.get_last_analysis()

    # Save and reload campaign
    state_file = tmpdir + "sc_state.json"
    my_campaign.save_state(state_file)
    new = uq.Campaign(state_file=state_file, work_dir=tmpdir)
    print(new)

    return results, sc_analysis

if __name__ == "__main__":

    results, sc_analysis = test_sc("/tmp/")
    mu = results['statistical_moments']['E']['mean']
    std = results['statistical_moments']['E']['std']

    print('=================================================')    
    print('Sobol indices:')
    print(results['sobol_indices']['E'])
    print('=================================================')    
