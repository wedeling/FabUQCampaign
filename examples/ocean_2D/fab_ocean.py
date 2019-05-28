import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import os

# author: Wouter Edeling
__license__ = "LGPL"

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
 
#    #Run execution using Fabsim (on the localhost)
#    sim_ID ='ade_example1'
#    Fab_home = '~/CWI/VECMA/FabSim3'
#    
#    cmd1 = "cd " + Fab_home + " && fab localhost campaign2ensemble:" + \
#            sim_ID + ",campaign_dir=" + my_campaign.campaign_dir
#    cmd2 = "cd " + Fab_home + " && fab localhost uq_ensemble:" + sim_ID
#    
#    print(cmd1)
#    print(cmd2)
#    
#    os.system(cmd1)
#    os.system(cmd2)
#    
#    os.system('cp -r ~/FabSim3/results/' + sim_ID + '_localhost_16/RUNS/Run_* ' + my_campaign.campaign_dir + '/runs')
    
    my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
        "./sc/ocean.py ocean_in.json"))

    my_campaign.collate()

    # Post-processing analysis
    sc_analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)

    my_campaign.apply_analysis(sc_analysis)

    results = my_campaign.get_last_analysis()

    return results, sc_analysis

if __name__ == "__main__":

    results, sc_analysis = test_sc("/tmp/")
    mu = results['statistical_moments']['E']['mean']
    std = results['statistical_moments']['E']['std']
    
    print('Mean energy =', mu)
    print('Standard dev energy =', std)
    print('Sobol indices of', [param for param in sc_analysis.sampler.vary.keys()])
    print(results['sobol_indices']['E'])
    
    plt.show()