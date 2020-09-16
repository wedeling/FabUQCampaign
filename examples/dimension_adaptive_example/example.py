import chaospy as cp
import numpy as np
import easyvvuq as uq
import os
import fabsim3_cmd_api as fab
import matplotlib.pyplot as plt

plt.close('all')

# author: Wouter Edeling
__license__ = "LGPL"

HOME = os.path.abspath(os.path.dirname(__file__))

#number of uncertain parameters
d = 15

#name of the FabSim3 cofig directory which contains the code
config='ohagan'  #the 15 dimensional test problem of O'Hagan
#identifier used to store campaign, sampler and analysis objects
ID = 'test_run'
#work directory
work_dir = '/tmp'

#choose a single QoI
output_columns = ["f"]

#start a new adaptive campaign or not
init = False

if init:
    
    # Set up a fresh campaign called "sc"
    campaign = uq.Campaign(name='adaptive_test', work_dir=work_dir)

    # Define parameter space
    params = {}
    for i in range(15):
        params["x%d" % (i + 1)] = {"type": "float",
                                   "default": 0.0}
    params["out_file"] = {"type": "string", "default": "output.csv"}
    output_filename = params["out_file"]["default"]

    # Create an encoder, decoder and collation element
    encoder = uq.encoders.GenericEncoder(
        template_fname=HOME + '/model/model2.template',
        delimiter='$',
        target_filename='model_in.json')
    decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                    output_columns=output_columns,
                                    header=0)
    collater = uq.collate.AggregateSamples()

    # Add the SC app (automatically set as current app)
    campaign.add_app(name="sc",
                        params=params,
                        encoder=encoder,
                        decoder=decoder,
                        collater=collater)

    #uncertain variables
    vary = {}
    for i in range(d):
        vary["x%d" % (i + 1)] = cp.Normal(0, 1)

    #=================================
    #create dimension-adaptive sampler
    #=================================
    #sparse = use a sparse grid (required)
    #growth = use a nested quadrature rule (not required)
    #dimension_adaptive = use a dimension adaptive sampler (required)
    sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=1,
                                    quadrature_rule="C",
                                    sparse=True, growth=True,
                                    dimension_adaptive=True)
    
    # Associate the sampler with the campaign
    campaign.set_sampler(sampler)

    # Will draw all (of the finite set of samples)
    campaign.draw_samples()
    campaign.populate_runs_dir()

    ##   Use this instead to run the samples using EasyVVUQ on the localhost
    # campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
    #     "./model/model2.py model_in.json"))

    ####################################
    # Ensemble execution using FabSim3 #
    ####################################

    # run the UQ ensemble
    fab.run_uq_ensemble(config, campaign.campaign_dir, script='ohagan',
                        machine="localhost", skip=0, PilotJob = False)

    #wait for jobs to complete and check if all output files are retrieved 
    #from the remote machine
    fab.verify(config, campaign.campaign_dir, 
               campaign._active_app_decoder.target_filename, 
               machine="localhost", PilotJob=False)

    #run the UQ ensemble
    fab.get_uq_samples(config, campaign.campaign_dir, 
                       number_of_samples = sampler._number_of_samples,
                       skip=0,
                       machine='localhost')

    ########################################
    # End ensemble execution using FabSim3 #
    ########################################

    campaign.collate()
    data_frame = campaign.get_collation_result()

    # Post-processing analysis
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
    campaign.apply_analysis(analysis)
else:
    #reload Campaign, sampler, analysis
    campaign = uq.Campaign(state_file="covid_easyvvuq_state" + ID + ".json", 
                           work_dir=work_dir)
    print('========================================================')
    print('Reloaded campaign', campaign.campaign_dir.split('/')[-1])
    print('========================================================')
    sampler = campaign.get_active_sampler()
    sampler.load_state("covid_sampler_state" + ID + ".pickle")
    campaign.set_sampler(sampler)
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
    analysis.load_state("covid_analysis_state" + ID + ".pickle")

max_samples = 1000
n_iter = 0

while sampler._number_of_samples < max_samples:
    #required parameter in the case of a Fabsim run
    skip = sampler.count

    print('Adaptation %d' % (n_iter+1))
    #look-ahead step (compute the code at admissible forward points)
    sampler.look_ahead(analysis.l_norm)

    #proceed as usual
    campaign.draw_samples()
    campaign.populate_runs_dir()

    #Use this to submit jobs using the ExecuteLocal subroutine of EasyVVUQ
    # campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
        # "./model/model2.py model_in.json"))
        
    ####################################
    # Ensemble execution using FabSim3 #
    ####################################

    # run the UQ ensemble
    fab.run_uq_ensemble(config, campaign.campaign_dir, script='ohagan',
                        machine="localhost", skip=skip, PilotJob = False)

    #wait for jobs to complete and check if all output files are retrieved 
    #from the remote machine
    fab.verify(config, campaign.campaign_dir, 
               campaign._active_app_decoder.target_filename, 
               machine="localhost", PilotJob=False)

    #run the UQ ensemble
    fab.get_uq_samples(config, campaign.campaign_dir, 
                       number_of_samples=sampler._number_of_samples,
                       skip=skip,
                       machine='localhost')

    ########################################
    # End ensemble execution using FabSim3 #
    ########################################

    campaign.collate()

    #compute the error at all admissible points, select direction with
    #highest error and add that direction to the grid
    data_frame = campaign.get_collation_result()
    analysis.adapt_dimension('f', data_frame, method='surplus_quad')

    #save everything
    campaign.save_state("covid_easyvvuq_state" + ID + ".json")
    sampler.save_state("covid_sampler_state" + ID + ".pickle")
    analysis.save_state("covid_analysis_state" + ID + ".pickle")

    n_iter += 1

#proceed as usual with analysis
campaign.apply_analysis(analysis)
results = campaign.get_last_analysis()

#mean ohagan 10**6 samples: 8.982735509649913
#std ohagan 10**6 samples:  7.757486464974213

print("======================================")
print("Number of samples = %d" % sampler._number_of_samples)
print("--------------------------------------")
print("Computed mean = %.4e" % results['statistical_moments']['f']['mean'])
print("--------------------------------------")
print("Computed standard deviation = %.4e" % results['statistical_moments']['f']['std'])
print("--------------------------------------")
print("First-order Sobol indices =", results['sobols_first']['f'])
print("--------------------------------------")

plt.plot(analysis.get_adaptation_errors())
analysis.plot_stat_convergence()
analysis.adaptation_table()