"""
Example on how to use FabSim3 inside a Python script to execute a dimension-adaptive
EasyVVUQ campaign
"""

import os
import chaospy as cp
import easyvvuq as uq
import matplotlib.pyplot as plt

############################################
# Import the FabSim3 commandline interface #
############################################
import fabsim3_cmd_api as fab

plt.close('all')

# author: Wouter Edeling
__license__ = "LGPL"

HOME = os.path.abspath(os.path.dirname(__file__))

#number of uncertain parameters
D = 15

#########
# FLAGS #
#########

# home directory
HOME = os.path.abspath(os.path.dirname(__file__))
# Work directory, where the easyVVUQ directory will be placed
WORK_DIR = '/tmp'
# FabSim3 config name
CONFIG = 'ohagan'
# Simulation identifier
ID = '_test'
# EasyVVUQ campaign name
CAMPAIGN_NAME = CONFIG + ID
# name and relative location of the output file name
TARGET_FILENAME = './output.csv'
# Use QCG PiltJob or not
PILOT_JOB = False
# machine to run ensemble on
MACHINE = "localhost"

#choose a single QoI
output_columns = ["f"]

#start a new adaptive campaign or not
INIT = True

if INIT:

    ##################################
    # Define (total) parameter space #
    ##################################
    params = {}
    for i in range(D):
        params["x%d" % (i + 1)] = {"type": "float",
                                   "default": 0.0}
    params["out_file"] = {"type": "string", "default": "output.csv"}

    ###########################
    # Set up a fresh campaign #
    ###########################

    encoder = uq.encoders.GenericEncoder(
        template_fname=HOME + '/model/model2.template',
        delimiter='$',
        target_filename='model_in.json')
    
    # actions for creating the run directories and encoding the input files
    actions = uq.actions.Actions(
        uq.actions.CreateRunDirectory(root=WORK_DIR, flatten=True),
        uq.actions.Encode(encoder),
    )

    # create an EasyVVUQ campaign
    campaign = uq.Campaign(
        name=CAMPAIGN_NAME,
        work_dir=WORK_DIR
    )

    campaign.add_app(
        name=CAMPAIGN_NAME,
        params=params,
        actions=actions
    )

    #######################
    # Specify input space #
    #######################
    vary = {}
    for i in range(D):
        vary["x%d" % (i + 1)] = cp.Normal(0, 1)

    #####################################
    # create dimension-adaptive sampler #
    #####################################

    #sparse = use a sparse grid (required)
    #growth = use a nested quadrature rule (not required)
    #dimension_adaptive = use a dimension adaptive sampler (required)

    sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=1,
                                    quadrature_rule="C",
                                    sparse=True, growth=True,
                                    dimension_adaptive=True)

    # Associate the sampler with the campaign
    campaign.set_sampler(sampler)

    ###############################
    # execute the defined actions #
    ###############################

    campaign.execute().collate()

    ####################################
    # Ensemble execution using FabSim3 #
    ####################################

    fab.run_uq_ensemble(CONFIG, campaign.campaign_dir, script='ohagan',
                        machine=MACHINE, PJ=PILOT_JOB)

    # wait for job to complete
    fab.wait(machine=MACHINE)

    # check if all output files are retrieved from the remote machine, returns a Boolean flag
    all_good = fab.verify(CONFIG, campaign.campaign_dir,
                          TARGET_FILENAME,
                          machine=MACHINE)

    if all_good:
        # copy the results from the FabSim results dir to the EasyVVUQ results dir
        fab.get_uq_samples(CONFIG, campaign.campaign_dir, sampler.n_samples, machine=MACHINE)
    else:
        print("Not all samples executed correctly")
        import sys
        sys.exit()

    #############################################
    # All output files are present, decode them #
    #############################################

    decoder = uq.decoders.SimpleCSV(
        target_filename=TARGET_FILENAME,
        output_columns=output_columns)

    actions_decode = uq.actions.Actions(
        uq.actions.Decode(decoder)
    )
    campaign.replace_actions(CAMPAIGN_NAME, actions_decode)

    ###########################
    # Execute decoding action #
    ###########################

    campaign.execute().collate()

    # Post-processing analysis
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
    campaign.apply_analysis(analysis)
# reload a previous adaptive campaign to refine it further
else:
    #reload Campaign, sampler, analysis
    DB_LOCATION = "sqlite:////tmp/ohagan_test2blvg2im/campaign.db" # change this to correct database file
    campaign = uq.Campaign(name=CAMPAIGN_NAME, db_location=DB_LOCATION)
    print("===========================================")
    print("Reloaded campaign {}".format(CAMPAIGN_NAME))
    print("===========================================")
    sampler = campaign.get_active_sampler()
    sampler.load_state("sampler_state" + ID + ".pickle")
    campaign.set_sampler(sampler, update=True)
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
    analysis.load_state("analysis_state" + ID + ".pickle")

    # also recreate the actions
    encoder = uq.encoders.GenericEncoder(
        template_fname=HOME + '/model/model2.template',
        delimiter='$',
        target_filename='model_in.json')

    actions = uq.actions.Actions(
        uq.actions.CreateRunDirectory(root=WORK_DIR, flatten=True),
        uq.actions.Encode(encoder),
    )

    # decoding actions
    decoder = uq.decoders.SimpleCSV(
        target_filename=TARGET_FILENAME,
        output_columns=output_columns)

    actions_decode = uq.actions.Actions(
        uq.actions.Decode(decoder)
    )

MAX_SAMPLES = 40
N_ITER = 0

while sampler.n_samples < MAX_SAMPLES:

    # the number of runs to skip in the FabSim SWEEP directory. Equals the number of
    # current runs. This prevents FabSim from submitting runs that already are computed.
    skip = sampler.count

    print('Adaptation %d' % (N_ITER + 1))

    ##################################################################
    # look-ahead step, evaluate the code at new candidate directions #
    ##################################################################

    sampler.look_ahead(analysis.l_norm)

    campaign.replace_actions(CAMPAIGN_NAME, actions)
    campaign.execute().collate()

    ####################################
    # Ensemble execution using FabSim3 #
    ####################################

    fab.run_uq_ensemble(CONFIG, campaign.campaign_dir, script='ohagan',
                        machine=MACHINE, PJ=PILOT_JOB, skip=skip)

    # wait for job to complete
    fab.wait(machine=MACHINE)

    # check if all output files are retrieved from the remote machine, returns a Boolean flag
    all_good = fab.verify(CONFIG, campaign.campaign_dir,
                          TARGET_FILENAME,
                          machine=MACHINE)

    if all_good:
        # copy the results from the FabSim results dir to the EasyVVUQ results dir
        fab.get_uq_samples(CONFIG, campaign.campaign_dir, sampler.n_samples, machine=MACHINE)
    else:
        print("Not all samples executed correctly")
        import sys
        sys.exit()

    #############################################
    # All output files are present, decode them #
    #############################################

    campaign.replace_actions(CAMPAIGN_NAME, actions_decode)

    ###########################
    # Execute decoding action #
    ###########################

    campaign.execute().collate()

    # get EasyVVUQ data frame
    data_frame = campaign.get_collation_result()

    analysis.adapt_dimension('f', data_frame, method='var')

    #save everything
    sampler.save_state("sampler_state" + ID + ".pickle")
    analysis.save_state("analysis_state" + ID + ".pickle")

    N_ITER += 1

#proceed as usual with analysis
campaign.apply_analysis(analysis)
results = campaign.get_last_analysis().raw_data

print("======================================")
print("Number of samples = %d" % sampler.n_samples)
print("--------------------------------------")
print("Computed mean = %.4e" % results['statistical_moments']['f']['mean'])
print("--------------------------------------")
print("Computed standard deviation = %.4e" % results['statistical_moments']['f']['std'])
print("--------------------------------------")
print("First-order Sobol indices =", results['sobols_first']['f'])
print("--------------------------------------")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(analysis.get_adaptation_errors())
plt.xlabel('iteration')
plt.ylabel('refinement error')
plt.tight_layout()

analysis.plot_stat_convergence()
analysis.adaptation_table()
