"""
Run a LAMMPS - EasyVVUQ UQ campaign on a remote machine using FabSim3
"""

import os
import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt

############################################
# Import the FabSim3 commandline interface #
############################################
import fabsim3_cmd_api as fab

plt.close('all')

# author: Wouter Edeling
__license__ = "LGPL"

#########
# FLAGS #
#########

# home directory
HOME = os.path.abspath(os.path.dirname(__file__))
# Work directory, where the easyVVUQ directory will be placed
WORK_DIR = '/tmp'
# FabSim3 config name
CONFIG = 'lammps_uq'
# Simulation identifier
ID = '_test'
# EasyVVUQ campaign name
CAMPAIGN_NAME = CONFIG + ID
# name and relative location of the output file name
TARGET_FILENAME = './output.'
# location of the EasyVVUQ database
DB_LOCATION = "sqlite:///" + WORK_DIR + "/campaign%s.db" % ID
# Use QCG PilotJob or not
PILOT_JOB = False
# machine to run ensemble on
MACHINE = "localhost"

##################################
# Define (total) parameter space #
##################################

# Define parameter space
params = {
    'dt': {
        'type': 'float',
        'default': 0.001},
    'n_init_steps': {
        'type': 'integer',
        'default': 500},
    'n_run_steps': {
        'type': 'integer',
        'default': 1000}
    }

###########################
# Set up a fresh campaign #
###########################

encoder = uq.encoders.GenericEncoder(
    template_fname= 'in_NEMD.template',
    delimiter='$',
    target_filename='in_NEMD.in')

actions = uq.actions.Actions(
    uq.actions.CreateRunDirectory(root=WORK_DIR, flatten=True),
    uq.actions.Encode(encoder),
)

campaign = uq.Campaign(
    name=CAMPAIGN_NAME,
    work_dir=WORK_DIR,
)

campaign.add_app(
    name=CAMPAIGN_NAME,
    params=params,
    actions=actions
)

#######################
# Specify input space #
#######################

vary = {
    "dt": cp.Uniform(0.0005, 0.001),
}

##################
# Select sampler #
##################

sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=2)

# Associate the sampler with the campaign
campaign.set_sampler(sampler)

###############################
# execute the defined actions #
###############################

campaign.execute().collate()

# ###############################################
# # run the UQ ensemble using FabSim3 interface #
# ###############################################

fab.run_uq_ensemble(CONFIG, campaign.campaign_dir, script='lammps_uq',
                    machine=MACHINE, PJ=PILOT_JOB)

# wait for job to complete
fab.wait(machine=MACHINE)

# # check if all output files are retrieved from the remote machine, returns a Boolean flag
# all_good = fab.verify(CONFIG, campaign.campaign_dir,
#                       TARGET_FILENAME,
#                       machine=MACHINE)

# if all_good:
#     # copy the results from the FabSim results dir to the EasyVVUQ results dir
#     fab.get_uq_samples(CONFIG, campaign.campaign_dir, sampler.n_samples, machine=MACHINE)
# else:
#     print("Not all samples executed correctly")
#     import sys
#     sys.exit()

# #############################################
# # All output files are present, decode them #
# #############################################

# output_columns = ["u"]
# # decoder = uq.decoders.SimpleCSV(
# #     target_filename=TARGET_FILENAME,
# #     output_columns=output_columns)

# decoder = uq.decoders.HDF5(
#     target_filename=TARGET_FILENAME,
#     output_columns=output_columns)

# actions = uq.actions.Actions(
#     uq.actions.Decode(decoder),
# )
# campaign.replace_actions(CAMPAIGN_NAME, actions)

# ###########################
# # Execute decoding action #
# ###########################

# campaign.execute().collate()

# # get EasyVVUQ data frame
# data_frame = campaign.get_collation_result()

# ############################
# # Post-processing analysis #
# ############################

# analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=["u"])
# results = analysis.analyse(data_frame=data_frame)

# plt.show()