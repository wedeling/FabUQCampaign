"""
Example on how to use FabSim3 inside a Python script to execute an EasyVVUQ campaign

Runs only the execution phase, without post processing.

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
CONFIG = 'ade'
# Simulation identifier
ID = '_test'
# EasyVVUQ campaign name
CAMPAIGN_NAME = CONFIG + ID
# name and relative location of the output file name
TARGET_FILENAME = './output.hdf5'
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
    "Pe": {
        "type": "float",
        "min": 1.0,
        "max": 2000.0,
        "default": 100.0},
    "f": {
        "type": "float",
        "min": 0.0,
        "max": 10.0,
        "default": 1.0},
    "out_file": {
        "type": "string",
        "default": "output.csv"}}

###########################
# Set up a fresh campaign #
###########################

encoder = uq.encoders.GenericEncoder(
    template_fname= HOME + '/../../config_files/ade/ade_config',
    delimiter='$',
    target_filename='ade_in.json')

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
    "Pe": cp.Uniform(100.0, 500.0),
    "f": cp.Uniform(0.9, 1.1)
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

###############################################
# run the UQ ensemble using FabSim3 interface #
###############################################

fab.run_uq_ensemble(CONFIG, campaign.campaign_dir, script='ade',
                    machine=MACHINE, PJ=PILOT_JOB)
