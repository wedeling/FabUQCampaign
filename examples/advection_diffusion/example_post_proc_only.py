"""
Example on how to use FabSim3 inside a Python script to execute an EasyVVUQ campaign

Runs only the post processing phase, after example_exec_only.py has been
executed

"""

import os
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
# Use QCG PilotJob or not
PILOT_JOB = False
# machine to run ensemble on
MACHINE = "localhost"

# =============================================================================
# IMPORTANT: SPECIFY THE FULL NAME OF THE EASYVVUQ CAMPAIGN DIRECTORY,
# YOU WILL NEED TO CHANGE THE VALUE BELOW
CAMPAIGN_DIRNAME = 'ade_testaglupw82'
# =============================================================================

# location of the EasyVVUQ database
DB_LOCATION = "sqlite:///%s/%s/campaign.db" % (WORK_DIR, CAMPAIGN_DIRNAME)

# Load EasyVVUQ Campaign
campaign = uq.Campaign(name=CAMPAIGN_NAME, db_location=DB_LOCATION)
print("===========================================")
print("Reloaded campaign {}".format(CAMPAIGN_NAME))
print("===========================================")
sampler = campaign.get_active_sampler()

# check if all output files are retrieved from the remote machine, 
# returns a Boolean flag, set wait = True if jobs are still running
all_good = fab.verify(CONFIG, campaign.campaign_dir,
                      TARGET_FILENAME,
                      machine=MACHINE,
                      wait=False)

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

output_columns = ["u"]
# decoder = uq.decoders.SimpleCSV(
#     target_filename=TARGET_FILENAME,
#     output_columns=output_columns)

decoder = uq.decoders.HDF5(
    target_filename=TARGET_FILENAME,
    output_columns=output_columns)

actions = uq.actions.Actions(
    uq.actions.Decode(decoder),
)
campaign.replace_actions(CAMPAIGN_NAME, actions)

###########################
# Execute decoding action #
###########################

campaign.execute().collate()

# get EasyVVUQ data frame
data_frame = campaign.get_collation_result()

############################
# Post-processing analysis #
############################

analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=["u"])
results = analysis.analyse(data_frame=data_frame)

###################################
# Plot the moments and SC samples #
###################################

mu = results.describe(output_columns[0], 'mean')
std = results.describe(output_columns[0], 'std')

x = np.linspace(0, 1, 301)

fig = plt.figure(figsize=[10, 5])
ax = fig.add_subplot(121, xlabel='location x', ylabel='velocity u',
                      title=r'code mean +/- standard deviation')
ax.plot(x, mu, 'b', label='mean')
ax.plot(x, mu + std, '--r', label='std-dev')
ax.plot(x, mu - std, '--r')

#####################################
# Plot the random surrogate samples #
#####################################

ax = fig.add_subplot(122, xlabel='location x', ylabel='velocity u',
                      title='Surrogate samples')

#generate n_mc samples from the input distributions
N_MC = 20
xi_mc = np.zeros([20,2])
for idx, dist in enumerate(sampler.vary.get_values()):
    xi_mc[:, idx] = dist.sample(N_MC)
    idx += 1

# evaluate the surrogate at these values
print('Evaluating surrogate model %d times' % (N_MC,))
for i in range(N_MC):
    ax.plot(x, analysis.surrogate(output_columns[0], xi_mc[i]), 'g')
print('done')

plt.tight_layout()

#######################
# Plot Sobol indices #
#######################

fig = plt.figure()
ax = fig.add_subplot(
    111,
    xlabel='location x',
    ylabel='Sobol indices',
    title='spatial dist. Sobol indices, Pe only important in viscous regions')

lbl = ['Pe', 'f']

sobols = results.raw_data['sobols_first'][output_columns[0]]

for idx, S_i in enumerate(sobols):
    ax.plot(x, sobols[S_i], label=lbl[idx])

leg = plt.legend(loc=0)
leg.set_draggable(True)

plt.show()