"""
Example on how to use FabSim3 inside a Python script to execute an EasyVVUQ campaign
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
TARGET_FILENAME = './output.csv'
# location of the EasyVVUQ database
DB_LOCATION = "sqlite:///" + WORK_DIR + "/campaign%s.db" % ID
# Use QCG PiltJob or not
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
    template_fname= HOME + '/sc/ade.template',
    delimiter='$',
    target_filename='ade_in.json')

actions = uq.actions.Actions(
    uq.actions.CreateRunDirectory(root=WORK_DIR, flatten=True),
    uq.actions.Encode(encoder),
)

campaign = uq.Campaign(
    name=CAMPAIGN_NAME,
    db_location=DB_LOCATION,
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

vary = {
    "Pe": cp.Uniform(100.0, 500.0),
    "f": cp.Uniform(0.9, 1.1)
}

##################
# Select sampler #
##################

sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=4)

# Associate the sampler with the campaign #
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

output_columns = ["u"]
decoder = uq.decoders.SimpleCSV(
    target_filename=TARGET_FILENAME,
    output_columns=output_columns)

actions = uq.actions.Actions(
    uq.actions.Decode(decoder)
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
