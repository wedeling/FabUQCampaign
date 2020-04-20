import chaospy as cp
import numpy as np
import easyvvuq as uq
import os
import fabsim3_cmd_api as fab
import matplotlib.pyplot as plt

plt.close('all')

# author: Wouter Edeling
__license__ = "LGPL"

# home directory of user
home = os.path.expanduser('~')
HOME = os.path.abspath(os.path.dirname(__file__))

# Set up a fresh campaign called "sc"
my_campaign = uq.Campaign(name='sc', work_dir='/tmp')

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

output_filename = params["out_file"]["default"]
output_columns = ["u"]

# Create an encoder, decoder and collation element
encoder = uq.encoders.GenericEncoder(
    template_fname=HOME + '/sc/ade.template',
    delimiter='$',
    target_filename='ade_in.json')
decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                output_columns=output_columns,
                                header=0)
collater = uq.collate.AggregateSamples()

# Add the SC app (automatically set as current app)
my_campaign.add_app(name="sc",
                    params=params,
                    encoder=encoder,
                    decoder=decoder,
                    collater=collater)

# Create the sampler
vary = {
    "Pe": cp.Uniform(100.0, 200.0),
    "f": cp.Uniform(0.95, 1.05)
}

"""
SPARSE GRID PARAMETERS
----------------------
- sparse = True: use a Smolyak sparse grid
- growth = True: use an exponential rule for the growth of the number
  of 1D collocation points per level. Used to make e.g. clenshaw-curtis
  quadrature nested.
"""
my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=2,
                                   quadrature_rule="G", 
                                   sparse=False, growth=False)

# Associate the sampler with the campaign
my_campaign.set_sampler(my_sampler)

# Will draw all (of the finite set of samples)
my_campaign.draw_samples()
my_campaign.populate_runs_dir()

##   Use this instead to run the samples using EasyVVUQ on the localhost
#my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
#    "sc_model.py ade_in.json"))

fab.run_uq_ensemble(my_campaign.campaign_dir, script_name='ade', machine='localhost')

fab.get_uq_samples(my_campaign.campaign_dir, machine='localhost')

my_campaign.collate()

# Post-processing analysis
analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)

my_campaign.apply_analysis(analysis)

results = my_campaign.get_last_analysis()

###################################
# Plot the moments and SC samples #
###################################

mu = results['statistical_moments']['u']['mean']
std = results['statistical_moments']['u']['std']

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
n_mc = 20
xi_mc = np.zeros([20,2])
idx = 0
for dist in my_sampler.vary.get_values():
    xi_mc[:, idx] = dist.sample(n_mc)
    idx += 1
    
# evaluate the surrogate at these values
print('Evaluating surrogate model', n_mc, 'times')
for i in range(n_mc):
    ax.plot(x, analysis.surrogate('u', xi_mc[i]), 'g')
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

lbl = ['Pe', 'f', 'Pe-f interaction']
idx = 0

for S_i in results['sobols']['u']:
    ax.plot(x, results['sobols']['u'][S_i], label=lbl[idx])
    idx += 1

leg = plt.legend(loc=0)
leg.set_draggable(True)

plt.tight_layout()

plt.show()