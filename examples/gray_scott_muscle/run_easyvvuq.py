import chaospy as cp
import numpy as np
import easyvvuq as uq
import os
import fabsim3_cmd_api as fab
import matplotlib.pyplot as plt

machine = 'eagle_vecma'
config = 'gray_scott_muscle'
work_dir = '/home/wouter/VECMA/Campaigns'

plt.close('all')

# author: Wouter Edeling
__license__ = "LGPL"

# home directory of usewr
HOME = os.path.abspath(os.path.dirname(__file__))

# Set up a fresh campaign called "sc"
campaign = uq.Campaign(name='gray_scott', work_dir=work_dir)

# Define parameter space
n_samples = 28
params = {}
for i in range(n_samples):
    params["feed%d" % i] = {"type": "float", "default": 0.02}
    params["kill%d" % i] = {"type": "float", "default": 0.05}

output_filename = 'output.csv'
output_columns = ["Q1", "Q2", "Q3", "Q4", "Q1_HF", "Q2_HF", "Q3_HF", "Q4_HF"]

# Create an encoder, decoder and collation element
encoder = uq.encoders.GenericEncoder(
    template_fname=HOME + '/gray_scott.template',
    delimiter='$',
    target_filename='gray_scott_in.json')
decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                output_columns=output_columns,
                                header=0)
collater = uq.collate.AggregateSamples()

# Add the SC app (automatically set as current app)
campaign.add_app(name="gray_scott",
                 params=params,
                 encoder=encoder,
                 decoder=decoder,
                 collater=collater)

# Create the sampler
vary = {}
for i in range(n_samples):
    vary["feed%d" % i] = cp.Uniform(0.02, 0.025)
    vary["kill%d" % i] = cp.Uniform(0.05, 0.055)

sampler = uq.sampling.RandomSampler(vary, max_num=4)

# Associate the sampler with the campaign
campaign.set_sampler(sampler)

# Will draw all (of the finite set of samples)
campaign.draw_samples()
campaign.populate_runs_dir()

# run the UQ ensemble
fab.run_uq_ensemble(config, campaign.campaign_dir, script='Gray_Scott_muscle',
                    machine=machine, PilotJob = False)

#wait for job to complete
fab.wait(machine=machine)

#wait for jobs to complete and check if all output files are retrieved 
#from the remote machine
# fab.verify(config, campaign.campaign_dir, 
#            campaign._active_app_decoder.target_filename, 
#            machine=machine, PilotJob=False)

#run the UQ ensemble
fab.fetch_results(machine=machine)
fab.get_uq_samples(config, campaign.campaign_dir, sampler.max_num,
                   skip=0, machine=machine)
# campaign.collate()

campaign.save_state("easyvvuq_state.json")