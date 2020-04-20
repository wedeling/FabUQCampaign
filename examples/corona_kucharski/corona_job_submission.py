import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import os
import fabsim3_cmd_api as fab

# author: Wouter Edeling
__license__ = "LGPL"

#home dir of this file    
HOME = os.path.abspath(os.path.dirname(__file__))
   
# Set up a fresh campaign called "sc"
my_campaign = uq.Campaign(name='corona', work_dir='/tmp')

# Define parameter space
params = {
    "incubation": {
        "type": "float",
        "min": 1.0,
        "max": 10.0,
        "default": 5.2},
    "r0": {
        "type": "float",
        "min": 1.0,
        "max": 10.0,
        "default": 2.5},
    "out_file": {
        "type": "string",
        "default": "posterior_I.csv"}}

output_filename = params["out_file"]["default"]
output_columns = ["V100"]

# Create an encoder, decoder and collation element
encoder = uq.encoders.GenericEncoder(
    template_fname= HOME + '/sc/theta_initial_conditions.csv',
    delimiter='$',
    target_filename='theta_initial_conditions.csv')
decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                output_columns=output_columns,
                                header=0)
collater = uq.collate.AggregateSamples(average=False)

# Add the SC app (automatically set as current app)
my_campaign.add_app(name="sc",
                    params=params,
                    encoder=encoder,
                    decoder=decoder,
                    collater=collater)    

# Create the sampler
vary = {
    "incubation": cp.Normal(5.2, 0.1),
    "r0": cp.Normal(2.5, 0.1)
}

my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=2)
# Associate the sampler with the campaign
my_campaign.set_sampler(my_sampler)

# Will draw all (of the finite set of samples)
my_campaign.draw_samples()

my_campaign.populate_runs_dir()


#Run execution using Fabsim 
fab.run_uq_ensemble(my_campaign.campaign_dir, 'corona', machine='localhost')

#Save the Campaign
my_campaign.save_state("campaign_state.json")