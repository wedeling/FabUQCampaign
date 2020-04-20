import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
import os
import fabsim3_cmd_api as fab
import tkinter as tk
from tkinter import filedialog

# author: Wouter Edeling
__license__ = "LGPL"

HOME = os.path.abspath(os.path.dirname(__file__))

# Set up a fresh campaign called "sc"
my_campaign = uq.Campaign(name='sikr', work_dir='/tmp')

# Define parameter space
params = {
    "R0": {
        "type": "float",
        "min": 0.0,
        "max": 10.0,
        "default": 2.0},
    "genWeibShape": {
        "type": "float",
        "min": 0.1,
        "max": 10.0,
        "default": 2.826027},
    "genWeibScale": {
        "type": "float",
        "min": 0.1,
        "max": 10.0,
        "default": 5.665302},
    "out_file": {
        "type": "string",
        "default": "output.csv"}}

output_filename = params["out_file"]["default"]
output_columns = ["S", "I", "R", "H", "D"]

# Create an encoder, decoder and collation element for PCE test app
encoder = uq.encoders.GenericEncoder(
    template_fname= HOME + '/model/sikr.template',
    delimiter='$',
    target_filename='sikr_in.json')
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
    # "R0": cp.Uniform(1.9, 2.1),
    "genWeibShape": cp.Normal(2.826027, 0.1),
    "genWeibScale": cp.Normal(5.665302, 0.1)
}

my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=2)
# Associate the sampler with the campaign
my_campaign.set_sampler(my_sampler)
    
# Will draw all (of the finite set of samples)
my_campaign.draw_samples()

my_campaign.populate_runs_dir()

#Run execution using Fabsim 
fab.run_uq_ensemble(my_campaign.campaign_dir, 'sikr', machine='localhost')

#Save the Campaign
my_campaign.save_state("campaign_state.json")
