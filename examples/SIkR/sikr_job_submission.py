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
my_campaign = uq.Campaign(name='sikr', work_dir = '/home/wouter/VECMA/Campaigns')

# Define parameter space
params = {
    "R0": {
        "type": "float",
        "min": 0.0,
        "max": 10.0,
        "default": 2.0},
    "genWeibShape": {
        "type": "float",
        "min": 0.0,
        "max": 10.0,
        "default": 2.826027},
    "genWeibScale": {
        "type": "float",
        "min": 0.0,
        "max": 10.0,
        "default": 5.665302},
    "recovered_perc": {
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.03},
    "gIH": {
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.026445154},
    "gHD": {
        "type": "float",
        "min": 0.0,
        "max": 10.0,
        "default": 0.003718003},
    "gHR": {
        "type": "float",
        "min": 0.0,
        "max": 10.0,
        "default": 0.096281997},    
    "incMeanLog": {
        "type": "float",
        "min": 0.0,
        "max": 10.0,
        "default": 1.644},
    "incSdLog": {
        "type": "float",
        "min": 0.0,
        "max": 10.0,
        "default": 0.363},
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
    "R0": cp.Normal(params['R0']['default'], 0.1*params['R0']['default']),
    # "genWeibShape": cp.Normal(params['genWeibShape']['default'], 0.1*params['genWeibShape']['default']),
    # "genWeibScale": cp.Normal(params['genWeibScale']['default'], 0.1*params['genWeibScale']['default']),
    # "incMeanLog": cp.Normal(params['incMeanLog']['default'], 0.1*params['incMeanLog']['default']),
    # "incSdLog": cp.Normal(params['incSdLog']['default'], 0.1*params['incSdLog']['default'])
    "gIH": cp.Normal(params['gIH']['default'], 0.1*params['gIH']['default']),
    "gHD": cp.Normal(params['gHD']['default'], 0.1*params['gHD']['default']),
    "gHR": cp.Normal(params['gHR']['default'], 0.1*params['gHR']['default']),
    "recovered_perc": cp.Normal(params['recovered_perc']['default'], 0.1*params['recovered_perc']['default'])   
}

my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=2, 
                                   sparse=True, quadrature_rule="G",
                                   growth=False)

print('*****************************************')
print('Sampling the code', my_sampler._number_of_samples, 'times.')
print('*****************************************')

# Associate the sampler with the campaign
my_campaign.set_sampler(my_sampler)
    
# # Will draw all (of the finite set of samples)
my_campaign.draw_samples()

my_campaign.populate_runs_dir()

#Run execution using Fabsim 
fab.run_uq_ensemble(my_campaign.campaign_dir, 'sikr', machine='localhost')

#Save the Campaign
my_campaign.save_state("campaign_state.json")