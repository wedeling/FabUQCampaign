import chaospy as cp
import numpy as np
import easyvvuq as uq
import os
import fabsim3_cmd_api as fab
import matplotlib.pyplot as plt

plt.close('all')

# author: Wouter Edeling
__license__ = "LGPL"

HOME = os.path.abspath(os.path.dirname(__file__))

# Set up a fresh campaign called "sc"
my_campaign = uq.Campaign(name='sc', work_dir='/tmp')

#number of uncertain parameters
d = 5

# Define parameter space
params = {}
for i in range(45):
    params["x%d" % (i + 1)] = {"type": "float",
                               "min": 0.0,
                               "max": 1.0,
                               "default": 0.5}
params["d"] = {"type": "integer", "default": d}
params["out_file"] = {"type": "string", "default": "output.csv"}
output_filename = params["out_file"]["default"]
output_columns = ["f"]

# Create an encoder, decoder and collation element
encoder = uq.encoders.GenericEncoder(
    template_fname=HOME + '/sc/poly.template',
    delimiter='$',
    target_filename='poly_in.json')
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

#uncertain variables
vary = {}
for i in range(d):
    vary["x%d" % (i + 1)] = cp.Uniform(0, 1)

#=================================
#create dimension-adaptive sampler
#=================================
#sparse = use a sparse grid (required)
#growth = use a nested quadrature rule (not required)
#dimension_adaptive = use a dimension adaptive sampler (required)
my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=1,
                                   quadrature_rule="C",
                                   sparse=True, growth=True,
                                   dimension_adaptive=True)

# Associate the sampler with the campaign
my_campaign.set_sampler(my_sampler)

# Will draw all (of the finite set of samples)
my_campaign.draw_samples()
my_campaign.populate_runs_dir()

##   Use this instead to run the samples using EasyVVUQ on the localhost
my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
    "./sc/poly_model.py poly_in.json"))
# fab.run_uq_ensemble(my_campaign.campaign_dir, script_name='poly_model', 
#                     machine='localhost')
# fab.get_uq_samples(my_campaign.campaign_dir, machine='localhost')

my_campaign.collate()
data_frame = my_campaign.get_collation_result()

# Post-processing analysis
analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
my_campaign.apply_analysis(analysis)

# how many adaptation to make
number_of_adaptations = 7
for i in range(number_of_adaptations):
    #required parameter in the case of a Fabsim run
    skip = my_sampler.count

    print('Adaptation %d' % (i+1))
    #look-ahead step (compute the code at admissible forward points)
    my_sampler.look_ahead(analysis.l_norm)

    #proceed as usual
    my_campaign.draw_samples()
    my_campaign.populate_runs_dir()
    my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
        "./sc/poly_model.py poly_in.json"))
    # fab.run_uq_ensemble(my_campaign.campaign_dir, script_name='poly_model',
    #                     machine='localhost', skip = skip)
    # fab.get_uq_samples(my_campaign.campaign_dir, machine='localhost')
    my_campaign.collate()

    #compute the error at all admissible points, select direction with
    #highest error and add that direction to the grid
    data_frame = my_campaign.get_collation_result()
    analysis.adapt_dimension('f', data_frame, interp_based_error=True)

#proceed as usual with analysis
my_campaign.apply_analysis(analysis)
results = my_campaign.get_last_analysis()

#some post-processing

#analytic mean and standard deviation
a = np.array([1/(2*(i+1)) for i in range(d)])
ref_mean = np.prod(a+1)/2**d
ref_std = np.sqrt(np.prod(9*a[0:d]**2/5 + 2*a[0:d] + 1)/2**(2*d) - ref_mean**2)

print("======================================")
print("Number of samples = %d" % my_sampler._number_of_samples)
print("--------------------------------------")
print("Analytic mean = %.4e" % ref_mean)
print("Computed mean = %.4e" % results['statistical_moments']['f']['mean'])
print("--------------------------------------")
print("Analytic standard deviation = %.4e" % ref_std)
print("Computed standard deviation = %.4e" % results['statistical_moments']['f']['std'])
print("--------------------------------------")
print("First order Sobol indices =", results['sobols_first']['f'])
print("--------------------------------------")

analysis.plot_grid()

analysis.adaptation_table()