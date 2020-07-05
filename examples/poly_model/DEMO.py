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
my_campaign = uq.Campaign(name='demo', work_dir='/tmp')

#number of uncertain parameters
d = 2

##########################
# Define parameter space #
##########################

params = {}
#total number of variables
for i in range(20):
    params["x%d" % (i + 1)] = {"type": "float", "default": 0.5}
    
params["d"] = {"type": "integer", "default": d}
params["out_file"] = {"type": "string", "default": "output.csv"}
output_filename = params["out_file"]["default"]
output_columns = ["f"]

####################################################
# Create an encoder, decoder and collation element #
####################################################

encoder = uq.encoders.GenericEncoder(
    template_fname=HOME + '/sc/poly.template',
    delimiter='$',
    target_filename='poly_in.json')

decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                output_columns=output_columns,
                                header=0)

collater = uq.collate.AggregateSamples()

#####################################
# Add everything to the current app #
#####################################

my_campaign.add_app(name="sc",
                    params=params,
                    encoder=encoder,
                    decoder=decoder,
                    collater=collater)

############################################
# Decide which variables to make uncertain #
############################################

vary = {}
for i in range(d):
    vary["x%d" % (i + 1)] = cp.Uniform(0.0, 1.0)

####################
# Create a sampler #
####################

my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=2,
                                   quadrature_rule="C",
                                   sparse=True, growth=True,
                                   midpoint_level1=True,
                                   dimension_adaptive=True)

# Associate the sampler with the campaign
my_campaign.set_sampler(my_sampler)

##########################
# Draw all (new) samples #
##########################

my_campaign.draw_samples()
my_campaign.populate_runs_dir()

#################################################################
# Interface with FabSim3 to compute samples on (remote) machine #
#################################################################

#run ensemble
fab.run_uq_ensemble(my_campaign.campaign_dir, script_name='poly_model', 
                    machine='localhost')

#retrieve results
fab.get_uq_samples(my_campaign.campaign_dir, machine='localhost')

#combine all samples in a dataframe
my_campaign.collate()
data_frame = my_campaign.get_collation_result()

############################
# Post-processing analysis #
############################

analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
my_campaign.apply_analysis(analysis)

# how many adaptation to make
number_of_refinements = 3
for i in range(number_of_refinements):
    #required parameter in the case of a Fabsim run
    skip = my_sampler.count

    ###############################
    # Refinement of sampling plan #
    ###############################
    
    if my_sampler.dimension_adaptive:
        my_sampler.look_ahead(analysis.l_norm)
    else:
        my_sampler.next_level_sparse_grid()

    ####################
    # Proceed as usual #
    ####################
    my_campaign.draw_samples()
    my_campaign.populate_runs_dir()
    fab.run_uq_ensemble(my_campaign.campaign_dir, script_name='poly_model',
                        machine='localhost', skip = skip)
    fab.get_uq_samples(my_campaign.campaign_dir, machine='localhost')
    my_campaign.collate()

    if my_sampler.dimension_adaptive:    
        #compute the error at all admissible points, select direction with
        #highest error and add that direction to the grid
        data_frame = my_campaign.get_collation_result()
        analysis.adapt_dimension('f', data_frame)

#proceed as usual with analysis
my_campaign.apply_analysis(analysis)
results = my_campaign.get_last_analysis()

#########################################
# Compare computed results to reference #
#########################################

#analytic mean and standard deviation for U[0,1]
a = np.array([1/(5**i) for i in range(d)])
ref_mean = np.prod(a+1)/2**d
ref_std = np.sqrt(np.prod(9*a[0:d]**2/5 + 2*a[0:d] + 1)/2**(2*d) - ref_mean**2)

print("======================================")
print("Number of samples = %d" % my_sampler._number_of_samples)
print("--------------------------------------")
print("Analytic mean = %.4e" % ref_mean)
print("Computed mean = %.4e" % results['statistical_moments']['f']['mean'])
print("--------------------------------------")
print("Analytic standard deiation = %.4e" % ref_std)
print("Computed standard deiation = %.4e" % results['statistical_moments']['f']['std'])
print("--------------------------------------")

analysis.plot_grid()
plt.show()
