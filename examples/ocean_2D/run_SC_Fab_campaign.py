"""
Perform stochastic collocation using EasyVVUQ, sample using FabSim3 campaign2ensemble.
Test problem: 2D ocean model
"""

import numpy as np
import matplotlib.pyplot as plt
import easyvvuq as uq
import os
import chaospy as cp
from collections import OrderedDict

# Input file containing information about parameters of interest
input_json = "ocean_input.json"
output_json = "ocean_output.json"

# 1. Initialize `Campaign` object which information on parameters to be sampled
#    and the values used for all sampling runs
my_campaign = uq.Campaign(name='ocean_test', state_filename=input_json)

# 2. Set which parameters we wish to include in the analysis and the
#    distribution from which to draw samples
#!NOTE: the variables are in the standard domain [-1, 1]. The mapping to
#       their physical range is done in ADE.py
m = 4 
my_campaign.vary_param("decay_time_nu", dist=cp.distributions.Uniform(-1, 1))
my_campaign.vary_param("decay_time_mu", dist=cp.distributions.Uniform(-1, 1))

# 3. Select the SC sampler to create a tensor grid
sc_sampler = uq.elements.sampling.SCSampler(my_campaign, m)
number_of_samples = sc_sampler.number_of_samples

my_campaign.add_runs(sc_sampler, max_num=number_of_samples)

# 4. Create directories containing inputs for each run containing the
#    parameters determined by the `Sampler`(s).
#    This makes use of the `Encoder` specified in the input file.
my_campaign.populate_runs_dir()

# 5. Run execution using Fabsim (on the localhost)
sim_ID ='ocean_example1'
Fab_home = '~/CWI/VECMA/FabSim3'

cmd1 = "cd " + Fab_home + " && fab localhost campaign2ensemble:" + \
        sim_ID + ",campaign_dir=" + my_campaign.campaign_dir
cmd2 = "cd " + Fab_home + " && fab localhost uq_ensemble:" + sim_ID

os.system(cmd1)
os.system(cmd2)

os.system('cp -r ~/FabSim3/results/' + sim_ID + '_localhost_16/RUNS/Run_* ' + my_campaign.campaign_dir + '/runs')

#"cd ~/CWI/VECMA/FabSim3 && fab localhost campaign2ensemble:ade-example1,campaign_dir=<full path>"
#"cd ~/CWI/VECMA/FabSim3 %% fab localhost ade_ensemble:ade-example1"

# 6. Aggregate the results from all runs.
#    This makes use of the `Decoder` selected in the input file to interpret the
#    run output and produce data that can be integrated in a summary pandas
#    dataframe.

output_filename = my_campaign.params_info['out_file']['default']
output_columns = ['u']

aggregate = uq.elements.collate.AggregateSamples(
                                                my_campaign,
                                                output_filename=output_filename,
                                                output_columns=output_columns,
                                                header=0,
                                                )
aggregate.apply()

# 7.  Post-processing analysis: computes the 1st two statistical moments and
#     gives the ability to use the SCAnalysis object as a surrogate, which
#     interpolated the code samples to unobserved parameter variables.
sc_analysis = uq.elements.analysis.SCAnalysis(
    my_campaign, value_cols=output_columns)
results, output_file = sc_analysis.get_moments(polynomial_order=m)  # moment calculation
# results, output_file = sc_analysis.apply()


# 8. Use the SC samples and integration weights to estimate the
#    (1-st order or all) Sobol indices. In this example, at x=1 the Sobol indices
#    are NaN, since the variance is zero here.

# get Sobol indices for free
#typ = 'first_order'
typ = 'all'
sobol_idx = sc_analysis.get_Sobol_indices(typ)

my_campaign.save_state(output_json)
###############################################################################

print(results)
print(sobol_idx)

plt.show()
