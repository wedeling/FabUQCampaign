# -*- coding: utf-8 -*-
#
# This source file is part of the FabSim software toolkit, which is distributed under the BSD 3-Clause license.
# Please refer to LICENSE for detailed information regarding the licensing.
#
# This file contains FabSim definitions specific to FabDummy.

from base.fab import *

# Add local script, blackbox and template path.
add_local_paths("FabUQCampaign")

@task
def run_UQ_sample(config,**args):
    """Submit a UQ sample job to the remote queue.
    The job results will be stored with a name pattern as defined in the environment,
    e.g. cylinder-abcd1234-legion-256
    config : config directory to use to define input files, e.g. config=cylinder
    Keyword arguments:
            cores : number of compute cores to request
            images : number of images to take
            steering : steering session i.d.
            wall_time : wall-time job limit
            memory : memory per node
    """
    update_environment(args)
    with_config(config)
    execute(put_configs,config)
    job(dict(script='run_UQ_sample', wall_time='0:15:0', memory='2G'),args)

@task
def uq_ensemble(config="dummy_test",**args):
    """
    Submits an ensemble of EasyVVUQ jobs.
    One job is run for each file in <config_file_directory>/dummy_test/SWEEP.
    """
    
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    env.script = 'run_UQ_sample'

    run_ensemble(config, sweep_dir, **args)
    
 
#import numpy as np
#import matplotlib.pyplot as plt
#import easyvvuq as uq
#import os
#import chaospy as cp
#from collections import OrderedDict
#
#@task
#def ocean2D(config, **args):
#  update_environment(args)
#  with_config(config)
#
#  ocean_path = "%s/examples/ocean_2D" % get_plugin_path("FabUQCampaign")
#  easyvvuq_workdir = "%s/tmp-easyvvuq" % get_plugin_path("FabUQCampaign")
#
#  # Input file containing information about parameters of interest
#  input_json = "%s/ocean_input.json" % env.job_config_path_local
#  output_json = "%s/ocean_output.json" % easyvvuq_workdir #TODO: This could be changed to results dir if we want?
#
#  # 1. Initialize `Campaign` object which information on parameters to be sampled
#  #    and the values used for all sampling runs
#  my_campaign = uq.Campaign(name='ocean_test', workdir=easyvvuq_workdir, state_filename=input_json)
#
#  # 2. Set which parameters we wish to include in the analysis and the
#  #    distribution from which to draw samples
#  #!NOTE: the variables are in the standard domain [-1, 1]. The mapping to
#  #       their physical range is done in ADE.py
#  m = 4 
#  my_campaign.vary_param("decay_time_nu", dist=cp.distributions.Uniform(-1, 1))
#  my_campaign.vary_param("decay_time_mu", dist=cp.distributions.Uniform(-1, 1))
#
#  # 3. Select the SC sampler to create a tensor grid
#  sc_sampler = uq.elements.sampling.SCSampler(my_campaign, m)
#  number_of_samples = sc_sampler.number_of_samples
#
#  my_campaign.add_runs(sc_sampler, max_num=number_of_samples)
#
#  # 4. Create directories containing inputs for each run containing the
#  #    parameters determined by the `Sampler`(s).
#  #    This makes use of the `Encoder` specified in the input file.
#  my_campaign.populate_runs_dir()
#
#  # 5. Run execution using Fabsim (on the localhost)
#  sim_ID = "ocean2D_%s" % env.label
#
#  #cmd1 = "fab localhost campaign2ensemble:" + \
#  #      sim_ID + ",campaign_dir=" + my_campaign.campaign_dir
#  campaign2ensemble(sim_ID, campaign_dir=my_campaign.campaign_dir)
#  #cmd2 = "fab localhost uq_ensemble:" + sim_ID
#  uq_ensemble(sim_ID)
#  fetch_results()
#
#  #local('cp -r ~/FabSim3/results/' + sim_ID + '_localhost_16/RUNS/Run_* ' + my_campaign.campaign_dir + '/runs')
#  ensemble2campaign('~/FabSim3/results/' + sim_ID + '_localhost_16', my_campaign.campaign_dir)
#
#  # 6. Aggregate the results from all runs.
#  #    This makes use of the `Decoder` selected in the input file to interpret the
#  #    run output and produce data that can be integrated in a summary pandas
#  #    dataframe.
#
#  output_filename = my_campaign.params_info['out_file']['default']
#  output_columns = ['E']
#
#  aggregate = uq.elements.collate.AggregateSamples(
#                                                my_campaign,
#                                                output_filename=output_filename,
#                                                output_columns=output_columns,
#                                                header=0,
#                                                )
#  aggregate.apply()
#
#  # 7.  Post-processing analysis: computes the 1st two statistical moments and
#  #     gives the ability to use the SCAnalysis object as a surrogate, which
#  #     interpolated the code samples to unobserved parameter variables.
#  sc_analysis = uq.elements.analysis.SCAnalysis(my_campaign, value_cols=output_columns)
#  results, output_file = sc_analysis.get_moments(polynomial_order=m)  # moment calculation
#  # results, output_file = sc_analysis.apply()
#
#
#  # 8. Use the SC samples and integration weights to estimate the
#  #    (1-st order or all) Sobol indices. In this example, at x=1 the Sobol indices
#  #    are NaN, since the variance is zero here.
#
#  # get Sobol indices for free
#  #typ = 'first_order'
#  typ = 'all'
#  sobol_idx = sc_analysis.get_Sobol_indices(typ)
#
#  my_campaign.save_state(output_json)
#  ###############################################################################
#
#  print(results)
#  print(sobol_idx)
