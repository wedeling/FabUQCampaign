# -*- coding: utf-8 -*-
#
# This source file is part of the FabSim software toolkit, which is distributed under the BSD 3-Clause license.
# Please refer to LICENSE for detailed information regarding the licensing.
#

from base.fab import *
import os

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
    job(dict(script='run_UQ_sample', job_wall_time='0:15:0', memory='2G'),args)

@task
def uq_ensemble(config="dummy_test", script="ERROR: PARAMETER script SHOULD BE DEFINED FOR TASK UQ_ENSEMBLE",**args):
    """
    Submits an ensemble of EasyVVUQ jobs.
    One job is run for each file in <config_file_directory>/dummy_test/SWEEP.
    """
    
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    env.script = script

    run_ensemble(config, sweep_dir, **args)

@task
def run_uq_ensemble(config, campaign_dir, script_name, skip=0, **args):
    """
    Generic subsmission of samples
    """

    campaign2ensemble(config, campaign_dir=campaign_dir)
    #here should be an option to remove certain runs from the sweep dir
    if int(skip) > 0:
        path_to_config = find_config_file_path(config)
        sweep_dir = path_to_config + "/SWEEP"
        for i in range(int(skip)):
            os.system('rm -r %s/Run_%s' %(sweep_dir, i+1))
    uq_ensemble(config, script_name)


@task
def get_uq_samples(config, campaign_dir, **args):
    """
    Fetches sample output from host, and copies results to EasyVVUQ work directory
    """
    
    fetch_results()

    #loop through all result dirs to find result dir of sim_ID
    found = False
    dirs = os.listdir(env.local_results)
    for dir_i in dirs:
        if config in dir_i:
            found = True
            break

    if found:
        print('Copying results from', env.local_results + '/' + dir_i + 'to' + campaign_dir)
        ensemble2campaign(env.local_results + '/' + dir_i, campaign_dir, **args)
    else:
        print('Campaign dir not found')