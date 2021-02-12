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
    job(dict(script='run_UQ_sample', job_wall_time='0:15:0', memory='2G'), args)

@task
def uq_ensemble(config="dummy_test", script="ERROR: PARAMETER script SHOULD BE DEFINED FOR TASK UQ_ENSEMBLE",**args):
    """
    Submits an ensemble of EasyVVUQ jobs.
    """

    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    # required by qcg-pj to distribute threads correctly
    env.task_model = 'threads'
    env.script = script
    with_config(config)
    run_ensemble(config, sweep_dir, **args)

@task
def run_uq_ensemble(config, campaign_dir, script, skip=0, **args):
    """
    Generic subsmission of samples
    """
    campaign2ensemble(config, campaign_dir=campaign_dir, skip=skip)
    uq_ensemble(config, script, **args)

@task
def get_uq_samples(config, campaign_dir, skip=0, **args):
    """
    Fetches sample output from host, and copies results to EasyVVUQ work directory
    """
    
    # fetch_results()
    #loop through all result dirs to find result dir of sim_ID
    found = False
    dirs = os.listdir(env.local_results)
    for dir_i in dirs:
        #We are assuming here that the name of the directory with the runs dirs
        #STARTS with the config name. e.g. <config_name>_eagle_vecma_28 and
        #not PJ_header_<config_name>_eagle_vecma_28
        config_i = dir_i.split('_' + args['machine'])[0]
        print(config_i)
        # if config == dir_i[0:len(config)]:
        if config == config_i:
            found = True
            break

    if found:
        results_dir = os.path.join(env.local_results, dir_i)
        print('Copying results from %s to %s' % (results_dir, campaign_dir))
        ensemble2campaign(results_dir, campaign_dir, skip=skip, **args)

    else:
        print('Config not found in FabSim3 results directory')


@task
def verify_last_ensemble(config, 
                         campaign_dir,
                         target_filename, **args):
    """
    Verify if last EasyVVUQ ensemble produced all required output files
    """
    #if filename contained '=', replace it back
    target_filename = target_filename.replace('replace_equal', '=')
    #config and sweep directory
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"

    #loop through all result dirs to find result dir of sim_ID
    found = False
    dirs = os.listdir(env.local_results)
    for dir_i in dirs:
        #We are assuming here that the name of the directory with the runs dirs
        #STARTS with the config name. e.g. <config_name>_eagle_vecma_28 and
        #not PJ_header_<config_name>_eagle_vecma_28
        config_i = dir_i.split('_' + args['machine'])[0]
        print(config_i)
        # if config == dir_i[0:len(config)]:
        if config == config_i:            
            found = True
            break    

    if found:
        #directory where FabSim3 copies results to from the remote machine
        results_dir = os.path.join(env.local_results, dir_i)
    
        #all runs in the sweep directory = last ensemble
        run_dirs = os.listdir(sweep_dir)
        all_good = True
        for run_dir in run_dirs:
            #if in one of the runs dirs the target output file is not found
            target = os.path.join(results_dir, 'RUNS', run_dir, target_filename)
            if not os.path.exists(target):
                print("Output for %s not found in %s" % (run_dir, target,))
                all_good = False
    
        #something went wrong
        if not all_good:
            print('Not all output files were found for last ensemble')
            # local("fab {} {}:{},script={}".format("eagle_vecma", "CovidSim_ensemble", config, "CovidSim"))
        #all output files are present
        else:
            print('Last ensemble executed correctly.')
    
        #write a flag to campaign_dir/check.dat
        fp = open(os.path.join(campaign_dir, 'check.dat'), 'w')
        fp.write('%d' % all_good)
        fp.close()
    else:
        print('Config not found in FabSim3 results directory')
