"""
This source file is part of the FabSim software toolkit, which is distributed
under the BSD 3-Clause license. Please refer to LICENSE for detailed information
regarding the licensing.
"""

import os
from fabsim.base.fab import *

# Add local script, blackbox and template path.
add_local_paths("FabUQCampaign")

@task
def uq_ensemble(config, script, **args):
    """
    Internal subroutine, do not call directly. Use run_uq_ensemble instead.

    Parameters
    ----------
    config : string
        Name of the config directory.
    script : string
        Name of the script to execute.

    Returns
    -------
    None.

    """

    with_config(config)
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
    Submit an EasyVVUQ ensemble

    Parameters
    ----------
    config : string
        Name of the config directory.
    campaign_dir : string
        Name of the EasyVVUQ campaign directory (campaign.campaign_dir)
    script : string
        Name of the script to execute.
    skip : int, optional
        Number of runs to skip. The default is 0. If skip=10, only run directories
        with name run_I, with I > 10, will be submitted to the machine. Used in adaptive
        sampling to avoid repeating already computed runs.
    **args : various
        Used to pass the machine name and the PilotJob flag.

    Returns
    -------
    None.

    """
    with_config(config)
    campaign2ensemble(config, campaign_dir=campaign_dir, skip=skip)
    uq_ensemble(config, script, **args)

@task
def get_uq_samples(config, campaign_dir, number_of_samples, skip=0):
    """
    Copies UQ ensemble results from the local FabSim directory to the local
    EasyVVUQ work directory. Does not fetch the results from the (remote)
    host. For this, use the fetch_results() subroutine.

    Parameters
    ----------
    config : string
        Name of the config directory.
    campaign_dir : string
        Name of the EasyVVUQ campaign directory (campaign.campaign_dir)
    number_of_samples : int
        The total number of samples in the ensemble.
    skip : int, optional
        Number of runs to skip. The default is 0. If skip=10, only run directories
        with name run_I, with I > 10, will be submitted to the machine. Used in adaptive
        sampling to avoid repeating already computed runs.

    Returns
    -------
    None.

    """

    #loop through all result dirs to find result dir of sim_ID
    found = False
    dirs = os.listdir(env.local_results)
    for dir_i in dirs:
        #We are assuming here that the name of the directory with the runs dirs
        #STARTS with the config name. e.g. <config_name>_eagle_vecma_28 and
        #not PJ_header_<config_name>_eagle_vecma_28
        if config == dir_i[0:len(config)]:
            found = True
            break

    if found:
        #This compies the entire result directory from the (remote) host back to the
        #EasyVVUQ Campaign directory
        print('Copying results from', env.local_results + '/' + dir_i + 'to' + campaign_dir)
        ensemble2campaign(env.local_results + '/' + dir_i, campaign_dir, skip)

        #If the same FabSim3 config name was used before, the statement above
        #might have copied more runs than currently are used by EasyVVUQ.
        #This removes all runs in the EasyVVUQ campaign dir (not the Fabsim results dir)
        #for which Run_X with X > number of current samples.
        dirs = os.listdir(path.join(campaign_dir, 'runs'))
        for dir_i in dirs:
            run_id = int(dir_i.split('_')[-1])
            if run_id > int(number_of_samples):
                local('rm -r %s/runs/run_%d' % (campaign_dir, run_id))
                print('Removing Run %d from %s/runs' % (run_id, campaign_dir))
    else:
        print('Campaign dir not found')

@task
def verify_last_ensemble(config,
                         campaign_dir,
                         target_filename,
                         machine='localhost'):
    """
    Verify if last EasyVVUQ ensemble produced all required output files

    Parameters
    ----------
    config : string
        Name of the config directory.
    campaign_dir : string
        Name of the EasyVVUQ campaign directory (campaign.campaign_dir)
    target_filename : string
        Name of an output file, whose existence is used as proof that the run
        executed sucesfully.
    machine : string, optional
        Name of the machine the ensemble was exeuted on. Default is localhost.

    Returns
    -------
    None.

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
        config_i = dir_i.split('_' + machine)[0]
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
        missing_outputs = []
        for run_dir in run_dirs:
            #if in one of the runs dirs the target output file is not found
            target = os.path.join(results_dir, 'RUNS', run_dir, target_filename)
            # print("Looking for output file %s in %s" % (target_filename, target))
            if not os.path.exists(target):
                print("Output for %s not found in %s" % (run_dir, target,))
                missing_outputs.append(target)
                all_good = False

        #something went wrong
        if not all_good:
            print('Not all output files were found for last ensemble')
        #all output files are present
        else:
            print('Last ensemble executed correctly.')

        #write a flag to campaign_dir/check.dat
        with open(os.path.join(campaign_dir, 'check.dat'), 'w') as file:
            file.write('%d' % all_good)

        #write missing output files to campaign_dir/missed_targets.dat
        with open(os.path.join(campaign_dir, 'missed_targets.dat'), 'w') as file:
            for target in missing_outputs:
                file.write('%s\n' % target)

    else:
        print('Config not found in FabSim3 results directory')

@task
def remove_succesful_runs(config, campaign_dir):
    """
    This command clears the succesful runs from the SWEEP directory.

    Parameters
    ----------
    config : string
        Name of the config directory.
    campaign_dir : string
        Name of the EasyVVUQ campaign directory (campaign.campaign_dir)

    Returns
    -------
    None.

    """

    #config and sweep directory
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"

    #list of all runs in the SWEEP directory
    succesful_runs = os.listdir(sweep_dir)

    #read missing output files to campaign_dir/missed_targets.dat
    with open(os.path.join(campaign_dir, 'missed_targets.dat'), 'r') as file:
        missing_outputs = file.readlines()

    print('There are %d failed runs.' % len(missing_outputs))

    for path in missing_outputs:
        # find "run_i" with i being the index of the failed run
        tmp = path.split('/')
        run_i = tmp[tmp.index('RUNS') + 1]
        # removes the failed run from the succesful list
        succesful_runs.remove(run_i)

    print('There are %d succesful runs' % len(succesful_runs))

    for run_i in succesful_runs:
        print('Removing %s/%s' % (sweep_dir, run_i))
        os.system('rm -r %s/%s' % (sweep_dir, run_i))
