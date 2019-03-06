# -*- coding: utf-8 -*-
#
# This source file is part of the FabSim software toolkit, which is distributed under the BSD 3-Clause license.
# Please refer to LICENSE for detailed information regarding the licensing.
#
# This file contains FabSim definitions specific to FabDummy.

from base.fab import *

# Add local script, blackbox and template path.
add_local_paths("FabSC")

@task
def sc(config,**args):
    """Submit a Dummy job to the remote queue.
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
    job(dict(script='ade', wall_time='0:15:0', memory='2G'),args)

@task
def sc_ensemble(config="dummy_test",**args):
    """
    Submits an ensemble of dummy jobs.
    One job is run for each file in <config_file_directory>/dummy_test/SWEEP.
    """
    
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    env.script = 'sc'

    run_ensemble(config, sweep_dir, **args)
    
