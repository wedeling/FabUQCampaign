"""
=============
QMC CAMPAIGN
=============
"""

def create_template(n_in):
    
    fname = '%s/templates/sade%d.template' % (home, n_in)
    
    if not os.path.exists(fname):
        print('Creating input template %s.' % fname)
        with open(fname, 'w') as fp:
            fp.write("z_a,z_kappa\n")
            for i in range(n_in):
                fp.write("$a%d,$k%d\n" % (i+1, i+1))
    else:
        print('Input template %s already exists.' % fname)

import os
import easyvvuq as uq
import chaospy as cp
import fabsim3_cmd_api as fab

home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["u_equilibrium"]

for N_INPUTS in range(100, 101):
    
    create_template(N_INPUTS)
    
    WORK_DIR = '/home/wouter/VECMA/Campaigns/SADE%d' % (N_INPUTS,)
    # WORK_DIR = '/tmp'
    
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)
    
    # FabSim3 config name 
    CONFIG = 'SADE'
    # Simulation identifier
    ID = '_%d' % (N_INPUTS,)
    # EasyVVUQ campaign name
    CAMPAIGN_NAME = CONFIG + ID
    # name and relative location of the output file name
    TARGET_FILENAME = 'output.csv'
    # location of the EasyVVUQ database
    DB_LOCATION = "sqlite:///" + WORK_DIR + "/campaign%s.db" % ID
    # Use QCG PiltJob or not
    PILOT_JOB = True
    
    # number of QMC samples
    N_SAMPLES = 5000
    # machine to run the samples on
    MACHINE = 'eagle_vecma'
    
    #set to True if starting a new campaign
    INIT = False
    WAIT_4_COMPLETE = True
    if INIT:
    
        ###########################
        # Set up a fresh campaign #
        ###########################
        
        params = {}
        for i in range(N_INPUTS):
            params["a%d" % (i+1)] = {"default":0.0, "type": "float"}
            params["k%d" % (i+1)] = {"default":0.0, "type": "float"}
        
        encoder = uq.encoders.GenericEncoder(
            template_fname='%s/templates/sade%d.template' % (home, N_INPUTS),
            delimiter='$',
            target_filename='sade_in.csv')
    
        actions = uq.actions.Actions(
            uq.actions.CreateRunDirectory(root=WORK_DIR, flatten=True),
            uq.actions.Encode(encoder),
        )
    
        campaign = uq.Campaign(
            name=CAMPAIGN_NAME,
            db_location=DB_LOCATION,
            work_dir=WORK_DIR,
            verify_all_runs=False
        )
    
        campaign.add_app(
            name=CAMPAIGN_NAME,
            params=params,
            actions=actions
        )
    
        vary = {}
        for i in range(N_INPUTS):
            vary["a%d" % (i+1)] = cp.Normal()   
            vary["k%d" % (i+1)] = cp.Normal()   
    
        # sampler = uq.sampling.RandomSampler(vary, max_num=4)
        sampler = uq.sampling.quasirandom.LHCSampler(vary, max_num=N_SAMPLES)
    
        ###########################################
        # Associate the sampler with the campaign #
        ###########################################
        campaign.set_sampler(sampler)
    
        #########################################
        # draw all of the finite set of samples #
        #########################################
        campaign.execute().collate()
    
        # run the UQ ensemble
        fab.run_uq_ensemble(CONFIG, campaign.campaign_dir, script='SADE',
                            machine=MACHINE, PJ=PILOT_JOB)
    
        if WAIT_4_COMPLETE:
            #wait for job to complete
            fab.wait(machine=MACHINE)
        
            #check if all output files are retrieved from the remote machine
            all_good = fab.verify(CONFIG, campaign.campaign_dir,
                                  TARGET_FILENAME,
                                  machine=MACHINE, PJ=PILOT_JOB)
        
            if all_good:
                #copy the results from the FabSim results dir to the
                fab.get_uq_samples(CONFIG, campaign.campaign_dir, sampler.max_num,
                                    machine=MACHINE)
            else:
                print("Not all samples executed correctly")
                import sys; sys.exit()
        
            #####################
            # execute collate() #
            #####################
            decoder = uq.decoders.SimpleCSV(
                target_filename=TARGET_FILENAME,
                output_columns=output_columns)
        
            actions = uq.actions.Actions(
                uq.actions.Decode(decoder)
            )
            campaign.replace_actions(CAMPAIGN_NAME, actions)
            campaign.execute().collate()
            
            fab.clear_results(MACHINE)    
    else:
    
        ###################
        # reload Campaign #
        ###################
        campaign = uq.Campaign(name=CAMPAIGN_NAME, db_location=DB_LOCATION)
        print("===========================================")
        print("Reloaded campaign {}".format(CAMPAIGN_NAME))
        print("===========================================")
    
        sampler = campaign.get_active_sampler()
        campaign.set_sampler(sampler, update=True)
    
        #wait for job to complete
        fab.wait(machine=MACHINE)
    
        #check if all output files are retrieved from the remote machine
        all_good = fab.verify(CONFIG, campaign.campaign_dir,
                              TARGET_FILENAME,
                              machine=MACHINE, PJ=PILOT_JOB)
    
        if all_good:
            #copy the results from the FabSim results dir to the
            fab.get_uq_samples(CONFIG, campaign.campaign_dir, sampler.max_num,
                                machine=MACHINE)
        else:
            print("Not all samples executed correctly")
            import sys; sys.exit()
    
        #####################
        # execute collate() #
        #####################
        decoder = uq.decoders.SimpleCSV(
            target_filename=TARGET_FILENAME,
            output_columns=output_columns)
    
        actions = uq.actions.Actions(
            uq.actions.Decode(decoder)
        )
        campaign.replace_actions(CAMPAIGN_NAME, actions)
        campaign.execute().collate()
        data_frame = campaign.get_collation_result()
        
        fab.clear_results(MACHINE)    
