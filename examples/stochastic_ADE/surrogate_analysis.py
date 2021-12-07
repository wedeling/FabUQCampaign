import os
import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
import easyvvuq as uq

plt.close('all')

# number of inputs for a(x) and kappa(x)
N_INPUTS = 100

home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["u_equilibrium"]
WORK_DIR = '/home/wouter/VECMA/Campaigns/SADE%d' % (N_INPUTS,)
# WORK_DIR = '/tmp'

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

###################
# reload Campaign #
###################
campaign = uq.Campaign(name=CAMPAIGN_NAME, db_location=DB_LOCATION)
print("===========================================")
print("Reloaded campaign {}".format(CAMPAIGN_NAME))
print("===========================================")
sampler = campaign.get_active_sampler()
campaign.set_sampler(sampler, update=True)

surr_campaign = es.Campaign()
params, samples = surr_campaign.load_easyvvuq_data(campaign, output_columns)
samples = samples[output_columns[0]]
if N_INPUTS == 8:
    params = np.delete(params, 1894, axis=0)
    
#train a DAS network
D = 2 * N_INPUTS
d = 10

test_fracs = np.arange(0.05, 0.0, -0.05)
final_training_error = np.zeros(test_fracs.size)
final_test_error = np.zeros(test_fracs.size)

for idx_frac, test_frac in enumerate(test_fracs):
    # size training data
    train_size = (1.0 - test_frac) * samples.size
    # minibatch size
    batch_size = 100
    # number of iterations for 1 epoch
    n_iter = int(train_size / batch_size)
    # max number of epochs
    n_epocs = 50
    
    #training and test error
    training_error = np.zeros(n_epocs)
    test_error = np.zeros(n_epocs)
    
    # test error avarged over the last 6 iterations
    avg_test_err = 99
    # tolerance in test error variation, below which to terminate
    test_error_tol = 1e-4
    
    for n in range(n_epocs):
        
        # first time creat surrogate method train for 1 epoch
        if n == 0:
            surrogate = es.methods.DAS_Surrogate()
            surrogate.train(params, samples, d, n_iter=n_iter, n_layers=4, n_neurons=100, test_frac = test_frac, 
                            batch_size = batch_size)
            
            dims = surrogate.get_dimensions()
        # subsequent time, just train for 1 epoch
        else:
            surrogate.neural_net.train(n_iter, store_loss = True)
    
        #########################
        # Compute error metrics #
        #########################
    
        # run the trained model forward at training locations
        n_mc = dims['n_train']
        pred = np.zeros([n_mc, dims['n_out']])
        for i in range(n_mc):
            pred[i,:] = surrogate.predict(params[i])
           
        train_data = samples[0:dims['n_train']]
        rel_err_train = np.linalg.norm(train_data - pred) / np.linalg.norm(train_data)
        
        # run the trained model forward at test locations
        pred = np.zeros([dims['n_test'], dims['n_out']])
        for idx, i in enumerate(range(dims['n_train'], dims['n_samples'])):
            pred[idx] = surrogate.predict(params[i])
        test_data = samples[dims['n_train']:]
        rel_err_test = np.linalg.norm(test_data - pred) / np.linalg.norm(test_data)
    
        # store errors
        training_error[n] = rel_err_train
        test_error[n] = rel_err_test
        
        # look for variation in the last 6 test errors
        if n > 6:
            check = np.mean(test_error[n-6:n])
            # if variation falls below 0.001
            if np.abs(check - avg_test_err) < test_error_tol:
                break
            avg_test_err = check
    
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='epochs')
    ax.plot(np.arange(1, n + 1), training_error[0:n], '-ro', label='relative training error')
    ax.plot(np.arange(1, n + 1), test_error[0:n], '-b*', label='relative test error')
    plt.legend(loc = 0)
    plt.tight_layout()
    
    IMAGE_DIR = '%s/images' % WORK_DIR
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    fig.savefig('%s/errors_test_frac%.2f.png' % (IMAGE_DIR, test_frac))
    plt.close()

    final_training_error[idx_frac] = training_error[n]    
    final_test_error[idx_frac] = test_error[n]    

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='epochs')
ax.plot(test_fracs, final_training_error, '-ro', label='relative training error')
ax.plot(test_fracs, final_test_error, '-b*', label='relative test error')
plt.legend(loc = 0)
plt.tight_layout()
fig.savefig('%s/FINAL_errors.png' % (IMAGE_DIR, ))

###########################
# Sensitivity experiments #
###########################

analysis = es.analysis.ANN_analysis(surrogate)

n_mc = 10**4
# params = np.array([p.sample(n_mc) for p in sampler.vary.get_values()]).T
idx, mean = analysis.sensitivity_measures(params)
params_ordered = np.array(list(sampler.vary.get_keys()))[idx[0]]

fig = plt.figure('sensitivity', figsize=[4, 8])
ax = fig.add_subplot(111)
ax.set_ylabel(r'$\int\frac{\partial ||y||^2_2}{\partial x_i}p({\bf x})d{\bf x}$', fontsize=14)
# find max quad order for every parameter
ax.bar(range(mean.size), height = mean[idx].flatten())
ax.set_xticks(range(mean.size))
ax.set_xticklabels(params_ordered)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
