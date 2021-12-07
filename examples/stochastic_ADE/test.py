import os
import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
import easyvvuq as uq
from scipy import linalg

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

# number of inputs for a(x) and kappa(x)
N_INPUTS = 5

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
samples = samples[output_columns[0]][:,-1].reshape([-1,1])
if N_INPUTS == 8:
    params = np.delete(params, 1894, axis=0)

#train a DAS network
D = 2 * N_INPUTS

surrogate = es.methods.ANN_Surrogate()
surrogate.train(params, samples, n_iter=10000, n_layers=4, n_neurons=50, test_frac = 0.0, 
                batch_size = 64)
dims = surrogate.get_dimensions()

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
print("Training error = %.4f" % rel_err_train)

# run the trained model forward at test locations
pred = np.zeros([dims['n_test'], dims['n_out']])
for idx, i in enumerate(range(dims['n_train'], dims['n_samples'])):
    pred[idx] = surrogate.predict(params[i])
test_data = samples[dims['n_train']:]
rel_err_test = np.linalg.norm(test_data - pred) / np.linalg.norm(test_data)
print("Test error = %.4f" % rel_err_test)

# n_mc = 10**4
# params = np.array([p.sample(n_mc) for p in sampler.vary.get_values()]).T
n_mc = params.shape[0]

surrogate.neural_net.set_batch_size(1)

C = 0.0
n_mc_samples = np.zeros([n_mc])

for i, param in enumerate(params):
    param_i = (param - surrogate.feat_mean) / surrogate.feat_std
    df_dx = surrogate.neural_net.d_norm_y_dX(param_i.reshape([1, -1]))
    n_mc_samples[i] = surrogate.predict(param)
    C += np.dot(df_dx, df_dx.T) / n_mc
    
eigvals, eigvecs = linalg.eigh(C)

# Sort the eigensolutions in the descending order of eigenvalues
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

d = 4
W1 = eigvecs[:, 0:d]
y = np.dot(W1.T, params.T).T

fig = plt.figure()
ax = fig.add_subplot(111)
ct = ax.tricontourf(y[:, 0], y[:, 1], n_mc_samples, 100)
plt.colorbar(ct)
plt.tight_layout()

idx = np.flipud(np.argsort(np.abs(W1[:,0])))
print(np.array(list(sampler.vary.get_keys()))[idx])

#########################

das_surrogate = es.methods.DAS_Surrogate()
das_surrogate.train(params, samples, d, n_iter=10000, n_layers=4, n_neurons=100, test_frac = 0.0, 
                batch_size = 64)

W1_das = das_surrogate.neural_net.layers[1].W

n_mc = params.shape[0]

das_surrogate.neural_net.set_batch_size(1)

C = 0.0
n_mc_samples = np.zeros([n_mc])
foo = 0.0

for i, param in enumerate(params):
    n_mc_samples[i] = das_surrogate.predict(param)
    param_i = (param - das_surrogate.feat_mean) / das_surrogate.feat_std
    df_dx = das_surrogate.neural_net.d_norm_y_dX(param_i.reshape([1, -1]))
    df_dh = das_surrogate.neural_net.layers[1].delta_hy.reshape([-1,1])
    foo += np.dot(df_dh, df_dh.T) / n_mc
    C += np.dot(df_dx, df_dx.T) / n_mc
    
eigvals, eigvecs = linalg.eigh(C)

# Sort the eigensolutions in the descending order of eigenvalues
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

fig = plt.figure()
ax = fig.add_subplot(111, title=r'$d=%d$' % d)
ax.set_ylabel(r'$\lambda_i$')
ax.set_ylabel(r'$i$')
ax.plot(eigvals, 'ro')
plt.tight_layout()

fig = plt.figure(figsize=[8,4])
ax = fig.add_subplot(121, title='AS')
W1 = eigvecs[:, 0:d]
W2 = eigvecs[:, d:]
y = np.dot(W1.T, params.T).T
ct = ax.tricontourf(y[:, 0], y[:, 1], n_mc_samples, 100)
ax.set_xlabel(r'$y_1$')
ax.set_ylabel(r'$y_2$')
plt.colorbar(ct)
#
ax = fig.add_subplot(122, title='DAS')
y_das = np.dot(W1_das.T, params.T).T
ct = ax.tricontourf(y_das[:, 0], y_das[:, 1], n_mc_samples, 100)
ax.set_xlabel(r'$y_1$')
ax.set_ylabel(r'$y_2$')
plt.colorbar(ct)
plt.tight_layout()

# R = np.dot(W1.T, W1_das)
# y_trans = np.dot(R, y.T).T

# ax = fig.add_subplot(133, title='TRANS')
# ct = ax.tricontourf(y_trans[:, 0], y_trans[:, 1], n_mc_samples, 100)
# plt.colorbar(ct)
# plt.tight_layout()

eigvals_red, eigvecs_red = linalg.eigh(foo)
order = eigvals_red.argsort()[::-1]
eigvals_red = eigvals_red[order]
eigvecs_red = eigvecs_red[:, order]

print(np.dot( W1_das, np.dot(foo, W1_das.T)) - C)
print(np.dot(eigvecs_red.T, np.dot(foo, eigvecs_red)) - np.diag(eigvals[0:d]))
print(np.dot(W1, np.dot(np.dot(eigvecs_red.T, np.dot(foo, eigvecs_red)), W1.T)) - C)

# correct_sign = np.array(np.sign([np.dot(np.dot(W1, eigvecs_red.T[:,i]), W1_das[:,i]) for i in range(d)]))
# eigvecs_red *= correct_sign
# print(W1_das - np.dot(W1, eigvecs_red.T))
# print(correct_sign)
# print(np.abs(eigvecs_red.T) - np.abs(np.dot(W1.T, W1_das)))
print(eigvecs_red)
print(np.dot(W1_das.T, W1))

# A1 = np.copy(eigvecs_red)
# A2 = np.copy(eigvecs_red)
# A3 = np.copy(eigvecs_red)
# A4 = np.copy(eigvecs_red)

# A2[:, 0] *= -1
# A3[:, 1] *= -1
# A4 *= -1

# print(W1_das - np.dot(W1, A1.T))
# print(W1_das - np.dot(W1, A2.T))
# print(W1_das - np.dot(W1, A3.T))
# print(W1_das - np.dot(W1, A4.T))

#####################################
# global gradient-based sensitivity #
#####################################

analysis = es.analysis.DAS_analysis(das_surrogate)

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

###################
# Activity scores #
###################


