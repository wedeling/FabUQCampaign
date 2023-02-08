# def draw():

#     plt.clf()
#     ax = fig.add_subplot(121)
#     ax.set_ylim(bottom=0.01, top=0.5)
#     ax.set_xlabel('training data size')
#     ax.set_ylabel('relative error e')
#     ax.set_title('training error')
#     sns.despine(top=True)
    
#     ax.plot(data_size, err_ANN[:,:,0].T, '.', color='mediumaquamarine', label='no dropout')
    
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = OrderedDict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys())
    
#     ax2 = fig.add_subplot(122, sharey=ax)
#     ax2.set_xlabel('training data size')
#     ax2.set_title('test error')
    
#     # make the plot using all samples, not with confidence intervals
#     ax2.plot(data_size, err_ANN[:,:,1].T, '.', color='mediumaquamarine')
    
#     sns.despine(left=True, ax=ax2)
#     ax2.get_yaxis().set_visible(False)
#     plt.tight_layout()
#     plt.pause(0.1)

def read_data(data_frame, params_df, features, target):

    QoI = 'binding_energy'

    runs = np.unique(data_frame['run'].values)

    for run in runs:
        samples = data_frame.loc[data_frame['run'] == run]
        samples = samples[QoI].values
        contains_nan = np.isnan(samples).any()
        if not contains_nan and samples.size > 0:
            # feats = np.repeat(params[run].reshape([1, -1]), n_replicas, axis=0)
            feats = params_df.loc[params_df['run'] == run][param_names].values
            features = np.append(features, feats, axis=0)
            sample_mean = np.mean(samples).reshape([-1, 1])
            target = np.append(target, sample_mean, axis=0)    
    
    return features, target

import numpy as np
# import matplotlib.pyplot as plt
import easysurrogate as es
import pandas as pd
# import seaborn as sns
# from collections import OrderedDict
# import os

# plt.close('all')
# plt.rcParams['image.cmap'] = 'seismic'
# fig = plt.figure(figsize=[8,4])

# HOME = os.path.abspath(os.path.dirname(__file__))

# In[] 

# Generate training data 
# params binding energy
params_df1 = pd.read_csv('/home/wouter/VECMA/Python/MD/binding_energy_FF_only/esmacs_ff.inputs.csv')
D = 153
param_names = params_df1.keys()[0:D]

params_df2 = pd.read_csv('/home/wouter/VECMA/Python/MD/binding_energy/esmacs_ff+sim.inputs.csv')
# D = 167
# param_names = params_df2.keys()[0:D]

features = np.empty((0, D))
target = np.empty((0, 1))

data_frame1 = pd.read_csv('/home/wouter/VECMA/Python/MD/binding_energy_FF_only/esmacs_ff.nonavg.outputs.csv')
# data_frame2 = pd.read_csv('./binding_energy/esmacs_ff+sim.nonavg.outputs.csv')

features, target = read_data(data_frame1, params_df1, features, target)
# features, target = read_data(data_frame2, params_df2, features, target)

# scale inputs within [-1, 1]
p_max = np.max(features, axis=0)
p_min = np.min(features, axis=0)
features = (features - 0.5 * (p_min + p_max)) / (0.5 * (p_max - p_min))
    
# In[] 

# number of neurons, replicas and number of test fractions
# n_neurons = 10
n_replicas = 2
n_test_fracs = 10
n_samples = target.shape[0]
n_epochs = 200
batch_size = 10

# In[]

# set test fractions
test_fracs = np.linspace(0.9, 0.1, n_test_fracs)

# size of training data used
data_size = np.ceil((1 - test_fracs) * n_samples)

# initialize arrays
err_ANN = np.zeros([n_replicas, n_test_fracs, 2])

# In[]

# compute errors
for r in range(n_replicas):

    for n, test_frac in enumerate(test_fracs):

        ########################################
        # Train an unconstrained ANN surrogate #
        ########################################

        n_iter = np.ceil(data_size[n] / batch_size * n_epochs).astype('int')
    
        surrogate_uc = es.methods.ANN_Surrogate()
        # train ANN. the input parameters are already scaled to [-1, 1], so no need to
        # standardize these        
        surrogate_uc.train(features, target, 
                        n_iter=n_iter, n_layers=3, n_neurons=20, 
                        test_frac = test_frac, batch_size = batch_size, 
                        standardize_X=False, standardize_y=True,
                        learning_rate=0.001, beta1=0.9,
                        dropout=True, dropout_prob = [0.8, 
                                                          0.8,
                                                          0.8,
                                                          0.8])

        #########################
        # Compute error metrics #
        #########################
        analysis = es.analysis.ANN_analysis(surrogate_uc)
        rel_err_train, rel_err_test = analysis.get_errors(features, target, relative=True)
        err_ANN[r, n, 0] = rel_err_train
        err_ANN[r, n, 1] = rel_err_test

campaign = es.Campaign()
campaign.store_data_to_hdf5({'err_train' : err_ANN[:, :, 0], 'err_test' : err_ANN[:, :, 1]},
                             file_path='output.hdf5')