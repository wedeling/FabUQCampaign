def p_box(samples):
    sample_sort = np.sort(samples[burn_in:].flatten())
    prob = np.linspace(0, 1, sample_sort.size)
    
    return sample_sort, prob

import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt

plt.close('all')

work_dir = '/home/wouter/VECMA/Campaigns'
#reload Campaign
campaign = uq.Campaign(state_file="easyvvuq_state.json", 
                       work_dir=work_dir)

data_frame = campaign.get_collation_result()

#burn-in period
burn_in = 5000

QoI = ['Q1', 'Q2', 'Q3', 'Q4']

p_boxes = {}

for q in QoI:
    p_boxes[q] = []
    for run in data_frame[q].keys():
        X, prob = p_box(data_frame[q][run])
        p_boxes[q].append(X)
    p_boxes[q] = np.array(p_boxes[q])

fig = plt.figure(figsize=[12,3])

i = 1
for q in QoI:
    Q = p_boxes[q]
    left_bound = np.min(Q, axis=0)
    right_bound = np.max(Q, axis=0)
    min_Q = np.min(Q); max_Q = np.max(Q)    

    ax = fig.add_subplot(1, 4, i, xlim=[0.9*min_Q, 1.1*max_Q])
    ax.set_xlabel(r'$Q_%d$' % i)
    if i == 1:
        ax.set_ylabel(r'probability')
    ax.set_xticks([min_Q, 0.5*(min_Q + max_Q), max_Q])
    ax.set_xticklabels(['%.2f' % min_Q, '%.2f' % (0.5*(min_Q + max_Q)), '%.2f' % max_Q])

    ax.plot(Q.T, prob, color='steelblue', alpha = 0.5)
    ax.plot(left_bound, prob, color='navy', linewidth=3)
    ax.plot(right_bound, prob, color='navy', linewidth=3)

    i += 1

plt.tight_layout()
plt.show()