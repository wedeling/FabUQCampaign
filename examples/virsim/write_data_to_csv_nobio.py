"""
@author: Federica Gugole
__license__= "LGPL"
"""

import numpy as np
import easyvvuq as uq
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter, NullFormatter
plt.rcParams.update({'font.size': 18, 'legend.fontsize': 15})
plt.rcParams['figure.figsize'] = 12,9

"""
*************
* Load data *
*************
"""
workdir = '/export/scratch1/federica/VirsimCampaigns'#'/tmp'

# home directory of this file    
HOME = os.path.abspath(os.path.dirname(__file__))

# Reload the FC campaign with biology
FC_campaign = uq.Campaign(state_file = "campaign_state_FC_MC2k.json", work_dir = workdir)
print('========================================================')
print('Reloaded campaign', FC_campaign.campaign_dir.split('/')[-1])
print('========================================================')

# collate output
FC_campaign.collate()
# get full dataset of data
FC_data = FC_campaign.get_collation_result()
#print(FC_data.columns)
FC_header = [i[0] for i in FC_data.columns]
FC_data.columns = FC_header
#print(FC_data.columns)
#print(type(FC_data)) #<class 'pandas.core.frame.DataFrame'>
FC_data.to_csv('FC_nobio_data.csv', columns=FC_data.columns, index=False)

# Reload the CT campaign with biology
CT_campaign = uq.Campaign(state_file = "campaign_state_CT_MC2k_newdistr.json", work_dir = workdir)
print('========================================================')
print('Reloaded campaign', CT_campaign.campaign_dir.split('/')[-1])
print('========================================================')

# collate output
CT_campaign.collate()
# get full dataset of data
CT_data = CT_campaign.get_collation_result()
#print(CT_data.columns)
CT_header = [i[0] for i in CT_data.columns]
CT_data.columns = CT_header
CT_data.to_csv('CT_nobio_data.csv', columns=CT_data.columns, index=False)

# Reload the IL campaign with biology
IL_campaign = uq.Campaign(state_file = "campaign_state_IL_nobio_MC2k.json", work_dir = workdir)
print('========================================================')
print('Reloaded campaign', IL_campaign.campaign_dir.split('/')[-1])
print('========================================================')

# collate output
IL_campaign.collate()
# get full dataset of data
IL_data = IL_campaign.get_collation_result()
#print(IL_data.columns)
IL_header = [i[0] for i in IL_data.columns]
IL_data.columns = IL_header
IL_data.to_csv('IL_nobio_data.csv', columns=IL_data.columns, index=False)

# Reload the PO campaign with biology
PO_campaign = uq.Campaign(state_file = "campaign_state_PO_MC2k.json", work_dir = workdir)
print('========================================================')
print('Reloaded campaign', PO_campaign.campaign_dir.split('/')[-1])
print('========================================================')

# collate output
PO_campaign.collate()
# get full dataset of data
PO_data = PO_campaign.get_collation_result()
#print(PO_data.columns)
PO_header = [i[0] for i in PO_data.columns]
PO_data.columns = PO_header
PO_data.to_csv('PO_nobio_data.csv', columns=PO_data.columns, index=False)

