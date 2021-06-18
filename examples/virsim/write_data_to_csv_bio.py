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
FC_bio_campaign = uq.Campaign(state_file = "campaign_state_FC_bio_cdf_2k.json", work_dir = workdir)
print('========================================================')
print('Reloaded campaign', FC_bio_campaign.campaign_dir.split('/')[-1])
print('========================================================')

# collate output
FC_bio_campaign.collate()
# get full dataset of data
FC_bio_data = FC_bio_campaign.get_collation_result()
#print(FC_bio_data.columns)
FC_bio_header = [i[0] for i in FC_bio_data.columns]
FC_bio_data.columns = FC_bio_header
#print(FC_bio_data.columns)
#print(type(FC_bio_data)) #<class 'pandas.core.frame.DataFrame'>
FC_bio_data.to_csv('FC_bio_data.csv', columns=FC_bio_data.columns, index=False)

# Reload the CT campaign with biology
CT_bio_campaign = uq.Campaign(state_file = "campaign_state_CT_bio_cdf_2k.json", work_dir = workdir)
print('========================================================')
print('Reloaded campaign', CT_bio_campaign.campaign_dir.split('/')[-1])
print('========================================================')

# collate output
CT_bio_campaign.collate()
# get full dataset of data
CT_bio_data = CT_bio_campaign.get_collation_result()
#print(CT_bio_data.columns)
CT_bio_header = [i[0] for i in CT_bio_data.columns]
CT_bio_data.columns = CT_bio_header
CT_bio_data.to_csv('CT_bio_data.csv', columns=CT_bio_data.columns, index=False)

# Reload the IL campaign with biology
IL_bio_campaign = uq.Campaign(state_file = "campaign_state_IL_bio_cdf_2k.json", work_dir = workdir)
print('========================================================')
print('Reloaded campaign', IL_bio_campaign.campaign_dir.split('/')[-1])
print('========================================================')

# collate output
IL_bio_campaign.collate()
# get full dataset of data
IL_bio_data = IL_bio_campaign.get_collation_result()
#print(IL_bio_data.columns)
IL_bio_header = [i[0] for i in IL_bio_data.columns]
IL_bio_data.columns = IL_bio_header
IL_bio_data.to_csv('IL_bio_data.csv', columns=IL_bio_data.columns, index=False)

# Reload the PO campaign with biology
PO_bio_campaign = uq.Campaign(state_file = "campaign_state_PO_bio_cdf_2k.json", work_dir = workdir)
print('========================================================')
print('Reloaded campaign', PO_bio_campaign.campaign_dir.split('/')[-1])
print('========================================================')

# collate output
PO_bio_campaign.collate()
# get full dataset of data
PO_bio_data = PO_bio_campaign.get_collation_result()
#print(PO_bio_data.columns)
PO_bio_header = [i[0] for i in PO_bio_data.columns]
PO_bio_data.columns = PO_bio_header
PO_bio_data.to_csv('PO_bio_data.csv', columns=PO_bio_data.columns, index=False)

