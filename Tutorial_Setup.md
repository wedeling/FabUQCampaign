Setup activities for the FabUQCampaign tutorials
=====

This document describes what you need to do to set up all the software required for the FabUQCampaign tutorials of the
advection-diffusion equation and 2D ocean model.

## Prerequisites

To perform the tutorials, you will require 
* Linux environment
* Python 3.6+
* Python libraries
   * numpy (see https://www.numpy.org)
   * EasyVVUQ (see https://github.com/UCL-CCS/EasyVVUQ)
   * matplotlib (see https://matplotlib.org)
   * h5py (only for ocean model tutorial, see https://github.com/h5py/h5py)
* The following software packages:
   * FabSim3
   * The FabUQCampaign plugin

Below you can find installation instructions for each of these packages.

### Installing EasyVVUQ

EasyVVUQ can be installed either via `pip3 install easyvvuq`, or manually from the source:
```
git clone https://github.com/UCL-CCS/EasyVVUQ.git
cd EasyVVUQ/
pip3 install -r requirements.txt
python3 setup.py install
```

For more details, see https://github.com/UCL-CCS/EasyVVUQ.

### Installing FabSim3

To install FabSim3, you need to install dependencies and clone the FabSim3 repository.
<br/> For detailed installation instructions, see https://github.com/djgroen/FabSim3/blob/master/INSTALL.md
```
git clone https://github.com/djgroen/FabSim3.git
```
We will assume that you will install FabSim3 in a directory called (FabSim3 Home), e.g. `~/FabSim3/`.

_NOTE: Please make sure both `machines.yml` and `machines_user.yml` are configured correctly based on the installation guide._

Once you have installed FabSim3, you can install FabUQCampaign by typing:
```
fabsim localhost install_plugin:FabUQCampaign
```
The FabUQCampaign plugin will appear in `~/FabSim3/plugins/FabUQCampaign`.
   
 ## 2. Main tutorial
 
 Once you have completed these tasks, you can do the advection-diffusion tutorial at:
 
 https://github.com/wedeling/FabUQCampaign/blob/master/README.md
 
 or the 2D ocean model tutorial at:
 
 https://github.com/wedeling/FabUQCampaign/blob/master/Tutorial_ocean.md
