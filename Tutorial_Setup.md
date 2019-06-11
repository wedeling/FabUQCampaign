Setup activities for the FabUQCampaign tutorials
=====

This document describes what you need to do to set up all the software required for the FabUQCampaign tutorials on the
advection-diffusion equation and 2D ocean model.

## Prerequisites

To perform this tutorial, you will require 
* Linux environment
* Python 3.6+
* Python libraries
   * numpy (see https://www.numpy.org)
   * EasyVVUQ (see https://github.com/UCL-CCS/EasyVVUQ)
   * matplotlib (see https://matplotlib.org)
* The following software packages:
   * FabSim3
   * The FabUQCampaign plugin

Below you can find installation instructions for each of these packages.

### Installing EasyVVUQ

EasyVVUQ can be installed either via `pip install easyvvuq', or manually from the source:
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


## 2. Configuration

Once you have installed the required dependencies, you will need to take a few small configuration steps:
1. Go to `(FabSim Home)/deploy`
2. Open `machines_user.yml`
3. Under the section `default:`, please add the following lines:
   <br/> a. `  flee_location=(Flee Home)`
   <br/> _NOTE: Please replace (Flee Home) with your actual install directory._
   <br/> b. `  flare_location=(Flare Home)`
   <br/> _NOTE: Please replace (Flare Home) with your actual install directory._
   
 ## 3. Main tutorial
 
 Once you have completed these tasks, you can do the main tutorial at https://github.com/djgroen/FabFlee/blob/master/doc/Tutorial.md
