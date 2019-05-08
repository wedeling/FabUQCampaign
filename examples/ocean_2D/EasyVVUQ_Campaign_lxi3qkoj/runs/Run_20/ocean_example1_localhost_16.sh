# 
# Copyright (C) University College London, 2007-2014, all rights reserved.
# 
# This file is part of FabSim and is CONFIDENTIAL. You may not work 
# with, install, use, duplicate, modify, redistribute or share this
# file, or any part thereof, other than as allowed by any agreement
# specifically made by you with University College London.
# 
# no batch system


cd /home/wouter/FabSim3/results/ocean_example1_localhost_16/RUNS/Run_20
echo Running...

/usr/bin/env > env.log

/home/wouter/anaconda3/bin/python3 /home/wouter/CWI/VECMA/FabSim3/plugins/FabUQCampaign/examples/ocean_2D/ocean.py ocean_in.json
