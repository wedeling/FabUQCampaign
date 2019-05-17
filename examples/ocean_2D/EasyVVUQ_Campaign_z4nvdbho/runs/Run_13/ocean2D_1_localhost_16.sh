# 
# Copyright (C) University College London, 2007-2014, all rights reserved.
# 
# This file is part of FabSim and is CONFIDENTIAL. You may not work 
# with, install, use, duplicate, modify, redistribute or share this
# file, or any part thereof, other than as allowed by any agreement
# specifically made by you with University College London.
# 
# no batch system


cd ~/FS3/results/ocean2D_1_localhost_16/RUNS/Run_13
echo Running...

/usr/bin/env > env.log

python3 /home/derek/FabSim3/plugins/FabUQCampaign/examples/ocean_2D/ocean.py ocean_in.json
