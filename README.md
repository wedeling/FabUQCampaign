# FabDummy
This is a dummy example plugin for FabSim3. It is meant to showcase a minimal implementation for a FabSim3 plugin.

## Installation
Simply type `fab localhost install_plugin:FabDummy` anywhere inside your FabSim3 install directory.

## Testing
1. To run a dummy job, type `fab localhost dummy:dummy_test`.
2. To run an ensemble of dummy jobs, type `fab localhost dummy_ensemble`.

## Explanation of files
* FabDummy.py - main file containing the ```fab localhost dummy``` command implementation.
* config_files/dummy_test - directory containing input data for the dummy command.
* templates/dummy - template file for running the dummy command on the remote machine.
