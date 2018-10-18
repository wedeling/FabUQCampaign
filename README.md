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

## Detailed Examples

### Executing a single job on localhost

1. To run a dummy job, type `fab localhost dummy:dummy_test`. This does the following:
  - Copy your job input, which is in `plugins/FabDummy/config_files/dummy_test`, to the remote location specified in the variable `remote_path_template` in `deploy/machines.yml`.
  - Copy the input to the remote results directory.
  - Start the remote job.
2. If your job runs without a scheduler (normally the case when running on localhost), you can simply wait for it to finish, or cancel the job using Ctrl+C.
3. After the job has finished, the terminal becomes available again, and a message is printing indicating where the output data resides remotely.
4. You can fetch the remote data using `fab localhost fetch_results`, and then use it as you see fit! Local results are typically locations in the `results/` subdirectory.


### Executing a single job on a remote host

1. Ensure the host is defined in machines.yml, and the user login information in `deploy/machines_user.yml`.
2. To run a dummy job, type `fab <machine name> dummy:dummy_test`. This does the following:
  - Copy your job input, which is in `plugins/FabDummy/config_files/dummy_test`, to the remote location specified in the variable `remote_path_template` in `deploy/machines.yml` (not it will substitute in machine-specific variables provided in the same file).
  - Copy the input to the remote results directory.
  - Start the remote job.
3. Uour job now likely runs using a scheduler, so your terminal becomes available again. Use `fab <machine> stat` to track its submission status, or `fab <machine> monitor` to poll periodically for the job status.
4. If the stat or monitor commands do not show your job being listed, then your job has finished (or successfully or unsuccessfully).
5. You can then fetch the remote data using `fab localhost fetch_results`, and investigate the output as you see fit. Local results are typically locations in the `results/` subdirectory.


