# FabUQCampaign 2D ocean model
This plugin runs the samples from an ![EasyVVUQ](https://github.com/UCL-CCS/EasyVVUQ) campaign using ![FabSim3](https://github.com/djgroen/FabSim3) via the `campaign2ensemble` subroutine.

## Installation
Simply type `fab localhost install_plugin:FabUQCampaign` anywhere inside your FabSim3 install directory.

## Explanation of files
+ `FabUQCampaign/FabUQCampaign.py`: contains the `run_UQ_sample` subroutine in which the job properties are specified, e.g. number of cores, memory, wall-time limit etc
+ `FabUQCampaign/templates/run_UQ_sample`: contains the command-line execution command for a single EasyVVUQ sample.
+ `FabUQCampaign/examples/ocean_2D/`: an example script, see below.

## Detailed Examples

### Executing an ensemble job on localhost
In the examples folder there is a script which runs an EasyVVUQ Stochastic Collocation (SC) campaign using FabSim3 for a 2D ocean model on a square domain with periodic boundary conditions. Essentially, the governing equations are the Navier-Stokes equations written in terms of the vorticity ![equation](https://latex.codecogs.com/gif.latex?%5Comega) and stream function ![equation](https://latex.codecogs.com/gif.latex?%5CPsi), plus an additional forcing term F:

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%5Comega%7D%7B%5Cpartial%20t%7D%20&plus;%20%5Cfrac%7B%5Cpartial%5CPsi%7D%7B%5Cpartial%20x%7D%5Cfrac%7B%5Cpartial%5Comega%7D%7B%5Cpartial%20y%7D%20-%20%5Cfrac%7B%5Cpartial%5CPsi%7D%7B%5Cpartial%20y%7D%5Cfrac%7B%5Cpartial%5Comega%7D%7B%5Cpartial%20x%7D%20%3D%20%7B%5Ccolor%7BRed%7D%20%5Cnu%7D%5Cnabla%5E2%5Comega%20&plus;%20%7B%5Ccolor%7BRed%7D%5Cmu%7D%5Cleft%28F-%5Comega%5Cright%29)

![equation](https://latex.codecogs.com/gif.latex?%5Cnabla%5E2%5CPsi%20%3D%20%5Comega)

The viscosities ![equation](https://latex.codecogs.com/gif.latex?%5Cnu) and ![equation](https://latex.codecogs.com/gif.latex?%5Cmu) are the uncertain parameters. Their values are computed in `ocean.py` by specifying a decay time. For ![equation](https://latex.codecogs.com/gif.latex?%5Cnu) we specify a uniformly distributed decay time between 1 and 5 days, and for ![equation](https://latex.codecogs.com/gif.latex?%5Cmu) between 85 and 90 days. The ocean model just runs for a simulation time of 1 day to limit the runtime of a single sample.


The first steps are the same as for an EasyVVUQ campaign that does not use FabSim to execute the runs:

 1. Create an EasyVVUQ campaign object: `my_campaign = uq.Campaign(name='sc', work_dir=tmpdir)`
 2. Define the parameter space of the ade model, comprising of the uncertain parameters Pe and f, plus the name of the output file of `ade_model.py`:
 
```python
    # Define parameter space
    params = {
        "decay_time_nu": {
            "type": "real",
            "min": "0.0",
            "max": "1000.0",
            "default": "5.0"},
        "decay_time_mu": {
            "type": "real",
            "min": "0.0",
            "max": "1000.0",
            "default": "90.0"},
        "out_file": {
            "type": "str",
            "default": "output.csv"}}
```
2. (continued): the `params` dict corresponds to the template file `examples/ocean_2D/sc/ocean.template`, which defines the input of a single model run. The content of this file is as follows:
```
{"outfile": "$out_file", "decay_time_nu": "$decay_time_nu", "decay_time_mu": "$decay_time_mu"}
```
2. (continued): Select which paramaters of `params` are assigned a Chaospy input distribution, and add these paramaters to the `vary` dict, e.g.:

```python
    vary = {
        "decay_time_nu": cp.Normal(5.0, 0.1),
        "decay_time_mu": cp.Normal(90.0, 1.0)
    }
```

3. Create an encoder, decoder and collation element. The encoder links the template file to EasyVVUQ and defines the name of the input file (`ocean_in.json`). The ade model `examples/ocean_2D/sc/ocean.py` writes the total energy (`E`) to a simple `.csv` file, hence we select the `SimpleCSV` decoder, where in this case we have a single output column:
```python
    output_filename = params["out_file"]["default"]
    output_columns = ["E"]
    
    encoder = uq.encoders.GenericEncoder(template_fname='./sc/ocean.template',
                                         delimiter='$',
                                         target_filename='ocean_in.json')
    decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                    output_columns=output_columns,
                                    header=0)
    collation = uq.collate.AggregateSamples()
```
 
 4. Now we have to select a sampler, in this case we use the Stochastic Collocation (SC) sampler:
 ```python
     my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=3)
     my_campaign.set_sampler(my_sampler)
 ```
 
 4. (continued) If left unspecified, the polynomial order of the SC expansion will be set to 4. If instead we wish te use a Polynomial Chaos Expansion (PCE) sampler, simply replace `SCSampler` with `PCESampler`.
 
 5. The following commands ensure that we draw all samples, and create the ensemble run directories which will be used in FabSim's `campaign2ensemble` subroutine:
 ```python 
     my_campaign.draw_samples()
     my_campaign.populate_runs_dir()
 ```
 
 6. To execute the runs (and collect the results), we can use a sequential approach on the localhost via
 ```python
     my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
        "./sc/ocean.py ocean_in.json"))
     my_campaign.collate()
 ```
 6. (continued) Note that this command contains the command line instruction for a single model run, i.e. `./sc/ocean.py ocean_in.json`. To allow `ocean.py` to be executed in this way, a shebang command is placed on the 1st line of `ocean.py` that links to the python interpreter that we wish to use, e.g. `#!/usr/bin/env python3`, or in the case of a Anaconda interpreter, use `#!/home/yourusername/anaconda3/bin/python`. Instead of EasyVVUQ's `ExecuteLocal`, we can also use FabSim to run the ensemble.

Only the fifth step is specific to FabSim. For now, several variables need to be hardcoded, i.e.: 
 + A simulation identifier (`$sim_ID`)
 + Your FabSim home directory (`$fab_home`)
 + The `FabUQCampaign/template/run_UQ_sample` file contains the command line instruction to run a single sample, in this case: `python3 $ocean_exec ocean_in.json`. Here, `ocean_in.json` is just the input file with the parameter values generated by EasyVVUQ. Furthermore, `$ocean_exec` is the full path to the Python script which runs the ocean model `ocean.py` at the parameters of `ocean_in.json`. It is defined in `deploy/machines_user.yml`, which in this case looks like
 
`localhost:`

 &nbsp;&nbsp;&nbsp;&nbsp;`ocean_exec: "$fab_home/plugins/FabUQCampaign/examples/ocean_2D/ocean.py"`
 
 The following two commands execute the ensemble run:
 
 1. `cd $fab_home && fab localhost campaign2ensemble:$sim_ID, campaign_dir=$campaign_dir`
 2. `cd $fab_home && fab localhost uq_ensemble:$sim_ID`
 
The run directory `$campaign_dir` is available from the EasyVVUQ object. The `campaign2ensemble` results directory (located in `~/FabSim3/results`) has (by design) the same structure as the EasyVVUQ run directory, so the results can simply be copied back, in this case via

`cp -r ~/FabSim3/results/$sim_ID_localhost_16/RUNS/Run_* $campaign_dir/runs`

7. Afterwards, post-processing tasks in EasyVVUQ can be undertaken via:
```python
    sc_analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
    my_campaign.apply_analysis(sc_analysis)
    results = my_campaign.get_last_analysis()
```
7. (continued) The `results` dict contains the first 2 statistical moments and Sobol indices for every quantity of interest defined in `output_columns`. If the PCE sampler was used, `SCAnalysis` should be replaced with `PCEAnalysis`.

### Executing an ensemble job on a remote host

To run the example script on a remote host, every instance of `localhost` must replaced by the `machine_name` of the remote host. Ensure the host is defined in `machines.yml`, and the user login information and `$ocean_exec` in `deploy/machines_user.yml`.

