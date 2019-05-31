# EasyVVUQ Stochastic Collocation tutorial
This tutorial describes how to create a Stochastic Collocation EasyVVUQ campaign. Creating a Polynomial Chaos campaign is very similar, and we indicate below where the code must be modified if Polynomial Chaos is preferred.

## Explanation of files
+ `tests/test_sc.py` (from EasyVVUQ root directory): the complete example script which is described below in detail.
+ `tests/sc/sc_model.py`: a finite element solver of the advection-diffusion equation with uncertain coefficients.
+ `tests/sc/sc.template`: the EasyVVUQ template of the input file for a single sample of `sc_model.py`.

### Executing an ensemble job on localhost
Thus, `tests/test_sc.py` is a script which runs an EasyVVUQ Stochastic Collocation (SC) campaign for a simple advection-diffusion equation (ade) finite-element solver on the localhost. The governing equations are:

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bdu%7D%7Bdx%7D%20&plus;%20%5Cfrac%7B1%7D%7BPe%7D%5Cfrac%7Bd%5E2u%7D%7Bdx%7D%20%3D%20f),

where the Peclet Number (Pe) and forcing term (f) are the uncertain SC parameters, and u is the velocity subject to Dirichlet boundary conditions u(0)=u(1)=0. The script executes the ensemble, computes the first two moments of the output, generates some random sample of the SC surrogate and computes the Sobol indices of Pe and f.

All steps are described below:

 1. Create an EasyVVUQ campaign object: `my_campaign = uq.Campaign(name='sc', work_dir=tmpdir)`
 2. Define the parameter space of the ade model, comprising of the uncertain parameters Pe and f, plus the name of the output file of `sc_model.py`:
 
```python
    params = {
        "Pe": {
            "type": "real",
            "min": "1.0",
            "max": "2000.0",
            "default": "100.0"},
        "f": {
            "type": "real",
            "min": "0.0",
            "max": "10.0",
            "default": "1.0"},
        "out_file": {
            "type": "str",
            "default": "output.csv"}}
```
2. (continued): the `params` dict corresponds to the template file `tests/sc/sc.template`, which defines the input of a single model run. The content of this file is as follows:
```
{"outfile": "$out_file", "Pe": "$Pe", "f": "$f"}
```
2. (continued): Select which paramaters of `params` are assigned a Chaospy input distribution, and add these paramaters to the `vary` dict, e.g.:

```python
    vary = {
        "Pe": cp.Uniform(100.0, 200.0),
        "f": cp.Normal(1.0, 0.1)
    }
```

3. Create an encoder, decoder and collation element. The encoder links the template file to EasyVVUQ and defines the name of the input file (`sc_in.json`). The ade model `tests/sc/sc_model.py` writes the velocity output (`u`) to a simple `.csv` file, hence we select the `SimpleCSV` decoder, where in this case we have a single output column:
```python
    output_filename = params["out_file"]["default"]
    output_columns = ["u"]
    
    encoder = uq.encoders.GenericEncoder(template_fname='tests/sc/ade.template',
                                         delimiter='$',
                                         target_filename='sc_in.json')
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
 
 5. The following commands ensure that we draw all samples, and creates the ensemble run directories:
 ```python 
     my_campaign.draw_samples()
     my_campaign.populate_runs_dir()
 ```
 
 6. To execute the runs (and collect the results), we can use a sequential approach on the localhost via
 ```python
     my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
        "tests/sc/sc_model.py sc_in.json"))
     my_campaign.collate()
 ```
 6. (continued) Note that this command contains the command line instruction for a single model run, i.e. `tests/sc/sc_model.py sc_in.json`. To allow `sc_model.py` to be executed in this way, a shebang command is placed on the 1st line of `sc_model.py` that links to the python interpreter that we wish to use, e.g. `#!/usr/bin/env python3`, or in the case of a Anaconda interpreter, use `#!/home/yourusername/anaconda3/bin/python`.  

7. Afterwards, post-processing tasks in EasyVVUQ can be undertaken via:
```python
    sc_analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
    my_campaign.apply_analysis(sc_analysis)
    results = my_campaign.get_last_analysis()
```
7. (continued) The `results` dict contains the first 2 statistical moments and Sobol indices for every quantity of interest defined in `output_columns`. If the PCE sampler was used, `SCAnalysis` should be replaced with `PCEAnalysis`.
