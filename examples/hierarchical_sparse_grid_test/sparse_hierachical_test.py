import matplotlib.pyplot as plt
import os
import easyvvuq as uq
import numpy as np
import chaospy as cp
import fabsim3_cmd_api as fab

def exact_sobols_poly_model():
    """
    Exact Sobol indices for the polynomial test model
    """
    S_i = np.zeros(d)

    for i in range(d):
        S_i[i] = 5**-(i+1)/((6/5)**d - 1)

    return S_i
    
# number of unknown variables
d = 2

# author: Wouter Edeling
__license__ = "LGPL"

HOME = os.path.abspath(os.path.dirname(__file__))

#An EasyVVUQ campaign on sobol_model.py. Takes the polynomial order as input
def run_campaign(poly_order, work_dir = '/tmp'):
    # Set up a fresh campaign called "sc"
    my_campaign = uq.Campaign(name='sc', work_dir=work_dir)
    
    # Define parameter space
    params = {
        "x1": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5},
        "x2": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5},
        "x3": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5},
        "x4": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5},
        "x5": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5},
        "x6": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5},
        "out_file": {
            "type": "string",
            "default": "output.csv"}}
    
    output_filename = params["out_file"]["default"]
    output_columns = ["f"]
    
    # Create an encoder, decoder and collation element
    encoder = uq.encoders.GenericEncoder(
        template_fname=HOME + '/sc/sobol.template',
        delimiter='$',
        target_filename='sobol_in.json')
    decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                    output_columns=output_columns,
                                    header=0)
    collater = uq.collate.AggregateSamples(average=False)
    
    # Add the SC app (automatically set as current app)
    my_campaign.add_app(name="sc",
                        params=params,
                        encoder=encoder,
                        decoder=decoder,
                        collater=collater)
    
    # Create the sampler
    vary = {
        "x1": cp.Uniform(0.0, 1.0),
        "x2": cp.Uniform(0.0, 1.0)}
        # "x3": cp.Uniform(0.0, 1.0),
        # "x4": cp.Uniform(0.0, 1.0),
        # "x5": cp.Uniform(0.0, 1.0)}
    
    #To use 'next_level_sparse_grid' below, we must select a nested 
    #sparse grid here
    my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=poly_order,
                                       quadrature_rule="C", sparse=True,
                                       growth=True)
    
    # Associate the sampler with the campaign
    my_campaign.set_sampler(my_sampler)
    
    print('Number of samples:', my_sampler._number_of_samples)
    
    # Will draw all (of the finite set of samples)
    my_campaign.draw_samples()
    my_campaign.populate_runs_dir()
    
    # Use this instead to run the samples using EasyVVUQ on the localhost
    # my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
    #     "sc/sobol_model.py sobol_in.json"))
    
    #Run execution using Fabsim 
    fab.run_uq_ensemble(my_campaign.campaign_dir, 'hier_sparse_test', 
                        machine='localhost')
    fab.get_uq_samples(my_campaign.campaign_dir, machine='localhost')
    # Use this instead to run the samples using EasyVVUQ on the localhost
    # my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
    #     "./sc/sobol_model.py sobol_in.json"))

    my_campaign.collate()
    
    # Post-processing analysis
    analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
    
    my_campaign.apply_analysis(analysis)
    results = my_campaign.get_last_analysis()
 
    #update the sparse grid to the next level
    my_sampler.next_level_sparse_grid()
    #get the number number from the previous grid
    number_of_old_runs = my_sampler.count
    
    #draw the new samples
    my_campaign.draw_samples()
    my_campaign.populate_runs_dir()
    
    #run ensemble with fabsim, this time 'skip' must be set equal
    #to the number of old samples, which prevents a re-execution of the 
    # samples of the previous sparse grid
    fab.run_uq_ensemble(my_campaign.campaign_dir, 'hier_sparse_test', 
                        machine='localhost', skip=number_of_old_runs)
    fab.get_uq_samples(my_campaign.campaign_dir, machine='localhost') 
    # my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
    #     "./sc/sobol_model.py sobol_in.json"))
    
    my_campaign.collate()
    my_campaign.apply_analysis(analysis)
    results = my_campaign.get_last_analysis()
    
    #check the computed Sobol indices against the analytical result
    for i in range(ref_sobols.size):
        print('Exact Sobol indices order %d = %.4f' %(i+1, ref_sobols[i]))
    print('Computed Sobol indices', results['sobols']['f'])
    
    return my_campaign, my_sampler, results

if __name__ == '__main__':

    #analytic 1st order Sobol indices
    ref_sobols = exact_sobols_poly_model()

    my_campaign, my_sampler, results = run_campaign(poly_order = 2)