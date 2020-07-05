# for the Sobol g function, the exact (1st-order)
# Sobol indices are known analytically
import matplotlib.pyplot as plt
import os
import easyvvuq as uq
import numpy as np
import chaospy as cp
import fabsim3_cmd_api as fab
from vvp import ensemble_vvp
import pandas as pd

#The VVP sample_testing_function, just reads the Sobol indices from a file
def load_sobols(dirname, **kwargs):

    df = pd.read_csv(dirname + '/sobols.csv')
    
    return df.to_numpy()[0][1:]

#The VVP agregation_function, compares the sobol indices (as function of 
#the polynomial order) with the reference values
def check_convergence(sobols, **kwargs):
    
    j = 0
    for sb in sobols:
        print('Polynomial order = %d' % kwargs['poly_orders'][j])
        j += 1
        for i in range(len(sb)):
            print('Sobol x' + str(i+1) + ' = %.3f' % sb[i], 
                  ', exact = %.3f' % ref_sobols[i], 
                  ', error = %.3f' % np.linalg.norm(sb[i]-ref_sobols[i]))
        print('=========================================================')

#Compute the analytic sobol indices for the test function of sobol_model.py
def exact_sobols_g_function():
    V_i = np.zeros(d)

    for i in range(d):
        V_i[i] = 1.0 / (3.0 * (1 + a[i])**2)

    V = np.prod(1 + V_i) - 1

    print('----------------------')
    print('Exact 1st-order Sobol indices: ', V_i / V)
    
    return V_i/V

def exact_sobols_poly_model():
    S_i = np.zeros(d)

    for i in range(d):
        S_i[i] = 5**-(i+1)/((6/5)**d - 1)

    return S_i
    

# number of unknown variables
d = 5

# parameters required by Sobol g test function
a = [0.0, 1.0, 2.0, 4.0, 8.0, 16]

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
        "x2": cp.Uniform(0.0, 1.0),
        "x3": cp.Uniform(0.0, 1.0),
        "x4": cp.Uniform(0.0, 1.0),
        "x5": cp.Uniform(0.0, 1.0)}
    
    """
    SPARSE GRID PARAMETERS
    ----------------------
    - sparse = True: use a Smolyak sparse grid
    - growth = True: use an exponential rule for the growth of the number
      of 1D collocation points per level. Used to make e.g. clenshaw-curtis
      quadrature nested.
    """
    
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
    fab.run_uq_ensemble(my_campaign.campaign_dir, 'sobol_test', machine='localhost')
    fab.get_uq_samples(my_campaign.campaign_dir, machine='localhost')
    # Use this instead to run the samples using EasyVVUQ on the localhost
    # my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
    #     "./sc/sobol_model.py sobol_in.json"))

    my_campaign.collate()
    
    # Post-processing analysis
    analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
    
    my_campaign.apply_analysis(analysis)
    
    results = my_campaign.get_last_analysis()
    
    my_campaign.save_state('campaign_state.json')

    #the unique ID of this Campaign
    ID = my_campaign.campaign_dir.split('/')[-1]
   
    #store the sobol indices of each campaign to the same results directory
    results_dir = work_dir + '/sobols/' + ID
    if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)
 
    #store the 1st order sobols indices to a CSV file
    sobols = pd.DataFrame(results['sobols_first']['f'])
    sobols.to_csv(results_dir + '/sobols.csv')
    
    mu, D, S_u = analysis.get_pce_analysis('f')
    print(S_u)

    return my_campaign, my_sampler, results, ID   

if __name__ == '__main__':

    items = []
    
    #analytic 1st order Sobol indices
    ref_sobols = exact_sobols_g_function()

    #perform campaigns, each time refining the polynomial order
    poly_orders = range(4, 5)
    for p in poly_orders:
        my_campaign, my_sampler, results, ID = run_campaign(p)
        items.append(ID)
      
    print('Ref sobols', ref_sobols)
    print(results['sobols']['f'])
      
    #Check the convergence of the SC Sobols indices with polynomial refinement.
    #items (the name of the results directories) must be specified since
    #the order is important in convergence studies.
    #poly_orders is passed as a kwarg for check_convergence.
    ensemble_vvp('/tmp/sobols', load_sobols, check_convergence, items=items,
                  poly_orders=poly_orders)