import os
import easyvvuq as uq
import numpy as np
import chaospy as cp
import fabsim3_cmd_api as fab

# author: Wouter Edeling
__license__ = "LGPL"

HOME = os.path.abspath(os.path.dirname(__file__))

def run_campaign(poly_order, d, quad_rule, growth=False):
    
    vary = {}
    for i in range(d):
        vary["x%d" % (i + 1)] = cp.Uniform(0.0, 1.0)

    # #Dense, full tensor product grid 
    # dense_sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=poly_order,
    #                                     rule=quad_rule, sparse=False,
    #                                     growth=growth)

    #Sparse grid
    sparse_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=poly_order,
                                       quadrature_rule=quad_rule, sparse=True,
                                       growth=growth)

    print('================================')
    print('Number of variables = %d' % d)
    print('poly_order = %d' % poly_order)
    print('1D quadrature rule = %s' % quad_rule)
    # print('Number of samples on dense grid = %d' % dense_sampler._number_of_samples)
    print('Number of samples on sparse grid = %d' % sparse_sampler._number_of_samples)

if __name__ == '__main__':
    
    #poly_order = polynomial order, serves as level of sparse grid too
    #d = number of uncertain parameters
    #quad_rule: which 1D quadrature rule is used to build the d-dimensional grid
    #           (e.g. "G", "clenshaw_curtis", "fejer", "leja", "newton_cotes")
    run_campaign(poly_order = 2, d = 8, quad_rule = "C", growth=True)