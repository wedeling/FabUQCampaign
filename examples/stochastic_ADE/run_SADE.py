"""
Solves u(x)_t + a(x) u' = kappa(x) u''(x), where a(x) and kappa(x) are 2*pi periodic
and randomly sampled using the Karhoenen-Loeve expansion.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stochastic_ADE import Stochastic_ADE

def draw():
    """
    plot while the simulation is running
    """
    plt.clf()
    ax1 = fig.add_subplot(111, xlim=[0, 2 * np.pi], ylim=[0,1], xlabel='x', ylabel='u')
    ax1.plot(solver.x, u)
    plt.pause(0.001)
    plt.tight_layout()

plt.close('all')
fig = plt.figure()
plot = True
store_output = True

# the random inputs for a(x) and kappa(x)
csv_input = sys.argv[1]
inputs = pd.read_csv(csv_input)
z_a = inputs['z_a'].values
z_kappa = inputs['z_kappa'].values

# The number of spatial points
N = 127
# The time step
dt = 0.1

# the mean value of a(x)
mean_a = 0.0
# the scale parameter of the autocorrelation value of a(x)
sigma_a = 10**-3
# the correlation length of the autocorrelation value of a(x)
l_a = 1.0
# the period of the autocorrelation value of a(x)
T_a = 2 * np.pi
# The truncation order of the KL expansion of a(x)
truncation_order_a = z_a.size

# likewise for the diffusion coefficient kappa(x)
mean_kappa = 10**-4
sigma_kappa = 10**-5
l_kappa = 1.0
T_kappa = 2 * np.pi
truncation_order_kappa = z_kappa.size

# the solver object
solver = Stochastic_ADE(N, dt,
                        mean_a, sigma_a, l_a, T_a,
                        mean_kappa, sigma_kappa, l_kappa, T_kappa,
                        truncation_order_a, truncation_order_kappa,
                        z_a = z_a, z_kappa = z_kappa)

# get the Fourier coefficients of the initial condition u(x, 0)
u_hat_n = solver.initial_condition()

# test value of u to check for convergence to a steady state
u_test = np.fft.ifft(u_hat_n).real

# some counters
max_steps = 1000000
n = 0

# the error wrt a steady state
err = 1.0

# when err < tolerance, assume steady state
tolerance = 1e-5

# time loop
while err > tolerance and n < max_steps:

    # evolve the system in time
    u_hat_n = solver.step_rk4(u_hat_n)

    # plot the solution to screen if plot = True
    if plot and np.mod(n, 1000) == 0:
        u = np.fft.ifft(u_hat_n).real
        draw()

    # compute the error between the current u(x, t) and u_test
    if np.mod(n, 100) == 0:
        u = np.fft.ifft(u_hat_n).real
        err = np.linalg.norm(u_test - u)
        print(err)
        u_test = u

        # if this error drops below the tolerance assume a steday state has been reached
        if err < tolerance:
            print("Solution is converged after %d time steps." % n)
    n += 1

# the steady-state solution
u_final = np.fft.ifft(u_hat_n).real

# write the equilibrium value of u and the error to a CSV file
if store_output:
    with open('output.csv', 'w') as fp:
        fp.write('u_equilibrium,error,time_steps\n')
        fp.write('%.4e,%.4e,%d\n' % (np.mean(u_final), err, n))

print(np.mean(u_final))
