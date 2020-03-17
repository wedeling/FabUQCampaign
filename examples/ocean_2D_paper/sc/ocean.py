#!/home/wouter/anaconda3/bin/python

"""
*************************
* S U B R O U T I N E S *
*************************
"""

#pseudo-spectral technique to solve for Fourier coefs of Jacobian
def compute_VgradW_hat(w_hat_n, P):
    
    #compute streamfunction
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    
    #compute jacobian in physical space
    u_n = np.fft.irfft2(-ky*psi_hat_n)
    w_x_n = np.fft.irfft2(kx*w_hat_n)

    v_n = np.fft.irfft2(kx*psi_hat_n)
    w_y_n = np.fft.irfft2(ky*w_hat_n)
    
    VgradW_n = u_n*w_x_n + v_n*w_y_n
    
    #return to spectral space
    VgradW_hat_n = np.fft.rfft2(VgradW_n)
    
    VgradW_hat_n *= P
    
    return VgradW_hat_n

#get Fourier coefficient of the vorticity at next (n+1) time step
def get_w_hat_np1(w_hat_n, w_hat_nm1, VgradW_hat_nm1, P, norm_factor, sgs_hat = 0.0):
    
    #compute jacobian
    VgradW_hat_n = compute_VgradW_hat(w_hat_n, P)
    
    #solve for next time step according to AB/BDI2 scheme
    w_hat_np1 = norm_factor*P*(2.0/dt*w_hat_n - 1.0/(2.0*dt)*w_hat_nm1 - \
                               2.0*VgradW_hat_n + VgradW_hat_nm1 + mu*F_hat - sgs_hat)
    
    return w_hat_np1, VgradW_hat_n

def draw():
    plt.subplot(111)
    plt.contourf(x, y, w_np1_HF, 100)
    plt.tight_layout()

#compute spectral filter
def get_P(cutoff):
    
    P = np.ones([N, int(N/2+1)])
    
    for i in range(N):
        for j in range(int(N/2+1)):
            
            if np.abs(kx[i, j]) > cutoff or np.abs(ky[i, j]) > cutoff:
                P[i, j] = 0.0
                
    return P

#compute spectral filter
def get_P_full(cutoff):

    P = np.ones([N, N])

    for i in range(N):
        for j in range(N):

            if np.abs(kx_full[i, j]) > cutoff or np.abs(ky_full[i, j]) > cutoff:
                P[i, j] = 0.0

    return P

#compute the energy and enstrophy at t_n
def compute_E_and_Z(w_hat_n, verbose=True):

    #compute stats using Fourier coefficients - is faster
    #convert rfft2 coefficients to fft2 coefficients
    w_hat_full = np.zeros([N, N]) + 0.0j
    w_hat_full[0:N, 0:int(N/2+1)] = w_hat_n
    w_hat_full[map_I, map_J] = np.conjugate(w_hat_n[I, J])
    w_hat_full *= P_full
    
    #compute Fourier coefficients of stream function
    psi_hat_full = w_hat_full/k_squared_no_zero_full
    psi_hat_full[0,0] = 0.0

    #compute energy and enstrophy (density)
    Z = 0.5*np.sum(w_hat_full*np.conjugate(w_hat_full))/N**4
    E = -0.5*np.sum(psi_hat_full*np.conjugate(w_hat_full))/N**4

    if verbose:
        #print 'Energy = ', E, ', enstrophy = ', Z
        print('Energy = ', E.real, ', enstrophy = ', Z.real)

    return E.real, Z.real

"""
***************************
* M A I N   P R O G R A M *
***************************
"""

import numpy as np
import os, h5py, sys, json
#import matplotlib.pyplot as plt
#from drawnow import drawnow

####################################################################################
# the json input file containing the values of the parameters, and the output file #
####################################################################################

json_input = sys.argv[1]

with open(json_input, "r") as f:
    inputs = json.load(f)
    
decay_time_nu = float(inputs['decay_time_nu'])
decay_time_mu = float(inputs['decay_time_mu'])

output_filename = inputs['outfile']

#decay_time_nu = 5.0
#decay_time_mu = 95.0
#output_filename = 'output.csv'

###############################################################################

#plt.close('all')
#plt.rcParams['image.cmap'] = 'seismic'

HOME = os.path.abspath(os.path.dirname(__file__))

#number of gridpoints in 1D
I = 7
N = 2**I

#2D grid
h = 2*np.pi/N
axis = h*np.arange(1, N+1)
axis = np.linspace(0, 2.0*np.pi, N)
[x , y] = np.meshgrid(axis , axis)

#frequencies
k = np.fft.fftfreq(N)*N

kx = np.zeros([N, int(N/2+1)]) + 0.0j
ky = np.zeros([N, int(N/2+1)]) + 0.0j

for i in range(N):
    for j in range(int(N/2+1)):
        kx[i, j] = 1j*k[j]
        ky[i, j] = 1j*k[i]

k_squared = kx**2 + ky**2
k_squared_no_zero = np.copy(k_squared)
k_squared_no_zero[0,0] = 1.0

kx_full = np.zeros([N, N]) + 0.0j
ky_full = np.zeros([N, N]) + 0.0j

for i in range(N):
    for j in range(N):
        kx_full[i, j] = 1j*k[j]
        ky_full[i, j] = 1j*k[i]

k_squared_full = kx_full**2 + ky_full**2
k_squared_no_zero_full = np.copy(k_squared_full)
k_squared_no_zero_full[0,0] = 1.0

#cutoff in pseudospectral method
Ncutoff = N/3
Ncutoff_LF = 2**(I-1)/3 

#spectral filter
P = get_P(Ncutoff)
P_LF = get_P(Ncutoff_LF)
P_U = P - P_LF

#spectral filter for the full FFT2 (used in compute_E_Z)
P_full = get_P_full(Ncutoff_LF)

#map from the rfft2 coefficient indices to fft2 coefficient indices
#Use: see compute_E_Z subroutine
shift = np.zeros(N).astype('int')
for i in range(1,N):
    shift[i] = np.int(N-i)
I = range(N);J = range(np.int(N/2+1))
map_I, map_J = np.meshgrid(shift[I], shift[J])
I, J = np.meshgrid(I, J)

#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

nu = 1.0/(day*Ncutoff**2*decay_time_nu)
mu = 1.0/(day*decay_time_mu)

#start, end time (in days) + time step
t = 0.0*day
t_end = t + 1*day
#initial time period during which no data is stored
t_burn = 0.0*day
dt = 0.01
n_burn = np.ceil((t_burn-t)/dt).astype('int')
n_steps = np.ceil((t_end-t)/dt).astype('int')

#constant factor that appears in AB/BDI2 time stepping scheme, multiplying the Fourier coefficient w_hat_np1
norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared + mu)

#############
# USER KEYS #
#############

sim_ID = 'run1'
#store the state at the end of the simulation
state_store = False
#restart from a stored state
restart = False
#plot the solution during executaion
plot = False
plot_frame_rate = np.floor(1.0*day/dt).astype('int')
#store data
store = True
store_frame_rate = np.floor(0.25*day/dt).astype('int')
#data lists
E = []; Z = []

#forcing term
F = 2**1.5*np.cos(5*x)*np.cos(5*y);
F_hat = np.fft.rfft2(F);

#restart from a previous stored state (set restart = True, and set t to the end time of previous simulation)
if restart == True:
    
    fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t/day,1)) + '.hdf5'
    
    #create HDF5 file
    h5f = h5py.File(fname, 'r')
    
    for key in h5f.keys():
        print(key)
        vars()[key] = h5f[key][:]
        
    h5f.close()
        
#start from initial condition
else:
    
    #initial condition
    w = np.sin(4.0*x)*np.sin(4.0*y) + 0.4*np.cos(3.0*x)*np.cos(3.0*y) + \
        0.3*np.cos(5.0*x)*np.cos(5.0*y) + 0.02*np.sin(x) + 0.02*np.cos(y)

    #initial Fourier coefficients at time n and n-1
    w_hat_n_HF = P*np.fft.rfft2(w)
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
   
    #initial Fourier coefficients of the jacobian at time n and n-1
    VgradW_hat_n_HF = compute_VgradW_hat(w_hat_n_HF, P)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)
    
    t = 0.0

print('Solving forced dissipative vorticity equations')
print('decay_time_nu = ', decay_time_nu)
print('decay_time_mu = ', decay_time_mu)
print('Grid = ', N, 'x', N)
print('t_begin = ', t/day, 'days')
print('t_end = ', t_end/day, 'days')

#some counters
j = 0; j2 = 0

#time loop
for n in range(n_steps):
    
    #solve for next time step
    w_hat_np1_HF, VgradW_hat_n_HF = get_w_hat_np1(w_hat_n_HF, w_hat_nm1_HF, VgradW_hat_nm1_HF, P, norm_factor)

    #plot solution every plot_frame_rate. Requires drawnow() package
    if j == plot_frame_rate and plot == True:
        j = 0

        w_np1_HF = np.fft.irfft2(w_hat_np1_HF)
        drawnow(draw)

    #store data
    if j2 == store_frame_rate and store == True:

        j2 = 0
        
        if n >= n_burn:
            E_n , Z_n = compute_E_and_Z(w_hat_np1_HF, verbose=False)
            E.append(E_n); Z.append(Z_n)

    #update variables
    t += dt; j += 1; j2 += 1
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
    w_hat_n_HF = np.copy(w_hat_np1_HF)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)
    
    if np.mod(n, np.round(day/dt)) == 0:
        print('n = ', n, 'of', n_steps)
    
#store the state of the system to allow for a simulation restart at t > 0
if state_store == True:
    
    keys = ['w_hat_nm1_HF', 'w_hat_n_HF', 'VgradW_hat_nm1_HF']
    
    if os.path.exists(HOME + '/restart') == False:
        os.makedirs(HOME + '/restart')
    
    #cPickle.dump(state, open(HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.pickle', 'w'))
    
    fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.hdf5'
    
    #create HDF5 file
    h5f = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for key in keys:
        qoi = eval(key)
        h5f.create_dataset(key, data = qoi)
        
    h5f.close()

if store == True:
    #output csv file    
    header = 'E_mean,Z_mean,E_std,Z_std'
    np.savetxt(output_filename, np.array([np.mean(E), np.mean(Z), np.std(E), np.std(Z)]).reshape([1,4]), 
               delimiter=", ", comments='',
               header=header)
#plt.show()
