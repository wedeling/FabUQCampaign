#########################
# SIkR model
# author: Job Heijmans
#########################

from scipy.stats import weibull_min
from scipy.stats import lognorm
import cmath
import math 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.integrate import quad

##############################################################################
# UQ Edit: load parameter values for uncertain coefficients from a json file
##############################################################################

import json, sys
json_input = sys.argv[1]

with open(json_input, "r") as f:
    inputs = json.load(f)
    
#Define the generation time distribution
# genWeibShape = 2.826027
# genWeibScale = 5.665302
# R0=2

R0 = float(inputs['R0'])

#Define the generation time distribution
genWeibShape = float(inputs['genWeibShape'])
genWeibScale = float(inputs['genWeibScale'])

#Percentage of people who have infected and are recovered now.
recovered_perc = float(inputs['recovered_perc'])

#Define the parameters values of the gammas (\gamma_{H,R} ->gHR) 
# #These values come from the science paper.
# gIH = 0.026445154
# #gHD = 0.001859002 #20 days on IC
# #gHR = 0.048140998 

# gHD = 0.003718003 #10 days on IC
# gHR = 0.096281997

gIH = float(inputs['gIH'])
gHD = float(inputs['gHD'])
gHR = float(inputs['gHR'])

#Define incubation time distribution
# incMeanLog = 1.644
# incSdLog = 0.363

incMeanLog = float(inputs['incMeanLog'])
incSdLog = float(inputs['incSdLog'])

output_filename = inputs['outfile']

##############################################################################

#By the Science paper, someone who is infected will remain infectious for 13 days.
K=13

# The number of deceased patients from 10-04 back to 28-03 is multiplied by 100 
# to estimate the number of infected patients from 28-03 back to 17-03
# The number of recovered patients is calculated by summing up the number of deceased
# patients up to 17-03 and multiply by 100 * 0.99
N = 17000000
# K=13
Ikt = [ [9800] , [9000],[12200],[13400],[14500],[14200],[14900],[15100],[15900],[15200],[15800],[14700],[11700]]
R = [ N*recovered_perc ] #Percentage of people who have infected and are recovered now.
H = [1232] #number of tested corona patients on the IC 10-04
D = [0] #We will look at the number of deceased patients starting the count on zero
Itotal = [sum([Ikt[i][-1] for i in range(K)])]
S = [N - Itotal[0] -R[0]-H[0]-D[0]]


#We will repeat the initialisation step every run, so this is just to get an 
#overview of the values.

# #Define the parameters values of the gammas (\gamma_{H,R} ->gHR) 
# #These values come from the science paper.
# gIH = 0.026445154
# #gHD = 0.001859002 #20 days on IC
# #gHR = 0.048140998 

# gHD = 0.003718003 #10 days on IC
# gHR = 0.096281997


#Set efficacy
epsI = 0
epsT = 0

#BetaTau = [(2*weibull_min(genWeibShape, 0, genWeibScale).pdf(x)) for x in range(K)]

# #Define incubation time distribution
# incMeanLog = 1.644
# incSdLog = 0.363
#incubTau = [(lognorm.cdf(x,incSdLog,0,math.exp(incMeanLog))) for x in range(0,K+1)]

def incub(tau): #get probability that incubation time is lower than tau
    return lognorm.cdf(tau,incSdLog,0,math.exp(incMeanLog))

def gener(tau): #probability density of generation time at tau
    return R0*weibull_min(genWeibShape, 0, genWeibScale).pdf(tau)

def f(ht,htp): #Here we mean ht=\hat{\tau} and htp=\hat{\tau}' (p of prime)
    if (htp >= ht): #This just function f as defined above
        return gener(ht)*(1-epsI*incub(ht))*(1-epsT+epsT*(1-incub(htp))/(1-incub(htp-ht))) 
    else:
        return 0
        
def G(tau,taup): #This is funciton g as defined above, taup= \tau'
    if (taup >= tau):
        return dblquad( lambda ht, htp : f(ht,htp), taup, taup+1, lambda x: tau, lambda x: min(x,tau +1))
    else:
        return [0]
    
#It takes a long time to calculate these, so we store them in a matrix such that we only have to calculate once.
g = [[G(tau,taup)[0] for taup in range(K)] for tau in range(K)]
#So g(\tau,\tau') is what we defined above.

#Set the values for S for t=-12,-11,...,0
Sbefore0 = [S[0]]
for k in range(K-1):
    Sbefore0.append(S[-1]+Ikt[k][0])
Sbefore0.reverse() #Now Sbefore0[0] = S[-12] -> Sbefore0[12]=S[0]

#Formula as stated above:
YttauI = [[Ikt[-t][0]*quad(gener, K-tau, K-tau+1)[0] for tau in range(K)] for t in range(1,K+1)]
#I for initialised, so we do not have to do this every time if we run the code below.

YttauI.append([0]*K) #make an empty row for Y*(0,:)
for tau in range(1,K):
    #sumhelp is what is in the sum given above, summ is the sum itself.
    sumhelp = [ g[tau][tauPr]*YttauI[12-tau][tauPr-tau] for tauPr in range(tau,K)]
    summ = sum(sumhelp)
    YttauI[-1][tau] = S[-1]/N*summ
    
Ikt = None; R=None; H=None; D=None; Itotal=None;S=None;Yttau=None; Yttot=None;

T = 210 #number of days we want to look ahead.

N = 17000000
Ikt = [ [9800] , [9000],[12200],[13400],[14500],[14200],[14900],[15100],[15900],[15200],[15800],[14700],[11700]]
R = [ N*recovered_perc]
H = [1232] #number of tested corona patients on the IC 10-04
D = [0]
Itotal = [sum([Ikt[i][-1] for i in range(K)])]
S = [N - Itotal[0] -R[0]-H[0]-D[0]]

#Yttau = [([(Sbefore0[t]/N*BetaTau[tau]) for tau in range(K)]) for t in range(K)]
Yttau = YttauI.copy()
Yttot = [sum(Yttau[-1])] #This is the total number of new infections at t=1
for t in range(0,T):
    
    for k in range(1,K):
        Ikt[k].append(Ikt[k-1][t]) #update every infection class with the previous day
    
    Ikt[0].append(Yttot[-1]) #new infections 
    S.append(S[-1] - Yttot[-1]) #remove the new infections from S
    Itotal.append(sum([Ikt[i][-1] for i in range(K)])) #Update total number of infections
    
    H.append(H[-1]*(1-gHR-gHD) + gIH*Ikt[K-1][t]) #Update hospitalized patients
    R.append(R[-1]+(1-gIH)*Ikt[K-1][t]+gHR*H[t]) #Update recovered patients
    D.append(D[-1]+gHD*H[t]) #Update deceased patients
    
    
    Yttau.append([0]*K) #make an empty row for Y*(t+12,:)
    for tau in range(1,K):
        #sumhelp is what is in the sum given above, summ is the sum itself.
        sumhelp = [ g[tau][tauPr]*Yttau[12+t-tau][tauPr-tau] for tauPr in range(tau,K)]
        summ = sum(sumhelp)
        Yttau[-1][tau] = S[-1]/N*summ
    #We have beta(0)=0, so there are no infections from patients that are infected on the day itself.
    Yttau[-1][0] = 0 #this is already the case.
    Yttot.append(sum(Yttau[-1]))
    
time = range(0,T+1)
# plt.figure() #Plot S, I and R in one plot.
# plt.plot(time, S, 'r-',time, Itotal,'b-',time, R, 'g-')
# plt.show()

# plt.figure() #Plot H and D in one plot
# plt.plot(time, H, 'b-', time, D, 'k-')
# plt.show()

# plt.figure()
# plt.plot(time,Itotal)
# plt.show()

###############################################################################
# UQ edit: write QoIs to output file
###############################################################################

#output csv file    
header = 't,S,I,R,H,D'
np.savetxt(output_filename, np.array([time, S, Itotal, R, H, D]).T, 
           delimiter=", ", comments='',
           header=header)