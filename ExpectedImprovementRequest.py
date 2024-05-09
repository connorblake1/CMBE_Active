import sys
import pandas as pd
import os
import subprocess
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import PairwiseKernel
import gpytorch
import torch
import joblib
import gpytorch.constraints as constraints
from gpytorch.models import ExactGP
from gpytorch.means import LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import scipy
from scipy.stats import norm
from itertools import product
from tqdm import tqdm
sys.path.append(r"C:\Users\yuanlongzheng\Desktop\CloudMBE\ExpectedImprovementTesting")
from ParsingUtils import *
# PART 1: LOADING DATA AND MODEL
# Read the data from the previously modified file and stack OR just read data to make prediction
processed_data_path = sys.argv[1]
calib_filename_noext, calib_extension = os.path.splitext(sys.argv[2])
target_filename_noext, target_extension = os.path.splitext(sys.argv[3])

calib_filename_noextdot = calib_filename_noext[2:] # TODO maybe?
target_filename_noextdot = target_filename_noext[2:]
scaler = joblib.load('scaler.pkl')
df = pd.read_excel(processed_data_path)

fkey = '689Rc'
X = df[['temperature', 'time', 'wavelength', fkey]]
yR = df['R'].to_numpy()
yA = df['A'].to_numpy()
print("Dataframe loaded.")
# Data standardization
X_scaled = scaler.transform(X)
# Convert scaled data and target to PyTorch tensors
X_train = torch.tensor(X_scaled, dtype=torch.float32)
y_train_R = torch.tensor(yR, dtype=torch.float32).reshape(-1)
y_train_A = torch.tensor(yA, dtype=torch.float32).reshape(-1)
print("Data transformed.")
# Save the tensors
# torch.save(X_train, 'X_train'+calib_filename_noextdot+'.pt')
# torch.save(y_train_R, 'y_train_R'+calib_filename_noextdot+'.pt')
# torch.save(y_train_A, 'y_train_A'+calib_filename_noextdot+'.pt')

class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean(input_size=train_x.size(-1))
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.size(-1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
likelihood_R = GaussianLikelihood()
likelihood_R.train()
model_R = GPModel(X_train, y_train_R, likelihood_R)
model_R.load_state_dict(torch.load('model_R.pth'))
model_R.eval()
with torch.no_grad():
    model_R(X_train)  # Populate cache for fantasy update
# model_R = model_R.get_fantasy_model(X_train, y_train_R)
torch.save(model_R.state_dict(), 'model_R.pth')

likelihood_A = GaussianLikelihood()
likelihood_A.train()
model_A = GPModel(X_train, y_train_A, likelihood_A)
model_A.load_state_dict(torch.load('model_A.pth'))
model_A.eval()
with torch.no_grad():
    model_A(X_train)  # Populate cache for fantasy update
# model_R = model_R.get_fantasy_model(X_train, y_train_R)
torch.save(model_A.state_dict(), 'model_A.pth')
print("Models built.")
# PART 2: LOADING IN TARGET
# Load in Target Info
df_target = pd.read_csv(sys.argv[3])
cwavR = df_target.set_index('wav')['cWavR'].to_dict()
cwavA = df_target.set_index('wav')['cWavA'].to_dict()
twavR = df_target.set_index('wav')['tWavR'].to_dict()
print("Target Loaded.")
def loss(RTi, Ai, R_stdi, A_stdi):  # convex wrt RT, A, RT and A are dicts with key as wavelength (int)
    L = 0
    for key in RTi:
        L += cwavR[key] * ((RTi[key] - twavR[key]) ** 2+ (2 * R_stdi[key])**2)
        # L += cwavA[key] * (Ai[key] ** 2 + (2 * A_stdi[key])**2)
    return L

print(calib_filename_noext+calib_extension)
calibrations = pull_calib(calib_filename_noext+calib_extension)
fixed = calibrations[fkey]
print(calibrations)
print(fkey + " Extracted from " + calib_filename_noextdot + " with value of " + str(np.round(fixed,3)))

wavelengths = [443, 514, 689, 781, 817]

Tmin = 820
Tmax = 900
tmin = 0
tmax = 30000

def loss_caller(x): # x = [time, Temp]
    RT, A, R_std, A_std = full_predictions(x)
    return loss(RT, A, R_std, A_std)

def full_predictions(x):
    ti = x[0]
    Ti = x[1]
    RT = dict()
    A = dict()
    R_std = dict()
    A_std = dict()
    wavcombos = []
    for wav in wavelengths:
        wavcombos.append((Ti,ti,wav,fixed))
    inputs = np.array(wavcombos)
    std_inputs = scaler.transform(inputs)
    std_inputs = torch.from_numpy(std_inputs).float()
    R_preds = model_R(std_inputs)
    A_preds = model_A(std_inputs)
    for i, wav in enumerate(wavelengths):
        RT[wav] = R_preds.mean.detach().numpy()[i]
        A[wav] = A_preds.mean.detach().numpy()[i]
        R_std[wav] = np.sqrt(R_preds.variance.detach().numpy()[i])
        A_std[wav] = np.sqrt(A_preds.variance.detach().numpy()[i])
    return [RT, A, R_std, A_std]

# PART 3: Expected Improvement Algorithm
# https://krasserm.github.io/2018/03/21/bayesian-optimization/
boundsIn = np.array([[tmin,tmax],[Tmin,Tmax]])
# min_val = 100000
# min_x = None
# n_restarts = 25
# xi = 0 # TODO.01
# returns mean and standard deviation of loss predictions
def Lstats(x):
    ti = x[0]
    Ti = x[1]
    wavcombos = []
    for wav in wavelengths:
        wavcombos.append((Ti, ti, wav, fixed))
    inputs = torch.from_numpy(scaler.transform(np.array(wavcombos))).float()
    R_preds = model_R(inputs)
    A_preds = model_A(inputs) # TODO
    Rmu = R_preds.mean.detach().numpy()
    Rsigma2 = R_preds.variance.detach().numpy()
    Amu = A_preds.mean.detach().numpy() #np.array([0]*len(wavelengths))
    Asigma2 = A_preds.variance.detach().numpy() # np.array([0]*len(wavelengths))
    # ask connor for derivation of these two things
    L_mu = 0
    for i,wav in enumerate(wavelengths):
        L_mu += cwavR[wav]*(twavR[wav]**2 - 2*twavR[wav]*Rmu[i] + Rmu[i]**2 + Rsigma2[i]) + cwavA[wav]*(Amu[i]**2 + Asigma2[i])
    L_sigma2 = 0
    for i,wav in enumerate(wavelengths):
        L_sigma2 += 4*(cwavR[wav]**2)*(twavR[wav]**2)*Rsigma2[i]+(cwavR[wav]**2)*(4*(Rmu[i]**2)*Rsigma2[i]) + (cwavA[wav]**2)*(4*(Amu[i]**2)*Asigma2[i])
    L_sigma = np.sqrt(L_sigma2)
    return L_mu,L_sigma

def fastlossmin(): # finds the minimum loss over all current samples
    # Nsamples x 5 x 3 (wav,RT,A) for each wav
    nwav= len(wavelengths)
    data = df[['wavelength', 'R', 'A']].values
    num_blocks = len(df) // nwav
    LambdaRA = data[:num_blocks * nwav].reshape(num_blocks, nwav, 3)
    L = []
    for i in range(len(LambdaRA)): # samples
        fL = 0
        for j in range(len(LambdaRA[0])): # wavelengths
            wav = int(LambdaRA[i,j,0])
            fL += cwavR[wav] * (LambdaRA[i,j,1]-twavR[wav])**2 + cwavA[wav]*LambdaRA[i,j,2]**2
        L.append(fL)
    ind = np.argmin(L)
    Topti,topti = df.at[ind*nwav,'temperature'],df.at[ind*nwav,'time']
    return np.min(L),Topti,topti
# mu_sample_opt,Topt,topt = fastlossmin()
# max_sample = -mu_sample_opt

def expected_improvement(x):
    ti = x[0]
    Ti = x[1]
    penalty = 0 # penalty for going out of bounds in time
    #if ti > 5000 + 100*(910-Ti):
    #    penalty = (ti-(5000+100*(910-Ti)))**2
    if ti > np.exp((-Ti-298)*0.020118)*1.75884*(10**14):
        penalty = (ti-(np.exp((-Ti-298)*0.020118)*1.75884*(10**14)))**2
    Lmu,Lsigma = Lstats(x)
    Lmu = -Lmu
    imp = Lmu-max_sample-xi
    val = 0
    if abs(Lsigma) > 1e-5:
        Z = imp/Lsigma
        val = (imp*norm.cdf(Z) + Lsigma*norm.pdf(Z)) - penalty
    else:
        val = -penalty
    # print("EI:")
    # print("Time: " + str(ti) + " Temperature: " + str(Ti))
    # print("Lmu: " + str(np.round(Lmu,3))+ " Lsigma: " + str(np.round(Lsigma,3)))
    # print("EI: " + str(np.round(val,3)))
    return -val
# Find the best optimum by starting from n_restart different random points.
# for x0 in np.random.uniform(boundsIn[:, 0], boundsIn[:, 1], size=(n_restarts, 2)):
#     # 5000 + 100 * (910 - temperature)
#     res = scipy.optimize.minimize(expected_improvement, x0=x0, bounds=boundsIn, method='L-BFGS-B')
#     if res.fun < min_val:
#         min_val = res.fun
#         min_x = res.x
print("Starting minimization.")
n_restarts = 25 # TODO 25
stupid_min_val = 1000
stupid_min_x = None
minshots = []
meth = "Nelder-Mead"
deltaT = 50
deltat = 2000
fatol = .1
for x0 in tqdm(np.random.uniform(boundsIn[:, 0], boundsIn[:, 1], size=(n_restarts, 2))):
    # 5000 + 100 * (910 - temperature)
    stupidsol = scipy.optimize.minimize(loss_caller, x0=x0, bounds=boundsIn, method=meth,options={'initial_simplex':[[x0[0],x0[1]],[x0[0]+deltat,x0[1]],[x0[0],x0[1]+deltaT]],'fatol':fatol})
    minshots.append((stupidsol.x[0],stupidsol.x[1],stupidsol.fun,x0[0],x0[1]))
    print(len(minshots))
    print("Min Sample Shot " + str(np.round(stupidsol.x[0],4)) + " " + str(np.round(stupidsol.x[1],4)) + ": " + str(np.round(stupidsol.fun,4)) + " retries: " + str(stupidsol.nfev))
    if stupidsol.fun < stupid_min_val:
        stupid_min_val = stupidsol.fun
        stupid_min_x = stupidsol.x
minshots = np.array(minshots)
headers = ['X', 'Y', 'Z',"t0","T0"]
np.savetxt(meth+"_shots_" + target_filename_noext[2:] + "_fatol" + str(np.round(fatol,3)) + "_deltaT" + str(np.round(deltaT,3)) + "_deltat" + str(np.round(deltat,3))+ "_" + str(n_restarts)  + ".csv", minshots, delimiter=',', header=','.join(headers), comments='')

print("Shots: " + str(n_restarts))
print("Mean: " + str(np.round(np.mean(minshots),4)))
print("Stddev: " + str(np.round(np.std(minshots),4)))

print()
# print("Best Known Sample:")
# print("Loss: " + str(np.round(mu_sample_opt,8)))
# print("Temp: " + str(np.round(Topt)) + " Time: " + str(np.round(topt)))
# print()
# print("Expected Improvement")
# print("EI: " + str(np.round(-min_val,8)))
# print("Temp: " + str(np.round(min_x[1])) + " Time: " + str(np.round(min_x[0])))
# print()
print("Pure Loss Min (no consideration of uncertainty)")
print("Loss: " + str(np.round(stupid_min_val,8)))
print("Temp: " + str(np.round(stupid_min_x[1])) + " Time: " + str(np.round(stupid_min_x[0])))
print()
print("Dicts")
for key in cwavR.keys():
    print("Wavelength: " + str(key) + " CwavR: " + str(cwavR[key]) + " CwavA: " + str(cwavA[key]) + " twavR: " + str(twavR[key]))

print("Saving Pure Loss Min...")
def r3(val): # utility to round things
    return np.round(val,3)

# TODO make min_x
# Save the DataFrame to an Excel file
import csv
output_log_name = sys.argv[4]
def write_to_csv(value):
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.isfile(output_log_name):
        with open(output_log_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Date and Time','Calibration Name','TargetName','TempC','TimeS','Loss','RT1','RT2','RT3','RT4','RT5','A1','A2','A3','A4','A5','RTs1','RTs2','RTs3','RTs4','RTs5','As1','As2','As3','As4','As5'])
    with open(output_log_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        RTout, Aout, Rstdout, Astdout = full_predictions(value)
        writer.writerow([current_datetime, calib_filename_noextdot,target_filename_noextdot,
                         np.round(value[1]),np.round(value[0]),r3(loss_caller(value)),
                         r3(RTout[443]),r3(RTout[514]),r3(RTout[689]),r3(RTout[781]),r3(RTout[817]),
                         r3(Aout[443]), r3(Aout[514]), r3(Aout[689]), r3(Aout[781]), r3(Aout[817]),
                         r3(Rstdout[443]), r3(Rstdout[514]), r3(Rstdout[689]), r3(Rstdout[781]), r3(Rstdout[817]),
                         r3(Astdout[443]), r3(Astdout[514]), r3(Astdout[689]), r3(Astdout[781]), r3(Astdout[817])])
write_to_csv(stupid_min_x)
print(f"Optimal growth saved to {output_log_name}")
