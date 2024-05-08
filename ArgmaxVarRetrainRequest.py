import sys
import pandas as pd
import os
import subprocess
from datetime import datetime
sys.path.append(r"C:\Users\yuanlongzheng\Desktop\CloudMBE\ExpectedImprovementTesting")
from ParsingUtils import *
# Read the data from the previously modified file and stack OR just read data to make prediction
if len(sys.argv) == 4:
    processed_data_path = sys.argv[1]
    calib_filename_noext, calib_extension = os.path.splitext(sys.argv[2])
    calib_filename_noextdot = calib_filename_noext[2:] # TODO maybe?
elif len(sys.argv) == 5:
    processed_data_path = sys.argv[1]
    calib_filename_noext, calib_extension = os.path.splitext(sys.argv[2])
    calib_filename_noextdot = calib_filename_noext[2:]
    # Stepper(processed_data_path)
elif len(sys.argv) == 2:
    # argv = [0th is something idk, Current excel name (if successively stacked already), output file name]
    old_filename_noext, extension = os.path.splitext(sys.argv[1])
    ProcessData(old_filename_noext+extension,old_filename_noext+"Processed.xlsx")
    df3 = pd.read_excel(old_filename_noext+"Processed.xlsx")
    processed_data_path = old_filename_noext+".xlsx"
    df3.to_excel(processed_data_path, index=False)
    exit()
else:
    print("ERROR: WRONG NUMBER OF ARGV INPUTS")
    print(f"ERROR: WRONG NUMBER OF ARGV INPUTS. Received {len(sys.argv)}, expected 4 or 5.")
    print(sys.argv)  # Print the received arguments for debugging

    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import PairwiseKernel
import gpytorch
import torch
import joblib
import os
import gpytorch.constraints as constraints
from gpytorch.models import ExactGP
from gpytorch.means import LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
# Regression
if len(sys.argv) == 4:
    df = pd.read_excel(processed_data_path)
    print("Loaded...")

    # Use all data for training (no test split)
    fkey = '689Rc'
    X = df[['temperature', 'time', 'wavelength', fkey]]
    y = df['R'].to_numpy()
    yA = df['A'].to_numpy()

    # Data standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.to_numpy())

    # Save the scaler for future inverse transformation or application
    joblib.dump(scaler, 'scaler.pkl')

    # Convert scaled data and target to PyTorch tensors
    X_train = torch.tensor(X_scaled, dtype=torch.float32)
    y_train_R = torch.tensor(y, dtype=torch.float32).reshape(-1)
    y_train_A = torch.tensor(yA,dtype=torch.float32).reshape(-1)

    # Save the tensors
    torch.save(X_train, 'X_train'+ calib_filename_noextdot +'.pt')
    torch.save(y_train_R, 'y_train_R'+ calib_filename_noextdot +'.pt')
    torch.save(y_train_A, 'y_train_A'+ calib_filename_noextdot +'.pt')

    # Define the GP Model
    class GPModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = LinearMean(input_size=train_x.size(-1))
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.size(-1)))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    # Initialize likelihood and model for R & A
    likelihood_R = GaussianLikelihood()
    model_R = GPModel(X_train, y_train_R, likelihood_R)
    likelihood_A = GaussianLikelihood()
    model_A = GPModel(X_train, y_train_A, likelihood_A)

    # Set model and likelihood to training mode
    model_R.train()
    likelihood_R.train()
    model_A.train()
    likelihood_A.train()

    # Use the Adam optimizer
    optimizer_R = torch.optim.Adam(model_R.parameters(), lr=0.1)
    mll_R = ExactMarginalLogLikelihood(likelihood_R, model_R)

    # Training loop
    print("R Optimization")
    prev_loss = float('inf')
    loss_change_threshold = 0  # Define a small threshold for early stopping
    for i in range(35):
        optimizer_R.zero_grad()
        output_R = model_R(X_train)
        loss_R = -mll_R(output_R, y_train_R)
        loss_R.backward()
        optimizer_R.step()

        # Replace the problematic print line with this:
        print(
            f"Iteration: {i}, Loss: {loss_R.item()}, Lengthscales: {model_R.covar_module.base_kernel.lengthscale.detach().numpy()}")

        if abs(prev_loss - loss_R.item()) < loss_change_threshold:
            print("Early stopping triggered")
            break
        prev_loss = loss_R.item()

    # Save the model state
    torch.save(model_R.state_dict(), 'model_R'+calib_filename_noextdot+'.pth')
    torch.save(model_R.state_dict(), 'model_R.pth')

    optimizer_A = torch.optim.Adam(model_A.parameters(), lr=0.1)
    mll_A = ExactMarginalLogLikelihood(likelihood_A, model_A)
    print("A Optimization")
    prev_loss = float('inf')
    loss_change_threshold = 0  # Define a small threshold for early stopping
    for i in range(35):
        optimizer_A.zero_grad()
        output_A = model_A(X_train)
        loss_A = -mll_A(output_A, y_train_A)
        loss_A.backward()
        optimizer_A.step()

        # Replace the problematic print line with this:
        print(
            f"Iteration: {i}, Loss: {loss_A.item()}, Lengthscales: {model_A.covar_module.base_kernel.lengthscale.detach().numpy()}")

        if abs(prev_loss - loss_A.item()) < loss_change_threshold:
            print("Early stopping triggered")
            break
        prev_loss = loss_A.item()

    # Save the model state
    torch.save(model_A.state_dict(), 'model_A'+calib_filename_noextdot+'.pth')
    torch.save(model_A.state_dict(), 'model_A.pth')


if len(sys.argv) == 5:
    
    scaler = joblib.load('scaler.pkl')
    df = pd.read_excel(processed_data_path)

    df_filtered = df
    fkey = '689Rc'
    X = df_filtered[['temperature', 'time', 'wavelength', fkey]]
    y = df_filtered['R'].to_numpy()
    yA = df_filtered['A'].to_numpy()
    # Data standardization
    X_scaled = scaler.transform(X)
    # Convert scaled data and target to PyTorch tensors
    X_train = torch.tensor(X_scaled, dtype=torch.float32)
    y_train_R = torch.tensor(y, dtype=torch.float32).reshape(-1)
    y_train_A = torch.tensor(yA,dtype=torch.float32).reshape(-1)
    # Save the tensors
    torch.save(X_train, 'X_train'+calib_filename_noextdot+'.pt')
    torch.save(y_train_R, 'y_train_R'+calib_filename_noextdot+'.pt')
    torch.save(y_train_A, 'y_train_A' + calib_filename_noextdot + '.pt')
    
    class GPModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = LinearMean(input_size=train_x.size(-1))
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.size(-1)))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    # Initialize likelihood and model for R
    likelihood_R = GaussianLikelihood()
    likelihood_R.train()
    model_R = GPModel(X_train, y_train_R, likelihood_R)
    model_R.load_state_dict(torch.load('model_R.pth'))
    model_R.eval()
    with torch.no_grad():
        model_R(X_train)  # Populate cache for fantasy update
    # model_R = model_R.get_fantasy_model(X_train, y_train_R)
    torch.save(model_R.state_dict(), 'model_R'+calib_filename_noextdot+'.pth')
    torch.save(model_R.state_dict(), 'model_R.pth')

    likelihood_A = GaussianLikelihood()
    likelihood_A.train()
    model_A = GPModel(X_train, y_train_A, likelihood_A)
    model_A.load_state_dict(torch.load('model_R.pth'))
    model_A.eval()
    with torch.no_grad():
        model_A(X_train)  # Populate cache for fantasy update
    # model_R = model_R.get_fantasy_model(X_train, y_train_R)
    torch.save(model_A.state_dict(), 'model_A'+calib_filename_noextdot+'.pth')
    torch.save(model_A.state_dict(), 'model_A.pth')

import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import product
import joblib
from tqdm import tqdm

# Define the parameters
temperatures = np.arange(820, 901, 1)  # Temperature range
wavelengths = [443, 514, 689, 781, 817]  # Wavelengths
#times = np.arange(1, 5001, 250)  # Time range
print(calib_filename_noext+calib_extension)
calibrations = pull_calib(calib_filename_noext+calib_extension)
print(calibrations)
fixed = calibrations[fkey]
print(fkey + "Extracted from " + calib_filename_noextdot + " with value of " + str(np.round(fixed,3)))
# Load the model and scaler
model_R.eval()
model_A.eval()
scaler = joblib.load("scaler.pkl")
all_combinations = []
for temperature in temperatures:
    #max_time = 5000 + 100 * (910 - temperature)
    max_time = np.exp((-temperature-298)*0.020118)*1.75884*(10**14)
    times = np.arange(1, max_time + 1, 250)  # Adjust times dynamically
    for time in times:
        for wavelength in wavelengths:
            all_combinations.append((temperature, time, wavelength, fixed))
batch_inputs = np.array(all_combinations)
    # Standardize input data
batch_inputs_standardized = scaler.transform(batch_inputs)
batch_inputs_standardized = torch.from_numpy(batch_inputs_standardized).float()
    # Process in batches
batch_size = 1000  # Adjust based on your system's capabilities
num_batches = len(batch_inputs_standardized) // batch_size + (0 if len(batch_inputs_standardized) % batch_size == 0 else 1)
variances_R = np.zeros(len(all_combinations))  # Linearized variances_R storage
variances_A = np.zeros(len(all_combinations))
with torch.no_grad():
    for i in tqdm(range(num_batches), desc="Processing Batches"):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(batch_inputs_standardized))
        batch_predictions_R = model_R(batch_inputs_standardized[batch_start:batch_end])
        extracted_variance_R = batch_predictions_R.variance.numpy()
        variances_R[batch_start:batch_end] = extracted_variance_R
        batch_predictions_A = model_A(batch_inputs_standardized[batch_start:batch_end])
        extracted_variance_A = batch_predictions_A.variance.numpy()
        variances_A[batch_start:batch_end] = extracted_variance_A
    # Sum variances_R by temperature
avg_of_variances_R = np.zeros(len(temperatures))
avg_of_variances_A = np.zeros(len(temperatures))

for i, temperature in enumerate(temperatures):
    indices = [idx for idx, comb in enumerate(all_combinations) if comb[0] == temperature]
    avg_of_variances_R[i] = np.mean(variances_R[indices])
    avg_of_variances_A[i] = np.mean(variances_A[indices])
# Plot Sum of variances_R vs Temperature
plt.figure(figsize=(10, 6))
plt.plot(temperatures, avg_of_variances_R, label='Avg of variances (R)')
plt.xlabel('Temperature')
plt.ylabel('Avg of Variances (R)')
plt.title('Avg of R Variances vs Temperature')
plt.legend()
plt.grid(True)
plt.savefig("AvgOfvariances_R_"+calib_filename_noextdot+".png")
plt.clf()
plt.plot(temperatures, avg_of_variances_A, label='Avg of variances (A)')
plt.xlabel('Temperature')
plt.ylabel('Avg of Variances (A)')
plt.title('Avg of A Variances vs Temperature')
plt.legend()
plt.grid(True)
plt.savefig("AvgOfvariances_A_"+calib_filename_noextdot+".png")
plt.clf()
# plt.show()


# Save the DataFrame to an Excel file
import pandas as pd
from datetime import datetime
import os
import csv
# Assuming sum_of_variances_R and temperatures are defined as before
# Find the temperature with the highest sum of variance
max_variance_index = np.argmax(avg_of_variances_R+avg_of_variances_A)
optimal_temperature = temperatures[max_variance_index]

output_log_name = sys.argv[3]
def write_to_csv(value):
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.isfile(output_log_name):
        with open(output_log_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Date and Time','Calibration Name','TempC'])
    with open(output_log_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([current_datetime, calib_filename_noextdot,value])
write_to_csv(optimal_temperature)

print(optimal_temperature)
print(f"Optimal temperature saved to {output_log_name}")
