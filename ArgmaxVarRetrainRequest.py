import sys
import pandas as pd
import os
import subprocess
from datetime import datetime
sys.path.append(r"C:\Users\theco\OneDrive\Desktop\YangLab\Enviro")
from ParsingUtils import *
# Read the data from the previously modified file and stack OR just read data to make prediction
if len(sys.argv) == 4:
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

df = pd.read_excel(processed_data_path)
print("Loaded...")

# Use all data for training (no test split)
X = df[['temperature', 'time', 'wavelength', '689Rc']]
y = df['R'].to_numpy()

# Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future inverse transformation or application
joblib.dump(scaler, 'scaler_R'+calib_filename_noextdot+'.pkl')

# Convert scaled data and target to PyTorch tensors
X_train_R = torch.tensor(X_scaled, dtype=torch.float32)
y_train_R = torch.tensor(y, dtype=torch.float32).reshape(-1)

# Save the tensors
torch.save(X_train_R, 'X_train_R'+calib_filename_noextdot+'.pt')
torch.save(y_train_R, 'y_train_R'+calib_filename_noextdot+'.pt')


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


# Initialize likelihood and model for R
likelihood_R = GaussianLikelihood()
model_R = GPModel(X_train_R, y_train_R, likelihood_R)

# Set model and likelihood to training mode
model_R.train()
likelihood_R.train()

# Use the Adam optimizer
optimizer_R = torch.optim.Adam(model_R.parameters(), lr=0.1)
mll_R = ExactMarginalLogLikelihood(likelihood_R, model_R)

# Training loop
print("R Optimization")
prev_loss = float('inf')
loss_change_threshold = 0  # Define a small threshold for early stopping
for i in range(50):
    optimizer_R.zero_grad()
    output_R = model_R(X_train_R)
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

import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import product
import joblib
from tqdm import tqdm

# Define the parameters
temperatures = np.arange(840, 911, 1)  # Temperature range
wavelengths = [443, 514, 689, 781, 817]  # Wavelengths
times = np.arange(1, 5001, 250)  # Time range
calibrations = pull_calib(calib_filename_noext+calib_extension)
print(calibrations)
fixed_689Rc = calibrations['689Rc']
print("689Rc Extracted from " + calib_filename_noextdot + " with value of " + str(np.round(fixed_689Rc,3)))

# Load the model and scaler
model_R.eval()
# scaler = joblib.load("scaler.pkl")

# Create a single large batch for all combinations
all_combinations = list(product(temperatures, times, wavelengths, [fixed_689Rc]))
batch_inputs = np.array(all_combinations)

# Standardize the input data
batch_inputs_standardized = scaler.transform(batch_inputs)
batch_inputs_standardized = torch.from_numpy(batch_inputs_standardized).float()

# Define batch size for progress tracking
batch_size = 1000  # Adjust this based on your system's capability
num_batches = len(batch_inputs_standardized) // batch_size + (0 if len(batch_inputs_standardized) % batch_size == 0 else 1)

# Initialize the array for storing variances
variances = np.zeros((len(temperatures), len(times), len(wavelengths)))

# Process in batches with a progress bar
with torch.no_grad():
    for i in tqdm(range(num_batches), desc="Processing Batches"):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(batch_inputs_standardized))
        batch_predictions = model_R(batch_inputs_standardized[batch_start:batch_end])
        # Extract variances
        extracted_variances = batch_predictions.variance.numpy()
        for j, idx in enumerate(range(batch_start, batch_end)):
            idx_unraveled = np.unravel_index(idx, (len(temperatures), len(times), len(wavelengths)))
            variances[idx_unraveled] += extracted_variances[j]

# Sum the variances for each temperature
sum_of_variances = variances.sum(axis=(1, 2))  # Sum over times and wavelengths

# Plot Sum of Variances vs Temperature
plt.figure(figsize=(10, 6))
plt.plot(temperatures, sum_of_variances, label='Sum of Variances')
plt.xlabel('Temperature')
plt.ylabel('Sum of Variances')
plt.title('Sum of Variances vs Temperature')
plt.legend()
plt.grid(True)
plt.savefig("SumOfVariances_"+calib_filename_noextdot+".png")
plt.clf()
# plt.show()


# Save the DataFrame to an Excel file
import pandas as pd
from datetime import datetime
import os
import csv
# Assuming sum_of_variances and temperatures are defined as before
# Find the temperature with the highest sum of variance
max_variance_index = np.argmax(sum_of_variances)
optimal_temperature = temperatures[max_variance_index]

output_log_name = sys.argv[-1]
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