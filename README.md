# Dependencies
pandas, pytorch, gpytorch, numpy, pickle
# Example Run
## Setup
1. python3 .\BatchGeneration.py "Data_2.07.csv"
    - generates sample datasets by substrate (with both calibration and full sets each)
2. python3 .\MatlabSim.py "Data_2.07"
    - generates m selected samples as specified within MatlabSim.py as a starting point
3. python3 ArgmaxVarRetrainRequest.py .\Current.csv
    - turns raw input from MatlabSim into processed data (.csv -> .xlsx)
## Active Training Cycle
1. python3 ArgmaxVarRetrainRequest.py .\Current.xlsx .\NewCalibData.csv .\outputlog.csv
    - using calibration data generate new model with training and make it predict where the most valuable next point is (argmax uncertainty sum)
    - output prediction is saved to outputlog.csv on the last line
(this is where Matlab actually grows the sample at target temperature)
2. python3 .\AppendToCurrent.py .\Current.xlsx .\FullData.csv
    - give it the full dataset for a given substrate after growth
3. (repeat)
## Active Testing Cycle
1. python3 ExpectedImprovementRequest.py .\Current.xlsx .\NewCalibData.csv .\sampletarget.csv .\outputlog.csv
