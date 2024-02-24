# Example Run:
## Setup
python .\BatchGeneration.py "Data_2.08.csv"
- generates sample datasets by substrate broken out by temperature (with both calibration and full sets each)
python .\MatlabSim.py "Data_2.07"
- generates m selected samples as specified within MatlabSim.py as a starting point
python ArgmaxVarRetrainRequest.py .\Current.csv
- turns raw input from MatlabSim into processed data (.csv -> .xlsx)
## Active Training Cycle
python ArgmaxVarRetrainRequest.py .\Current.xlsx .\NewCalibData.csv .\outputlog.csv
- using calibration data generate new model with training and make it predict where the most valuable next point is (argmax uncertainty sum)
- output prediction is saved to outputlog.csv on the last line
(this is where Matlab actually grows the sample at target temperature)
python .\AppendToCurrent.py .\Current.xlsx .\FullData.csv
- give it the full dataset for a given substrate after growth
(repeat)
## Active Testing Cycle
WIP
