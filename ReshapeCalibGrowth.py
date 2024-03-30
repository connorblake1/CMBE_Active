import sys
import os
import pandas as pd
sys.path.append(r"C:\Users\yuanlongzheng\Desktop\CloudMBE\ExpectedImprovementTesting")
# drop pre measurments in active
df2 = pd.read_csv(sys.argv[2])
df2 = df2[df2['time'] != -1]
df2.to_csv(sys.argv[2], index=False)
from ParsingUtils import *
excel_csv_stacker(sys.argv[1],sys.argv[2],sys.argv[3])
Stepper(sys.argv[3])
