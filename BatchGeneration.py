import pandas as pd
import os
import sys
sys.path.append(r"C:\Users\yuanlongzheng\Desktop\CloudMBE\ExpectedImprovementTesting")
from ParsingUtils import *
csv_file = sys.argv[1]
Stepper(csv_file)
file_raw,file_ext = os.path.splitext(csv_file)
df = pd.read_csv(csv_file)
unique_values = df['substrate'].unique()
for value in unique_values:
    subset_df = df[df['substrate'] == value]
    last_row_temperature = subset_df['T_cell'].iloc[-1]
    splitname = f"{file_raw}_Full_T{last_row_temperature}.csv"
    subset_df.to_csv(splitname, index=False)
    df_calib = subset_df[subset_df['step'] < 2]
    df_calib.to_csv(f"{file_raw}_Calib_T{last_row_temperature}.csv",index=False)

