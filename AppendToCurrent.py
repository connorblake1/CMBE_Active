import sys
import os
import pandas as pd
sys.path.append(r"C:\Users\theco\OneDrive\Desktop\YangLab\Enviro")
from ParsingUtils import *
# argv = [0th is something idk, old excel name (processed), new data (raw)]
new_filename_noext, extension = os.path.splitext(sys.argv[2])
print("Reading " + new_filename_noext +" ...")
ProcessData(new_filename_noext+extension,new_filename_noext+"Processed.xlsx")
df1 = pd.read_excel(sys.argv[1])
df2 = pd.read_excel(new_filename_noext+"Processed.xlsx")
result_df = pd.concat([df1, df2], ignore_index=True)
result_df.to_excel(sys.argv[1], index=False)