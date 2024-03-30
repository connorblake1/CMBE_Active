import sys
import subprocess
import pandas as pd
batch_name = sys.argv[1]

def getfname(valt):
    val, iscal = valt
    if iscal:
        return f"{batch_name}_Calib_T{val}.csv"
    else:
        return f"{batch_name}_Full_T{val}.csv"
def build_from_list(fname,datalist):
    result_df = pd.read_csv(getfname(datalist[0]))
    for i in range(1,len(datalist)):
        df2 = pd.read_csv(getfname(datalist[i]))
        result_df = pd.concat([result_df, df2],ignore_index=True)
        result_df.to_csv(fname,index=False)
    return result_df

def stack(fname,basedf,value):
    df2 = pd.read_csv(getfname(value))
    result_df = pd.concat([basedf,df2],ignore_index=True)
    result_df.to_csv(fname,index=False)
data_start = [(830,False),(840,False),(845,False),(850,False),(855,False),(860,False),(865,False),(870,False),(875,False),(880,False),(885,False),(890,False),(895,False),(900,False),(910,False)]
data_start = [(830,False),(840,False),(850,False),(860,False),(870,False),(880,False),(890,False),(900,False),(910,False)]
#data_start = [(830,False),(860,False),(890,False),(900,False),(910,False)]
PassiveDF = build_from_list("Current.csv",data_start)


# test commit

#
# command = "python RetrainRequest.py "
# result = subprocess.run(command, shell=True, capture_output=True, text=True)
#
#
#
#
#
#
#
#
#
# df1 = pd.read_excel(old_file_path)
# df2 = pd.read_excel(new_file_path)
# result_df = pd.concat([df1, df2], ignore_index=True)
# result_df.to_excel(old_file_path, index=False)
# file_path = old_file_path
#
# # Open the CSV file for reading
# with open('your_file.csv', 'r') as file:
#     # Initialize variable to store the last non-empty line
#     last_non_empty_line = None
#
#     # Iterate over each line in the file
#     for line in file:
#         # Check if the line is not empty
#         if line.strip():
#             # Store the non-empty line
#             last_non_empty_line = line
#
# # Print the last non-empty line
# print(last_non_empty_line)
