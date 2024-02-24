import sys
import subprocess
import pandas as pd
batch_name = sys.argv[1]
valid = [840,845,850,855,860,865,870,875,880]
avail_dict = {key: True for key in valid}

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

data_start = [(875,True),(875,False),(845,True),(845,False),
              (860,True),(860,False),(840,True),(840,False)]

for key in data_start:
    avail_dict[key[0]] = False
print(avail_dict)
PassiveDF = build_from_list("Current.csv",data_start)




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
