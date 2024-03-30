def pull_calib(raw_input_name):
    import sys
    import pandas as pd
    import os
    Stepper(raw_input_name)
    data = pd.read_csv(raw_input_name)


    # Extract the temperature value from this row
    last_row_with_step_1 = data[data['step'] == 1].iloc[-1]
    time_calib = last_row_with_step_1['time']

    # Create a new dataframe with the desired structure
    output = pd.DataFrame(columns=['substrate', 'temperature', 'time', 'wav', 'back', 'T', 'R', 'P', 'step'])

    for index, group in data.groupby(['substrate', 'time', 'wavelength']):
        new_row = {
            'substrate': index[0],
            'temperature': group['T_cell'].iloc[0],  # Using T_cell value directly as temperature
            'time': index[1],
            'wav': index[2],
            'back': group['r_background'].iloc[0],
            'T': group[group['measurement (t,r,p)'] == 1]['mW'].values[0],
            'R': group[group['measurement (t,r,p)'] == 2]['mW'].values[0],
            'P': group[group['measurement (t,r,p)'] == 3]['mW'].values[0],
            'step': group['step'].iloc[0]
        }
        # <2/5 Edit CB>
        # OLD
        # output = output.append(new_row, ignore_index=True)
        # NEW
        output = pd.concat([output, pd.DataFrame([new_row])],
                           ignore_index=True)  # THIS ROW WAS CHANGED FROM OLD to fit modern pandas conventions
        # </2/5 Edit CB>

    output['Reflectivity'] = (output['R'] - output['back']) / (output['R'] - output['back'] + output['T'])

    # Extract Powerinitial and Pinitial
    power_initials = output[output['time'] == -1].set_index(['substrate', 'wav'])['R'] + \
                     output[output['time'] == -1].set_index(['substrate', 'wav'])['T']
    p_initials = output[output['time'] == -1].set_index(['substrate', 'wav'])['P']

    # Calculate Power for each row
    output['Power'] = output.apply(
        lambda row: power_initials[row['substrate'], row['wav']] * row['P'] / p_initials[row['substrate'], row['wav']],
        axis=1)

    # Calculate Absorption for each row
    output['Absorption'] = (output['Power'] - output['T'] - output['R'] + output['back']) / output['Power']

    # Generate final output
    final_output_data = []
    unique_sub_time = output[['substrate', 'time']].drop_duplicates().values

    for sub, t in unique_sub_time:
        row_data = {'substrate': sub, 'time': t,
                    'step': output[(output['substrate'] == sub) & (output['time'] == t)]['step'].iloc[0]}
        subset = output[(output['substrate'] == sub) & (output['time'] == t)]

        # Extract the temperature from the subset
        temperature_value = subset['temperature'].values[0] if not subset['temperature'].empty else None
        row_data['temperature'] = temperature_value

        for wav in [443, 514, 689, 781, 817]:
            wav_data = subset[subset['wav'] == wav]
            if not wav_data.empty:
                row_data[f'{wav}R'] = wav_data['Reflectivity'].values[0]
                row_data[f'{wav}A'] = wav_data['Absorption'].values[0]
        final_output_data.append(row_data)

    final_output_df = pd.DataFrame(final_output_data)

    import numpy as np
    import pickle

    # Load the data
    df = final_output_df
    filtered_df = df[(df['step'] == 1) & (df['time'] == time_calib)].tail(1)

    # If the filtered dataframe is not empty, store the required values
    if not filtered_df.empty:
        values = filtered_df[['443R', '443A', '514R', '514A', '689R', '689A', '781R', '781A', '817R', '817A']].iloc[0]
        # Storing values in a dictionary for later recall
        stored_values = {f"{col}c": values[col] for col in values.index}
    else:
        stored_values = None
    return stored_values


def excel_csv_stacker(in1, in2, out):
    import pandas as pd
    import os
    # Determine file type
    file_ext1 = os.path.splitext(in1)[1]
    file_ext2 = os.path.splitext(in2)[1]

    # Read the data
    if file_ext1 == '.csv':
        df1 = pd.read_csv(in1)
    elif file_ext1 in ['.xls', '.xlsx']:
        df1 = pd.read_excel(in1)
    else:
        raise ValueError("Unsupported file format for input 1")

    if file_ext2 == '.csv':
        df2 = pd.read_csv(in2)
    elif file_ext2 in ['.xls', '.xlsx']:
        df2 = pd.read_excel(in2)
    else:
        raise ValueError("Unsupported file format for input 2")
    stacked_df = pd.concat([df1, df2], ignore_index=True)
    stacked_df.to_csv(out, index=False)


def ProcessData(raw_input_name, processed_output_name):
    import sys
    import pandas as pd
    import os
    file_name_raw, file_ext1 = os.path.splitext(raw_input_name)

    if file_ext1 == '.csv':
        data = pd.read_csv(raw_input_name)
    elif file_ext1 in ['.xls', '.xlsx']:
        data = pd.read_excel(raw_input_name)
    else:
        raise ValueError("Unsupported file format for input 1")



    # Create a new dataframe with the desired structure
    output = pd.DataFrame(columns=['substrate', 'temperature', 'time', 'wav', 'back', 'T', 'R', 'P', 'step'])

    for index, group in data.groupby(['substrate', 'time', 'wavelength']):
        new_row = {
            'substrate': index[0],
            'temperature': group['T_cell'].iloc[0],  # Using T_cell value directly as temperature
            'time': index[1],
            'wav': index[2],
            'back': group['r_background'].iloc[0],
            'T': group[group['measurement (t,r,p)'] == 1]['mW'].values[0],
            'R': group[group['measurement (t,r,p)'] == 2]['mW'].values[0],
            'P': group[group['measurement (t,r,p)'] == 3]['mW'].values[0],
            'step': group['step'].iloc[0]
        }
        # <2/5 Edit CB>
        # OLD
        # output = output.append(new_row, ignore_index=True)
        # NEW
        output = pd.concat([output, pd.DataFrame([new_row])],
                           ignore_index=True)  # THIS ROW WAS CHANGED FROM OLD to fit modern pandas conventions
        # </2/5 Edit CB>

    output['Reflectivity'] = (output['R'] - output['back']) / (output['R'] - output['back'] + output['T'])

    # Extract Powerinitial and Pinitial
    power_initials = output[output['time'] == -1].set_index(['substrate', 'wav'])['R'] + \
                     output[output['time'] == -1].set_index(['substrate', 'wav'])['T']
    p_initials = output[output['time'] == -1].set_index(['substrate', 'wav'])['P']

    # Calculate Power for each row
    output['Power'] = output.apply(
        lambda row: power_initials[row['substrate'], row['wav']] * row['P'] / p_initials[row['substrate'], row['wav']],
        axis=1)

    # Calculate Absorption for each row
    output['Absorption'] = (output['Power'] - output['T'] - output['R'] + output['back']) / output['Power']

    # Generate final output
    final_output_data = []
    unique_sub_time = output[['substrate', 'time']].drop_duplicates().values

    for sub, t in unique_sub_time:
        row_data = {'substrate': sub, 'time': t,
                    'step': output[(output['substrate'] == sub) & (output['time'] == t)]['step'].iloc[0]}
        subset = output[(output['substrate'] == sub) & (output['time'] == t)]

        # Extract the temperature from the subset
        temperature_value = subset['temperature'].values[0] if not subset['temperature'].empty else None
        row_data['temperature'] = temperature_value

        for wav in [443, 514, 689, 781, 817]:
            wav_data = subset[subset['wav'] == wav]
            if not wav_data.empty:
                row_data[f'{wav}R'] = wav_data['Reflectivity'].values[0]
                row_data[f'{wav}A'] = wav_data['Absorption'].values[0]
        final_output_data.append(row_data)

    final_output_df = pd.DataFrame(final_output_data)
    #print(final_output_df.columns.tolist())

    # final_output_df.to_excel(file_name_raw+"stupid.xlsx",index=False)
    import numpy as np
    import pickle

    # Load the data
    df = final_output_df
    last_row_with_step_1 = df[df['step'] == 1].iloc[-1]
    time_calib = last_row_with_step_1['time']
    filtered_df = df[(df['step'] == 1) & (df['time'] == time_calib)].tail(1)
    # If the filtered dataframe is not empty, store the required values
    if not filtered_df.empty:
        values = filtered_df[['443R', '443A', '514R', '514A', '689R', '689A', '781R', '781A', '817R', '817A']].iloc[0]
        # Storing values in a dictionary for later recall
        stored_values = {f"{col}c": values[col] for col in values.index}
    else:
        stored_values = None
    stored_values  # can call by stored_values['443Rc']
    # <2/5 new cb>
    # with open(folder_prefix + "stored_values.pkl", 'wb') as file:
    #     pickle.dump(stored_values, file)
    # </2/5 new cb>
    # Step 1: Extract corrected values when step=1 and time=time_calib
    corrected_rows = df[(df['step'] == 1) & (df['time'] == time_calib)]
    corrected_rows = corrected_rows.drop_duplicates(subset='substrate').set_index('substrate')
    corrected_cols = {col: f'{col}c' for col in corrected_rows.columns if 'R' in col or 'A' in col}
    corrected_rows.rename(columns=corrected_cols, inplace=True)

    # Step 2: Remove rows where step=1 and step=0
    df = df[~df['step'].isin([0, 1])]

    # Step 3: Merge the corrected values with the original dataframe
    df = df.join(corrected_rows[corrected_cols.values()], on='substrate')
    print(df.columns.tolist())

    # Define wavelengths and transformation function
    wavelengths = {
        443: ('443R', '443A'),
        514: ('514R', '514A'),
        689: ('689R', '689A'),
        781: ('781R', '781A'),
        817: ('817R', '817A')
    }

    def transform_data(row):
        transformed_rows = []
        for wavelength in wavelengths:
            R, A = wavelengths[wavelength]
            if pd.notna(row[R]) and pd.notna(row[A]):
                new_row = row[['substrate', 'temperature', 'time']].to_dict()
                new_row.update({f'{wvl}Rc': row[f'{wvl}Rc'] for wvl in wavelengths})
                new_row.update({f'{wvl}Ac': row[f'{wvl}Ac'] for wvl in wavelengths})
                new_row['wavelength'] = wavelength
                new_row['R'] = row[R]
                new_row['A'] = row[A]
                transformed_rows.append(new_row)
        return transformed_rows

    # Step 4: Transform the data
    transformed_data = [new_row for _, row in df.iterrows() for new_row in transform_data(row)]
    final_df = pd.DataFrame(transformed_data)
    print(final_df.columns.tolist())
    # Filter values below 0 or above 1
    columns_to_check = ['443Rc', '514Rc', '689Rc', '781Rc', '817Rc', '443Ac', '514Ac', '689Ac', '781Ac', '817Ac', 'R',
                        'A']

    # Replace values in each column that are <0 or >1 with NaN
    for col in columns_to_check:
        final_df[col] = final_df[col].apply(lambda x: 0 if x < 0 else (x if x <= 1 else np.nan))

    # Save the final dataframe to an Excel file
    final_df.to_excel(processed_output_name, index=False)


def Stepper(raw_input_name):
    import pandas as pd
    import sys
    import os

    file_name_raw, file_ext1 = os.path.splitext(raw_input_name)

    if file_ext1 == '.csv':
        data = pd.read_csv(raw_input_name)
    elif file_ext1 in ['.xls', '.xlsx']:
        data = pd.read_excel(raw_input_name)
    else:
        raise ValueError("Unsupported file format for input 1")

    if 'step' not in data.columns:
        data['step'] = 2
        for substrate, group_df in data.groupby('substrate'):
            timehold = -1
            for index, row in group_df.iterrows():
                if row['time'] < 0:
                    data.loc[index, 'step'] = 0
                elif row['time'] >= timehold:
                    data.loc[index, 'step'] = 1
                else:
                    break
                timehold = row['time']
        if file_ext1 == '.csv':
            data.to_csv(raw_input_name, index=False)
        else:
            data.to_excel(raw_input_name)

