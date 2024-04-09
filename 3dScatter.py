import sys
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
file_path = sys.argv[1]
data = pd.read_csv(file_path + ".csv")

# Extract x, y, z coordinates
x = data['X']
y = data['Y']
z = data['Z']

# Create scatter plots from multiple perspectives
perspectives = [
    {'elev': 30, 'azim': 30, 'suffix': 'rotated_30_30'},
    {'elev': 30, 'azim': 60, 'suffix': 'rotated_30_60'},
    {'elev': 0, 'azim': 90, 'suffix': 'rotated_0_90_XZ_only'},
    {'elev': 0, 'azim': 0, 'suffix': 'rotated_0_0_XY_only'}
    ]


# Create scatter plots from different perspectives
for perspective in perspectives:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.view_init(elev=perspective['elev'], azim=perspective['azim'])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Time')
    ax.set_zlabel('Loss')
    
    # Save plot to PNG file with specified suffix for each perspective
   
    plt.savefig(sys.argv[1] + "_" + perspective["suffix"]+".png")

    # Show plot
