import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the path to the directory containing the text files
txt_path = 'runs/track/exp44/tracks/simple'

# Define a list of colors for each id
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# Define a function to plot 3D trajectory for each id
def plot_trajectory_3d():
    # Loop over each id and plot its trajectory
    for id in range(1, len(colors) + 1):
        # Read the text file for the given id
        data = np.loadtxt(txt_path+ '.txt', delimiter=' ')

        # Extract x, y, z coordinates
        x = data[:, 2]
        y = data[:, 3]
        z = data[:, 4]

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory with a different color for each id
        ax.plot(x, y, z, color=colors[id-1], label='id ' + str(id))
        ax.legend()

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectory for id ' + str(id))

    # Show all the plots
    plt.show()

# Call the function
plot_trajectory_3d()
