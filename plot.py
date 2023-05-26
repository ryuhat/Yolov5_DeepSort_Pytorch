import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils import get_latest_exp_number

def plot_fish_movement_velocity(exp_number=None, track_folder="./runs/track"):
    # Construct the file path
    file_path = f'./runs/track/exp{exp_number}/tracks/i-id-x-y-z.txt'

    # Read the text file
    data = np.loadtxt(file_path)

    # Extract relevant columns
    frame_idx = data[:, 0]
    v_xy = data[:, 1]

    # Calculate time values (t) from frame indices
    t = frame_idx * 0.1  # Assuming each frame corresponds to 0.1 seconds

    # Plot the (t, v_xy) data as a line graph
    plt.plot(t, v_xy)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (v_xy)')
    plt.title(f'Fish Movement Velocity (Experiment {exp_number})')
    plt.show()

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Plot fish movement velocity.')
    parser.add_argument('--exp', type=int, help='Experiment number', default=None)
    parser.add_argument('--track_folder', type=str, help='Track folder path', default="./runs/track")
    args = parser.parse_args()

    # Determine the experiment number
    exp_number = args.exp if args.exp is not None else get_latest_exp_number(args.track_folder)

    # Call the plot function
    plot_fish_movement_velocity(exp_number, args.track_folder)
