import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils import get_latest_exp_number

class FishMovementPlotter:
    def __init__(self, exp_number=None, track_folder="./runs/track"):
        self.exp_number = exp_number
        self.track_folder = track_folder

    def get_latest_exp_number(self):
        track_path = Path(self.track_folder)
        exp_folders = [f for f in track_path.iterdir() if f.is_dir() and f.name.startswith("exp") and f.name[3:].isdigit()]
        if not exp_folders:
            return 0
        latest_exp_folder = max(exp_folders, key=lambda f: int(f.name[3:]))
        latest_exp_number = int(latest_exp_folder.name[3:]) + 1
        return latest_exp_number

    def plot_velocity(self):
        # Determine the experiment number
        exp_number = self.exp_number if self.exp_number is not None else self.get_latest_exp_number()

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

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Plot fish movement velocity.')
parser.add_argument('--exp', type=int, help='Experiment number', default=None)
parser.add_argument('--track_folder', type=str, help='Track folder path', default="./runs/track")
args = parser.parse_args()

# Create the FishMovementPlotter object
plotter = FishMovementPlotter(exp_number=args.exp, track_folder=args.track_folder)

# Plot the velocity
plotter.plot_velocity()
