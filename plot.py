import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils import get_latest_exp_number

def plot_fish_v_xy_by_id(exp_number=None, track_folder="./runs/track"):
    # Construct the file path
    file_path = f'./runs/track/exp{exp_number}/tracks/i-id-x-y-z.txt'

    # Read the text file
    data = np.loadtxt(file_path)

    # Extract relevant columns
    frame_idx = data[:, 0] - 1  # Subtract 1 to get the actual frame index
    id_vals = data[:, 1]
    x = data[:, 2]
    y = data[:, 3]

    unique_ids = np.unique(id_vals)  # Get unique IDs

    # Plot velocity for each unique ID
    for id_val in unique_ids:
        # Filter data for the current ID
        mask = id_vals == id_val
        x_id = x[mask]
        y_id = y[mask]

        # Calculate velocity (v_xy)
        dx = np.diff(x_id)
        dy = np.diff(y_id)
        v_xy = np.sqrt(dx**2 + dy**2)

        # Adjust the dimensions of v_xy to match frame_idx
        v_xy = np.concatenate([[0], v_xy])  # Add a zero velocity at the beginning

        # Plot the (frame_idx, v_xy) data as a line graph
        plt.plot(frame_idx[mask], v_xy, label=f'ID {int(id_val)}')

    plt.xlabel('Frame Index')
    plt.ylabel('Velocity (v_xy)')
    plt.title(f'Fish Movement Velocity by ID (Experiment {exp_number})')
    plt.legend()
    plt.show()

def plot_fish_v_xyz_by_id(exp_number=None, track_folder="./runs/track"):
    # Construct the file path
    file_path = f'./runs/track/exp{exp_number}/tracks/i-id-x-y-z.txt'

    # Read the text file
    data = np.loadtxt(file_path)

    # Extract relevant columns
    frame_idx = data[:, 0] - 1  # Subtract 1 to get the actual frame index
    id_vals = data[:, 1]
    x = data[:, 2]
    y = data[:, 3]
    z = data[:, 4]

    unique_ids = np.unique(id_vals)  # Get unique IDs

    # Plot velocity differences (v_xyz) for each unique ID
    for id_val in unique_ids:
        # Filter data for the current ID
        mask = id_vals == id_val
        x_id = x[mask]
        y_id = y[mask]
        z_id = z[mask]

        # Calculate velocity differences (v_xyz)
        dx = np.diff(x_id)
        dy = np.diff(y_id)
        dz = np.diff(z_id)
        v_xyz = np.sqrt(dx**2 + dy**2 + dz**2)

        # Adjust the dimensions of v_xyz to match frame_idx
        v_xyz = np.concatenate([[0], v_xyz])  # Add a zero velocity at the beginning

        # Plot the (frame_idx, v_xyz) data as a line graph
        plt.plot(frame_idx[mask], v_xyz, label=f'ID {int(id_val)}')

    plt.xlabel('Frame Index')
    plt.ylabel('Velocity Difference (v_xyz)')
    plt.title(f'Fish Movement Velocity Difference by ID (Experiment {exp_number})')
    plt.legend()
    plt.show()

def calculate_velocity_difference(x, y, z=None):
    dx = np.diff(x)
    dy = np.diff(y)

    if z is not None:
        dz = np.diff(z)
        v_xyz = np.sqrt(dx**2 + dy**2 + dz**2)
        return v_xyz
    else:
        v_xy = np.sqrt(dx**2 + dy**2)
        return v_xy


def calculate_sma(data, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(data, weights, 'valid')
    return sma

def calculate_velocity_difference(x, y, z):
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    v_xyz = np.sqrt(dx**2 + dy**2 + dz**2)
    return v_xyz

def calculate_sma(data, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(data, weights, 'valid')
    return sma

def plot_fish_v_xyz_sma_by_id(exp_number, track_folder, sma_window=5):
    # Construct the file path
    file_path = f'{track_folder}/exp{exp_number}/tracks/i-id-x-y-z.txt'

    # Read the text file
    data = np.loadtxt(file_path)

    # Extract relevant columns
    frame_idx = data[:, 0] - 1  # Subtract 1 to get the actual frame index
    id_vals = data[:, 1]
    x = data[:, 2]
    y = data[:, 3]
    z = data[:, 4]

    unique_ids = np.unique(id_vals)  # Get unique IDs

    # Plot SMA of velocity difference (v_xyz) for each unique ID
    for id_val in unique_ids:
        # Filter data for the current ID
        mask = id_vals == id_val
        x_id = x[mask]
        y_id = y[mask]
        z_id = z[mask]

        # Calculate velocity difference (v_xyz)
        v_xyz = calculate_velocity_difference(x_id, y_id, z_id)

        # Calculate SMA of v_xyz
        v_xyz_sma = calculate_sma(v_xyz, sma_window)

        # Adjust the dimensions of v_xyz_sma to match frame_idx
        v_xyz_sma = np.concatenate([[0] * (sma_window - 1), v_xyz_sma])  # Add zeros at the beginning

        # Plot the (frame_idx, v_xyz_sma) data as a line graph
        plt.plot(frame_idx[mask][:-1], v_xyz_sma, label=f'ID {int(id_val)}')

    plt.xlabel('Frame Index')
    plt.ylabel(f'SMA of Velocity Difference (v_xyz) (Window Size: {sma_window})')
    plt.title(f'Fish Movement Velocity Difference SMA by ID (Experiment {exp_number})')
    plt.legend()
    plt.show()


def calculate_velocity_difference_xyz(x, y, z):
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    v_xyz = np.sqrt(dx**2 + dy**2 + dz**2)
    return v_xyz

def calculate_velocity_difference_xy(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    v_xy = np.sqrt(dx**2 + dy**2)
    return v_xy

def plot_fish_velocity_subplots(exp_number, track_folder, sma_window=5):
    # Construct the file path
    file_path = f'{track_folder}/exp{exp_number}/tracks/i-id-x-y-z.txt'

    # Read the text file
    data = np.loadtxt(file_path)

    # Extract relevant columns
    frame_idx = data[:, 0] - 1  # Subtract 1 to get the actual frame index
    id_vals = data[:, 1]
    x = data[:, 2]
    y = data[:, 3]
    z = data[:, 4]

    unique_ids = np.unique(id_vals)  # Get unique IDs

    # Create subplots for v_xy, v_xyz, and v_xyz_sma
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    # Plot v_xy for each unique ID
    axes[0].set_title('Velocity Difference (v_xy)')
    for id_val in unique_ids:
        # Filter data for the current ID
        mask = id_vals == id_val
        x_id = x[mask]
        y_id = y[mask]

        # Calculate velocity difference (v_xy)
        v_xy = calculate_velocity_difference_xy(x_id, y_id)

        # Plot the (frame_idx, v_xy) data as a line graph
        axes[0].plot(frame_idx[mask][:-1], v_xy, label=f'ID {int(id_val)}')

    axes[0].set_xlabel('Frame Index')
    axes[0].set_ylabel('Velocity Difference')

    # Plot v_xyz for each unique ID
    axes[1].set_title('Velocity Difference (v_xyz)')
    for id_val in unique_ids:
        # Filter data for the current ID
        mask = id_vals == id_val
        x_id = x[mask]
        y_id = y[mask]
        z_id = z[mask]

        # Calculate velocity difference (v_xyz)
        v_xyz = calculate_velocity_difference_xyz(x_id, y_id, z_id)

        # Plot the (frame_idx, v_xyz) data as a line graph
        axes[1].plot(frame_idx[mask][:-1], v_xyz, label=f'ID {int(id_val)}')

    axes[1].set_xlabel('Frame Index')
    axes[1].set_ylabel('Velocity Difference')

    # Plot v_xyz_sma for each unique ID
    axes[2].set_title('Velocity Difference SMA (v_xyz_sma)')
    for id_val in unique_ids:
        # Filter data for the current ID
        mask = id_vals == id_val
        x_id = x[mask]
        y_id = y[mask]
        z_id = z[mask]

        # Calculate velocity difference (v_xyz)
        v_xyz = calculate_velocity_difference_xyz(x_id, y_id, z_id)

        # Calculate SMA of v_xyz
        v_xyz_sma = calculate_sma(v_xyz, sma_window)

        # Adjust the dimensions of v_xyz_sma to match frame_idx
        v_xyz_sma = np.concatenate([[0] * (sma_window - 1), v_xyz_sma])  # Add zeros at the beginning

        # Plot the (frame_idx, v_xyz_sma) data as a line graph
        axes[2].plot(frame_idx[mask][:-1], v_xyz_sma, label=f'ID {int(id_val)}')

    axes[2].set_xlabel('Frame Index')
    axes[2].set_ylabel('Velocity Difference SMA')
         # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the legend for each subplot
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    # Show the plot
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
    # plot_fish_v_xy_by_id(exp_number, args.track_folder)
    # plot_fish_v_xyz_by_id(exp_number, args.track_folder)
    # plot_fish_v_xy_sma_by_id(args.exp, args.track_folder)
    # plot_fish_v_xyz_sma_by_id(args.exp, args.track_folder, sma_window=30)
    plot_fish_velocity_subplots(args.exp, args.track_folder)
