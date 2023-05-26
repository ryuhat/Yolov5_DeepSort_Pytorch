import argparse
import matplotlib.pyplot as plt


def read_data(file_path, c):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line_data = line.strip().split()
            frame_idx = int(line_data[0])
            id = int(line_data[1])
            x = float(line_data[2])
            y = float(line_data[3])
            w = float(line_data[4])
            h = float(line_data[5])
            z = c / ((w ** 2 + h ** 2) ** 0.5)

            data.append((frame_idx, id, x, y, z))

    return data


def calculate_velocities(data):
    id_velocities = {}
    for i in range(1, len(data)):
        frame_idx, id, x, y, z = data[i]
        prev_frame_idx, prev_id, prev_x, prev_y, prev_z = data[i - 1]

        # Skip calculation if frame indices are the same
        if frame_idx == prev_frame_idx:
            continue

        # Calculate velocity
        v_xy = ((x - prev_x) ** 2 + (y - prev_y) ** 2) ** 0.5 / (frame_idx - prev_frame_idx)
        v_xyz = ((x - prev_x) ** 2 + (y - prev_y) ** 2 + (z - prev_z) ** 2) ** 0.5 / (frame_idx - prev_frame_idx)

        # Update velocities dictionary
        if id not in id_velocities:
            id_velocities[id] = {'frame_idx': [], 'v_xy': [], 'v_xyz': []}
        id_velocities[id]['frame_idx'].append(frame_idx)
        id_velocities[id]['v_xy'].append(v_xy)
        id_velocities[id]['v_xyz'].append(v_xyz)

    return id_velocities


def calculate_sma(velocities, window_size):
    velocities_with_sma = {}
    for id, data in velocities.items():
        frame_idx = data['frame_idx']
        v_xy = data['v_xy']
        v_xyz = data['v_xyz']

        # Calculate simple moving averages
        v_xy_sma = [sum(v_xy[max(0, i - window_size):i]) / min(i, window_size) for i in range(1, len(v_xy) + 1)]
        v_xyz_sma = [sum(v_xyz[max(0, i - window_size):i]) / min(i, window_size) for i in range(1, len(v_xyz) + 1)]

        velocities_with_sma[id] = {
            'frame_idx': frame_idx,
            'v_xy': v_xy,
            'v_xyz': v_xyz,
            'v_xy_sma': v_xy_sma,
            'v_xyz_sma': v_xyz_sma
        }

    return velocities_with_sma


def plot_velocities(velocities, window_size):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    for id, data in velocities.items():
        frame_idx = data['frame_idx']
        v_xy = data['v_xy']
        v_xyz = data['v_xyz']
        v_xy_sma = data['v_xy_sma']
        v_xyz_sma = data['v_xyz_sma']

        # Plot v_xy
        axs[0].plot(frame_idx, v_xy, label=f'ID {id}')
        axs[0].set_title('v_xy')
        axs[0].set_xlabel('Frame Index')
        axs[0].set_ylabel('Velocity')

        # Plot v_xyz
        axs[1].plot(frame_idx, v_xyz, label=f'ID {id}')
        axs[1].set_title('v_xyz')
        axs[1].set_xlabel('Frame Index')
        axs[1].set_ylabel('Velocity')

        # Plot v_xy_sma
        axs[2].plot(frame_idx, v_xy_sma, label=f'ID {id}')
        axs[2].set_title(f'v_xy_sma (Window Size = {window_size})')
        axs[2].set_xlabel('Frame Index')
        axs[2].set_ylabel('Velocity')

        # Plot v_xyz_sma
        axs[3].plot(frame_idx, v_xyz_sma, label=f'ID {id}')
        axs[3].set_title(f'v_xyz_sma (Window Size = {window_size})')
        axs[3].set_xlabel('Frame Index')
        axs[3].set_ylabel('Velocity')

    # Add legend
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()

    # Adjust subplot spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def main(file_path, window_size, c):
    # Read data from file
    data = read_data(file_path, c)

    # Calculate velocities
    velocities = calculate_velocities(data)

    # Calculate simple moving averages
    velocities_with_sma = calculate_sma(velocities, window_size)

    # Plot velocities
    plot_velocities(velocities_with_sma, window_size)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, nargs='?', default='./runs/track/exp2/labels/733.txt',
                        help='Path to the data file')
    parser.add_argument('window_size', type=int, nargs='?', default=30,
                        help='Window size for the simple moving average')
    parser.add_argument('--c', type=float, default=100000, help='Constant value c')
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.file_path, args.window_size, args.c)
