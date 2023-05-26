import matplotlib.pyplot as plt


def plot_velocity(file_path, window_size):
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
            c = 100000
            z = c / ((w ** 2 + h ** 2) ** 0.5)

            data.append((frame_idx, id, x, y, z))

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
    # Plot velocities
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    for idx, (id, velocities) in enumerate(id_velocities.items()):
        frame_idx = velocities['frame_idx']
        v_xy = velocities['v_xy']
        v_xyz = velocities['v_xyz']

        # Calculate simple moving averages
        v_xy_sma = [sum(v_xy[max(0, i - window_size):i]) / min(i, window_size) for i in range(1, len(v_xy) + 1)]
        v_xyz_sma = [sum(v_xyz[max(0, i - window_size):i]) / min(i, window_size) for i in range(1, len(v_xyz) + 1)]

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


if __name__ == '__main__':
    file_path = './runs/track/exp2/labels/733.txt'
    window_size = 5  # Adjust the window size for the simple moving average

    plot_velocity(file_path, window_size)
       
