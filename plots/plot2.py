import plotly.graph_objects as go


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

    # Create velocity plots
    fig = go.Figure()

    for id, velocities in id_velocities.items():
        frame_idx = velocities['frame_idx']
        v_xy = velocities['v_xy']
        v_xyz = velocities['v_xyz']

        # Calculate simple moving averages
        v_xy_sma = [sum(v_xy[max(0, i - window_size):i]) / min(i, window_size) for i in range(1, len(v_xy) + 1)]
        v_xyz_sma = [sum(v_xyz[max(0, i - window_size):i]) / min(i, window_size) for i in range(1, len(v_xyz) + 1)]

        # Add v_xy trace
        fig.add_trace(go.Scatter(x=frame_idx, y=v_xy, mode='lines', name=f'ID {id} v_xy'))

        # Add v_xyz trace
        fig.add_trace(go.Scatter(x=frame_idx, y=v_xyz, mode='lines', name=f'ID {id} v_xyz'))

        # Add v_xy_sma trace
        fig.add_trace(go.Scatter(x=frame_idx, y=v_xy_sma, mode='lines', name=f'ID {id} v_xy_sma'))

        # Add v_xyz_sma trace
        fig.add_trace(go.Scatter(x=frame_idx, y=v_xyz_sma, mode='lines', name=f'ID {id} v_xyz_sma'))

    # Set layout
    title_font = dict(size=18)
    axis_font = dict(size=14)
    fig.update_layout(
        title="Velocity Plots",
        xaxis=dict(title="Frame Index", tickfont=axis_font),
        yaxis=dict(title="Velocity", tickfont=axis_font),
        font=title_font,
        legend=dict(font=dict(size=12)),
        grid=dict(visible=False),
        showlegend=True,
        height=600,
        width=1000,
    )

    # Display the plot
    fig.show()


if __name__ == '__main__':
    file_path = './runs/track/exp2/labels/733.txt'
    window_size = 5  # Adjust the window size for the simple moving average

    plot_velocity(file_path, window_size)
