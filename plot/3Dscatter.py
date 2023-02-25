import plotly.graph_objs as go
import pandas as pd

# Load the data from a CSV file
df = pd.read_csv('data.csv')

# Create the plotly figure
fig = go.Figure(data=[go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], mode='markers',
                                   marker=dict(size=5, color=df['color'], opacity=0.8))])

# Set the axis labels
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Show the plot
fig.show()
