import plotly.express as px
import pandas as pd

# Load the data from the text file
df = pd.read_csv('fish_trajectories.txt', delimiter=' ')

# Create a 3D scatter plot
fig = px.scatter_3d(df, x='x', y='y', z='z', color='id', animation_frame='time', hover_name='id')

# Set the axis labels
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Show the plot
fig.show()
