import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly
from scipy.interpolate import Rbf

def upscale_array(arr, scale_factor):
    # Determine the original dimensions
    original_height, original_width = arr.shape
    # New dimensions
    new_height = original_height * scale_factor
    new_width = original_width * scale_factor
    # Create an upscaled array with zeros
    upscaled_arr = np.zeros((new_height, new_width), dtype=arr.dtype)

    print(upscaled_arr)
    # Place original array values in the new array
    for i in range(original_height):
        for j in range(original_width):
            upscaled_arr[i * scale_factor, j * scale_factor] = arr[i, j]
    return upscaled_arr

# Step 1: Read the CSV File
fname = 'circle.csv'
df = pd.read_csv(fname, header=None)

# Step 2: Process the Data
df.fillna(0, inplace=True)
# Convert the DataFrame to a NumPy array
# Assuming the data is already in a grid format suitable for a topographic map
potentials = df.values

potentials = upscale_array(potentials, 5)

print(potentials)

y, x = np.nonzero(potentials)
values = potentials[y, x]



rbf = Rbf(x, y, values, function='cubic')

x_grid, y_grid = np.meshgrid(np.arange(potentials.shape[1]), np.arange(potentials.shape[0]))

interpolated_values = rbf(x_grid, y_grid)
potentials[potentials == 0] = interpolated_values[potentials == 0]

np.savetxt(f"{fname}.interpolated.csv", potentials, delimiter=",")

# Create a 3D surface plot
fig = go.Figure(data=[
    go.Surface(
        z=potentials,
        x=x_grid,
        y=y_grid,
        contours={
            'z': {  # Contour lines for z dimension
                'show': True,
                'usecolormap': False,
                'highlightcolor': "limegreen",
                'project': {'z': True}  # Project contours onto the 'z' plane
            }
        }
    )
])

fig.update_layout(
    title=f"3D surface of {fname} with interpolated values",
    autosize=False,
    width=1440,
    height=900,
    margin=dict(l=65, r=50, b=65, t=90),
    scene=dict(
        zaxis=dict(
            title='VOLTS'
        )
    )
)

# Show the plot
plotly.offline.plot(fig, filename=f"{fname}.html")
