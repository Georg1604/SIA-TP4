### Ejercicio Europa
### 1.1 Red de Kohonen
### 1.2 Modelo de Oja

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from collections import defaultdict

# Load dataset with specified data types
dtypes = {'Country': 'str', 
          'Area': 'int64', 
          'GDP': 'int64', 
          'Inflation': 'float64', 
          'Life.expect': 'float64', 
          'Military': 'float64', 
          'Pop.growth': 'float64', 
          'Unemployment': 'float64'}

df = pd.read_csv('europe.csv', dtype=dtypes)

# Separate country names since it's not numerical
country_names = df['Country']
df_numeric = df.drop('Country', axis=1)

# Normalize numerical data
df_numeric = (df_numeric - df_numeric.mean()) / df_numeric.std()

""" Self-Organizing Map (SOM) / Kohonen map """
# Define SOM parameters
x = 5  # Width of map
y = 5  # Height of map
input_len = df_numeric.shape[1]  # Number of features in data
sigma = 1.0 # Neighbourhood radius
learning_rate = 0.5

# Create SOM
som = MiniSom(x, y, input_len, sigma, learning_rate) # create SOM
som.random_weights_init(df_numeric.values) # initial random weights

# Train SOM
iterations = 10000
som.train_random(df_numeric.values, iterations)

# Plot heat-map (SOM)
plt.pcolor(som.distance_map().T, cmap='coolwarm') # cmap=cmap='hot', cmap='binary', cmap='afmhot', https://matplotlib.org/stable/gallery/color/colormap_reference.html
plt.colorbar()
plt.title('SOM Clustering of Dataset')
plt.show()

""" Unified Distance Matrix  (U-Matrix) """
# Calculate Unified distance matrix (U-Matrix)
umat = som.distance_map()

# Plot U-Matrix as heatmap
plt.pcolor(umat.T, cmap='viridis')
plt.colorbar()
plt.title('U-Matrix: Average Distances Between Neurons')
plt.show() # Darker regions in the U-Matrix indicate larger distances between clusters

""" Winning neuron. Elements assigned to each neuron """
## Calculate element counts for each neuron
neuron_counts = defaultdict(int)

for sample in df_numeric.values:
    winning_neuron = som.winner(sample)
    neuron_counts[winning_neuron] += 1

# Reshape counts into grid, handling potential dead neurons
counts_grid = np.zeros((x, y))  # Initialize to zeros
for coordinates, count in neuron_counts.items():
    x_coord, y_coord = coordinates
    counts_grid[y_coord][x_coord] = count

# Initialize plot
fig, ax = plt.subplots()

# Plot the base SOM grid (to put count on)
plt.pcolor(som.distance_map().T, cmap='coolwarm', alpha=0.5)

# Overlay count labels on neurons
for x in range(som.get_weights().shape[0]):
    for y in range(som.get_weights().shape[1]):
        ax.text(x + 0.5, y + 0.5, str(counts_grid[y][x]),
                ha='center', va='center', size='x-large')

ax.set_title('Element-Count per Neuron')
plt.colorbar()
plt.show()
