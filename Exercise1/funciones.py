import numpy as np

class SOM:
    def __init__(self, x, y, input_len, sigma, learning_rate):
        """ Initialises Self-Organizing Map (SOM) """
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.weights = np.random.rand(x, y, input_len)

    def random_weights_init(self, data):
        """ Initializes the weights of the SOM picking random samples from data"""

        # Ensure enough data points for initialization
        if len(data) < self.x * self.y:
            raise ValueError("Too few data points to initialize weights.")

        # Sample random data points to initialize weights
        random_indices = np.random.choice(len(data), size=self.x * self.y, replace=False)
        self.weights = data[random_indices].reshape(self.x, self.y, self.input_len)

        # Sample from data's range
        #for i in range(self.x):
        #    for j in range(self.y):
        #        self.weights[i, j] = np.random.uniform(low=data.min(axis=0), high=data.max(axis=0))

    def winner(self, x):
        """ Finds winning neuron (Best Matching Unit - BMU) """
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def update(self, x, win, t, max_iterations):
        """ Updates the weights of the SOM based on the winning neuron and iteration """

        # Calculate the radius of the neighborhood
        radius = self.sigma * np.exp(-t / max_iterations)
        # Calculate the learning rate at time t
        lr = self.learning_rate * np.exp(-t / max_iterations)

        # Iterate over the map
        for i in range(self.x):
            for j in range(self.y):
                # Calculate the distance from the winning neuron
                dist_to_win = np.linalg.norm([i - win[0], j - win[1]])
                # If the neuron is inside the neighborhood, update its weight
                if dist_to_win <= radius:
                    influence = np.exp(-dist_to_win**2 / (2 * radius**2))
                    self.weights[i, j] += lr * influence * (x - self.weights[i, j])

    def train_random(self, data, iterations):
        """ Trains the SOM using random input samples """
        for t in range(iterations):
            i = np.random.randint(len(data))
            x = data[i]
            win = self.winner(x)
            self.update(x, win, t, iterations)

    def distance_map(self):
        """ Calculates the U-Matrix (unified distance matrix) of the SOM """
        um = np.zeros((self.x, self.y))
        for x in range(self.x):
            for y in range(self.y):
                neighbors = []
                for i in range(x - 1, x + 2):
                    for j in range(y - 1, y + 2):
                        if i >= 0 and j >= 0 and i < self.x and j < self.y and (i, j) != (x, y):
                            neighbors.append(self.weights[i, j])
                if neighbors:
                    um[x, y] = np.mean(np.linalg.norm(self.weights[x, y] - neighbors, axis=1))
        return um

    def get_weights(self):
        """ Returns the weights of the neural network"""
        return self.weights.copy()

class Oja:
    def __init__(self, input_dim, learning_rate):
        """ Initializes the Oja network."""
        self.weights = np.random.rand(input_dim)
        self.learning_rate = learning_rate

    def train(self, data, iterations):
        """ Trains the Oja network using the learning rule."""
        for _ in range(iterations):
            for x in data:
                y = np.dot(x, self.weights)
                self.weights += self.learning_rate * y * (x - y * self.weights)

    def get_weights(self):
        """ Returns the weights of the Oja network."""
        return self.weights.copy()

# weight_new = weight_old + learning_rate * output * (input - output * weight_old)

#class Hopfield:
    #def __init__(self, x):
