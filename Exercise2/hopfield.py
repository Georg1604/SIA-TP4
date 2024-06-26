import numpy as np
import random
import matplotlib.pyplot as plt

def read_letters_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    letters = {}
    for block in content.strip().split('\n\n'):
        if block:
            letter, pattern = block.split('=')
            letter = letter.strip(':')
            pattern_lines = pattern.strip().split('\n')
            matrix = np.array([[1 if char == 'X' else -1 for char in line] for line in pattern_lines])
            letters[letter] = matrix

    return letters


def select_random_patterns(patterns, n):
    if n > len(patterns):
        raise ValueError("n is too big to select random patterns")

    selected_items = random.sample(list(patterns.items()), n)
    selected_patterns = dict(selected_items)
    return selected_patterns
# Define patterns
patterns = read_letters_from_file('letters.txt')

def add_noise(pattern, noise_percentage):
    noisy_pattern = pattern.copy()
    num_pixels = pattern.size
    num_noisy_pixels = int(noise_percentage * num_pixels)
    indices = np.random.choice(num_pixels, num_noisy_pixels, replace=False)

    flat_pattern = noisy_pattern.flatten()
    flat_pattern[indices] *= -1  # Inverse the selected pixels
    noisy_pattern = flat_pattern.reshape(pattern.shape)

    return noisy_pattern




def hopfield_model(patterns):
    first_pattern = next(iter(patterns.values()))
    num_features = first_pattern.size
    weight_matrix = np.zeros((num_features, num_features))

    for pattern in patterns.values():
        pattern_vector = pattern.flatten()
        weight_matrix += np.outer(pattern_vector, pattern_vector)

    np.fill_diagonal(weight_matrix, 0)

    return weight_matrix

def plot_pattern(pattern, title, epoch=None):
    plt.imshow(pattern, cmap='binary', vmin=-1, vmax=1)
    if epoch is not None:
        plt.title(f'{title} (Epoch {epoch})')
    else:
        plt.title(title)
    plt.axis('off')
    plt.show()


def calculate_energy(pattern, weight_matrix):
    return -0.5 * np.dot(pattern.flatten(), np.dot(weight_matrix, pattern.flatten()))




def retrieve_pattern(weight_matrix, pattern, max_epochs, plot=True):
    epoch = 0
    current_pattern = pattern
    s_history = [current_pattern]
    energy_history = [calculate_energy(current_pattern, weight_matrix)]

    while epoch < max_epochs:
        new_pattern = np.sign(np.dot(weight_matrix, current_pattern.flatten()))
        new_pattern = np.where(new_pattern >= 0, 1, -1).reshape(current_pattern.shape)
        s_history.append(new_pattern)
        energy_history.append(calculate_energy(new_pattern, weight_matrix))

        if plot:
            plot_pattern(new_pattern, "Pattern", epoch)

        if len(s_history) > 2 and np.array_equal(s_history[-2], s_history[-1]):
            # print(f"Converged at epoch {epoch}")
            break

        epoch += 1
        current_pattern = new_pattern

    return new_pattern, energy_history


def retrieve_pattern_dynamic(weight_matrix, pattern, max_epochs):
    epoch = 0
    current_pattern = pattern
    s_history = [current_pattern]
    energy_history = [calculate_energy(current_pattern, weight_matrix)]

    while epoch < max_epochs:
        new_pattern = np.sign(np.dot(weight_matrix, current_pattern.flatten()))
        new_pattern = np.where(new_pattern >= 0, 1, -1).reshape(current_pattern.shape)
        s_history.append(new_pattern)
        energy = calculate_energy(new_pattern, weight_matrix)
        energy_history.append(energy)

        if len(s_history) > 2 and np.array_equal(s_history[-2], s_history[-1]):
            break

        epoch += 1
        current_pattern = new_pattern

    return s_history, energy_history




