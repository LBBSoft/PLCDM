import numpy as np

def multilayer_generation(edge_list, number_of_layers, edge_selection_prob=0.8):
    layer_data = {}
    for layer in range(number_of_layers):
        layer_data[layer] = []

    for edge in edge_list:
        for i in range(number_of_layers):
            if np.random.choice([True, False], 1, p=[edge_selection_prob, 1 - edge_selection_prob])[0]:
                layer_data[i].append(edge)
    return layer_data

