from single_layer_modular_graph_generation import *
from multilayer_simulation_from_single_layer_graph import *
from utitlities import *

import networkx as nx
import random as rd
import math


def make_multilayer(layer_data, node_lst, interlayer_factor=1, directed=False):
    multilayer_network = {}
    intralayer_transition_probabilities = {}
    interlayer_transition_probabilities = {}

    for i in layer_data.keys():
        if directed:
            multilayer_network[i] = nx.DiGraph()
        else:
            multilayer_network[i] = nx.Graph()

        for edge in layer_data[i]:
            multilayer_network[i].add_edge(edge[0], edge[1])
        multilayer_network[i].add_nodes_from(node_lst)

        # making transition probability matrix for each layer initialized by zero
        intralayer_transition_probabilities[i] = [0 for x in range(len(node_lst))]
        for j in range(len(node_lst)):
            intralayer_transition_probabilities[i][j] = [0 for x in range(len(node_lst))]

        for node in node_lst:
            node_neighbors = list(multilayer_network[i].neighbors(node))
            for neighbor in node_neighbors:
                if intralayer_transition_probabilities[i][node][neighbor] != 0:
                    continue
                neighbor_neighbors = list(multilayer_network[i].neighbors(neighbor))
                common_neighbors = set(node_neighbors) & set(neighbor_neighbors)
                total_weight = 0
                for l in layer_data:
                    if (node, neighbor) in layer_data[l] or (neighbor, node) in layer_data[l]:
                        total_weight += 1
                total_weight = (total_weight + 1) / (len(layer_data.keys()) + 2)
                total_weight = total_weight * math.log2(total_weight)
                probability = total_weight * ((len(common_neighbors) + 1) / (len(node_neighbors) + 1)) * ((len(common_neighbors) + 1) / (len(neighbor_neighbors) + 1))
                intralayer_transition_probabilities[i][node][neighbor] = round(probability, 2)

    for n in node_lst:
        interlayer_transition_probabilities[n] = []
        for l in list(layer_data.keys()):
            interlayer_transition_probabilities[n].append([0 for x in range(len(layer_data.keys()))])

    for n in node_lst:
        for l1 in list(layer_data.keys()):
            for l2 in list(layer_data.keys()):
                if l1 == l2:
                    continue
                l1_neighbors = set(multilayer_network[l1].neighbors(n))
                l2_neighbors = set(multilayer_network[l2].neighbors(n))
                common_nodes = l1_neighbors & l2_neighbors
                if len(l1_neighbors | l2_neighbors) == 0:
                    interlayer_transition_probabilities[n][l1][l2] = 0
                    interlayer_transition_probabilities[n][l2][l1] = 0
                else:
                    interlayer_transition_probabilities[n][l1][l2] = \
                        interlayer_transition_probabilities[n][l2][l1] = \
                        ((len(common_nodes) + 1) / (len(l1_neighbors | l2_neighbors) + 1)) * interlayer_factor

    return multilayer_network, intralayer_transition_probabilities, interlayer_transition_probabilities


def random_walk(multilayer_network,
                intralayer_transition_probabilities,
                interlayer_transition_probabilities,
                node_lst,
                seed,
                iteration_count=1000,
                layer_change_prob=0.3,
                random_jump_prob=0.3):
    layer_lst = sorted(list(multilayer_network.keys()))
    scores = {}

    for i in layer_lst:
        scores[i] = {}

    for i in layer_lst:
        for x in node_lst:
            scores[i][x] = 0

    current_layer = rd.choice(layer_lst)
    current_node = seed
    scores[current_layer][current_node] += 1

    # region checking if seed is alone in all layers
    seed_is_alone = True
    for l in layer_lst:
        if len(list(multilayer_network[l].neighbors(seed))) != 0:
            seed_is_alone = False
    if seed_is_alone:
        return scores
    # endregion

    for i in range(iteration_count):
        while len(list(multilayer_network[current_layer].neighbors(current_node))) == 0:
            current_node = seed
            layer_probs = interlayer_transition_probabilities[current_node][current_layer]
            if sum(layer_probs) == 0:
                print('yes')
            layer_probs = [x / sum(layer_probs) for x in layer_probs]
            layer_probs = [math.pow(x, 2) for x in layer_probs]
            layer_probs = [x / sum(layer_probs) for x in layer_probs]
            current_layer = np.random.choice(layer_lst, 1, p=layer_probs)[0]
        else:
            neighbors_trans_probs = intralayer_transition_probabilities[current_layer][current_node]
            neighbors_trans_probs = [x / sum(neighbors_trans_probs) for x in neighbors_trans_probs]
            neighbors_trans_probs = [math.pow(x, 2) for x in neighbors_trans_probs]
            neighbors_trans_probs = [x / sum(neighbors_trans_probs) for x in neighbors_trans_probs]
            current_node = np.random.choice(node_lst, 1, p=neighbors_trans_probs)[0]

        scores[current_layer][current_node] += 1

        layer_change = np.random.choice([True, False], 1, p=[layer_change_prob, 1 - layer_change_prob])[0]
        if layer_change:
            layer_probs = interlayer_transition_probabilities[current_node][current_layer]
            if sum(layer_probs) != 0:
                layer_probs = [x / sum(layer_probs) for x in layer_probs]
                layer_probs = [math.pow(x, 2) for x in layer_probs]
                layer_probs = [x / sum(layer_probs) for x in layer_probs]
                current_layer = np.random.choice(layer_lst, 1, p=layer_probs)[0]

        rj_prob = np.random.choice([True, False], 1, p=[random_jump_prob, 1 - random_jump_prob])[0]
        if rj_prob:
            current_node = seed
            scores[current_layer][current_node] += 1
    return scores


def module_selection(multilayer_scores, nodes_lst, layer_data):
    scores = {}
    for n in nodes_lst:
        temp = 0
        for l in list(layer_data.keys()):
            temp += multilayer_scores[l][n]
        scores[n] = temp

    avg_score = np.mean(list(scores.values()))
    std = np.std(list(scores.values()))

    transformed_scores = {}
    for item in scores:
        transformed_scores[item] = (scores[item] - avg_score) / std

    final_module = []
    for item in transformed_scores.keys():
        if transformed_scores[item] > 0:
            final_module.append(item)

    return final_module, transformed_scores

# sample 1: community detection for the multilayer network simulated from single modular graph generation algorithm

nodes_lst, edge_lst, modules = single_modular_net_generation(100, 2, 15, in_to_in_prob=0.8)
layer_data = multilayer_generation(edge_lst, 5, edge_selection_prob=.5)
multilayer_network, intralayer_transition_probabilities, interlayer_transition_probabilities = make_multilayer(
    layer_data, nodes_lst, interlayer_factor=1,directed=False)
multilayer_scores = random_walk(multilayer_network, intralayer_transition_probabilities,
                                interlayer_transition_probabilities, nodes_lst, seed=20,
                                iteration_count=1000, random_jump_prob=.5, layer_change_prob=0.5)
result_module, scores = module_selection(multilayer_scores, nodes_lst, layer_data)
print(result_module)

# sample 2: community detection for the multilayer network (from file)

nodes, edge_lst, layer_data = load_multilayer_network('multilayer.tsv')
nodes_lst = sorted(list(nodes.values()))

multilayer_network, intralayer_transition_probabilities, interlayer_transition_probabilities = make_multilayer(
        layer_data, nodes_lst, interlayer_factor=1)

custom_seed = nodes['IGKV3-20']
multilayer_scores = random_walk(multilayer_network, intralayer_transition_probabilities,
                                interlayer_transition_probabilities, nodes_lst, seed=custom_seed,
                                iteration_count=1000, random_jump_prob=.7, layer_change_prob=0.5)
del multilayer_network, intralayer_transition_probabilities, interlayer_transition_probabilities
result_module, scores = module_selection(multilayer_scores, nodes_lst, layer_data)

names = {}
for n in nodes:
    names[nodes[n]] = n
for item in result_module:
    print(names[item], end= ' ')
