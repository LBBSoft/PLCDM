import numpy as np


def module_number_of_a_node(node, module_size):
    return node // module_size


def is_a_module_node(node, number_of_modules, module_size):
    if node < number_of_modules * module_size:
        return True
    else:
        return False


def in_same_module(node1, node2, module_size):
    return module_number_of_a_node(node1, module_size) == module_number_of_a_node(node2, module_size)


def single_modular_net_generation(
        n=100,
        number_of_modules=2,
        module_size=10,
        in_to_in_prob=0.5,
        out_to_in_prob=0.04,
        out_to_out_prob=0.01):
    nodes_lst = list(range(n))
    edge_lst = []

    modules = {}
    for i in range(number_of_modules):
        modules[i] = list(range(i * module_size, (i * module_size) + module_size))

    for i in nodes_lst:
        for j in nodes_lst:
            if i == j:
                continue

            if is_a_module_node(i, number_of_modules, module_size) \
                    and is_a_module_node(j, number_of_modules, module_size):
                if in_same_module(i, j, module_size):
                    val = np.random.choice([True, False], 1, p=[in_to_in_prob, 1 - in_to_in_prob])[0]
                    if val:
                        edge_lst.append((i, j))
                else:
                    if np.random.choice([True, False], 1, p=[out_to_in_prob, 1 - out_to_in_prob])[0]:
                        edge_lst.append((i, j))
            elif is_a_module_node(i, number_of_modules, module_size) \
                    or is_a_module_node(j, number_of_modules, module_size):
                if np.random.choice([True, False], 1, p=[out_to_in_prob, 1 - out_to_in_prob])[0]:
                    edge_lst.append((i, j))
            else:
                if np.random.choice([True, False], 1, p=[out_to_out_prob, 1 - out_to_out_prob])[0]:
                    edge_lst.append((i, j))

    return nodes_lst, edge_lst, modules





