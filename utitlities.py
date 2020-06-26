
def load_multilayer_network(input_file):
    layer_data = {}
    nodes_lst = []
    edge_lst = []
    nodes = {}

    with open(input_file, 'r') as _file:
        for line in _file:
            line = line.strip()
            data = [x.strip() for x in line.split('\t')]
            nodes_lst.append(data[0])
            nodes_lst.append(data[1])
    _file.close()

    nodes_lst = list(set(nodes_lst))
    for i in range(len(nodes_lst)):
        nodes[nodes_lst[i]] = i

    with open(input_file, 'r') as _file:
        for line in _file:
            line = line.strip()
            data = [x.strip() for x in line.split('\t')]
            if int(data[2]) not in layer_data.keys():
                layer_data[int(data[2])] = []
            layer_data[int(data[2])].append((nodes[data[0]], nodes[data[1]]))
            edge_lst.append((nodes[data[0]], nodes[data[1]]))
    _file.close()

    return nodes, edge_lst, layer_data

