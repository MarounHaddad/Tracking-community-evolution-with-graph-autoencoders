"""Not used: Just for testing"""

import datapreparation.preprocess as pp
import numpy as np

data_directory = "..\\..\\data\\Dancer\\t"
pp.preprocess_data_DANCER(data_directory, True, True)


attribute_memory = {}

for timestep_index in range(0,len(pp.timesteps_graphs)):
    graph = pp.timesteps_graphs[timestep_index]
    node_list = np.array(list(graph.nodes()))
    for cluster in pp.clusters[timestep_index]:
        for node in cluster:
            node_index = np.where(node_list == node)[0][0]
            if node not in attribute_memory.keys():
                attribute_memory[node] = pp.attributes[timestep_index][node_index]
            else:
                pp.attributes[timestep_index][node_index] = [attribute +10.0 for attribute in attribute_memory[node]]

    with open("t"+str(timestep_index)+".graph", 'w') as f:
        for node_index in range(0,len(pp.attributes[timestep_index])):
            f.write("%s\n" % pp.attributes[timestep_index][node_index])