"""
A list of functions that are used for preprocessing and analyzing the data
"""
import os
import time

import networkx as nx
import numpy as np

def load_graph_file(edges_file_path, nodes_file_path, top_lines_to_remove, split_char='\t', print_details=False):
    """
    this function loads a graph file into memory as a networkx graph object
    the graph file should be a list of edges of the following format:

    node1 "Separator" node2

    :param edges_file_path: the graph file path
    :param top_lines_to_remove: the number of lines at the top of the file to be ignored
    :param split_char: the separation character between two nodes (default is TAB)
    :param print_details: print the details of the graph (number of nodes and number of edges)
    :return: Networkx graph object
    """

    edges_file = open(edges_file_path, "r")
    data = edges_file.read().split("\n")

    # remove top lines
    if top_lines_to_remove > 0:
        for index in range(0, top_lines_to_remove):
            data.pop(0)

    if nodes_file_path:
        nodes_file = open(nodes_file_path, "r")
        nodes_data = nodes_file.read().split("\n")
        nodes = [i for i in nodes_data if i != ""]
        nodes = sorted(set(nodes))
    else:
        nodes = [i.split(split_char, 1)[0] for i in data if i != ""]
        nodes = sorted(set(nodes))

    g = nx.Graph()
    g.add_nodes_from(nodes)

    # adding edges to the graph
    for index in range(0, len(data)):
        adjacent_nodes = data[index].split(split_char)
        if adjacent_nodes[0] == "":
            continue
        if adjacent_nodes[1] == "":
            continue
        node_1 = adjacent_nodes[0]
        node_2 = adjacent_nodes[1]

        g.add_edge(node_1, node_2)

    if print_details:
        print("graph loaded:")
        print("-------------")
        print("nodes:" + str(g.number_of_nodes()))
        print("edges:" + str(g.number_of_edges()))
        print("-------------")

    return g


def load_graph_directory(edges_directory_path, nodes_directory_path, files_extension, top_lines_to_remove,
                         split_char='\t',
                         print_details=False,
                         timesteps_graphs_file_path=""):
    """
    load a list of graphs from a directory
    the files are read one by one and a graph networkx object is created for each file

    :param edges_directory_path: the folder where the graph  edges  exist (Mendatory)
    :param edges_directory_path: the folder where the graph  nodes  exist (optional if nodes are derived from edges)
    :param files_extension: the extension of the graph files
    :param top_lines_to_remove: number of lines to be ignored from the top of each file
    :param split_char: the separation character used between nodes (node1 "SEPARATION" node 2)
    :param print_details: print the details of each graph
    :param timesteps_graphs_file_path: where to save the list of graphs as a numpy file
    :return: list of time steps where each timestep is a graph object
    """

    start = time.time()

    timesteps_graphs = []
    print("-----------------------------------")
    print("loading timesteps...")
    for file in os.listdir(edges_directory_path):
        if file.endswith(files_extension):

            if nodes_directory_path:
                nodes_file_path = os.path.join(nodes_directory_path, file)
            else:
                nodes_file_path = ""

            graph = load_graph_file(os.path.join(edges_directory_path, file), nodes_file_path, top_lines_to_remove,
                                    split_char,
                                    print_details)
            timesteps_graphs.append(graph)

    # save numpy file
    if timesteps_graphs_file_path:
        nx.write_gpickle(timesteps_graphs,timesteps_graphs_file_path)
        # np.save(timesteps_graphs_file_path, timesteps_graphs)

    end = time.time()
    print("completed: (" + str(len(timesteps_graphs)) + " timesteps in " + get_elapsed(start, end) + " )")
    print("-----------------------------------")

    return timesteps_graphs


def load_clusters_file(file_path, top_lines_to_remove):
    """
    load the list of clusters file
    each line is a cluster there the nodes are separated by commas and surrounded by brackets
    the file should be of the format:
    [node0,node1,...]
    [node2,node3,...]
    [...]

    :param file_path: clusters file path
    :param top_lines_to_remove: the number of lines to ignore from the top of the file
    :return: list of clusters for that time step
    """

    file = open(file_path, "r")
    data = file.read().split("\n")
    if top_lines_to_remove > 0:
        for index in range(0, top_lines_to_remove):
            data.pop(0)
    all_clusters = []
    for line in data:
        if line == "": continue
        if '\t' in line:
            clusters = line.split('\t')[1].replace('[', '').replace(']', '').replace(' ', '')
        else:
            clusters = line.replace('[', '').replace(']', '').replace(' ', '')

        clusters = clusters.split(',')
        clusters = [i for i in clusters]
        all_clusters.append(clusters)
    return all_clusters


def load_clusters_directory(directory_path, files_extension, top_lines_to_remove,
                            clusters_file_path=""):
    """
    reads a list of files in a directory where each file corresponds to a cluster of a time steps
    and extract the clusters from each file. the timesteps are read according to the order of the files in the directory.
    Each file should have the format:
    [node0,node1,...]
    [node2,node3,...]
    [...]

    :param directory_path: the clusters folder path
    :param files_extension: the extension of the cluster files
    :param top_lines_to_remove: the number of lines to ignore from the top of the file
    :param clusters_file_path: the path where to save numpy file of the clusters per timesteps
    :return: numpy array with the clusters of each timestep
    """
    start = time.time()
    all_clusters = []
    print("-----------------------------------")
    print("loading clusters...")
    for file in os.listdir(directory_path):
        if file.endswith(files_extension):
            cluster = load_clusters_file(os.path.join(directory_path, file), top_lines_to_remove)
            all_clusters.append(cluster)

    if clusters_file_path:
        np.save(clusters_file_path, all_clusters)

    end = time.time()
    count = sum([len(listElem) for listElem in all_clusters])
    print("completed: (" + str(count) + " clusters in " + get_elapsed(start, end) + " )")
    print("-----------------------------------")
    return all_clusters


def load_attributes_file(file_path, top_lines_to_remove):
    """
    load an attribute file related to a graph into memory
    :param file_path: the path of the attribute file
    :param top_lines_to_remove: the number of lines to ignore from the top of the file
    :return:list of attributes of the graph
    """
    file = open(file_path, "r")
    data = file.read().split("\n")
    if top_lines_to_remove > 0:
        for index in range(0, top_lines_to_remove):
            data.pop(0)
    all_attributes = []
    for line in data:
        if line == "": continue
        attributes = line.split(',')

        for index in range(0, len(attributes)):
            attributes[index] = attributes[index].replace('[', '').replace(']', '').replace(' ', '')
            attributes[index] = float(attributes[index])

        all_attributes.append(attributes)

    return all_attributes


def load_attributes_directory(directory_path, files_extension, top_lines_to_remove, attributes_file_path=""):
    """
    loads a directory that contains a list of attribute files (where each file contains the attributes that belong to a graph)
    :param directory_path: the attributes folder
    :param files_extension: the extension of the attribute files
    :param top_lines_to_remove: the number of lines to ignore from the top of the file
    :param attributes_file_path: where to save the numpy file containing the list of all attributes
    :return:
    """
    start = time.time()
    all_attributes = []
    print("-----------------------------------")
    print("loading attributes...")
    for file in os.listdir(directory_path):
        if file.endswith(files_extension):
            attributes = load_attributes_file(os.path.join(directory_path, file), top_lines_to_remove)
            all_attributes.append(attributes)

    if attributes_file_path:
        np.save(attributes_file_path, all_attributes)

    end = time.time()
    count = sum([len(listElem) for listElem in all_attributes])
    print("completed: (" + str(count) + " rows in " + get_elapsed(start, end) + " )")
    print("-----------------------------------")
    return all_attributes


def load_numpy_file(file_path):
    """
    load a numpy file into memory
    :param file_path: numpy file path
    :return: numpy array
    """
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    numpy_array = np.load(file_path)
    np.load = np_load_old
    return numpy_array


def get_elapsed(start, end):
    """
    used to calculate the time between two time stamps

    :param start: start time
    :param end: end time
    :return: a string in minutes or seconds for the elapsed time
    """
    elapsed = end - start
    if elapsed < 60:
        return '{0:.2g}'.format(end - start) + " seconds"
    else:
        return '{0:.2g}'.format((end - start) / 60.0) + " minutes"


