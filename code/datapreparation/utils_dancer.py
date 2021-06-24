"""
# This file contains a list of functions for extracting
# the Dancer files into files with a format that is readable by Trackgcn
"""

import os
import time
import datapreparation.utils as ut
import shutil
import numpy as np




def extract_data_directory(data_directory, ground_truth_file_path, files_extension, split_char=';'):
    """
    this function takes a directory containing a list of graph files generated by Dancer and extracts the details
    of each file:
            - the graph of each timestep
            - the attributes of each graph
            - the clusters of each timestep
            - ground truth series
    :param data_directory: the path of the graphs generated by Dancer (file.graph)
    :param ground_truth_file_path: where to save the ground truth numpy file extracted from the Dancer files
    :param files_extension: the extension of the Dancer files (default .graph)
    :param split_char: the separation character used in the dancer file (default ";")
    """

    ground_truth_clusters = []
    ground_truth_text_file_name = data_directory + "\\groundtruth.groundtruth"
    if not os.path.isdir(data_directory + "\\clusters"):
        os.mkdir(data_directory + "\\clusters")
    if not os.path.isdir(data_directory + "\\attributes"):
        os.mkdir(data_directory + "\\attributes")
    if not os.path.isdir(data_directory + "\\nodes"):
        os.mkdir(data_directory + "\\nodes")
    if not os.path.isdir(data_directory + "\\edges"):
        os.mkdir(data_directory + "\\edges")
    if not os.path.isdir(data_directory + "\\archive"):
        os.mkdir(data_directory + "\\archive")

    start = time.time()
    timesteps_graphs = []
    print("-----------------------------------")
    print("extracting Dancer data...")

    # for through each Dancer file in the directory and extract its data
    for file in os.listdir(data_directory):
        if file.endswith(files_extension):
            ground_truth_clusters.extend(extract_data(os.path.join(data_directory, file)))

    # write the ground truth text file readable by humans (groundtruth.groundtruth)
    with open(ground_truth_text_file_name, 'w') as f:
        for cluster in ground_truth_clusters:
            f.write("%s\n" % ','.join(map(repr, cluster)))

    # write the ground truth numpy file
    np.save(ground_truth_file_path, ground_truth_clusters)

    end = time.time()
    print("completed: (" + str(len(timesteps_graphs)) + " timesteps in " + ut.get_elapsed(start, end) + " )")
    print("-----------------------------------")


def extract_data(file_path, split_char=';'):
    """
    this function extracts the info in one graph file generated by Dancer
    the Dancer file contains:
        - the attributes
        - the graph info: list of edges
        - the cluster to which belong each node

    :param file_path: the .graph file generated by Dancer
    :param split_char: the separation character used in the dancer file (default ";")
    :return: list of clusters for this timestep (the value of each clusters is the series ID to which the cluster belong)
    """

    file = open(file_path, "r")
    data = file.read().split("\n")
    # dictionary of clusters where the key is the cluster ID and the value is a list of nodes
    clusters = {}
    # dictionary of attributes where the key is the node ID and the value is a list of attributes
    attributes = {}
    edges = []
    ground_truth_clusters = []
    nodes = []

    # extract attributes and clusters
    # Dancer file format: node_ID;attribute1|attribute2|...|attributeN;cluster_ID
    nodex_index = 0
    for index in range(0, len(data)):
        if data[index] == "# Vertices":
            continue
        if data[index] == "#":
            nodex_index = index + 3
            break
        columns = data[index].split(split_char)
        if columns[2] not in clusters.keys():
            clusters[columns[2]] = []
        clusters[columns[2]].append(int(columns[0]))
        attributes[columns[0]] = [float(x) for x in columns[1].split("|")]
        nodes.append(columns[0])

    clusters_file_name = os.path.dirname(os.path.realpath(file_path)) + "\\clusters\\" + os.path.basename(file_path)
    attributes_file_name = os.path.dirname(os.path.realpath(file_path)) + "\\attributes\\" + os.path.basename(file_path)
    graph_file_name = os.path.dirname(os.path.realpath(file_path)) + "\\" + "graph_" + os.path.basename(file_path)
    archive_file_name = os.path.dirname(os.path.realpath(file_path)) + "\\archive\\" + os.path.basename(file_path)
    nodes_file_name = os.path.dirname(os.path.realpath(file_path)) + "\\nodes\\" + os.path.basename(file_path)
    edges_file_name = os.path.dirname(os.path.realpath(file_path)) + "\\edges\\" + os.path.basename(file_path)

    # extract edges for the graph file
    for index in range(nodex_index, len(data)):
        if (data[index]) == "": continue
        edges.append([int(x) for x in data[index].split(split_char)])

    # write clusters file
    with open(clusters_file_name, 'w') as f:
        for cluster_id in clusters.keys():
            f.write("%s\n" % clusters[cluster_id])
            ground_truth_clusters.append(cluster_id)

    # write attributes file
    with open(attributes_file_name, 'w') as f:
        for node_id in sorted(attributes.keys()):
            f.write("%s\n" % attributes[node_id])

    # write edges file
    with open(edges_file_name, 'w') as f:
        for edge in edges:
            f.write("%s\n" % ','.join(map(repr, edge)))

    # write nodes file
    with open(nodes_file_name, 'w') as f:
        for node_id in sorted(nodes):
            f.write("%s\n" % int(node_id))

    file.close()

    # save the old Dancer graph file into an archive folder
    shutil.move(file_path, archive_file_name)
    return ground_truth_clusters