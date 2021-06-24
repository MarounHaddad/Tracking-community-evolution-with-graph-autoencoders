"""
This file contains the list of functions used to generate the clusters for every snapshot
"""

# libraries
import time

import infomap
import markov_clustering as mc
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
from cdlib.algorithms import ilouvain

import datapreparation.utils as ut


def cluster_markov(g, inflation=2, print_details=False):
    """
    this function clusters the graph of a timestep using Markov clustering
    :param g: networkx graph
    :param inflation: parameter used by Markov clustering
    :param print_details: if True the system will print the clusters
    :return: a list of clusters
    """

    mc_cluster = mc.run_mcl(nx.to_scipy_sparse_matrix(g), inflation=inflation)
    clusters = mc.get_clusters(mc_cluster)

    for k in range(0, len(clusters)):
        nodes = list(clusters[k])

        for item in range(0, len(nodes)):
            nodes[item] = list(g.nodes())[nodes[item]]
        clusters[k] = tuple(nodes)

    clusters = [cluster for cluster in clusters if len(cluster) > 4]

    if print_details:
        for k in range(len(clusters)):
            print(k, len(clusters[k]), clusters[k])

    return clusters


def cluster_infomap(g, print_details=False):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id and return number of communities found.
    """

    infomapWrapper = infomap.Infomap()

    print("Building Infomap network from a NetworkX graph...")
    nodes_list = np.array(list(g.nodes()))
    for e in g.edges():
        infomapWrapper.addLink(np.where(nodes_list == e[0])[0][0],np.where(nodes_list == e[1])[0][0])

    print("Find communities with Infomap...")
    infomapWrapper.run();
    clusters = []
    clusters.append([])
    for node in infomapWrapper.iterTree():
        if node.isLeaf():
            if node.moduleIndex() > len(clusters) - 1:
                clusters.append([])
            clusters[node.moduleIndex()].append(nodes_list[node.physicalId])

    clusters = [cluster for cluster in clusters if len(cluster) > 4]

    if print_details:
        for k in range(len(clusters)):
            print(k, len(clusters[k]), clusters[k])
    return clusters

def cluster_ilouvain(g,attributes, print_details=False):
    """Partition the graph with the louvain algorithm"""
    labels = dict()
    node_index = 0
    for node in g.nodes():
        labels[node] = {"l" + str(k): v for k, v in enumerate(attributes[node_index])}
        node_index += 1

    id = dict()
    for n in g.nodes():
        id[n] = n

    communities = ilouvain(g, labels, id)
    clusters = [cluster for cluster in communities.communities if len(cluster) > 4]
    if print_details:
        for k in range(len(clusters)):
            print(k, len(clusters[k]), clusters[k])
    return clusters

def cluster_multiple_timesteps(timesteps_graphs, attributes, inflation, print_details=False, clusters_file_path=""):
    """
    this function takes a list of graphs and applies Markov clustering to each graph and saves the results

    :param timesteps_graphs: list of networkx graphs (a graph of every timestep)
    :param inflation:  parameter used by Markov clustering
    :param print_details: if true the system will print the details of each cluster
    :param clusters_file_path: the file path where the result will be saved
    :return: a list of timesteps where each timestep is a list of clusters
    """

    start = time.time()

    clusters = []
    timestep_index = 0

    print("-----------------------------------")
    print("clustering timesteps...")

    for g in timesteps_graphs:
        if print_details:
            print("")
            print("-------------")
            print("Timestep: " + str(timestep_index))
            print("-------------")
        # g = KNN_enanched(g,attributes[timestep_index],50)
        # clusters.append(cluster_ilouvain(g, attributes[timestep_index], print_details))
        clusters.append(cluster_markov(g, inflation, print_details))
        # clusters.append(cluster_infomap(g, print_details))
        timestep_index += 1

    if clusters_file_path:
        np.save(clusters_file_path, clusters)

    end = time.time()
    print("completed: (" + str(len(timesteps_graphs)) + " timesteps in " + ut.get_elapsed(start, end) + " )")
    print("-----------------------------------")

    return clusters


def build_clusters_lookup(clusters, clusters_lookup_file_path=""):
    """
    this function takes a list of clusters (for all timesteps) and builds a lookup wit the following info:
    timestep ID | cluster index in this timestep | cluster size
    the lookup is saved under the path specified by clusters_lookup_file_path

    :param clusters: list of timesteps where each timestep is a list of clusters
    :param clusters_lookup_file_path: where to save the lookup
    :return: clusters lookup (a list with all the clusters)
    """
    start = time.time()
    clusters_lookup = []

    print("-----------------------------------")
    print("building clusters lookup...")

    for timestep in range(0, len(clusters)):
        timestep_cluster_index = 0
        for cluster in clusters[timestep]:
            clusters_lookup.append([timestep, timestep_cluster_index, len(cluster)])
            timestep_cluster_index += 1

    if clusters_lookup_file_path:
        np.save(clusters_lookup_file_path, clusters_lookup)

    end = time.time()
    print("completed: (" + str(len(clusters_lookup)) + " clusters in " + ut.get_elapsed(start, end) + " )")
    print("-----------------------------------")

    return clusters_lookup


def print_timesteps_clusters(clusters):
    """
    this function takes a list of timesteps and prints their details
    overall cluster index | timestep cluster index | cluster size

    :param clusters: list of all timesteps with their clusters
    """
    overall_index = 0
    for timestep in range(0, len(clusters)):
        print("")
        print("--------------")
        print("Timestep:" + str(timestep))
        print("--------------")
        timestep_index = 0
        for cluster in clusters[timestep]:
            print("O: " + str(overall_index) + " \t T: " + str(timestep_index) + " \t size: " + str(
                len(cluster)) + " \t ", cluster)
            timestep_index += 1
            overall_index += 1


def KNN_enanched(g, attributes, k):
    """This function augments the graph with more edeges based on the proximity of the attributes
    before applying the clustering
    (For testing only: not used in the final implementation)
    """
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(attributes)
    A = neigh.kneighbors_graph(attributes)
    A = A.toarray()
    g_KNN = nx.from_numpy_array(A)
    enhanced_g = g.copy(as_view=False)
    nodes_list = list(enhanced_g.nodes())
    for edge in g_KNN.edges():
        enhanced_g.add_edge(nodes_list[edge[0]], nodes_list[edge[1]],weight=1)
    return enhanced_g