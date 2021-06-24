"""
An implementation of the Netsimile features for clusters topological structure comparision.
Reference: NetSimile: A Scalable Approach to Size-Independent Network Similarity [Berlingerio et al. 2012]
"""

# libraries
import time
import networkx as nx
import numpy as np
from scipy import stats as st

from datapreparation import utils as ut


def average_neighbors_clustering(graph):
    """
    Calculates the average clustering score of the neighbors of each node
    :param graph: source graph
    :return: average neighbors clustering (calculated per node)
    """
    average_neighbors_clustering = {}
    for node in graph.nodes:
        neighbors_clustering = nx.clustering(graph, graph.neighbors(node))
        average = np.mean(list(neighbors_clustering.values()), axis=0)
        average_neighbors_clustering[node] = round(average, 5)
    return average_neighbors_clustering


def number_edges_egonet(graph):
    """
    calculates the number of edges in the egonet
    (egonet: the subgraph formed by the node and its neighbors)
    :param graph: source graph
    :return: number of edges of the egonet (calculated per node)
    """
    number_edges_egonet = {}
    for node in graph.nodes:
        egonet = nx.subgraph(graph, graph.neighbors(node))
        number_edges_egonet[node] = len(egonet.edges())
    return number_edges_egonet


def number_out_edges_egonet(graph):
    """
    calculates the number of out edeges of the egonets
    (the edges that connect the egonet subgraph to other nodes)
    :param graph: source graph
    :return: number of out edges of the egonet (calculated per node)
    """
    number_out_edges_egonet = {}
    for node in graph.nodes:
        egonet = nx.subgraph(graph, graph.neighbors(node))
        inner_edges = len(egonet.edges())
        all_edges = len(graph.edges(graph.neighbors(node)))
        number_out_edges_egonet[node] = all_edges - inner_edges
    return number_out_edges_egonet


def number_neighbors_egonet(graph):
    """
    the number of nodes that are connected to the egonet that are not part of the egonet.
    (i.e. the nodes connected to the neighbors of the node and are not connected to latter)
    :param graph: source graph
    :return:  (calculated per node)
    """
    number_neighbors_egonet = {}
    for node in graph.nodes:
        all_egonet_edges = graph.edges(graph.neighbors(node))
        all_nodes = [inner_node for edge in all_egonet_edges for inner_node in edge]
        all_nodes = list(set(all_nodes))
        number_neighbors_egonet[node] = len([x for x in all_nodes if x not in graph.neighbors(node)])
    return number_neighbors_egonet


def calculate_features(graph):
    """
    Calculate topological features
    """
    topo_features = []
    topo_features.append(dict(nx.degree(graph)))
    topo_features.append(nx.clustering(graph))
    # topo_features.append(nx.average_neighbor_degree(graph))
    # topo_features.append(average_neighbors_clustering(graph))
    # topo_features.append(number_edges_egonet(graph))
    # topo_features.append(number_out_edges_egonet(graph))
    # topo_features.append(number_neighbors_egonet(graph))
    topo_features.append(nx.katz_centrality_numpy(graph))
    topo_features.append(nx.betweenness_centrality(graph))
    return topo_features


def build_nodes_features(graph):
    """
    stack the features of the node
    :param graph:
    :return:
    """
    topo_features = calculate_features(graph)
    nodes_topo_features = []
    for node in graph.nodes():
        node_topo_features = []
        for feature in topo_features:
            node_topo_features.append(feature[node])
        nodes_topo_features.append(node_topo_features)

    return nodes_topo_features


def build_clusters_netsimile_features(data_directory, clusters, timesteps_graphs):
    """
    calculate the clusters Netsimile features
    The NetSimile features are caclculated by first calculating topological features for the node to form vectors of features.
    Then for every cluster the feature vectors of the nodes are aggregated using multiple aggregation functions (Mean, Median, ...)
    The aggregated features of every aggregation function are then concatenated to form the Netsimile features.
    :param data_directory:
    :param clusters:
    :param timesteps_graphs:
    :return:
    """
    start = time.time()
    print("-----------------------------------")
    print("calculating NetSimile Features...")

    clusters_features = []
    timestep_index = 0
    # Loop through all the snapshots
    for timestep in clusters:
        graph = timesteps_graphs[timestep_index]
        node_list = np.array(list(graph.nodes()))
        nodes_topo_features = build_nodes_features(graph)
        # loop through the clusters in the snapshot
        for cluster in timestep:
            cluster_topo_features = []
            # calculate the features for every node in the cluster
            for node in cluster:
                cluster_topo_features.append(nodes_topo_features[np.where(node_list == node)[0][0]])
            mean_features = np.mean(cluster_topo_features, axis=0)
            median_features = np.median(cluster_topo_features, axis=0)
            std_features = np.std(cluster_topo_features, axis=0)
            skew_features = st.skew(cluster_topo_features, axis=0)
            kurtosis_features = st.kurtosis(cluster_topo_features, axis=0)
            # concatenate the aggregations to form the netsimile features
            clusters_features.append(
                mean_features + median_features + std_features + skew_features + kurtosis_features)
        timestep_index += 1

    # save the numpy array
    np.save(data_directory + "\\processed\\clusters_topo_features", clusters_features)

    end = time.time()
    print("completetd: ( in " + ut.get_elapsed(start, end) + " )")
    print("-----------------------------------")
    return clusters_features
