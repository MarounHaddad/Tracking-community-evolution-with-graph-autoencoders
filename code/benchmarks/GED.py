"""
This file contains the implementation of GED: the method for group evolution discovery in social networks [Brodka et al. 2012]
"""
import time

import networkx as nx

import datapreparation.preprocess as pp
import datapreparation.utils as ut
import scores.scores as sc


def intersection(lst1, lst2):
    """Gets the list of intersecting nodes between two clusters"""
    return list(set(lst1) & set(lst2))


def degree_centrality(nodesdegreecent, nodes):
    """Filters and sums the degree centrality of a list of nodes"""
    value = 0
    for node in nodes:
        value += nodesdegreecent[node]
    return value


def quantity_measure(intersection_nodes, cluster1_nodes, cluster2_nodes):
    """
    calculates the jaccard index between the front and the compared cluster
    :param intersection_nodes: the nodes that are common between the Front and the Compared cluser
    :param cluster1_nodes: the nodes of the first cluster
    :param cluster2_nodes: the nodes of the second cluster
    :return: Quantity Measure (Jaccard Index)
    """
    return (len(intersection_nodes) / (len(cluster1_nodes) + len(cluster2_nodes) - len(intersection_nodes)))


def quality_measure(intersection_nodes, cluster1_nodes, nodesdegreecent):
    """
    Calculates the quality measure (which can be any centrality measure)
    In our case we choose Degree Centrality
    :param intersection_nodes: The nodes that are common between the Front and the compared cluster
    :param cluster1_nodes: The nodes of the first cluster
    :param nodesdegreecent: The degree centrality of the nodes
    :return: quality measure
    """
    intersectionquality = degree_centrality(nodesdegreecent, intersection_nodes)
    cluster1quality = degree_centrality(nodesdegreecent, cluster1_nodes)
    return intersectionquality / cluster1quality


def inclusion(nodesdegreecent, intersection_nodes, cluster1_nodes, cluster2_nodes):
    """
    Calculates the inclusion score of two clusters Which is the (quality score X the quantity score)
    Caculated twice inclusion (Front,Cluster) and inclusion (Cluster,Front)
    * Front is the last cluster in every sequence*
    :param nodesdegreecent: the degree centrality of the nodes
    :param intersection_nodes: the nodes that are common between the front and the cluster
    :param cluster1_nodes: the nodes of the first cluster
    :param cluster2_nodes: the nodes of the second cluster
    :return: Inclusion score
    """
    quantity = quantity_measure(intersection_nodes, cluster1_nodes, cluster2_nodes)
    quality = quality_measure(intersection_nodes, cluster1_nodes, nodesdegreecent)
    return quantity * quality


def get_fronts(sequences):
    """gets the last cluster in every sequence"""
    fronts = []
    for index in range(0, len(sequences)):
        fronts.append(sequences[index][-1])
    return fronts


def apply_GED_similarity(alpha, beta):
    """
    Generates the sequences of clusters using the Inclusion score
    The inclusion score is calculated twice:
    Inclusion(Front,Cluster)>alpha
    Inclusion(Cluster,Front)>beta
    The method gets the front of every sequence then calculates the inclusion between the front and the clusters
    of every snapshot to build the sequences.
    :param alpha: the similarity threshold for Inclusion(Front,Cluster)
    :param beta: the similarity threshold for Inclusion(Cluster,Front)
    """
    # normalize burt matrix

    sequences = []

    start = time.time()

    clusters_timestep0 = pp.get_clusters_in_timestep(0)

    # The clusters of the first snapshot are considered the start of the sequences.
    for index in range(0, len(clusters_timestep0)):
        sequences.append([index])

    # calculate the degree centrality of all the nodes
    nodesdegreecent = []
    for timestep in range(0, pp.number_timesteps):
        nodesdegreecent.append(nx.degree_centrality(pp.timesteps_graphs[timestep]))

    # loop through the snapshots (timesteps)
    for timestep in range(1, pp.number_timesteps):
        # get the fronts or the last cluster in every sequence
        fronts = get_fronts(sequences)
        for cluster in pp.get_clusters_in_timestep(timestep):
            sequence_index = 0
            new_sequence = True
            for front in fronts:
                # calculate the inclusion between the front and the cluster (two ways: inclusion (Front,cluster) and inclusion (cluster,Front)
                front_nodes = list(pp.clusters[pp.clusters_lookup[front][0]][pp.clusters_lookup[front][1]])
                cluster_nodes = list(pp.clusters[pp.clusters_lookup[cluster][0]][pp.clusters_lookup[cluster][1]])
                intersection_nodes = intersection(front_nodes, cluster_nodes)
                if inclusion(nodesdegreecent[pp.clusters_lookup[front][0]], intersection_nodes, front_nodes,
                             cluster_nodes) > alpha and \
                        inclusion(nodesdegreecent[pp.clusters_lookup[cluster][0]], intersection_nodes, cluster_nodes,
                                  front_nodes) > beta:
                    # if inclusion larger than threshholds add cluster to the sequence.
                    sequences[sequence_index].append(cluster)
                    new_sequence = False
                    break
                sequence_index += 1
            if new_sequence:
                sequences.append([cluster])

    end = time.time()
    print("========================================")
    print("                 GED                    ")
    print("========================================")
    print("\n alpha:", alpha, "beta:", beta)
    print("\n done in : ", ut.get_elapsed(start, end))
    sc.get_scores("GED", sequences)
