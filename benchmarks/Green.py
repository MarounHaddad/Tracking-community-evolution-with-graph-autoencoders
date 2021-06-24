"""
This file contains the implementation of "Tracking the evolution of communities in dynamic social networks"[Green et al. 2010]
The method relies on the Jaccard index to perform the matching.
Note: Unlike like the original paper, we do not allow for intersecting sequences to be more conform with the other benchmarks.
"""
import time

import datapreparation.preprocess as pp
import datapreparation.utils as ut
import scores.scores as sc


def jaccard_measure(i, j):
    """
    calculates the jaccard index for two different clusters
    it uses the burt matrix which contains info about the number of shared nodes
    between two clusters
    :param i: index of cluster 1
    :param j: index of cluster 2
    :return: jaccard index
    """
    return (pp.burt_matrix[i][j] / (pp.burt_matrix[i][i] + pp.burt_matrix[j][j] - pp.burt_matrix[i][j]))


def get_fronts(sequences):
    """
    Get the cluster in every sequence as the front of the sequence
    :param sequences: the list of the sequences
    :return: fronts
    """
    fronts = []
    for index in range(0, len(sequences)):
        fronts.append(sequences[index][-1])
    return fronts


def apply_green_similarity(threshold):
    """
    generates the sequences using the Jaccard index for matching
    :param threshold: the similarity threshold (overwhich the cluster is added to the sequence)
    :return:
    """
    sequences = []
    fronts = []

    start = time.time()

    clusters_timestep0 = pp.get_clusters_in_timestep(0)

    # conside every cluster in the first snapshot as a sequence
    for index in range(0, len(clusters_timestep0)):
        sequences.append([index])

    # loop through the snapshots (timesteps)
    for timestep in range(1, pp.number_timesteps):
        # get the fronts of every sequence
        fronts = get_fronts(sequences)
        for cluster in pp.get_clusters_in_timestep(timestep):
            sequence_index = 0
            new_sequence = True
            for front in fronts:
                if jaccard_measure(front, cluster) > threshold:
                    # if the jaccard measure is larger than the threshold add the cluster to the sequence
                    sequences[sequence_index].append(cluster)
                    new_sequence = False
                    break
                sequence_index += 1
            if new_sequence:
                sequences.append([cluster])

    end = time.time()
    print("========================================")
    print("                 GREEN                  ")
    print("========================================")

    print("\n Threshold:", threshold)

    print("\n done in : ", ut.get_elapsed(start, end))
    sc.print_predicted_sequences(sequences)
    sc.get_scores("GREEN", sequences)
