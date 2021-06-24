"""
Silhouette Jaccard: new score introduced in this work that evaluates the sequences using the Jaccard index
The Jaccard is calculated using the silhouette method which penalizes the false membership of the clusters in the sequences.
i.e. if the cluster has on average a Jaccard index that is higher with clusters in other sequences than its own the score will diminish.
"""

# libaries
import numpy as np


def silhouette_jaccard_sequence(cluster, sequence, burt):
    """
    calculates the mean jaccard index between a cluster and all other clusters in a certain sequence
    :param cluster: the target cluster
    :param sequence: the sequence of the current cluster
    :param burt: the burt matrix
    :return:
    """
    return np.mean([
        float(burt[cluster][other_cluster]) / float(
            burt[other_cluster][other_cluster] + burt[cluster][cluster] - burt[cluster][other_cluster]) for
        other_cluster in sequence if other_cluster != cluster])


def silhouette_jaccard_sample(cluster, cluster_sequence, other_sequences, burt):
    """
    Calculates the Jaccard sihouette score for a cluster

    :param cluster: The cluster to whom we want to calculate the jaccrd sihouette score
    :param cluster_sequence: the sequence of the current cluster
    :param other_sequences: The list of all other sequences
    :param burt: the burt matrix
    :return: the silhouette score of a single cluster
    """
    intra_sequence = silhouette_jaccard_sequence(cluster, cluster_sequence, burt)
    inter_sequence_values = []
    for other_sequence in other_sequences:
        inter_sequence_values.append(silhouette_jaccard_sequence(cluster, other_sequence, burt))
    inter_sequence = float(np.max(inter_sequence_values))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (intra_sequence - inter_sequence) / np.maximum(intra_sequence, inter_sequence)


def silhouette_jaccard_samples(sequences, burt):
    """
    Calculates the Jaccard for an entire sequence
    :param sequences: The list of sequences
    :param burt: The burt matrix
    :return: the silhouette score for one sequence
    """
    samples = [0] * len(burt)
    sequence_index = 0
    for sequence in sequences:
        if len(sequence) == 1:
            samples[sequence[0]] = 0
            sequence_index += 1
            continue
        for cluster in sequence:
            other_sequences = [other_sequence for i, other_sequence in enumerate(sequences) if
                               i != sequence_index]
            samples[cluster] = silhouette_jaccard_sample(cluster, sequence, other_sequences, burt)
        sequence_index += 1
    return samples


def silhouette_jaccard_score(sequences, burt, print_result=True):
    """
    Calculates the Silhouette Jaccard score
    (Main function to be called)
    :param sequences: The list of generated sequences
    :param burt: the burt matrix (used to calculate the Jaccard index)
    :param print_result: if should print the output or not
    :return: silhouette jaccard
    """
    # The score is the mean of the scores of all the sequences
    score = np.mean(silhouette_jaccard_samples(sequences, burt))
    if print_result:
        print("\n Silhouette Jaccard: ", round(score, 3))
    return score
