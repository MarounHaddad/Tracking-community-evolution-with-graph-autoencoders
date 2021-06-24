"""
Used for experiments only
Generate the sequences by applying markov clustering on the supergraph using the embeddings generated in the pretraining step
"""
import networkx as nx

import datapreparation.cluster as cl
import datapreparation.preprocess as pp
import scores.scores as sc


def extract_sequences_by_clustering():
    g = nx.from_numpy_array(pp.burt_matrix)
    enhanced_g = cl.KNN_enanched(g, pp.all_clusters_embeddings, 1)
    predicted_sequences = cl.cluster_markov(enhanced_g, 2, True)
    sc.get_scores("MARKOV", predicted_sequences)
