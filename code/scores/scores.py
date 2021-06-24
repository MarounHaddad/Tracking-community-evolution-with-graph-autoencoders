"""
This file contains the list of score functions for evaluating the sequences
"""

# Libraries
import os

import numpy as np
import scipy.stats as st
import sklearn.metrics as sk
from tabulate import tabulate

import datapreparation.preprocess as pp
import track.events as ev
import scores.silhouette_jaccard as sj

# the number of decimal points to be printed
precision = 3

dic_score = {}
dic_sequences = {}
dic_events = {}
dic_events_count = {}

def adjusted_rand_score(predicted_labels, print_details=True):
    """
    calculates the adjusted random score for the predicted sequences of clusters with the ground truth sequences
    :param ground_truth: ground truth sequences
    :param predicted_sequences: predicted sequences of clusters
    :return: ajusted random score
    """
    adj_rand_score = sk.adjusted_rand_score(pp.ground_truth, predicted_labels)
    if print_details:
        print("\n Adjusted random index score : ", round(adj_rand_score, precision))

    return adj_rand_score

def adjusted_mutual_information_score(predicted_labels):
    """
    Calculates the mutual information score of the predicted sequences and the ground truth sequences
    Used only when the ground truth sequences are provided
    :param predicted_labels:
    :return:
    """
    mutual_inf_score = sk.adjusted_mutual_info_score(pp.ground_truth, predicted_labels)
    print("\n Adjusted mutual Information score : ", round(mutual_inf_score, precision))
    return mutual_inf_score

def pearson_correlation(predicted_sequences, print_result=True):
    """
    This function calculates the average  correlation between the burt vectors of two clusters in the sequence.
    :param predicted_sequences:
    :param print_result:
    :return:
    """
    predicted_sequences_sizeL2 = [seq for seq in predicted_sequences if len(seq) > 1]
    if len(predicted_sequences_sizeL2) == 0:
        pearson_cor_all = -1
    else:
        number_clusters = sum(len(seq) for seq in predicted_sequences_sizeL2)
        pearson_cor_all = 0
        sequence_index = 0
        for seq in predicted_sequences_sizeL2:
            pearson_cor = 0
            for clust in seq:
                if clust == seq[0]:
                    continue
                pearson_cor += st.pearsonr(pp.burt_norm[seq[0]], pp.burt_norm[clust])[0]
            pearson_cor_all += pearson_cor
            sequence_index += 1
        pearson_cor_all = pearson_cor_all / number_clusters
    if print_result:
        print("\n Pearson correlation : ", round(pearson_cor_all, precision))
    return pearson_cor_all

def apnp(predicted_sequences, print_result=True):
    """
    Average proportion of nodes persisting in the clusters.
    This function calculates the number of nodes that are being preserved on average from cluster
    to cluster in the sequence
    :param predicted_sequences: the list of predicted sequences
    :param print_result: Print the output of the score
    :return: Average proportion of nodes persisting
    """
    predicted_sequences_sizeL2 = [seq for seq in predicted_sequences if len(seq) > 1]
    if len(predicted_sequences_sizeL2) == 0:
        apnp_all = -1
    else:
        number_clusters = sum(len(x) for x in predicted_sequences_sizeL2)
        apnp_all = 0
        for seq in predicted_sequences_sizeL2:
            apnp_num = 0
            for clust in seq:
                if clust == seq[0]:
                    continue
                apnp_num += pp.burt_matrix[seq[0]][clust]
            apnp_seq = (apnp_num / pp.burt_matrix[seq[0]][seq[0]])
            apnp_all += apnp_seq
        apnp_all = apnp_all / number_clusters
    if print_result:
        print("\n Average Proportion of Nodes Persisting: ", round(apnp_all, precision))
    return apnp_all


def silhouette_score(predicted_labels, data, metric, print_result=True):
    """
    Calculates the silhouette score of the sequence in reference to some features of the clusters
    :param predicted_labels: the sequences
    :param data: the features (could be attributes or Netsimile)
    :param metric: the distance used in the score
    :param print_result: whether to print the output
    :return:
    """
    if len(set(predicted_labels)) == len(predicted_labels):
        silhouette = -1
    else:
        silhouette = sk.silhouette_score(data, predicted_labels,metric=metric)
    if print_result:
        print("\n Silhouette Score: ", round(silhouette, precision))
    return silhouette


def get_scores(model_name, predicted_sequences):
    """
    Calculate all the scores for the generated sequences.
    :param model_name: the name of the tracking model
    :param predicted_sequences: the generated sequences
    """
    global dic_score
    global dic_sequences
    global dic_events
    global dic_events_count

    predicted_labels = convert_predicted_sequences_to_labels(predicted_sequences)
    events = ev.get_events(predicted_sequences)

    if model_name not in dic_score.keys():
        dic_sequences[model_name] = []
        dic_events[model_name] = []
        dic_score[model_name] = {}
        dic_events_count[model_name] = {}

        if pp.with_ground_truth_sequences:
            dic_score[model_name]["adjusted_rand_score"] = []
            dic_score[model_name]["adjusted_mutual_information"] = []

        dic_score[model_name]["silhouette_score_attributes"] = []
        dic_score[model_name]["silhouette_score_topo_features"] = []
        dic_score[model_name]["silhouette_score_jaccard"] = []


        dic_events_count[model_name][ev.event_type.MISSING] = []
        dic_events_count[model_name][ev.event_type.FORM] = []
        dic_events_count[model_name][ev.event_type.CONTINUE] = []
        dic_events_count[model_name][ev.event_type.GROW] = []
        dic_events_count[model_name][ev.event_type.SHRINK] = []
        dic_events_count[model_name][ev.event_type.SPLIT] = []
        dic_events_count[model_name][ev.event_type.MERGE] = []
        dic_events_count[model_name][ev.event_type.DISSOLVE] = []
        dic_events_count[model_name][ev.event_type.REFORM] = []

    # if the ground truth sequences are provided calculate the adjusted random index and the mutual information score
    if pp.with_ground_truth_sequences:
        dic_score[model_name]["adjusted_rand_score"].append(adjusted_rand_score(predicted_labels))
        dic_score[model_name]["adjusted_mutual_information"].append(adjusted_mutual_information_score(predicted_labels))

    # if the graph is attributed calculate the silhouette for the attributes of the clusters
    # else use the generated embeddings
    if pp.with_attributes:
        dic_score[model_name]["silhouette_score_attributes"].append(
            silhouette_score(predicted_labels, pp.clusters_attributes,'euclidean'))
    else:
        dic_score[model_name]["silhouette_score_attributes"].append(
            silhouette_score(predicted_labels, pp.all_clusters_embeddings,'euclidean'))

    # calculate the silhouette score for the topo features
    dic_score[model_name]["silhouette_score_topo_features"].append(
        silhouette_score(predicted_labels, pp.clusters_topo_features,'cosine'))

    # calculate the silhouette score for the Jaccard index
    dic_score[model_name]["silhouette_score_jaccard"].append(
        sj.silhouette_jaccard_score(predicted_sequences,pp.burt_matrix))

    dic_sequences[model_name].append(predicted_sequences)
    dic_events[model_name].append(events)

    # Count the total number of events
    dic_events_count[model_name][ev.event_type.MISSING].append(
        sum(sequence.count(ev.event_type.MISSING) for sequence in events))
    dic_events_count[model_name][ev.event_type.FORM].append(
        sum(sequence.count(ev.event_type.FORM) for sequence in events))
    dic_events_count[model_name][ev.event_type.CONTINUE].append(
        sum(sequence.count(ev.event_type.CONTINUE) for sequence in events))
    dic_events_count[model_name][ev.event_type.GROW].append(
        sum(sequence.count(ev.event_type.GROW) for sequence in events))
    dic_events_count[model_name][ev.event_type.SHRINK].append(
        sum(sequence.count(ev.event_type.SHRINK) for sequence in events))
    dic_events_count[model_name][ev.event_type.SPLIT].append(
        sum(sequence.count(ev.event_type.SPLIT) for sequence in events))
    dic_events_count[model_name][ev.event_type.MERGE].append(
        sum(sequence.count(ev.event_type.MERGE) for sequence in events))
    dic_events_count[model_name][ev.event_type.DISSOLVE].append(
        sum(sequence.count(ev.event_type.DISSOLVE) for sequence in events))
    dic_events_count[model_name][ev.event_type.MISSING].append(
        sum(sequence.count(ev.event_type.MISSING) for sequence in events))
    dic_events_count[model_name][ev.event_type.REFORM].append(
        sum(sequence.count(ev.event_type.REFORM) for sequence in events))


def convert_predicted_sequences_to_labels(predicted_sequences):
    # convert sequences to labels, since some score functions only take a labels vector as input
    predicted_labels = []
    index = 0
    number_clusters = sum(len(x) for x in predicted_sequences)
    for index in range(0, number_clusters):
        sequence_index = 0
        for sequence in predicted_sequences:
            if index in sequence:
                predicted_labels.append(sequence_index)
            sequence_index += 1
        index += 1
    return predicted_labels


def print_predicted_sequences(predicted_sequences):
    # print extracted  sequences
    print("------------------------")
    print("Extracted Sequences")
    print("------------------------")
    index_sequence = 0
    for sequence in predicted_sequences:
        print(index_sequence, sequence)
        index_sequence += 1


def print_ground_truth():
    # print ground truth sequences
    print("------------------------")
    print("Ground Truth Sequences")
    print("------------------------")

    ground_truth_print = []
    for value in set(pp.ground_truth):
        sequence = [i for i, x in enumerate(pp.ground_truth) if x == value]
        ground_truth_print.append(sequence)
    ground_truth_print.sort()
    for sequence in ground_truth_print:
        print(sequence)


def print_events(events):
    print("------------------------")
    print("Events")
    print("------------------------")
    all_sequences_print = []
    for sequence_index in range(0, len(events)):
        sequence_print = (sequence_index,)
        for time_step in range(0, pp.number_timesteps):
            str_event = str(events[sequence_index][time_step]).replace("event_type.", "")
            sequence_print = sequence_print + (str_event,)
        all_sequences_print.append(sequence_print)

    header_print = ["sequence"]
    for time_step in range(0, pp.number_timesteps):
        header_print.append(time_step)

    print(tabulate(all_sequences_print, headers=header_print))


def save_results(number_runs):
    global dic_score
    global dic_sequences
    global dic_events

    results_path = pp.data_directory + "\\results\\" + str(number_runs)

    if not os.path.isdir(pp.data_directory + "\\results"):
        os.mkdir(pp.data_directory + "\\results")

    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    np.save(results_path + "\\dic_scores", dic_score)
    np.save(results_path + "\\dic_sequences", dic_sequences)
    np.save(results_path + "\\dic_events", dic_events)
    np.save(results_path + "\\dic_events_count", dic_events_count)


def print_scores(dic_score, run):
    all_scores = []

    for embedding_model in dic_score:
        scores = (embedding_model,)
        for score in dic_score[embedding_model]:
            if run == -1:
                value = np.round(np.mean(dic_score[embedding_model][score]), 3)
            else:
                value = np.round(dic_score[embedding_model][score][run], 3)
            scores = scores + (str(value),)
        all_scores.append(scores)

    if pp.with_ground_truth_sequences:
        print(tabulate(all_scores,
                       headers=["model", "ARI", "AMI", "SL_attributes", "SL_topo_features", "SL_jaccard"]))
    else:
        print(tabulate(all_scores,
                       headers=["model", "SL_attributes", "SL_topo_features", "SL_jaccard"]))


def print_events_count(dic_events_count, run):
    all_counts = []

    for embedding_model in dic_events_count:
        scores = (embedding_model,)
        for event in dic_events_count[embedding_model]:
            if run == -1:
                value = np.round(np.mean(dic_events_count[embedding_model][event]), 3)
            else:
                value = np.round(dic_events_count[embedding_model][event][run], 3)
            scores = scores + (str(value),)
        all_counts.append(scores)

    print(tabulate(all_counts,
                   headers=["MISSING", "FORM", "CONTINUE", "GROW", "SHRINK", "SPLIT", "MERGE", "DISSOLVE",
                            "REFORM"]))


def print_results(number_runs):
    global dic_score
    global dic_sequences
    global dic_events
    global dic_events_count

    results_path = pp.data_directory + "\\results\\" + str(number_runs)

    dic_score = np.load(results_path + "\\dic_scores.npy", allow_pickle=True).item()
    dic_sequences = np.load(results_path + "\\dic_sequences.npy", allow_pickle=True).item()
    dic_events = np.load(results_path + "\\dic_events.npy", allow_pickle=True).item()
    dic_events_count = np.load(results_path + "\\dic_events_count.npy", allow_pickle=True).item()

    for run in range(0, number_runs):
        print("")
        print("Experimental run: " + str(run + 1))
        print("---------------------")
        for embedding_model in dic_score:
            print("========================")
            print(embedding_model)
            print("========================")
            if pp.with_ground_truth_sequences:
                print_ground_truth()
            print_predicted_sequences(dic_sequences[embedding_model][run])
            print_events(dic_events[embedding_model][run])
        print("-------------------------")
        print("Scores run:" + str(run + 1))
        print("-------------------------")
        print_scores(dic_score, run)
        print("-------------------------")
        print("events count run:" + str(run + 1))
        print("-------------------------")
        print_events_count(dic_events_count, run)
        print("======================================================================================")

    print("************************************************************************************************")
    print("************************************************************************************************")
    print("************************************************************************************************")
    print("")
    print("-------------------------")
    print("Final Scores over " + str(number_runs) + " run(s)")
    print("-------------------------")
    print_scores(dic_score, -1)
    print("")
    print("-------------------------")
    print("Final event count over " + str(number_runs) + " run(s)")
    print("-------------------------")
    print_events_count(dic_events_count, -1)


def exclude_single_cluster_sequences(predicted_labels):
    sequences = set(predicted_labels)
    long_sequences = []
    for sequence in sequences:
        if predicted_labels.count(sequence) > 1:
            long_sequences.append(sequence)

    evaluation_clusters = []
    for cluster_index in range(0, len(predicted_labels)):
        if predicted_labels[cluster_index] in long_sequences:
            evaluation_clusters.append(cluster_index)
    print(len(predicted_labels))
    print(len(evaluation_clusters))
    return evaluation_clusters

def get_closest_sequences_by_order(sequence, data):
    dist = sk.pairwise.cosine_distances(data, data)
    print( sorted((round(e,3),i) for i,e in enumerate(dist[sequence])))
    # print(np.argsort(dist[sequence]))