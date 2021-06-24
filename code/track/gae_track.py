"""
This function contains the list of funtions of TrackGNN for generating the sequences.
The model is trained on the supergraph (a graph formed from the BURT matrix) where the nodes are the clusters of the different timesteps
and teh edges are weighted with the number of nodes that are shared between the clusters.
The attributes are the cluster embeddings generated in the first step.
The model used a GAE with the SUM aggregation and a novel pruning mechanism for detaching the inter-sequence edges.
"""

# Libraries
import random
import time

import dgl
import math
import networkx as nx
import numpy as np
import sklearn.metrics as sk
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

import datapreparation.preprocess as pp
import datapreparation.utils as ut
import scores.scores as sc
import scores.silhouette_jaccard as sj
import scipy.cluster.hierarchy as hcluster

from datapreparation.finch import FINCH


def gcn_message(edges):
    """
    the message passing function
    :param edges: list of edges
    """
    return {'m': edges.src['h'], 'w': edges.data['weight'].float(), 's': edges.src['deg'], 'd': edges.dst['deg']}


def gcn_reduce(nodes):
    """
    the reduce function that aggregates the messages it takes the mean of the messages * the weights on the edges
    :param nodes: list of nodes
    :return: the new message of the node
    """
    return {
        'h': torch.sum(nodes.mailbox['m'] * nodes.mailbox['w'].unsqueeze(2), dim=1)}
    # return {
    #     'h': torch.mean(nodes.mailbox['m'], dim=1)}


class EncoderLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout):
        super(EncoderLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=True)
        self.activation = activation
        self.norm = nn.BatchNorm1d(out_feats)
        self.drop = nn.Dropout(dropout)

    def forward(self, g: dgl.graph, input):
        # g is the graph and the inputs is the input node features
        # first set the node features
        g.ndata['h'] = input
        g.update_all(gcn_message, gcn_reduce)
        # get the result node features
        h = g.ndata.pop('h')
        # perform linear transformation
        h = self.linear(h)
        h = self.activation(h)
        h = self.norm(h)
        h = self.drop(h)
        return h


class DecoderLayer(nn.Module):
    def __init__(self, num_features):
        super(DecoderLayer, self).__init__()
        self.var = torch.var
        self.norm = nn.BatchNorm1d(num_features)

    def forward(self, inputs):
        # multiply the embedding by its transpose Z*Z^T
        h = torch.mm(inputs, inputs.t())
        # apply sigmoid element wise
        h = F.sigmoid(h)
        h = self.norm(h)
        return h


class GAE(nn.Module):
    def __init__(self, number_nodes, input_size, hidden_size, encoded_size):
        super(GAE, self).__init__()
        self.enc1 = EncoderLayer(input_size, hidden_size, torch.tanh, 0)
        self.enc2 = EncoderLayer(hidden_size, encoded_size, torch.tanh, 0)
        self.dec = DecoderLayer(number_nodes)

    def forward(self, g, inputs):
        encoded1 = self.enc1.forward(g, inputs)
        encoded2 = self.enc2.forward(g, encoded1)

        # the embedding is the concatenation of all the encoder layers
        embedding = torch.cat((encoded1, encoded2),
                              dim=1)
        # embedding = encoded2
        decoded = self.dec.forward(embedding)

        return decoded, embedding


def train(graph, inputs, input_size, hidden_size, embedding_size, epochs, early_stopping,
          pruning_rate, early_stopping_pruning, sample_percentage, print_progress=True):
    """
    trains the graph autoencoder using the burt matrix as an adjacency matrix of the super graph
    during the training, the weights of the edges between the nodes of the super graph
    are adjusted according to the distance between their embeddings. the further the distance the more the weights
    are reduced until the clusters are separated. The remaining sub-graphs are retrieved as the sequences

    :param graph: the super graph (using burt matrix as adjacency matrix)
    :param inputs: the attributes of the clusters (generic one hot encoding or clusters embeddings)
    :param input_size: the size of the input layer
    :param hidden_size: the size of the hidden layer
    :param embedding_size: the size of the embedding layer
    :param epochs: the number of training epochs
    :param early_stopping: the number of epochs for early stopping (relying on the loss change)
    :param pruning_rate: the rate oat which the weights are adjusted on the edges
    :param early_stopping_pruning: number of epochs for early stopping (relying on the number of edges)
    :param sample_percentage: the percentage of edges to sample for pruning
    :param print_progress: print the progress of the training
    :return: the embedding and the new pruned graph
    """

    # generated dgl graph from the networkx graph
    dgl_graph = dgl.DGLGraph()

    # print(nx.adjacency_matrix(graph).todense())
    adj = nx.adjacency_matrix(graph).todense()
    avg = np.nanmean(np.where(adj != 0, adj, np.nan))
    print(avg)

    dist = sk.pairwise.cosine_distances(pp.clusters_attributes, pp.clusters_attributes)
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(pp.clusters_attributes)

    A = neigh.kneighbors_graph(pp.clusters_attributes)
    A = A.toarray()
    g_KNN = nx.from_numpy_array(A)
    nodes_list = list(graph.nodes())
    for edge in g_KNN.edges():
        if not graph.has_edge(nodes_list[edge[0]], nodes_list[edge[1]]):
            graph.add_edge(nodes_list[edge[0]], nodes_list[edge[1]],
                           weight=avg)
        else:
            if nodes_list[edge[0]] != nodes_list[edge[1]]:
                graph.get_edge_data(nodes_list[edge[0]], nodes_list[edge[1]])["weight"] = \
                    graph.get_edge_data(nodes_list[edge[0]], nodes_list[edge[1]])["weight"] + avg

    dist = sk.pairwise.cosine_distances(pp.all_clusters_embeddings, pp.all_clusters_embeddings)
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(pp.all_clusters_embeddings)

    A = neigh.kneighbors_graph(pp.all_clusters_embeddings)
    A = A.toarray()
    g_KNN = nx.from_numpy_array(A)
    nodes_list = list(graph.nodes())
    for edge in g_KNN.edges():
        if not graph.has_edge(nodes_list[edge[0]], nodes_list[edge[1]]):
            graph.add_edge(nodes_list[edge[0]], nodes_list[edge[1]],
                           weight=avg)
        else:
            if nodes_list[edge[0]] != nodes_list[edge[1]]:
                graph.get_edge_data(nodes_list[edge[0]], nodes_list[edge[1]])["weight"] = \
                    graph.get_edge_data(nodes_list[edge[0]], nodes_list[edge[1]])["weight"] + avg


    # dist = sk.pairwise.cosine_distances(pp.clusters_attributes, pp.clusters_attributes)
    # heat_kernel  = np.exp(- dist ** 2 / (2. * (dist.max()-dist.min()) ** 2))
    # print(heat_kernel)
    # A = nx.adjacency_matrix(graph).todense() + heat_kernel
    # print(A)
    # graph =  nx.Graph(A)

    dgl_graph.from_networkx(graph, edge_attrs=['weight'])

    adjcency = torch.tensor(dgl_graph.adjacency_matrix().to_dense())
    adjcency = adjcency / adjcency.sum(axis=1)[:, None]

    dgl_graph.ndata['deg'] = dgl_graph.out_degrees(dgl_graph.nodes()).float()

    gae = GAE(graph.number_of_nodes(), input_size, hidden_size, embedding_size)

    # setup the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gae = gae.to(device)
    dgl_graph.to(device)
    inputs = inputs.to(device)
    adjcency = adjcency.to(device)

    # setup the optimizer
    optimizer = torch.optim.Adam(gae.parameters(), lr=0.01)

    min_loss = 1000
    stop_index = 0

    # used for the early stopping if the number of edges is not changing
    old_edges_count = len(dgl_graph.edata["weight"])
    pruning_stop_index = 0

    dgl_graph.edata['weight'] = dgl_graph.edata['weight'].float()

    # track all epochs graphs and embedding (for testing)
    all_epochs_embeddings = []
    all_epochs_graphs = []

    for epoch in range(epochs):

        reconstructed, embedding = gae.forward(dgl_graph, inputs)

        embedding = embedding.to(device)
        reconstructed = reconstructed.to(device)

        loss = F.mse_loss(reconstructed, adjcency)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # prune edges
        removed_edges = 0
        removed_edges = prune(dgl_graph, embedding, pruning_rate, sample_percentage)

        # branch new edges
        new_edges = 0
        # new_edges = branch(dgl_graph, embedding, True, sample_percentage)

        epoch_graph = nx.Graph((dgl_graph.adjacency_matrix().to_dense()).detach().numpy())

        # track epoch graph and embedding (for testing)
        all_epochs_graphs.append(epoch_graph)
        all_epochs_embeddings.append(embedding.detach().cpu())

        # retreive sub graphs
        # sub_graphs = connected_component_subgraphs(epoch_graph)
        # predicted_sequences = []
        # for sub_graph in sub_graphs:
        #     sequences = list(sub_graph.nodes)
        #     predicted_sequences.append(sorted(sequences))

        c, num_clust, predicted = FINCH(embedding.detach().cpu().numpy())
        clusters = []
        for datapoint in c:
            clusters.append(datapoint[-1])

        predicted_sequences = []
        seqs = set(clusters)
        for seq in seqs:
            sequence = []
            for cluster_index in range(0, len(clusters)):
                if clusters[cluster_index] == seq:
                    sequence.append(cluster_index)
            predicted_sequences.append(sequence)

        predicted_labels = sc.convert_predicted_sequences_to_labels(predicted_sequences)
        if len(set(predicted_labels)) > 1:
            score_attributes = sc.silhouette_score(predicted_labels, pp.clusters_attributes, 'euclidean', False)
            score_topo_features = sc.silhouette_score(predicted_labels, pp.clusters_topo_features, 'cosine', False)
            score_jaccard = sj.silhouette_jaccard_score(predicted_sequences, pp.burt_matrix, False)
        else:
            score_attributes = -1
            score_topo_features = -1
            score_jaccard = -1

        # track the pruning early stopping index
        if old_edges_count == len(dgl_graph.edata["weight"]):
            pruning_stop_index += 1
        else:
            pruning_stop_index = 0
            old_edges_count = len(dgl_graph.edata["weight"])

        # track the loss and print epoch metrics
        if loss < min_loss:
            if print_progress:
                print(
                    'Epoch %d | old Loss: %.4f | New Loss:  %.4f | Edges:  %d | New Edges :  %d | Removed Edges :  %d | A:  %.4f | T:  %.4f | J:  %.4f | Number Sequences:   %d ' % (
                        epoch, min_loss, loss.item(), len(dgl_graph.edata["weight"]), new_edges, removed_edges,
                        score_attributes,
                        score_topo_features, score_jaccard, len(set(predicted_labels))))

            save_emb = embedding
            min_loss = loss
            stop_index = 0
        else:
            if print_progress:
                print(
                    'Epoch %d | New Loss: %.4f | old Loss:  %.4f | Edges:  %d | New Edges :  %d | Removed Edges :  %d | A:  %.4f | T:  %.4f | J:  %.4f | Number Sequences:   %d (No improvement %d)' % (
                        epoch, loss.item(), min_loss, len(dgl_graph.edata["weight"]), new_edges, removed_edges,
                        score_attributes, score_topo_features, score_jaccard, len(set(predicted_labels)),
                        stop_index))
            stop_index += 1

        # check for early stopping by loss and by number of edges
        if stop_index == early_stopping or pruning_stop_index == early_stopping_pruning:
            if print_progress:
                print("Early Stopping!")
            break

    # embedding to be retrived
    save_emb = save_emb.detach().cpu()
    save_emb = save_emb.numpy()

    # new graph after pruning and branching
    new_adj = (dgl_graph.adjacency_matrix().to_dense()).detach().numpy()
    new_graph = nx.Graph(new_adj)

    return save_emb, new_graph


def prune(dgl_graph, embedding, pruning_rate, sample_percentage):
    """
    This function calculates the distance between the embeddings of two connected nodes
    and adjusts the weight of the edge according to the distance
    if the distance is less than 1 then the weight is increased by log(distance)*pruning_rate
    if the distance is larger than 1 than the weight is decrease by  log(distance)*pruning_rate
    if the weight on the edge ever reaches 0, the edge is removed
    :param dgl_graph: the graph to be pruned
    :param embedding: the embedding matrix of the nodes
    :param pruning_rate: the pruning rate at which the weights will be updated
    :param sample_percentage: the percentage of edges to sample for pruning
    :return: number of pruned edges
    """

    if pruning_rate == 0.0 or sample_percentage ==0.0:
        return 0

    # sample random edges
    edge_indices = random.sample(range(0, len(dgl_graph.edata["weight"])),
                                 int(len(dgl_graph.edata["weight"]) * sample_percentage))

    nodes = dgl_graph.find_edges(edge_indices)
    zero_edges = []
    edge_index = 0

    pdist = nn.PairwiseDistance(p=2)
    dist = pdist(embedding[nodes[0]], embedding[nodes[1]])

    # calculate the distance between the embeddings of the nodes per edge
    # and adjust the weight of the edge according to the distance
    for edge_id in edge_indices:
        if nodes[0][edge_index] == nodes[1][edge_index]:
            edge_index += 1
            continue

        if dist[edge_index].item() == 0.0:
            edge_index += 1
            continue


        if math.log10(dgl_graph.edata["weight"][edge_id].item()) <=1:
            _pruning_rate = 0.1
        else:
            _pruning_rate = abs(math.log10(dgl_graph.edata["weight"][edge_id].item()))


        before = dgl_graph.edata["weight"][edge_id].item()
        dgl_graph.edata["weight"][edge_id] -= (math.log10(dist[edge_index].item()) * _pruning_rate)
        # if math.log10(dist[edge_index].item()) > 0:
        #     print(before,dgl_graph.edata["weight"][edge_id].item())
        node_0_weight = pp.burt_matrix[nodes[0][edge_index], nodes[0][edge_index]]
        node_1_weight = pp.burt_matrix[nodes[1][edge_index], nodes[1][edge_index]]

        # if the weight on the edge is 0, remove the edge
        if dgl_graph.edata["weight"][edge_id].item() <= 0:
            zero_edges.append(edge_id)
        # elif dgl_graph.edata["weight"][edge_id].item() < node_0_weight * 0.8 or \
        #         dgl_graph.edata["weight"][edge_id].item() < node_1_weight * 0.8:
        #     zero_edges.append(edge_id)

        edge_index += 1

    # the edges to be removed are accumulated and removed in one shot for speed
    dgl_graph.remove_edges(zero_edges)
    return len(zero_edges)


def branch(dgl_graph, embedding, check_neighbors, sample_percentage):
    """
    This function calculates the distance between the embeddings of a pair list of random unconnected vertices
    if the distance is less than 1 an edge is added between the vertices
    if the parameter check_neighbors is True, then add an edge only for the pair of vertices that already have a neighbour
    the weight of the new edge is the number of neighbours the nodes share
    (Not used only for experiments)
    :param dgl_graph: the graph to be branched with new edges
    :param embedding: the embedding matrix of the nodes
    :param check_neighbors: check if the pair of nodes already share a neighbour before adding the edge
    :return: number of new edges
    """
    sample_nodes = int(len(dgl_graph.nodes()) * sample_percentage)
    nodes1 = random.sample(range(0, sample_nodes), sample_nodes)
    nodes2 = random.sample(range(0, sample_nodes), sample_nodes)
    new_edges = []

    # pdist = nn.PairwiseDistance(p=2)
    # distances = pdist(embedding[nodes1], embedding[nodes2])

    pdist = nn.CosineSimilarity()
    distances = pdist(embedding[nodes1], embedding[nodes2])

    node_index = 0
    for dist in distances:
        if dist.item() <= 1:
            neighbors = len(
                np.intersect1d(dgl_graph.in_edges(nodes1[node_index])[0], dgl_graph.in_edges(nodes2[node_index])[0]))
            if not dgl_graph.has_edge_between(nodes1[node_index], nodes2[node_index]) and (
                    (neighbors > 0 and check_neighbors) or not check_neighbors):
                new_edges.append([nodes1[node_index], nodes2[node_index], neighbors])
        node_index += 1

    # print(new_edges)
    for new_edge in new_edges:
        dgl_graph.add_edge(new_edge[0], new_edge[1])
        dgl_graph.edata["weight"][dgl_graph.edge_id(new_edge[0], new_edge[1])] = new_edge[2]
    return len(new_edges)


def generate_sequences(with_pretrain_embeddings, hidden_size,
                       embedding_size, epochs,
                       early_stopping,
                       pruning_rate,
                       early_stopping_pruning,
                       sample_percentage,
                       print_progress=True):
    """
    Generates the list of cluster sequences:
    First load the burt matrix and use it as an adjacency matrix for the super graph
    Use the generated cluster embeddings in the pre-train phase as attributes of the super graph
    Apply TrackGAE on the supergraph with pruning and branching
    Retrieve the separated sub graphs as the sequences.
    Calculate the scores for the grenerated sequences
    :param data_directory: The path to load the data
    :param with_pretrain_embeddings: whether to use the cluster embeddings generated in the pretrainsp phase or not
    :param hidden_size: the size of the hidden layers
    :param embedding_size:the size of the embedding
    :param epochs:the number of epochs
    :param early_stopping: early stopping for the loss
    :param pruning_rate: the rate of prunining (for testing)
    :param early_stopping_pruning: the early stopping for the number of pruned edges
    :param print_progress: print the progress of the training
    """
    start = time.time()

    # generate the super graph from the burt matrix
    g = nx.Graph(pp.burt_matrix)

    # if the pre-training cluster embeddings will be used as the super graph attributes
    if with_pretrain_embeddings:
        inputs = torch.tensor(pp.all_clusters_embeddings).float()
    else:
        inputs = torch.eye(len(g.nodes))

    # apply trackgae on the super graph
    final_embedding, final_graph = train(g, inputs, len(inputs[0]),
                                         hidden_size,
                                         embedding_size, epochs,
                                         early_stopping,
                                         pruning_rate,
                                         early_stopping_pruning,
                                         sample_percentage,
                                         print_progress)
    end = time.time()

    print("\n done in : ", ut.get_elapsed(start, end))

    print("----------")
    print("Extracted sequences")
    print("----------")
    predicted_sequences = []

    c, num_clust, predicted = FINCH(final_embedding)
    clusters = []
    for datapoint in c:
        clusters.append(datapoint[-1])

    # thresh = 1.0
    # clusters = hcluster.fclusterdata(final_embedding, thresh, criterion="distance",metric="cosine")

    seqs = set(clusters)
    for seq in seqs:
        sequence = []
        for cluster_index in range(0,len(clusters)):
            if clusters[cluster_index]==seq:
                sequence.append(cluster_index)
        predicted_sequences.append(sequence)

    # sub_graphs = connected_component_subgraphs(final_graph)
    # for sub_graph in sub_graphs:
    #     sequences = list(sub_graph.nodes)
    #     predicted_sequences.append(sorted(sequences))

    # calculate the score
    if with_pretrain_embeddings:
        model_name = "TRACKGAE"
    else:
        model_name = "TRACKGAE-NP"

    sc.get_scores(model_name, predicted_sequences)


def connected_component_subgraphs(G):
    """Used for testing: We consider the sequences the disconnected subgraphs in the supergraph"""
    for c in nx.connected_components(G):
        yield G.subgraph(c)
