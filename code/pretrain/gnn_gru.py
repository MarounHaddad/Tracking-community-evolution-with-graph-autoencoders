"""
This file contains the functions used by the graph auto-encoder used as a preprocssing step
to generate an embedding for each cluster to be used as attributes by the track gcn
"""

# libraries
import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datapreparation.preprocess as pp
from dgl.nn.pytorch import GraphConv
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as sk


# standard message function
gcn_message = fn.copy_src(src='h', out='m')

# reduce function (uses sum of the attributes)
gcn_reduce = fn.sum(msg='m', out='h')

class GATLayer(nn.Module):
    def __init__(self,  in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self,g: dgl.graph, h):
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')

def edge_attention(self, edges):
    # edge UDF for equation (2)
    z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
    a = self.attn_fc(z2)
    return {'e': F.leaky_relu(a)}


def reduce_func(self, nodes):
    # reduce UDF for equation (3) & (4)
    # equation (3)
    alpha = F.softmax(nodes.mailbox['e'], dim=1)
    # equation (4)
    h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    return {'h': h}


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g: dgl.graph, h):
        head_outs = [attn_head(g,h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class EncoderLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout):
        super(EncoderLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=True)
        self.activation = activation
        self.norm = nn.BatchNorm1d(out_feats)
        self.drop = nn.Dropout(dropout)
        # torch.nn.init.zeros_(self.linear.weight)
    def forward(self, g: dgl.graph, input):
        # g is the graph and the inputs is the input node features
        # first set the node features
        h = self.linear(input)
        g.ndata['h'] = h
        g.update_all(gcn_message, gcn_reduce)
        # get the result node features
        h = g.ndata.pop('h')
        # perform linear transformation
        # h = self.linear(h)
        h = self.activation(h)
        h = self.norm(h)
        h = self.drop(h)
        return h


class DecoderAdjacencyLayer(nn.Module):
    def __init__(self, activation, num_features):
        super(DecoderAdjacencyLayer, self).__init__()
        self.activation = activation
        self.var = torch.var
        self.norm = nn.BatchNorm1d(num_features)

    def forward(self, inputs):
        # the decoder reconstructs the adjacency by mulitplying the output of the
        # encoder with its transpose
        h = torch.mm(inputs, inputs.t())
        h = self.activation(h)
        h = self.norm(h)
        return h


class DecoderAttributesLayer(nn.Module):
    def __init__(self, activation, in_feats, out_feats):
        super(DecoderAttributesLayer, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias=True)
        self.dropout = nn.Dropout(0.25)
        self.norm = nn.BatchNorm1d(out_feats)

    def forward(self, inputs):
        # the decoder reconstructs the attributes
        h = self.linear(inputs)
        h = self.activation(h)
        # h = torch.mm(inputs, inputs.t())
        # h = self.activation(h)
        # h = self.norm(h)
        return h


class classifier(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(classifier, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=True)
        self.dropout = nn.Dropout(0.25)
        # torch.nn.init.zeros_(self.linear.weight)

    def forward(self, inputs):
        h = self.linear(inputs)
        h = F.softmax(h, 1)
        return h


class cluster_layer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(cluster_layer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=True)
        # torch.nn.init.zeros_(self.linear.weight)

    def forward(self, input):
        h1 = self.linear(input)
        h2 = self.linear(input)
        att = F.softmax(h1, 0)
        h = F.tanh(torch.mul(att, h2))
        return h


class GAE(nn.Module):
    def __init__(self, number_nodes, input_size, hidden_size, encoded_size, embedding_size, number_clusters):
        super(GAE, self).__init__()
        self.enc1 = EncoderLayer(input_size, hidden_size, torch.tanh, 0)
        self.enc2 = EncoderLayer(hidden_size, encoded_size, torch.tanh, 0)
        # self.enc1 = GraphConv(input_size, hidden_size, norm="both", bias=True, activation=F.tanh)
        # self.enc2 =  GraphConv(hidden_size, hidden_size, norm="both", bias=True, activation=F.tanh)
        # self.enc1 = GATLayer(input_size, hidden_size)
        # self.enc2 = GATLayer(hidden_size, encoded_size)
        # self.enc1 = MultiHeadGATLayer(input_size, hidden_size,2)
        # self.enc2 = MultiHeadGATLayer(hidden_size*2, encoded_size,1)

        self.gru = nn.GRUCell(embedding_size, embedding_size, bias=True)
        self.decAdjacency = DecoderAdjacencyLayer(torch.sigmoid, number_nodes)
        self.decattributes = DecoderAttributesLayer(torch.sigmoid, embedding_size,1)
        self.clas = classifier(embedding_size, number_clusters)
        self.clust = cluster_layer(embedding_size, embedding_size, )
        self.linear = nn.Linear(input_size, hidden_size, bias=True)


    def forward(self, g, inputs, previous_state, device):
        encoded1 = self.enc1.forward(g, inputs)
        encoded2 = self.enc2.forward(g, encoded1)

        current_state = torch.cat((encoded1, encoded2),
                                          dim=1)
        # current_state = encoded1

        if len(previous_state) != 0:
            # print(current_state)
            # print(previous_state)
            embedding = self.gru(current_state, previous_state)
            # print(embedding)
        else:
            embedding = current_state


        decoded_adjacency = self.decAdjacency.forward(embedding)
        decoded_attributes = self.decattributes.forward(embedding)
        classification = self.clas.forward(embedding)
        return decoded_adjacency, embedding, classification,decoded_attributes


def train(hidden_size, encoded_size, embedding_size, epochs, early_stopping, print_progress=True):
    dgl_graphs = []
    adjacency_matrices = []
    ground_truth_clusters = []
    models = []
    optimizers = []
    losses = [[]] * len(pp.timesteps_graphs)
    embeddings = [[]] * len(pp.timesteps_graphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attributes = []
    inputs = []

    for timestep_index in range(0, len(pp.timesteps_graphs)):
        g = dgl.DGLGraph(nx.adjacency_matrix(pp.timesteps_graphs[timestep_index]))
        g = dgl.add_self_loop(g)
        dgl_graphs.append(g)
        adjacency_matrices.append(torch.tensor(dgl_graphs[-1].adjacency_matrix().to_dense()))
        ground_truth_clusters.append(
            convert_clusters_to_matrix(pp.timesteps_graphs[timestep_index], pp.clusters[timestep_index]))
        if pp.with_attributes:
            inputs.append(torch.tensor(pp.attributes[timestep_index]).float())
        else:
            inputs.append(torch.eye(len(g.nodes)))
        models.append(
            GAE(dgl_graphs[timestep_index].number_of_nodes(), len(inputs[timestep_index][0]), hidden_size, encoded_size,
                embedding_size,
                ground_truth_clusters[-1].shape[1]))

        scaler = MinMaxScaler()
        scaler.fit(pp.attributes[timestep_index])
        timestep_attributes = scaler.transform(pp.attributes[timestep_index])
        attributes.append(torch.tensor(timestep_attributes).float())

        # dist = sk.pairwise.euclidean_distances(pp.attributes[timestep_index], pp.attributes[timestep_index])
        # heat_kernel = np.exp(- dist ** 2 / (2. * (dist.max()-dist.min()) ** 2))
        # attributes.append(torch.tensor(heat_kernel).float())

        # for GPU
        dgl_graphs[timestep_index].to(device)
        inputs[timestep_index] = inputs[timestep_index].to(device)
        adjacency_matrices[timestep_index] = adjacency_matrices[timestep_index].to(device)
        ground_truth_clusters[timestep_index] = ground_truth_clusters[timestep_index].to(device)
        attributes[timestep_index]= attributes[timestep_index].to(device)
        # optimizer used for training
        optimizers.append(torch.optim.Adam(models[timestep_index].parameters(), lr=0.01))

    min_loss = [1000] * len(pp.timesteps_graphs)
    stop_index = [0] * len(pp.timesteps_graphs)
    for epoch in range(epochs):
        if sum(stop_index) >= early_stopping * len(pp.timesteps_graphs):
            break
        print('Epoch :', epoch)
        for timestep_index in range(0, len(pp.timesteps_graphs)):
            if stop_index[timestep_index] >= early_stopping:
                if print_progress:
                    print('Timestep:', timestep_index, " > Early Stopping!")
                continue

            if len(embeddings[timestep_index - 1]) == 0:
                previous_state = []
            else:
                previous_state = align_states(pp.timesteps_graphs[timestep_index],
                                              pp.timesteps_graphs[timestep_index - 1],
                                              embedding_size,
                                              embeddings[timestep_index - 1].detach())
                previous_state = previous_state.to(device)

            models[timestep_index] = models[timestep_index].to("cuda")

            decoded_adjacency, embedding, classification,decoded_attributes = models[timestep_index].forward(
                dgl_graphs[timestep_index],
                inputs[timestep_index],
                previous_state, device)

            decoded_adjacency = decoded_adjacency.to(device)
            classification = classification.to(device)
            decoded_attributes = decoded_attributes.to(device)

            losses[timestep_index] = F.mse_loss(decoded_adjacency, adjacency_matrices[timestep_index])/3.0 + \
                                     F.mse_loss(classification, ground_truth_clusters[timestep_index])/3.0+ \
                                     F.mse_loss(decoded_attributes,attributes[timestep_index])/3.0
            # losses[timestep_index] = F.mse_loss(decoded_adjacency, adjacency_matrices[timestep_index])
            # losses[timestep_index] =  F.mse_loss(decoded_attributes,attributes[timestep_index])

            optimizers[timestep_index].zero_grad()
            losses[timestep_index].backward()
            optimizers[timestep_index].step()

            if losses[timestep_index] < min_loss[timestep_index]:
                if print_progress:
                    print('> Timestep %d | old Loss: %.4f | New Loss:  %.4f ' % (
                        timestep_index, min_loss[timestep_index], losses[timestep_index].item()))

                # we only save the embedding if there is an improvement in training
                embeddings[timestep_index] = embedding
                min_loss[timestep_index] = losses[timestep_index]
                stop_index[timestep_index] = 0
            else:
                if print_progress:
                    print('> Timestep %d | No improvement | Loss: %.4f | old Loss :  %.4f ' % (
                        timestep_index, losses[timestep_index].item(), min_loss[timestep_index]))
                stop_index[timestep_index] += 1

            models[timestep_index] = models[timestep_index].to("cpu")

    embeddings = [embedding.detach().cpu() for embedding in embeddings]
    embeddings = [embedding.numpy() for embedding in embeddings]
    return embeddings


def convert_clusters_to_matrix(graph, clusters_per_timestep):
    ground_truth_clusters = torch.zeros(len(graph.nodes), len(clusters_per_timestep))
    nodes_list = np.array(list(graph.nodes()))
    for cluster in range(0, len(clusters_per_timestep)):
        for node in graph.nodes():
            node_index = np.where(nodes_list == str(node))[0][0]
            if node in clusters_per_timestep[cluster]:
                ground_truth_clusters[node_index, cluster] = 1
    return ground_truth_clusters


def align_states(new_graph, old_graph, embedding_size, old_state):
    aligned_old_state = torch.zeros(len(new_graph.nodes), embedding_size)
    new_nodes_list = np.array(list(new_graph.nodes()))
    old_nodes_list = np.array(list(old_graph.nodes()))
    for node in new_graph.nodes():
        new_node_index = np.where(new_nodes_list == str(node))[0][0]
        old_node_index = np.where(old_nodes_list == str(node))
        if len(old_node_index[0]) > 0:
            old_node_index = np.where(old_nodes_list == str(node))[0][0]
            aligned_old_state[new_node_index] = old_state[old_node_index]
        else:
            aligned_old_state[new_node_index] = get_node_neighbors_state(node, new_graph, old_graph, old_state,
                                                                         embedding_size)
    return aligned_old_state


def get_node_neighbors_state(node, new_graph, old_graph, old_state, embedding_size):
    neighbor_nodes = new_graph.neighbors(node)
    nodes_list = np.array(list(old_graph.nodes()))
    aggregated_state = []
    for neighbor_node in neighbor_nodes:
        neighbor_node_index = np.where(nodes_list == str(neighbor_node))
        if len(neighbor_node_index[0]) > 0:
            neighbor_node_index = np.where(nodes_list == str(neighbor_node))[0][0]
            aggregated_state.append(old_state[neighbor_node_index])
    if len(aggregated_state) > 0:
        return torch.mean(torch.stack(aggregated_state), dim=0)
    else:
        return torch.zeros(embedding_size)
