# Tracking community evolution with graph autoencoders
In this preliminary work, we apply deep learning on graphs for the task of cluster tracking in dynamic networks. While most cluster tracking methods rely on node membership to track the clusters from different timesteps, in our work we look at all three characteristics when performing the tracking: node membership, cluster attributes, and cluster structure. Our method is comprised of two steps. First, we generate representative embeddings of the clusters using Graph Neural Networks supplemented with GRUs for temporal learning. Next, we build an attributed, weighted Supergraph out of the clusters. In the Supergraph the nodes are the clusters, the edge-weights are the number of nodes shared between the clusters, and the attributes are the cluster embeddings generated in the first step. Subsequently, we transform the problem of cluster tracking into a clustering task on the Supergraph, where each cluster represents a sequence of clusters. In the second step, we apply a graph autoencoder (TrackGAE) on the Supergraph. TrackGAE is supplemented with a novel pruning mechanism that detaches the weak inter-sequence edges and reinforces the intra-sequence edges in the Supergraph. Finally, Finch, an agglomerative clustering algorithm is applied on the representations of TrackGAE in order to generate the sequences.

## Problem definition

Network data structures are a natural choice for modeling the relationships between interacting entities in a multitude of domains. Although many tools from graph theory are employed to extract information from these networks, clustering or community detection remains a key component for obtaining  meaningful insights about their underlying patterns. However, most studies concentrate on community detection in a static network, while real-world networks are dynamic in nature. This dynamism leads to changes in the structure and content of the networks which entails changes in their internal communities. Tracking and analysing the behaviour of these communities as they evolve over time is a crucial task in many fields of study that are interested in the dynamics of groups rather than the individuals, such as modeling the immune response to a virus in a population or group interactions in social networks.

While detecting high-quality communities in a static network remains a challenging task by itself, tracking the changes of these communities over time offers a new and unique set of problems. Over the past two decades, different techniques have been developed in order to tackle this task and different approaches were proposed based on varying assumptions on the nature of the evolution of the clusters, each with its advantages and limitations. However, in this study, we adopt the method of slicing the evolutionary history of the graph into multiple snapshots that are called time-steps and then generating community sequences by matching clusters from different time-steps. Each sequence of matching clusters represents the life cycle of a single evolving community. This method is generally referred to in the literature as Independent community detection and matching []. 

<p align="center">
  <img width="65%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/Tracking.png">
</p>
 <p align="center"><em>Figure 1 - Example of cluster tracking in a dynamic network.</em></p>

Figure 1 displays an example of cluster sequences generated on a dynamic graph of three timesteps. As the example shows, multiple combinations of sequences are possible for the same lifecycle of the graph. Finding the sequence that is most representative of the evolving cluster is the main challenge of the cluster tracking task. Furthermore, we notice that it is possible to define events on the evolving sequences. In this work, we identify 8 different events that can occur on the sequence:  
1. Birth: The sequence starts in this timestep.
2. Death: The sequence ends in this timestep (Has not clusters that belong to this timestep).
3. Shrinking: The number of nodes in the sequence of the current timestep is less than the number of nodes in the sequence of the previous timestep.
4. Growing: The number of nodes in the sequence of the current timestep is larger than the number of nodes in the sequence of the previous timestep.
5. Splitting: The number of clusters in the sequence of the current timestep is larger than the number of clusters in the sequence of the previous timestep.
6. Merging: The number of clusters in the sequence of the current timestep is less than the number of clusters in the sequence of the previous timestep.
7. Continuing: The same number of clusters and nodes in the sequence of the previous timestep is maintained in the current timestep.
8. Resurgence: A sequence dies and is missing for some timesteps and then is reborn again.

## Motivation

<p align="center">
  <img width="50%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/motivation.png">
</p>
 <p align="center"><em>Figure 2 - Three caracteristics to be evaluated for cluster matching.</em></p>
 
Over the years multiple techniques were introduced in the literature for independent community detection and matching. However, the majority of these methods rely on node membership to perform the matching of the clusters from different timesteps. Green et al. [], one of the earliest techniques, relied on the Jaccard Index as a similarity measure. In this method, the Jaccard index is calculated between the last cluster of each sequence (the fronts) and every cluster in the current timestep. If the Jaccard Index of two clusters is higher than a certain manually defined similarity threshold the clusters are placed in the same sequence. However, looking at the node membership alone might not be sufficient to properly represent the changes in the cluster. Take for example the image in row two of Figure 2. Two clusters from different timesteps can have the same nodes, however, the internal structure of the clusters is completely different. The same can be observed in row three of Figure 2. Two clusters from different timesteps can have the same nodes and same structure, however, the attributes on the nodes have completely changed from one timestep to another. In these two cases, can we consider these clusters as the same evolving cluster? or should they be separated into different sequences? We believe a reliable tracking algorithm should consider all three characteristics: node membership, cluster structure, and cluster attributes.  


## Methodolody
<p align="center">
  <img width="75%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/TrackGAE%20cycle.png">
</p>
 <p align="center"><em>Figure 3 - A two-step cycle of TrackGAE</em></p>
 
Our method consists of two main steps  1) Pretraining and 2) Tracking. However, first, we start by building a Supergraph out of the clusters from the different timesteps. In order to build the Supergraph, we generate the Burt matrix (B). The Burt Matrix indicates the number of nodes shared between the clusters. The diagonal of the Burt Matrix indicates the number of nodes per cluster. The Burt matrix is built by simply multiplying the Membership matrix (A) with its transpose. The membership matrix is a binary matrix that indicates whether a node belongs to a cluster or not. The main insight of this work is that the Burt matrix can be considered as an adjacency matrix for a weighted Supergraph, where the nodes are the clusters and the edge-weights indicate the number of shared nodes between the clusters. We transform the problem of generating the sequences of clusters as a clustering task on the Supergraph.  

After building the Supergraph, we start with the first step: Pretraining. In this step, we generate representations for the clusters using temporal graph autoencoders applied on each timestep. The cluster embeddings are then used as attributes on the Supergraph. In the second step, denoted Tracking, a graph autoencoder (GAE), supplemented with a pruning mechanism, is trained on the Supergraph in order to generate the sequences. Figure 3 details the different steps of the TrackGAE framework.  

## TrackGAE Pretrain
<p align="center">
  <img width="65%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/TrackGAE%20Pretrain.png">
</p>
 <p align="center"><em>Figure 4 - TrackGAE pretraining architecture (Step-1), used to generate clusters embeddings.</em></p>
 
In the pretraining step, N Graph autoencoders are trained on the N timesteps to generate the embeddings of the nodes. The encoder section of the GAE is formed of two layers that use the SUM aggregation rule with a size of 64. The hyperbolic tangent function (Tanh) is used on the hidden layers. The embedding H is formed by concatenating the outputs of the first and second layers. The output of the GAE from the previous timestep is passed through a GRU (Gated Recurrent Unit) with the output of GAE of the current timestep to form the embedding Z such as Z<sub>n</sub> = GRU(Z<sub>n-1</sub>, H<sub>n</sub>).  

The Decoder side of the GAE performs three tasks: Reconstruct the adjacency matrix, reconstruct the attributes matrix, and classify the nodes according to their cluster membership. The purpose is to capture all three characteristics in the embeddings, that is node membership, node attributes, and graph structure.  We train the models in parallel on 200 epochs with a patience of 10 for early stopping.  

To form the embeddings of the clusters, we aggregate the mean of the embeddings of the nodes that form each cluster according to the timestep of the cluster. We finally stack all the cluster embeddings to form the attributes of the Supergraph to be used in the second step.  

## TrackGAE Generate Sequences
<p align="center">
  <img width="65%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/TrackGAE%20Track.png">
</p>
 <p align="center"><em>Figure 5 - TrackGAE tracking architecture (Step-2), used to generate the sequences.</em></p>

<p align="center">
  <img width="50%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/pruning.gif">
</p>
 <p align="center"><em>Figure 6 - A pruning/reinforcement example.</em></p>
 
 
## Evaluation of Sequences
<p align="center">
  <img width="60%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/evaluation.png">
</p>
 <p align="center"><em>Figure 7 - The sequence evaluation metrics should penalize the malformed sequences.</em></p>
 
## Benchmarks

## Preliminary results
<p align="center">
<img  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/example1.png">
</p>
 
<p align="center">
<img   src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/example2.png">
</p>

## Conclusion

## References

## Note
This work is part of a preliminary reseach on dynamic graphs done at UQAM (University of Quebec at Montreal) under the supervision of Dr. Mohamad Bouguessa.  
  
Maroun Haddad 2020.
