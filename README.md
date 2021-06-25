# Tracking community evolution with graph autoencoders
In this preliminary work, we apply deep learning on graphs for the task of cluster tracking in dynamic networks. While most cluster tracking methods rely on node membership to track the clusters from different timesteps, in our work we look at all three characteristics when performing the tracking: node membership, node attributes, and cluster structure. Our method is comprised of two steps. First, we generate representative embeddings of the clusters using Graph Neural Networks supplemented with GRUs for temporal learning. Next, we build an attributed, weighted Supergraph out of the clusters. In the Supergraph the nodes are the clusters, the edge-weights are the number of nodes shared between the clusters, and the attributes are the cluster embeddings generated in the first step. Subsequently, we transform the problem of cluster tracking into a clustering task on the Supergraph, where each cluster represents a sequence of clusters. In the second step, we apply a graph autoencoder (TrackGAE) on the Supergraph. TrackGAE is supplemented with a novel pruning mechanism that detaches the weak inter-sequence edges and reinforces the intra-sequence edges in the Supergraph. Finally, Finch, an agglomerative clustering algorithm is applied on the representations of TrackGAE in order to generate the sequences.

## Problem definition

Network data structures are a natural choice for modeling the relationships between interacting entities in a multitude of domains. Although many tools from graph theory are employed to extract information from these networks, clustering or community detection remains a key component for obtaining  meaningful insights about their underlying patterns. However, most studies concentrate on community detection in a static network, while real-world networks are dynamic in nature. This dynamism leads to changes in the structure and content of the networks which entails changes in their internal communities. Tracking and analysing the behaviour of these communities as they evolve over time is a crucial task in many fields of study that are interested in the dynamics of groups rather than the individuals, such as modeling the immune response to a virus in a population [] or group interactions in social networks [].

While detecting high quality communities in a static network remains a challenging task by itself, tracking the changes of these communities over time offers a new and unique set of problems. Over the past two decades different techniques have been developed in order to tackle this task and different approaches were proposed based on varying assumptions on the nature of the evolution of the clusters, each with its advantages and limitations. However, in this study we adopt the method of slicing the of the evolutionary history of the graph into multiple snapshots that are called time-steps and then generating community sequences by matching clusters from different time-steps. Each sequence of matching clusters represent the life-cycle of a single evolving community. This methode is generally referred to in litterature as Independent community detection and matching []. 

<p align="center">
  <img width="300" height="400" src="https://github.com/MarounHaddad/Restaurant-recommendation-with-augmented-relational-graph-neural-networks/blob/main/images/data%20augementation.png">
</p>
 <p align="center"><em>Figure 2 - Data augmentation example for the 4 stars edges.</em></p>

Figure 1 displays an example of cluster sequences generated on a dynamic graph of three timesteps. As the example shows, multiple 


## Motivation


## Methodolody

## TrackGAE Pretrain

## TrackGAE Generate Sequences

## Evaluation of Sequences

## Conclusion

## References

## Note
This work is part of a preliminary reseach on dynamic graphs done at UQAM (University of Quebec at Montreal) under the supervision of Dr. Mohamad Bouguessa.
Maroun Haddad 2020.
