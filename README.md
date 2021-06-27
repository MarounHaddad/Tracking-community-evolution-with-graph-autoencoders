# Tracking Community Evolution with Graph Autoencoders
In this preliminary work, we apply deep learning on graphs for the task of cluster tracking in dynamic networks. Tracking evolving clusters is a crucial task in many domains that are interested in the dynamic of the group rather than the individual. Over the years, many different models were developed for tackling this problem. However, most of them were limited in their approach, as they only relied on superficial topological characteristics or the node membership to perform the tracking of the cluster. In recent years, great advancements in deep learning on graphs have produced state-of-the-art results on many classical graph learning tasks.  
Motivated by these advancements, we propose TrackGAE, a Graph Neural Network framework that relies on three important characteristics to perform the tracking: node membership, cluster attributes, and cluster structure. To achieve this goal we 1) Generate descriptive representations of the dynamic clusters and 2) Employ these representations to group the semantically similar clusters into temporal sequences. Our method starts by building a synthetic graph that we denote Supergraph out of the inter-cluster relationships from different time-steps. Subsequently, TrackGAE is applied on the Supergraph. TrackGAE is supplemented with a clustering functionality that we denote Deep-Pruning that allows it to detach the edges between the diverging clusters while grouping the semantically similar clusters into temporal sequences.

## Problem definition

Network data structures are a natural choice for modeling the relationships between interacting entities in a multitude of domains. Although many tools from graph theory are employed to extract information from these networks, clustering or community detection remains a key component for obtaining meaningful insights about their underlying patterns. However, most studies concentrate on community detection in a static network, while real-world networks are dynamic by nature. This dynamism leads to changes in the structure and content of the networks which entails changes in their internal communities. Tracking and analyzing the behavior of these communities as they evolve is a crucial task in many fields of study that are interested in the dynamics of groups rather than the individuals, such as modeling the immune response to a virus in a population or group interactions in social networks.  

While detecting high-quality communities in a static network remains a challenging task by itself, tracking the changes of these communities over time offers a new and unique set of problems. Over the past two decades, different techniques have been developed in order to tackle this task and different approaches were proposed based on varying assumptions on the nature of the evolution of the clusters, each with its advantages and limitations. However, in this study, we adopt the method of slicing the evolutionary history of the graph into multiple snapshots that are called time-steps and then generating community sequences by matching clusters from different time-steps. Each sequence of matching clusters represents the life cycle of a single evolving community. This method is generally referred to in the literature as Independent community detection and matching [4].  

<p align="center">
  <img width="65%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/Tracking.png">
</p>
 <p align="center"><em>Figure 1 - Example of cluster tracking in a dynamic network.</em></p>

Figure 1 displays an example of cluster sequences generated on a dynamic graph of three timesteps. As the example shows, multiple combinations of sequences are possible for the same lifecycle of the graph. Finding the sequence that is most representative of the evolving cluster is the main challenge of the cluster tracking task. Furthermore, we notice that it is possible to define events on the evolving sequences. In this work, we identify 8 different events that can occur on the sequence:  
1. Birth: The sequence starts in this timestep.
2. Death: The sequence ends in this timestep (i.e. has no clusters that belong to this timestep).
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
 
Over the years multiple techniques were introduced in the literature for independent community detection and matching. However, the majority of these methods rely on node membership to perform the matching of the clusters from different timesteps. Green et al. [5], one of the earliest techniques, relied on the Jaccard Index as a similarity measure. In this method, the Jaccard index is calculated between the last cluster of each sequence (the fronts) and every cluster in the current timestep. If the Jaccard Index of two clusters is higher than a certain manually defined similarity threshold the clusters are placed in the same sequence. However, looking at the node membership alone might not be sufficient to properly represent the changes in the cluster. Take for example row two of Figure 2. Two clusters from different timesteps can have the same nodes, however, the internal structure of the clusters is completely different. The same can be observed in row three of Figure 2. Two clusters from different timesteps can have the same nodes and same structure, however, the attributes on the nodes have completely changed from one timestep to another. In these two cases, can we consider these clusters as the same evolving cluster? or should they be separated into different sequences? We believe a reliable tracking algorithm should consider all three characteristics: node membership, cluster structure, and cluster attributes to properly represent the identity of the evolving clusters.  


## Methodolody
<p align="center">
  <img width="75%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/TrackGAE%20cycle.png">
</p>
 <p align="center"><em>Figure 3 - A two-step cycle of TrackGAE</em></p>
 
Our method consists of two main steps  1) Pretraining and 2) Tracking. However, first, we start by building a Supergraph out of the clusters from the different timesteps. In order to build the Supergraph, we generate the Burt matrix (B). The Burt Matrix indicates the number of nodes shared between the clusters. The diagonal of the Burt Matrix indicates the number of nodes per cluster. The Burt matrix is built by simply multiplying the Membership matrix (A) with its transpose. The membership matrix is a binary matrix that indicates whether a node belongs to a cluster or not. The main insight of this work is that the Burt matrix can be considered as an adjacency matrix for a weighted Supergraph, where the nodes are the clusters and the edge-weights indicate the number of shared nodes between the clusters. We transform the problem of generating the sequences of clusters as a clustering task on the Supergraph.  

After building the Supergraph, we start with the first step: Pretraining. In this step, we generate representations for the clusters using temporal graph autoencoders applied on each timestep. The cluster embeddings are then used as attributes on the Supergraph. In the second step, denoted Tracking, a graph autoencoder (TrackGAE), supplemented with a pruning mechanism, is trained on the Supergraph in order to generate the sequences. Figure 3 details the different steps of the TrackGAE framework.  

## Step 1 - Pretrain
<p align="center">
  <img width="65%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/TrackGAE%20Pretrain.png">
</p>
 <p align="center"><em>Figure 4 - TrackGAE pretraining architecture (Step-1), used to generate clusters embeddings.</em></p>
 
In the pretraining step, N Graph autoencoders are trained on the N timesteps to generate the embeddings of the nodes. The encoder section of the GAE is formed of two layers that use the SUM aggregation rule with a size of 64. The hyperbolic tangent function (Tanh) is used on the hidden layers. The embedding H is formed by concatenating the outputs of the first and second layers. The output of the GAE from the previous timestep is passed through a GRU (Gated Recurrent Unit) with the output of GAE of the current timestep to form the embedding Z such as Z<sub>n</sub> = GRU(Z<sub>n-1</sub>, H<sub>n</sub>).  

The Decoder side of the GAE performs three tasks: Reconstruct the adjacency matrix, reconstruct the attributes matrix, and classify the nodes according to their cluster membership. The purpose is to capture all three characteristics in the embeddings, that is node membership, node attributes, and graph structure.  We train the models in parallel on 200 epochs with a patience of 10 for early stopping.  

To form the embeddings of the clusters, we aggregate the mean of the embeddings of the nodes that form each cluster according to the timestep of the cluster. We finally stack all the cluster embeddings to form the attributes of the Supergraph to be used in the second step.  

## Step 2 - Generate Sequences
<p align="center">
  <img width="65%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/TrackGAE%20Track.png">
</p>
 <p align="center"><em>Figure 5 - TrackGAE tracking architecture (Step-2), used to generate the sequences.</em></p>


In the second step, we apply a graph autoencoder (TrackGAE) on the Supergraph built at the start. We use the embeddings of the clusters generated in the first step as attributes on the nodes of the Supergraph. The encoder section of TrackGAE is formed of two GNN layers that use the SUM aggregation rule. However, when we aggregate the messages of the neighbors, we multiply them with the weight on the edges:
<p align="center">
  <img width="20%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/aggregationrule.png">
</p>
 
The layers have a size of 64 and we use the hyperbolic tangent function (Tanh)  as an activation function. The embedding is formed by concatenating the output of the first and second layers of the encoder. The decoder reconstructs the adjacency matrix by multiplying the embedding with its transpose and applying a sigmoid element-wise. We train the model for 300 epochs with an Adam optimized having a learning rate of 0.01.
 
We notice in the supergraph that due to the nodes migrating from cluster to cluster between timesteps, the Supergraph has a lot of edges that represent minor interactions between the clusters. The red lines in Figure5 represent the inter-sequence edges. If we can detach these inter-sequence edges, we can retrieve better representative sequences. To achieve this goal, we supplement the TrackGAE with a pruning mechanism:  
<p align="center">
  <img width="35%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/pruningrule.png">
</p>

In every epoch, we sample a random percentage of the edges in the Supergraph. We calculate the distance between the embeddings of the two nodes of the edge. the log of the distance is then subtracted from the weight on the edge. If the distance is larger than 1, then the weight on the edge will be reduced, and eventually, if the weight is <=0, we detach the edge. However, if the distance is less than 1 then the result of the log is negative and the weight will increase, reinforcing the connection. The pruning rate $\rho$ controls the speed of the pruning. Figure 6 represents a mockup example of the pruning and reinforcement operations during the training.  

<p align="center">
  <img width="50%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/pruning.gif">
</p>
 <p align="center"><em>Figure 6 - A pruning/reinforcement example.</em></p>

To generate the sequences, we apply FINCH[6] on the embeddings generated by the graph autoencoder trained on the Supergraph. During our testing, we experimented with multiple concepts as well:
1. Branching: It is the opposite of pruning, where we calculate the distance between the embeddings of two disconnected nodes and connect them during the training if the distance is < 1.
2. Dynamic pruning rate: Instead of using a fixed pruning rate which can be hard to determine from graph to graph, a dynamic pruning rate is calculated as the |log(edge_Weight)|.
3. Pruning Patience: If the number of edges that are being pruned is 0 for a consecutive number of epochs, we stop the training.
4. Accelerated pruning: Instead of waiting till the weight on the edge is <=0 to detach it, if the edge weight is less than a certain percentage of the size of the nodes that form the edge, we detach it. (Size of the nodes = the number of nodes in the clusters pair that form the edge)
5. Augmenting the Supergraph: Before the training on the Supergraph, we can add extra edges by using KNN on the cluster embeddings generated in step 1 or the cluster attributes  (The cluster attributes are calculated by aggregating the attributes of the nodes of the cluster).
 
## Evaluation of Sequences
<p align="center">
  <img width="60%"  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/evaluation.png">
</p>
 <p align="center"><em>Figure 7 - The sequence evaluation metrics should penalize the malformed sequences.</em></p>
 
Evaluating the sequences remains a challenge that requires further investigation. However, in our test, to evaluate the generated sequences, we relied on three metrics that measure, to an extent, the three characteristics that we wanted to preserve: node membership, cluster attributes, and cluster structure:  

- For the node membership, we use the Jaccard Index.   

- For the cluster structure, we use NetSimile features [2]. The NetSimile features are generated by first calculating the topological features for the nodes and arranging them in vector form. Then for every cluster, the topological features vectors of the nodes are aggregated using multiple aggregation functions (Mean, Median, ...). The aggregated features are then concatenated to form the Netsimile features.  

- For the cluster attributes, we calculate the homogeneity between the cluster attributes of every sequence. The cluster attributes are formed by aggregating the attributes of the nodes that form the cluster.  

It is important for any metric that we use to consider all the generated sequences and not ignore the sequences of size 1. Several metrics that measure the cluster homogeneity, such as Calinski Harabasz and Davies Bouldin, ignore the clusters of size 1 (3rd image in Figure 7). However, if a cluster is placed in a sequence on its own or in a sequence where it does not belong, the evaluation metric should penalize the placement of this cluster. We found that the Silhouette Score to properly achieve this. Therefore, we use it for all three characteristics. However, for the Jaccard index, we developed a new Silhouette Jaccard score that considers the Jaccard index instead of the distance.  


## Benchmarks
We compare TrackGAE to three well-known methods for community tracking:

- Green et al[5], this method uses the Jaccard Index as a similarity measure and a manually defined similarity threshold. 
- GED[3], this method calculates the Inclusion metric which evaluates the quantity and quality of the compared clusters. The quantity is the Jaccard Index while the quality metric can be any node centrality measure. The Inclusion measure is calculated as the product Quantity x Quality. GED uses two manually defined similarity thresholds alpha and beta. 
- Mutual Transition [7], this method uses a normalized Burt matrix in order to calculate a similarity measure between the clusters. The Mutual Transition score is calculated as the sum of the harmonic mean between the normalized Burt matrix vectors of two clusters. Unlike the other methods, in [7] the similarity threshold is derived automatically from the Mutual Transition scores calculated between all clusters.  

## Preliminary results
Below we display some preliminary results with TrackGAE applied on dynamic graphs generated with Dancer[1]. Dancer is a tool that allows the user to generate synthetic dynamic graphs with an emphasis on dynamic clusters. The Supergraph images displayed below are generated by Dancer, the IDs of the clusters are highlighted in red.   
We display the sequences generated by each model, plus the evaluation metrics calculated for each result. We also display the events generated by TrackGAE and the summary of the total events generated by every model.

### Example 1
<p align="center">
<img  src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/example1.png">
</p>

 <hr /> 
 
 ### Example 2
<p align="center">
<img   src="https://github.com/MarounHaddad/Tracking-community-evolution-with-graph-autoencoders/blob/main/images/example2.png">
</p>

## Conclusion

The preliminary results of TrackGAE remain to an extent unsatisfactory. We believe we need a new approach for the evaluation of the generated sequences. This can be probably achieved through the application of a downstream task on the generated sequences, such as "critical event prediction". This approach would be more reliable at evaluating the sequences than using three disconnected evaluation metrics. 

## Code: Prerequisites
- python 3.6
- dgl-cu90 0.4.3
- torch 1.4
- networkx 2.4
- numpy 1.18.1
- scikit-learn 0.22.2
- cuda 9.0

## Code: Test Example
In order to run a test using Dancer generated data, you can use the example in "Test.py":  

```
data_directory = "..\\data\\Dancer\\test1"
pp.preprocess_data_DANCER(data_directory, True, True)
ex.run_experiment(1, 1, 1.0)
sc.print_results(1)
```

For running a TrackGAE on a new Dancer example, the "t.graph" files generated by Dancer should be placed in a new folder referenced by the variable "data_directory".  

run_expriment takes 3 arguments:  
- Number of test runs (The result is the mean)
- Pruning rate
- Percentage of sampled edges for pruning

The scores of each number of runs are saved separately. To display the results, call print_results(number_of_runs).  

## Note
This work is part of a preliminary research on dynamic graphs done at UQAM (University of Quebec at Montreal) under the supervision of Dr. Mohamad Bouguessa.  
  
Maroun Haddad 2020.

## References
[1] Benyahia, O., Largeron, C., Jeudy, B. et Zaïane, O. R. (2016). Dancer : Dynamic attributed network with community structure generator. In Machine Learning and Knowledge Discovery in Databases,41–44.  
[2] Berlingerio, M., Koutra, D., Eliassi-Rad, T. et Faloutsos, C. (2012). Netsi-mile : A scalable approach to size-independent network similarity.  
[3] Bródka, P., Saganowski, S. et Kazienko, P. (2012). GED : the method for group evolution discovery in social networks.
[4] Dakiche, N., Tayeb, F. B., Slimani, Y. et Benatchba, K. (2019). Tracking community evolution in social networks : A survey. In Inf. Process. Manag.,56(3), 1084–1102.  
[5] Greene, D., Doyle, D. et Cunningham, P. (2010). Tracking the evolution of communities in dynamic social networks. In N. Memon et R. Alhajj (dir.). International Conference on Advances in Social Networks Analysis and Mining.  
[6] Sarfraz, M. S., Sharma, V. et Stiefelhagen, R. (2019). Efficient parameter-freeclustering using first neighbor relations. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 8934–8943.  
[7] Tajeuna, E. G., Bouguessa, M. et Wang, S. (2015). Tracking the evolution ofcommunity structures in time-evolving social networks. In IEEE International Conference on Data Science and Advanced Analytics.  
