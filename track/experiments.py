"""
This file contains the list of functions used to run the experiments
"""

# libraries
import warnings

import benchmarks.GED as gd
import benchmarks.Green as gr
import benchmarks.mutual_transition as mt
import datapreparation.preprocess as pp
import track.gae_track as tg
import scores.scores as sc
import pretrain.pretrain as pg

warnings.filterwarnings("ignore")


def run_experiment(number_runs, pruning_rate, sample_percentage):
    """
    Run the experiments for a certain number of runs
    We take the mean of the results
    :param number_runs: number of runs
    :param pruning_rate: the pruning rate of TrackGNN
    :param sample_percentage: the percentage of edges to sample for pruning
    """
    for run in range(0, number_runs):
        pg.generate_clusters_embeddings(64, 64, 128, 200, 10)
        pp.load_embeddings()

        # generate the sequences with TrackGNNusing pretraining embedding
        tg.generate_sequences(True, 32, 64, 300, 200, pruning_rate, 200, sample_percentage)
        # generate the sequences with TrackGNN wihtout using the pretraining embedding (we used one hot encoding for the supergraph instead)
        tg.generate_sequences(False, 32, 64, 300, 200, pruning_rate, 200, sample_percentage)

        # generate the sequences using Mutual Transition
        mt.apply_mutual_transition()

        # generate the sequences using Jaccard Index (Green et al.)
        gr.apply_green_similarity(0.3)

        # generate the sequence susing GED Method
        gd.apply_GED_similarity(0.5, 0.5)

    sc.save_results(number_runs)
    sc.print_results(number_runs)
