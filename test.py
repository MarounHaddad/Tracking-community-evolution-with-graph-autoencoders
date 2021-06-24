import datapreparation.preprocess as pp
import scores.scores as sc
import track.experiments as ex


data_directory = "..\\data\\Dancer\\test1"
pp.preprocess_data_DANCER(data_directory, True, True)
ex.run_experiment(1, 1, 1.0)
sc.print_results(1)

