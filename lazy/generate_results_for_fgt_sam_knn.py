import os
import sys

from data import FileStream
from evaluation.fgt_evaluate_prequential import FGTEvaluatePrequential
from lazy.fgt_sam_knn import FGTSAMKNN

cwd = os.getcwd()
sys.path.insert(0, cwd)



"""
This is a python program to generate evaluation results for 4 datasets with forgetting feature.
The aim is to compare models with the same k value and maximum window size
that forget different number of samples every 1000 samples seen.
For that purpose there are 5 forgetting values for each model: 0, 0.1, 0.25, 0.50, 0.75
"""

"""
initializing streams by passing csv datasets to FileStream constructor
"""
# stream_interchanging = FileStream('fgt-sam-knn-tests/1000-recent/interchanging-rbf/interchanging.csv')
# stream_squares = FileStream('fgt-sam-knn-tests/1000-recent/moving_squares/squares.csv')
stream_chessboard = FileStream('fgt-sam-knn-tests/1000-recent/chessboard/chessboard.csv')
# stream_poker = FileStream('fgt-sam-knn-tests/1000-recent/poker/poker.csv')

"""
preparing each data stream for use
"""
# stream_interchanging.prepare_for_use()
# stream_squares.prepare_for_use()
stream_chessboard.prepare_for_use()
# stream_poker.prepare_for_use()


def generate_samknn_models(k, wind_size):
    # list with number of samples to be forgotten
    n_samples_fgt = [0.1, 0.25, 0.50, 0.75]
    # initialize list of SAMKNN with model which won't have data forgotten - 25% not forget
    samknn_models = [FGTSAMKNN(n_neighbors=k, max_window_size=wind_size, fgt=False)]
    # loop to add samknn's that will have data forgotten
    for i in range(4):
        samknn_models.append(FGTSAMKNN(n_neighbors=k, max_window_size=wind_size, fgt_n_instances=n_samples_fgt[i]))
    return samknn_models


def generate_different_k_values_sam_knn_models(starting_k_value):
    # list that will contain lists with samknn models, each with its own 'k' values
    samknn_models_nested = []
    # loop to append to samknn_models_nested samknn models with k = [3, 5]
    for j in range(starting_k_value, 4, 2):
        samknn_models_nested.append(generate_samknn_models(j, 5000))
    return samknn_models_nested


def evaluate(dataset_name, stream, starting_k_value=3):
    samknn_list = generate_different_k_values_sam_knn_models(starting_k_value)
    for i in range(len(samknn_list)):
        file_name = 'results_k=' + str(samknn_list[i][0].n_neighbors) + '_ws=' + str(samknn_list[i][0].max_wind_size)
        evaluator = FGTEvaluatePrequential(max_samples=2000000,
                                           show_plot=False,
                                           pretrain_size=samknn_list[i][0].n_neighbors,
                                           n_wait=100,
                                           metrics=['accuracy'],
                                           output_file='fgt-sam-knn-tests/1000-recent/' + dataset_name + '/' + file_name + '.csv',
                                           fgt_freq=1000)

        evaluator.evaluate(stream=stream, model=samknn_list[i],
                           image_name='fgt-sam-knn-tests/' + dataset_name + '/' + file_name,
                           model_names=['0', '0.1', '0.25', '0.5', '0.75'])


"""
generate results for each dataset
"""
# evaluate('interchanging', stream_interchanging)
# evaluate('squares', stream_squares)
evaluate('chessboard', stream_chessboard)
# evaluate('poker', stream_poker)
