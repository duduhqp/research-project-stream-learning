import random
from abc import ABC

import numpy as np
from lazy.sam_knn import SAMKNNClassifier


class FGTSAMKNN(SAMKNNClassifier, ABC):
    def __init__(self,
                 fgt=True,
                 fgt_n_instances=100,
                 fgt_from_sub_set_length=1000,
                 n_neighbors=5,
                 weighting='distance',
                 max_window_size=5000,
                 ltm_size=0.4,
                 min_stm_size=50,
                 stm_size_option='maxACCApprox',
                 use_ltm=True):
        super().__init__(n_neighbors=n_neighbors,
                         weighting=weighting,
                         max_window_size=max_window_size,
                         ltm_size=ltm_size,
                         min_stm_size=min_stm_size,
                         stm_size_option=stm_size_option,
                         use_ltm=use_ltm)

        self.fgt = fgt
        self.fgt_n_instances = fgt_n_instances
        self.fgt_from_sub_set_length = fgt_from_sub_set_length

    def delete_element_at_index(self, i):
        """ Delete element at a given index i from the sample window """
        self.window._n_samples -= 1
        self.window._buffer = np.concatenate((self.window._buffer[:i, :], self.window._buffer[i + 1:, :]))

    def get_last_random(self, n_samples, sub_set_length):
        """ get 'n_samples' randomly from the newest 'sub_set_length' elements """
        window_length = self.window.n_samples
        last_random_instances = []
        last_samples_starting_position = window_length - sub_set_length
        last_samples_range = range(last_samples_starting_position, window_length)
        random_indexes = (random.sample(last_samples_range, n_samples))
        for i in range(n_samples):
            index = random_indexes[i]
            random_instance = (self.window.buffer[index])
            last_random_instances.append(random_instance)
        return last_random_instances

    def delete_by_instance(self, instances):
        """ looks for 'instances' given in the window and deletes them """
        for i in range(len(instances)):
            for j in range(len(self.window._buffer) - 1, -1, -1):
                if np.array_equal(instances[i], self.window._buffer[j]):
                    self.delete_element_at_index(j)
                    break

    def forget_last_random(self):
        """ deletes newest random instances """
        instances = self.get_last_random(self.fgt_n_instances, self.fgt_from_sub_set_length)
        self.delete_by_instance(instances)
