from kohonen import KohonenNet

import numpy as np


class KohonenFrequencyNet(KohonenNet):
    def __init__(self, first_layer_size, second_layer_size, learning_rate):
        super().__init__(first_layer_size, second_layer_size, learning_rate)
        self.wins = np.zeros(self.second_layer_size)
        self.weights_updates_number = 1
    
    def _find_winner(self, sample) -> int:
        distances = np.array([
            np.linalg.norm(self.weights[:, j] - sample) 
            for j in range(self.second_layer_size)
        ])

        winner_index = np.argmin(distances)        
        
        return winner_index
    
    def _update_weights(self, sample: np.array) -> int:
        winner_index = self._find_winner(sample=sample)

        for ind in range(self.second_layer_size):
            self.weights[ind][winner_index] += self.learning_rate * (sample[ind] - self.weights[ind][winner_index])

        self.wins[winner_index] += 1
        self.weights_updates_number += 1
        
        return winner_index