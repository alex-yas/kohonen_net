import numpy as np

class KohonenNet():
    def __init__(self, first_layer_size: int, second_layer_size: int, learning_rate: float):
        self.first_layer_size = first_layer_size
        self.second_layer_size = second_layer_size
        self.learning_rate = learning_rate

        self.weights = np.random.normal(size=(first_layer_size, second_layer_size))
        self.wins = np.zeros(self.second_layer_size)
    
    def _update_weights(self, sample: np.array) -> int:
        winner_index = self._find_winner(sample=sample)

        for ind in range(self.second_layer_size):
            self.weights[ind][winner_index] += self.learning_rate * (sample[ind] - self.weights[ind][winner_index])
        
        return winner_index
    
    def _find_winner(self, sample: np.array) -> int:
        second_layer_output = np.empty(self.second_layer_size)

        for j in range(self.second_layer_size):
            second_layer_output[j] = sum([self.weights[i][j] * sample[i] for i in range(self.first_layer_size)])
        
        winner_index = np.argmax(second_layer_output)

        return winner_index
    
    def fit(self, features: np.array):
        for sample in features:
            self._update_weights(sample=sample)
    
    def predict(self, features) -> np.array:
        predictions = np.array([self._find_winner(sample=sample) for sample in features])

        return predictions