import numpy as np

from kohonen_frequency import KohonenFrequencyNet

def main():
    learning_rate = 0.01
    first_layer_size = 5
    second_layer_size = 5

    kohonen_net = KohonenFrequencyNet(
        first_layer_size=first_layer_size,
        second_layer_size=second_layer_size,
        learning_rate=learning_rate
    )
    
    features_train = np.array(
        [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1]
        ]
    )

    kohonen_net.fit(features=features_train)

    features_test = np.array(
        [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1]
        ]
    )

    predictions = kohonen_net.predict(features=features_test)
    print(predictions)


if __name__ == "__main__":
    main()