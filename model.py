from random import uniform
from sklearn.model_selection import train_test_split
import numpy as np


class Input(object):

    '''A class which is responsible for creating and feeding the input to the network.

    Attributes:
        x_train: numpy array with all the training inputs
        x_validation: numpy array with all the validation inputs
        x_test: numpy array with all the test inputs
        y_train: list with all the training labels
        y_validation: list with all the validation labels
        y_test: list with all the test labels
    '''

    def __init__(self, nr_examples):

        self.x_train = []
        self.x_test = []
        self.x_validation = []
        self.y_train = []
        self.y_test = []
        self.y_validation = []
        self.split_data(self.generate_data(nr_examples))
        self.x_train = np.array(self.x_train)
        self.x_validation = np.array(self.x_validation)
        self.x_test = np.array(self.x_test)


    def generate_data(self, nr_examples):
        x = []
        y = []
        for i in range(0, nr_examples):
            a = uniform(0, 1)
            b = uniform(0, 1)
            x.append([a, b])
            y.append(a*b)
        return x, y

    def split_data(self, data,  percentage=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data[0], data[1], test_size=percentage, random_state=69)
        self.x_train, self.x_validation, self.y_train, self.y_validation = train_test_split(self.x_train, self.y_train, test_size= 0.1, random_state=11)

    def get_batch(self, phase, batch_size=50):

        if(phase == "train"):
            x = self.x_train
            y = self.y_train
        elif (phase == "validation"):
            x = self.x_validation
            y = self.y_validation
        elif (phase == "test"):
            x = self.x_test
            y = self.y_test
        else:
            raise ValueError("The given case does not match any of the alternatives")

        for i in range(0, x.shape[0] - batch_size, batch_size):
            yield x[i:i + batch_size, :], y[i:i + batch_size]
