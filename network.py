import theano
import lasagne
import logging
import time
from math import exp

from model import Input


def main():

    logging.basicConfig(filename="network.log", level=logging.INFO)
    logging.info("Started")
    ANN(100)


class ANN(object):

    def create(self, num_epochs=10):

        input_var = theano.tensor.matrix('inputs')
        target_var = theano.tensor.vector('targets')
        network = self.build(input_var)
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = loss.mean()
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=0.01)
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
        test_loss = test_loss.mean()
        train_acc = theano.tensor.mean(theano.tensor.le(abs(prediction - target_var), exp(-4)))
        test_acc = theano.tensor.mean(theano.tensor.le(abs(test_prediction - target_var), exp(-4)))
        train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates)
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_accuracy = 0
            train_batches = 0
            start_time = time.time()
            for inputs, targets in self.input.get_batch("train"):
                err, acc = train_fn(inputs, targets)
                train_err += err
                train_accuracy += acc
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for inputs, targets in self.input.get_batch("validation"):
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we log the results for this epoch: # Finally, launch the training loop.
            logging.info("Epoch %d of %d took %.3f s - Training loss %.9f acc %.3f%% Validation loss %.9f acc %.3f%%", (epoch + 1), num_epochs, (time.time() - start_time), (train_err / train_batches), (train_accuracy / train_batches) * 100, (val_err / val_batches), (val_acc / val_batches) * 100)
        # After training, we compute the test loss and accuracy
        test_err = 0
        test_acc = 0
        test_batches = 0
        for inputs, targets in self.input.get_batch("test"):
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1

        logging.info("Final results: Test loss %.9f acc %.2f", (test_err / test_batches), (test_acc / test_batches) * 100)

    def build(self, input_var=None):
        
        network = lasagne.layers.InputLayer(shape=(50, 2), input_var=input_var)
        network = lasagne.layers.DenseLayer(network, num_units=16, W=lasagne.init.GlorotUniform(), nonlinearity=lasagne.nonlinearities.sigmoid)
        network = lasagne.layers.DenseLayer(network, num_units=1, W=lasagne.init.GlorotUniform(), nonlinearity=lasagne.nonlinearities.sigmoid)
        return network

    def __init__(self, nr_epochs):
        
        self.input = Input(1000000)
        self.create(nr_epochs)


if __name__ == "__main__":
    main()
