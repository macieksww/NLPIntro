import numpy as np
from keras.datasets import mnist
from sklearn.utils import shuffle

class NeuralNet:
    def __init__(self, train_data, hidden_layers_sizes=[8, 4], output_layer_size=2,
                 epochs=10, batch_size=16, learning_rate=0.1):
        self.train_input = train_data[0]
        self.train_output = train_data[1]
        self.train_data = train_data
        self.input_layer_size = np.prod(np.shape(self.train_input[0]))
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_layer_size = output_layer_size
        self.layers = []
        self.layers_inputs = []
        self.weighs = []
        self.biases = []
        self.overall_error = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_layers()
        self.init_weights_and_biases()
        self.debug = True

    def init_layers(self):
        """
        self.layers is a list of all layers that nn consists of
        """
        hls = ([None] * size for size in self.hidden_layers_sizes)
        self.layers.append([None] * self.input_layer_size)
        [self.layers.append(hl) for hl in hls]
        self.layers.append([None] * self.output_layer_size)
        for i in range(len(self.layers)):
            self.layers[i] = np.transpose(np.array(self.layers[i]))
        for i in range(1, len(self.layers)):
            self.layers_inputs[i] = np.transpose(np.array(self.layers[i]))

    def init_weights_and_biases(self):
        """
        Initialisation of nn weights matrices and biases vectors
        """
        for i in range(len(self.layers)-1):
            self.weighs.append(np.random.random((len(self.layers[i+1]), len(self.layers[i]))))
        for i in range(1, len(self.layers)):
            self.biases.append(np.random.random((len(self.layers[i]))))

        for weigh in self.weighs:
            print(np.shape(weigh))

    def sigmoid(self, v):
        return 1.0/(1.0+np.exp(-v))

    def sigmoid_der(self, v):
        return (np.exp(-v)/((np.exp(-v)+1)**2))

    def feedforward(self, input_layer):
        # Transposing input for matrix multiplication
        current_layer = np.transpose(np.array(input_layer))
        # Calculating output vector using matrix multiplication
        for i in range(len(self.weighs)):
            current_layer = self.sigmoid(np.add(np.matmul(self.weighs[i], current_layer), self.biases[i]))
            # Saving vectors of inputs to consecutive net layers, that will later
            # be used during the error backpropagation algorithm
            self.layers_inputs.append(np.add(np.matmul(self.weighs[i], current_layer), self.biases[i]))
            # Saving vectors of activations of consecutive net layers, that will later
            # be used during the error backpropagation algorithm
            self.layers.append(self.sigmoid(np.add(np.matmul(self.weighs[i], current_layer), self.biases[i])))
        return current_layer

    def train(self):
        for i in range(self.epochs):
            epoch_error = self.train_epoch()
            print(f'{"Epoch: "}{i}{" / "}{self.epochs}')
            print(f'{"Error: "}{epoch_error}')

    def train_epoch(self):
        epoch_error = 0
        batches = self.create_batches()
        for batch in batches:
            epoch_error += self.train_batch(batch)
        # Adding epoch cost to the list containing all losses across all epochs
        self.overall_error.append(epoch_error)
        return epoch_error

    def train_batch(self, batch):
        batch_output_error = 0
        batch_error_vectors = [np.zeros(len(layer)) for layer in self.layers][:-1]
        for i, o in batch:
            batch_output_error += self.example_error(self.feedforward(i), self.net_output(o))
            # Running backpropagation algorithm to calculate errors on
            # weights and biases on every layer
            example_error_vectors = self.backprop(self.example_error(self.feedforward(i), self.net_output(o)))
            # Updating overall batch error with example error values
            for i in range(len(batch_error_vectors)):
                batch_error_vectors[i] += example_error_vectors[i]

        # After all batch layers errors are calculated, SGD is performed
        # to update weights and biases
        self.sgd(batch_error_vectors)
        return batch_output_error

    def backprop(self, output_error_vector):
        example_error_vectors = []
        current_error_vector = output_error_vector
        for i in range(len(self.layers)-1, 1, -1):
            current_error_vector = np.multiply(np.matmul(np.transpose(self.weighs[i]), current_error_vector),
                                             np.array(list(map(self.sigmoid_der, self.layers_inputs[i]))))
            example_error_vectors.append(current_error_vector)
        return example_error_vectors

    def sgd(self, batch_error_vectors):
        for i in range(len(self.weighs)):
            # Update the values in the vector of weights in the steepest gradient direction
            self.weighs[i] = np.subtract(self.weighs[i], np.multiply(self.learning_rate/self.batch_size, np.sum(
                            np.matmul(batch_error_vectors[i], np.transpose(self.layers[i])))))
            # Update the values in the vector of biases in the steepest gradient direction
            self.biases[i] = np.subtract(self.biases[i], np.multiply(self.learning_rate/self.batch_size, np.sum(
                            batch_error_vectors[i])))

    def example_error(self, net_output, expected_output):
        return np.divide(np.square(np.subtract(net_output, expected_output)), 2)

    def create_batches(self):
        batches = []
        # Shuffling training dataset
        sh_t_in, sh_t_out = shuffle(self.train_input, self.train_output, random_state=0)
        # Sampling a mini-batch from the dataset
        i = 0
        while i < len(sh_t_in-self.batch_size):
            batches.append((sh_t_in[i:i+self.batch_size], sh_t_out[i:i+self.batch_size]))
            i += self.batch_size
        return batches

    def net_output(self, output):
        net_output = np.zeros(10)
        net_output[output] = 1
        return net_output

if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print(np.shape(train_X[0]))
    print(type(train_y[0]))
    nn = NeuralNet((train_X, train_y))


