import copy
import numpy as np
from keras.datasets import mnist
from sklearn.utils import shuffle
np.seterr(all='raise')

class NeuralNet:
    def __init__(self, train_data, hidden_layers_sizes=[32], activ_functions=['relu', 'sig'],
                 output_layer_size=10, epochs=50, batch_size=16, learning_rate=1):
        self.debug = False
        self.train_input = train_data[0]
        self.train_output = train_data[1]
        self.train_data = train_data
        self.input_layer_size = np.prod(np.shape(self.train_input[0]))
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_layer_size = output_layer_size
        self.activ_functions = activ_functions
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
        self.layers_inputs = self.layers[1:]

    def init_weights_and_biases(self):
        """
        Initialisation of nn weights matrices and biases vectors
        """
        # for i in range(len(self.layers) - 1):
        #     self.weighs.append(np.random.random((len(self.layers[i + 1]), len(self.layers[i]))))
        # for i in range(1, len(self.layers)):
        #     self.biases.append(np.random.random((len(self.layers[i]))))
        for i in range(len(self.layers) - 1):
            self.weighs.append(np.random.uniform(low=-0.3, high=0.3, size=(len(self.layers[i + 1]), len(self.layers[i]))))
        for i in range(1, len(self.layers)):
            self.biases.append(np.random.uniform(low=-0.3, high=0.3, size=(len(self.layers[i]))))
        if self.debug:
            for weigh in self.weighs:
                print(np.shape(weigh))

    def activ_function(self, v, activ_function_type):
        if activ_function_type == 'relu':
            return self.relu(v)
        elif activ_function_type == 'sig':
            return self.sigmoid(v)

    def activ_der(self, v, activ_function_type):
        if activ_function_type == 'relu':
            return self.relu_der(v)
        elif activ_function_type == 'sig':
            return self.sigmoid_der(v)

    def sigmoid(self, v):
        # Sigmoid function flattens very fast. Raising to power v which can be very high
        # is computationally demanding and slows down the training
        sigmoid_v = []
        for e in v:
            if e > 10:
                e = 10
            if e < -10:
                e = -10
            sigmoid_v.append(1.0 / (1.0 + np.exp(-e)))
        return np.array(sigmoid_v)

    def sigmoid_der(self, v):
        # Sigmoid function flattens very fast. Raising to power v which can be very high
        # is computationally demanding and slows down the training
        sigmoid_der_v = []
        for e in v:
            if e > 10:
                e = 10
            if e < -10:
                e = -10
            sigmoid_der_v.append(np.exp(-e) / ((np.exp(-e) + 1) ** 2))
        return np.array(sigmoid_der_v)

    def relu(self, v):
        relu_v = []
        for e in v:
            if e > 0:
                relu_v.append(e)
            if e <= 0:
                relu_v.append(0)
        return np.array(relu_v)

    def relu_der(self, v):
        relu_der_v = []
        for e in v:
            if e > 0:
                relu_der_v.append(1)
            if e <= 0:
                relu_der_v.append(0)
        return np.array(relu_der_v)


    def feedforward(self, input_layer):
        # Transposing input for matrix multiplication
        current_layer = np.transpose(np.array(input_layer))
        # Assigning activations in first layer to input values
        self.layers[0] = current_layer
        # Calculating output vector using matrix multiplication
        for i in range(len(self.weighs)):
            # Saving vectors of inputs to consecutive net layers, that will later
            # be used during the error backpropagation algorithm
            self.layers_inputs[i] = np.add(np.matmul(self.weighs[i], current_layer), self.biases[i])
            # Saving vectors of activations of consecutive net layers, that will later
            # be used during the error backpropagation algorithm
            # index is i+1 since there is 1 more layer than weight vectors (first layer)
            # so we update all layers instead of first. First was assigned to input layer
            # before the loop.
            self.layers[i + 1] = self.activ_function(self.layers_inputs[i], self.activ_functions[i])
            current_layer = self.layers[i + 1]
        return current_layer

    def train(self):
        for i in range(self.epochs):
            epoch_error = self.train_epoch()
            print(f'{"Epoch: "}{i}{" / "}{self.epochs}')
            print(f'{"Error: "}{epoch_error}')
            # print("LAST LAYER OF WEIGHTS")
            # print(self.weighs[-1])


    def train_epoch(self):
        epoch_error = 0
        batches = self.create_batches()
        for batch in batches:
            epoch_error += self.train_batch(batch)
        # Adding epoch cost to the list containing all losses across all epochs
        self.overall_error.append(epoch_error)
        return epoch_error

    def train_batch(self, batch):
        inputs = batch[0]
        outputs = batch[1]
        batch_output_error = 0
        batch_error_vectors = [np.zeros(len(layer)) for layer in self.layers][1:]
        for k in range(len(inputs)):
            # Inputs in batch
            i = self.flatten_matrix(inputs[k])
            # Outputs in batch
            o = self.flatten_matrix(outputs[k])
            self.ff_net_output = self.feedforward(i)
            self.expected_output = self.net_output(o)
            example_error, error_to_backprop = self.example_error(self.ff_net_output, self.expected_output)
            batch_output_error += example_error
            # Running backpropagation algorithm to calculate errors on
            # weights and biases on every layer
            example_error_vectors = self.backprop(error_to_backprop)
            # Updating overall batch error with example error values
            for i in range(len(batch_error_vectors)):
                batch_error_vectors[i] += example_error_vectors[len(batch_error_vectors) - i - 1]
        # print("BATCH OUTPUT ERROR")
        # print(batch_output_error)
        # After all batch layers errors are calculated, SGD is performed
        # to update weights and biases
        self.sgd(batch_error_vectors)
        return batch_output_error

    def backprop(self, output_error_vector):
        example_error_vectors = []
        current_error_vector = output_error_vector
        # Appending the vector of errors in output layer
        # to the example_error_vectors
        example_error_vectors.append(current_error_vector)
        for i in range(len(self.layers) - 1, 1, -1):
            # print("I")
            # print(i)
            # print("LEN OD LAYER OD I")
            # print(np.size(self.layers[i]))
            #
            # print("SHAPE OF WEIGHTS")
            # print(np.shape(self.weighs[i - 1]))
            # print("SHAPE OF current_error_vector")
            # print(np.shape(current_error_vector))
            # print("LAYERS INPUTS I-2 SHAPE")
            # print(np.shape(self.layers_inputs[i - 2]))
            # print("LAYERS INPUTS I-2 ")
            # print(self.layers_inputs[i - 2])
            # print("OUTPUT ERROR")
            # print(output_error_vector)
            # print("SELF WEIGTHS -1")
            # print(self.weighs[i - 1])
            # print("SELF SIGMOID DER")
            # print(self.activ_der(self.layers_inputs[i - 2], self.activ_functions[i-1]))
            # print("CURRENT ERROR VECTOR WITHOUT DER")
            # print(np.matmul(np.transpose(self.weighs[i - 1]), current_error_vector))

            current_error_vector = np.multiply(np.matmul(np.transpose(self.weighs[i - 1]), current_error_vector),
                                    np.array(self.activ_der(self.layers_inputs[i - 2], self.activ_functions[i-1])))

            # print("CURRENT ERROR VECTOR")
            # print(current_error_vector)


            example_error_vectors.append(current_error_vector)
        return example_error_vectors

    def sgd(self, batch_error_vectors):
        for i in range(len(self.weighs)):
            # Update the values in the vector of weights in the steepest gradient direction
            # print("WEIGHTS UPDATE")
            # print(np.multiply(self.learning_rate / self.batch_size,
            #     np.matmul(batch_error_vectors[i], np.transpose(self.layers[i + 1]))))
            # print("BIASES UPDATE")
            # print(np.multiply(self.learning_rate / self.batch_size,
            #         batch_error_vectors[i]))

            # print(np.multiply(self.learning_rate / self.batch_size, np.sum(
            #     np.matmul(batch_error_vectors[i], np.transpose(self.layers[i + 1])))))
            # self.weighs[i] = np.subtract(self.weighs[i], np.multiply(self.learning_rate / self.batch_size, np.sum(
            #     np.matmul(batch_error_vectors[i], np.transpose(self.layers[i + 1])))))
            # Update the values in the vector of biases in the steepest gradient direction
            # self.biases[i] = np.subtract(self.biases[i], np.multiply(self.learning_rate / self.batch_size, np.sum(
            #     batch_error_vectors[i])))
            # self.weighs[i] = np.subtract(self.weighs[i], np.multiply(self.learning_rate / self.batch_size,
            #     np.matmul(batch_error_vectors[i], np.transpose(self.layers[i + 1]))))
            # self.biases[i] = np.subtract(self.biases[i], np.multiply(self.learning_rate / self.batch_size,
            #     batch_error_vectors[i]))
            # print("WEIGHT UPDATE")
            # print(np.shape(batch_error_vectors[i]))
            # print(np.shape(self.layers[i + 1]))
            # print(np.shape(self.weighs[i]))
            # print(np.shape(np.outer(batch_error_vectors[i], np.transpose(self.layers[i]))))      # print(np.shape(np.multiply(self.learning_rate / self.batch_size,

            self.weighs[i] = np.subtract(self.weighs[i], np.multiply(self.learning_rate / self.batch_size,
                            np.outer(batch_error_vectors[i], np.transpose(self.layers[i]))))
            self.biases[i] = np.subtract(self.biases[i], np.multiply(self.learning_rate / self.batch_size,
                            batch_error_vectors[i]))

            #                   np.matmul(batch_error_vectors[i], np.transpose(self.layers[i + 1])))))
            # print(np.multiply(self.learning_rate / self.batch_size,
            #             np.matmul(batch_error_vectors[i], np.transpose(self.layers[i + 1]))))
            # self.weighs[i] = np.subtract(self.weighs[i], np.multiply(self.learning_rate / self.batch_size,
            #     np.matmul(batch_error_vectors[i], np.transpose(self.layers[i + 1]))))
            # self.biases[i] = np.subtract(self.biases[i], np.multiply(self.learning_rate / self.batch_size,
            #     batch_error_vectors[i]))


    def example_error(self, net_output, expected_output):
        # print("EXAMPLE ERROR")
        # print(np.divide(np.square(np.subtract(net_output, expected_output)), 2),
        #                    self.sigmoid_der(self.layers_inputs[-1]))
        # print("EXAMPLE ERROR AFTER MULTIPLICATION VIA SIGMOID DER OF LAST LAYER INPUT")
        # print(np.multiply(np.divide(np.square(np.subtract(net_output, expected_output)), 2),
        #                    self.sigmoid_der(self.layers_inputs[-1])))
        # print("LAST LAYER INPUT")
        # print(self.layers_inputs[-1])
        # print("LAST LAYER OUTPUT")
        # print(self.layers[-1])
        # print("NET OUTPUT")
        # print(net_output)
        # print("EXPECTED OUTPUT")
        # print(expected_output)
        error = np.divide(np.square(np.subtract(net_output, expected_output)), 2)
        return (error, np.multiply(error, self.activ_der(self.layers_inputs[-1], self.activ_functions[-1])))
        # return np.divide(np.square(np.subtract(net_output, expected_output)), 2)

    def create_batches(self):
        batches = []
        # Shuffling training dataset
        sh_t_in, sh_t_out = shuffle(self.train_input, self.train_output, random_state=0)
        # Sampling a mini-batch from the dataset
        i = 0
        # while i < len(sh_t_in - self.batch_size):
        while i < len(sh_t_in) - self.batch_size:
            batches.append((sh_t_in[i:i + self.batch_size], sh_t_out[i:i + self.batch_size]))
            i += self.batch_size
        return batches

    def net_output(self, output):
        net_output = np.zeros(10)
        net_output[output] = 1
        return net_output

    def flatten_matrix(self, matrix):
        if len(np.shape(matrix)) > 1:
            return matrix.flatten()
        else:
            return matrix

if __name__ == "__main__":
    dataset_size = 3000
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X[:dataset_size]
    train_y = train_y[:dataset_size]
    print("Preprocessing MNIST")
    prep_train_x = []
    for i in range(len(train_X)):
        prep_train_x.append((np.array(train_X[i]) / 255.0))
    prep_train_x = np.array(prep_train_x)
    train_X = prep_train_x
    print("Preprocessing finished")

    nn = NeuralNet((train_X, train_y))
    nn.train()
