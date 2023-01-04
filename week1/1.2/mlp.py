import numpy as np
import copy

from perceptron import Perceptron
from activation import ActivationFunction

import matplotlib.pyplot as plt

class Sigmoid(ActivationFunction):
    """
       Sigmoid activation: `f(x) = 1/(1+e^(-x))`
    """
    def forward(self, x):
        """
           Activation function output.
           DONE: Change the function to return the correct value, given input `x`.
        """
        return 1 / (1 + np.exp(-1 * x))

    def gradient(self, x):
        """
           Activation function derivative.
           DONE: Change the function to return the correct value, given input `x`.
        """
        a = self.forward(x)
        return a * (1 - a)


class LinearActivation(ActivationFunction):
    """
       Linear activation: `f(x) = x`
    """
    def forward(self, x):
        """
           Activation function output.
           DONE: Change the function to return the correct value, given input `x`.
        """
        return x

    def gradient(self, x):
        """
           Activation function derivative.
           DONE: Change the function to return the correct value, given input `x`.
        """
        return 1


class Layer:
    def __init__(self, num_inputs, num_units, act_f):
        """
           Initialize the layer, creating `num_units` perceptrons with `num_inputs` each.
        """
        # DONE Create the perceptrons required for the layer

        self.num_units = num_units
        self.ps = [Perceptron(num_inputs, act_f) for _ in range(num_units)]

    def activation(self, x):
        """ Returns the activation `a` of all perceptrons in the layer, given the input vector`x`. """
        return np.array([p.activation(x) for p in self.ps])

    def output(self, a):
        """ Returns the output `o` of all perceptrons in the layer, given the activation vector `a`. """
        return np.array([p.output(ai) for p, ai in zip(self.ps, a)])

    def predict(self, x):
        """ Returns the output `o` of all perceptrons in the layer, given the input vector `x`. """
        return np.array([p.predict(x) for p in self.ps])

    def gradient(self, a):
        """ Returns the gradient of the activation function for all perceptrons in the layer, given the activation vector `a`. """
        return np.array([p.gradient(ai) for p, ai in zip(self.ps, a)])

    def update_weights(self, dw):
        """
        Update the weights of all of the perceptrons in the layer, given the weight change of each.
        Input size: (n_inputs+1, n_units)
        """
        for i in range(self.num_units):
            self.ps[i].w += dw[:, i]

    @property
    def w(self):
        """
           Returns the weights of the neurons in the layer.
           Size: (n_inputs+1, n_units)
        """
        return np.array([p.w for p in self.ps]).T

    def import_weights(self, w):
        """
           Import the weights of all of the perceptrons in the layer.
           Input size: (n_inputs+1, n_units)
        """
        for i in range(self.num_units):
            self.ps[i].w = w[:, i]


class MLP:
    """
       Multi-layer perceptron class

    Parameters
    ----------
    n_inputs : int
       Number of inputs
    n_hidden_units : int
       Number of units in the hidden layer
    n_outputs : int
       Number of outputs
    alpha : float
       Learning rate used for gradient descent
    """
    def __init__(self, num_inputs, n_hidden_units, n_outputs, alpha=1e-3):
        self.num_inputs = num_inputs
        self.n_hidden_units = n_hidden_units
        self.n_outputs = n_outputs

        self.alpha = alpha

        # DONE: Define a hidden layer and the output layer
        self.l1 = Layer(num_inputs, n_hidden_units, Sigmoid) # hidden layer 1
        self.l_out = Layer(n_hidden_units, n_outputs, LinearActivation) # output layer

    def predict(self, x):
        """
        Forward pass prediction given the input x
        DONE: Write the function
        """
        y = self.l1.predict(x)
        y = self.l_out.predict(y)
        return y

    def train(self, inputs, outputs):
        """
           Train the network

        Parameters
        ----------
        `x` : numpy array
           Inputs (size: n_examples, n_inputs)
        `t` : numpy array
           Targets (size: n_examples, n_outputs)

        DONE: Write the function to iterate through training examples and apply gradient descent to update the neuron weights
        """

        N = len(inputs)

        dw3 = np.zeros([self.n_hidden_units+1, self.n_outputs])
        dw1 = np.zeros([self.num_inputs+1, self.n_hidden_units])

        # Loop over training examples
        for i, t in enumerate(outputs):
            # Forward pass
            a1 = self.l1.activation(inputs[i])
            o1 = self.l1.output(a1)
            a2 = self.l_out.activation(o1)
            o2 = self.l_out.output(a2)

            # Backpropagation
            err2 = self.l_out.gradient(a2) * (o2 - t)
            err1 = np.multiply(self.l1.gradient(a1), self.l_out.w[:-1] @ err2)

            delta_out = -1 * err2 * self.alpha / N
            delta1 = -1 * err1 * self.alpha / N

            inp = inputs[i]

            # Add weight change contributions to temporary array
            o0 = np.insert(inp, 0, 1)
            o1 = np.insert(o1, 0, 1)

            dw1 += delta1.reshape(-1,1).dot(o0.reshape(1,-1)).T
            dw3 += delta_out.reshape(-1,1).dot(o1.reshape(1,-1)).T

        # Update weights

        self.l1.update_weights(dw1)
        self.l_out.update_weights(dw3)

    def export_weights(self):
        return [self.l1.w, self.l_out.w]

    def import_weights(self, ws):
        if ws[0].shape == (self.l1.n_units, self.n_inputs+1) and ws[1].shape == (self.l_out.n_units, self.l1.n_units+1):
            print("Importing weights..")
            self.l1.import_weights(ws[0])
            self.l_out.import_weights(ws[1])
        else:
            print("Sizes do not match")


def calc_prediction_error(model, x, t):
    """ Calculate the average prediction error """
    # DONE: Write the function
    loss = 0
    for i, inp in enumerate(x):
        loss += np.linalg.norm(model.predict(inp) - t[i])**2
    return loss / len(x)


if __name__ == "__main__":
    # DONE: Test new activation functions
    print("Activation function test: ")
    fn_sigmoid = Sigmoid()
    print("Sigmoid(2) = {}".format(fn_sigmoid.forward(2)))
    print("Sigmoid'(2) = {}".format(fn_sigmoid.gradient(2)))
    fn_linear = LinearActivation()
    print("Linear(2) = {}".format(fn_linear.forward(2)))
    print("Linear'(2) = {}".format(fn_linear.gradient(2)))

    # DONE: Test Layer class init
    l = Layer(2, 5, LinearActivation)
    print("Layer test:")
    print("Prediction to [pi, 1] = {}".format(l.predict([np.pi, 1])))
    print("Weights = {}".format(l.w))



    # DONE: Test MLP class init
    m = MLP(2, 3, 1)
    print("MLP test:")
    print("Prediction to [pi, 1] = {}".format(m.predict([np.pi, 1])))
    print("Weights = {}".format(m.export_weights()))

    example_x = np.array([[1, 1], [2, 2], [3, 3]])
    example_t = np.array([[1], [2], [3]])
    print("Test loss function = {}".format(calc_prediction_error(m, example_x, example_t)))


    # DONE: Training data
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    t = np.array([[0], [1], [1], [0]])


    # DONE: Initialization
    modele2 = MLP(2, 2, 1, alpha=1e-2)
    modele3 = copy.deepcopy(modele2)
    modele3.alpha = 1e-3
    modele4 = copy.deepcopy(modele2)
    modele4.alpha = 1e-4



    # DONE: Write a for loop to train the network for a number of iterations. Make plots.
    lossese2 = []
    lossese3 = []
    lossese4 = []
    for i in range(5000):
        modele2.train(x, t)
        modele3.train(x, t)
        modele4.train(x, t)
        lossese2.append(calc_prediction_error(modele2, x, t))
        lossese3.append(calc_prediction_error(modele3, x, t))
        lossese4.append(calc_prediction_error(modele4, x, t))


    plt.plot(lossese2, label="a = 1e-2")
    plt.plot(lossese3, label="a = 1e-3")
    plt.plot(lossese4, label="a = 1e-4")
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    pass
