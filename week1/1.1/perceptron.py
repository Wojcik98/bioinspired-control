import numpy as np
from activation import ActivationFunction
import matplotlib.pyplot as plt


class SignActivation(ActivationFunction):
    """
          Sign activation: `f(x) = 1 if x > 0, 0 if x <= 0`
    """
    def forward(self, x):
        """
           This is the output function.
           TODO: Define the correct return function, given input `x`
        """
        return 1.0 if x > 0 else 0.0

    def gradient(self, x):
        """
              Function derivative.
              Define the correct return value (derivative), given input `x`
        """
        return 0


class Perceptron:
    """
       Perceptron neuron model
       Parameters
       ----------
       n_inputs : int
          Number of inputs
       act_f : Subclass of `ActivationFunction`
          Activation function
    """
    def __init__(self, n_inputs, act_f):
        """
           Perceptron class initialization
           TODO: Write the code to initialize weights and save the given activation function
        """
        if not isinstance(act_f, type) or not issubclass(act_f, ActivationFunction):
            raise TypeError('act_f has to be a subclass of ActivationFunction (not a class instance).')
        # weights
        mean, std = 0.0, 1.0
        self.w = np.random.normal(mean, std, n_inputs + 1)
        # activation function
        self.f = act_f()

        if self.f is not None and not isinstance(self.f, ActivationFunction):
            raise TypeError("self.f should be a class instance.")

    def activation(self, x):
        """
           It computes the activation `a` given an input `x`
           TODO: Fill in the function to provide the correct output
           NB: Remember the bias
        """
        a = np.dot(np.append(x, [1]), self.w)
        return a

    def output(self, a):
        """
           It computes the neuron output `y`, given the activation `a`
           TODO: Fill in the function to provide the correct output
        """
        y = self.f.forward(a)
        return y

    def predict(self, x):
        """
           It computes the neuron output `y`, given the input `x`
           TODO: Fill in the function to provide the correct output
        """
        return self.output(self.activation(x))

    def gradient(self, a):
        """
           It computes the gradient of the activation function, given the activation `a`
        """
        return self.f.gradient(a)


if __name__ == '__main__':
    data = np.array([
        [0.5, 0.5, 0],
        [1.0, 0, 0],
        [2.0, 3.0, 0],
        [0, 1.0, 1],
        [0, 2.0, 1],
        [1.0, 2.2, 1]
    ])
    xdata = data[:, :2]
    ydata = data[:, 2]
    print(xdata)
    print(ydata)

    # TODO Test your activation function
    a = SignActivation()
    print(a.forward(2))
    "print(a.forward(0))"

    # TODO Test perceptron initialization
    p = Perceptron(2, SignActivation)
    print(p.predict(xdata[0, :]))

    # TODO Learn the weights
    r = 0.001     # learning rate
    # calculate the error and update the weights
    epochs = 1000

    for epoch in range(epochs):
        for i in range(xdata.shape[0]):
            x = xdata[i, :]
            y_pred = p.predict(x)
            y_target = ydata[i]
            p.w += r * (y_target - y_pred) * np.append(x, [1])

    print(p.w)
    # TODO plot points and linear decision boundary

    y_preds = np.array([p.predict(x) for x in xdata])
    print(ydata)
    print(y_preds)

    c1 = ydata > 0.5
    c2 = ydata <= 0.5
    bad = y_preds != ydata

    plt.plot(xdata[c1][:, 0], xdata[c1][:, 1], 'go')
    plt.plot(xdata[c2][:, 0], xdata[c2][:, 1], 'bo')
    plt.plot(xdata[bad][:, 0], xdata[bad][:, 1], 'rx', markersize=20)

    x1 = np.linspace(-0.5, 2.5)
    x2 = (-p.w[2] - p.w[0]*x1) / p.w[1]
    plt.plot(x1, x2, 'm--')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

