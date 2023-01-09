import numpy as np

from .bases import MultiInputSecondOrderBases

class AdaptiveFilterCerebellum:
    def __init__(self, dt, n_inputs, n_outputs, num_bases, beta):
        self.dt = dt
        self.beta = beta
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        # Initialize cortical bases
        self.tau_r = np.random.uniform(low=2.0, high=50.0, size=num_bases)*1e-3 # ms
        self.tau_d = np.random.uniform(low=50.0, high=750.0, size=num_bases)*1e-3 # ms
        self.p = MultiInputSecondOrderBases(dt, n_inputs, self.tau_r, self.tau_d)

        # Weights
        self.weights = np.zeros((n_inputs*num_bases, n_outputs))

        # Signals
        self.C = np.zeros(n_outputs)
        self.internal_error = np.zeros(n_outputs)

    def step(self, x, error):
        # Gives p_r, p_d and p at time t
        self.p.step(x)

        # Output of microcircuit
        self.C = np.dot(self.weights.T, self.p.value)

        # Update error signal and weights
        self._update_weights(error) # update before or after calc?

        return self.C


    def _update_weights(self, error):
        self.weights += self.beta * np.outer(self.p.value, error)

    @property
    def output(self):
        return self.C