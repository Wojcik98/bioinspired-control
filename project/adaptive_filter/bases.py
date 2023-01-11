
import numpy as np

class FirstOrderBases:
    '''
        First order low pass filter
        Transfer function: G = 1 / ( tau*s + 1 )
    '''
    def __init__(self, dt, tau):
        self.ky1 = np.exp(-dt/tau)
        self.ku1 = 1-self.ky1

        self.reset()

    def step(self, input_signal):
        self.value = self.ky1 * self.y1  +  self.ku1 * self.u1

        self.y1 = self.value
        self.u1 = input_signal

    def reset(self):
        self.value = 0
        self.y1 = 0
        self.u1 = 0

class DoubleFirstOrderBases:
    def __init__(self, dt, tau1, tau2=None):
        self.dt = dt

        if tau2 is None: # same time constant
            self.y1 = FirstOrderBases(dt, tau1)
            self.y2 = FirstOrderBases(dt, tau1)
        else:
            self.y1 = FirstOrderBases(dt, tau1)
            self.y2 = FirstOrderBases(dt, tau2)

        self.value = 0

    def step(self, input_signal):
        self.y1.step(input_signal)
        self.y2.step(self.y1.value)
        self.value = self.y2.value

    def reset(self):
        self.y1.reset()
        self.y2.reset()


class MultiInputSecondOrderBases:
    def __init__(self, dt, n_inputs, tau_r, tau_d):
        self.n_inputs = n_inputs
        self.n_bases = len(tau_r)

        self.tau_r = np.tile(tau_r, n_inputs) # repeats the whole array a number of times
        self.tau_d = np.tile(tau_d, n_inputs)
        self.p = DoubleFirstOrderBases(dt, self.tau_r, self.tau_d)

    def step(self, input_signal):
        # input signal has length num_inputs

        # p's have length num_inputs*num_bases
        # Repeat each element of input signal 'num_bases' times
        input_repeated = np.repeat(input_signal, self.n_bases)

        self.p.step(input_repeated)

    def reset(self):
        self.p.reset()

    @property
    def value(self):
        return self.p.value

if __name__ == '__main__':

    tau_r = np.array([50e-3, 60e-3])
    tau_d = np.array([100e-3, 110e-3])
    b = MultiInputSecondOrderBases(1e-3, n_inputs=2, tau_r=tau_r, tau_d=tau_d)
    b.step([0.5, 1])
    b.step([0.5, 1])
    b.step([0.5, 1])
    b.step([0.5, 1])
    print(b.value)