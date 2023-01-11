import numpy as np

class SingleLink:
    def __init__(self, dt):
        self.theta = 0
        self.omega = 0

        self.m = 0.25 # link mass
        self.l = 1.0 # link lengths
        self.I = 1/3*self.m*self.l**2 # link inertia, rod

        self.b = 5 # link 
        self.g_const = 9.82 # m/s^2

        self.sim_step = 1e-3 # time step for Euler integration

        if dt < self.sim_step:
            self.sim_step = dt
        self.dt = dt # time step of step function

    def step(self, tau):
        t = 0
        while t < self.dt:
            alpha = (tau - self.b*self.omega - self.g) / self.I

            self.theta += self.omega*self.sim_step
            self.omega +=      alpha*self.sim_step
            t += self.sim_step

    @property
    def g(self):
        return self.m*self.g_const*(0.5*self.l)*np.cos(self.theta)

if __name__ == '__main__':
    r = SingleLink(1e-3)

    theta = []
    for _ in range(20*1000):
        r.step(0)
        theta.append(r.theta)

    print(theta[-1], -np.pi/2)
    import matplotlib.pyplot as plt
    plt.plot(theta)
    plt.show()
