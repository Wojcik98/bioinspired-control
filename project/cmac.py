import numpy as np

def GaussianBasisFunction(x, mu, sigma):
    return np.exp(-(x-mu)**2/(sigma**2))

class CMAC:
    def __init__(self, n_rfs, xmin, xmax, beta=1e-3):
        """ Initialize the basis function parameters and output weights """
        self.n_rfs = n_rfs

        self.mu = np.zeros((2, self.n_rfs))
        self.sigma = np.zeros(2)
        crossval = 0.8 # has to be between 0 and 1 !

        for k in range(2):
            self.sigma[k] = 0.5/np.sqrt(-np.log(crossval)) * (xmax[k] - xmin[k])/(self.n_rfs-1) # RFs cross at phi = crossval
            self.mu[k] = np.linspace(xmin[k], xmax[k], self.n_rfs)
        
        self.w = np.random.normal(loc=0.0, scale=0.2, size=(self.n_rfs, self.n_rfs))

        self.beta = beta

        self.B = None
        self.y = None

    def predict(self, x):
        """ Predict yhat given x
            Saves activations `B` for later weight update
        """
        phi = np.zeros((2, self.n_rfs))
        for k in range(2):
            phi[k] = GaussianBasisFunction(x[k], self.mu[k], self.sigma[k]) # for i in phi_ki at the same time

        self.B = np.zeros((self.n_rfs, self.n_rfs))
        for i in range(self.n_rfs):
            for j in range(self.n_rfs):
                self.B[i,j] = phi[0][i] * phi[1][j]

        yhat = np.dot(self.w.ravel(), self.B.ravel()) # Element-wise multiplication and summing of all elements

        return yhat

    def learn(self, e):
        """ 
        Update the weights using the covariance learning rule
        For all weights at once.
        """
        self.w += self.beta*e*self.B


if __name__ == '__main__':
    n_rfs = 11

    xmin = [0, 0]
    xmax = [1, 1]

    c = CMAC(n_rfs, xmin, xmax, 1e-2)
    print(c.w.shape)

    for _ in range(1000):
        e_vec = []
        for x1 in np.linspace(0, 1, 11):
            for x2 in np.linspace(0, 1, 11):
                x = [x1, x2]

                yhat = c.predict(x)

                yd = np.arctan2(x[0], x[1])

                e = yd - yhat

                c.learn(e)
                e_vec.append(e**2)

        print(np.mean(e_vec))

    # Test values
    x = [0.5, 0.5]
    print(c.predict(x), np.arctan2(x[0], x[1]))

    x = [0.2, 0.5]
    print(c.predict(x), np.arctan2(x[0], x[1]))