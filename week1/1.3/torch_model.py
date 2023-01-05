import torch


class MLPtorch(torch.nn.Module):
    def __init__(self):
        super(MLPtorch, self).__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 2)

    def forward(self, x):
        res = torch.sigmoid(self.fc1(x))
        res = self.fc2(res)
        return res


class MLPNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLPNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = self.predict(x)
        return x


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_feature, n_hidden),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class RNNNet(torch.nn.Module):
    def __init__(self, n_feature=4, n_ffnn=20, layers_rnn=2, n_RNN=10, n_latent=6, n_target=4, n_decoder=50, n_output=2):
        super(RNNNet, self).__init__()
        self.ffnn = torch.nn.Sequential(
            torch.nn.Linear(n_feature, n_ffnn),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_ffnn, n_ffnn),
            torch.nn.Sigmoid(),
        )
        self.RNN = torch.nn.RNN(n_ffnn, n_RNN, layers_rnn, batch_first=True)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_RNN, n_latent),
            torch.nn.Sigmoid(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_latent + n_target, n_decoder),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_decoder, n_decoder),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_decoder, n_output),
            torch.nn.Sigmoid(),
        )

    def forward(self, measurement, target, train=True):
        x = self.encode(measurement)
        x = self.decode(target, x=x, train=train)
        return x

    def encode(self, measurement):
        x = self.ffnn(measurement)
        x = self.RNN(x)[0]
        x = self.encoder(x)
        return x

    def decode(self, target, x=None, train=True):
        if x is None:
            x_shape = list(target.shape)
            x_shape[2] = 6
            x = torch.zeros(x_shape).cuda()
        if train:
            x = torch.cat((x, target), 2)
        else:
            x = torch.cat((x, target), 1)  # the input isn't batched
        x = self.decoder(x)
        return x

