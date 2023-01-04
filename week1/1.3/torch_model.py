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
            torch.nn.Linear(n_feature, n_hidden),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_feature, n_hidden),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        x = self.net(x)
        return x
