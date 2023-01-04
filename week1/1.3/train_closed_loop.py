import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_model

from tqdm import tqdm  # progress bar



angles_diff = np.array([[0, 0]])
inp = np.array([[0, 0, 0, 0]])


datasets = ["training_data.p"]
for i in range(2, 31):
    datasets.append("training_data"+str(i)+".p")

for dataset in datasets:
    print("Processing {}".format(dataset))
    # Load data
    data = pickle.load(open(dataset, "rb"))
    angles = data[:, :2]
    angles /= 90.0
    end_pos = data[:, 2:]
    end_pos = (end_pos - 200) / 200

    for i, angle in enumerate(angles):
        for j, angle2 in enumerate(angles):
            if i!=j:
                angles_diff = np.concatenate((angles_diff, [angle2]))
                inp = np.concatenate((inp, [np.concatenate((end_pos[j] - end_pos[i], angle))]))
with open('processed.p', 'wb') as file:
    pickle.dump((inp, angles_diff), file)
# with open('processed.p', 'rb') as file:
#     inp, angles_diff = pickle.load(file)


inp = inp[1:, :]
angles_diff = angles_diff[1:, :]
print(inp)
print(angles_diff)
print("Training set generated")
# Use GPU?
device = 'cpu'
if torch.cuda.is_available():
    print("Using GPU")
    device = 'cuda'
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

x = torch.from_numpy(inp).float()
y = torch.from_numpy(angles_diff).float()
print(y)
if device == 'cuda':
    x = x.cuda()
    y = y.cuda()
# DONE split the training set and test set
print(len(x.cpu()))
train, test = torch.utils.data.random_split(range(len(x)), [0.8, 0.2])
x_train, y_train = x[train.indices], y[train.indices]
x_test, y_test = x[test.indices], y[test.indices]
# Eventually normalize the data

# Define neural network - an example
model = torch_model.Net(4, 50, 2)
model.to(device)
# model = torch_model.Net(n_feature=2, n_hidden1=h, n_hidden2=h, n_output=2)
# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()
num_epochs = 500

# h = 16
g = 0.5
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=g)

l_vec = np.zeros(num_epochs)
l_test_vec = np.zeros(num_epochs)

for t in tqdm(range(num_epochs)):
    # TODO Train network
    model.train()
    x, y = x_train, y_train
    prediction = model(x)  # Forward pass prediction. Saves intermediary values required for backwards pass
    loss = loss_func(prediction, y)  # Computes the loss for each example, using the loss function defined above
    optimizer.zero_grad()  # Clears gradients from previous iteration
    loss.backward()  # Backpropagation of errors through the network
    optimizer.step()  # Updating weights
    scheduler.step()

    l = loss.data
    if device == 'cuda':
        l = l.cpu()
    # print(l.numpy())
    l_vec[t] = l.numpy()

    model.eval()
    y_pred = model(x_test)
    test_loss = loss_func(y_pred, y_test)
    l_test_vec[t] = test_loss

# TODO Test the network
model.eval()
y_pred = model(x_test)
test_loss = loss_func(y_pred, y_test)
print(f'Test loss: {test_loss}')

plt.plot(l_vec)
plt.plot(l_test_vec)
plt.legend(['train', 'test'])
plt.yscale('log')

torch.save(model.state_dict(), 'closed_loop_trained_model.pth')
plt.show()

# Parameter TIPS - Try
# Adam with lr = 0.001
# StepLR scheduler with step_size=100, and gamma = 0.999
# Two hidden layers with 18 units each
