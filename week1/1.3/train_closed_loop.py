import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_model

from tqdm import tqdm  # progress bar

N_measurements = 400 # how long is the sequence
N_measurement_series = 10 # how many sequences generated from each dataset

datasets = ["training_data.p"]
for i in range(2, 32):
    datasets.append("training_data"+str(i)+".p")

training_data = np.array([[[0, 0, 0, 0] for _ in range(N_measurements)]])

for dataset in datasets:
    print("Processing {}".format(dataset))
    # Load data
    data = pickle.load(open(dataset, "rb"))
    angles = data[:, :2]
    angles /= 90.0
    end_pos = data[:, 2:]
    end_pos = (end_pos - 200) / 200

    for i in range(N_measurement_series):
        all_measurements = np.array([np.concatenate((end_pos, angles), 1)])
        idx = np.random.randint(all_measurements.shape[1], size=N_measurements)
        chosen_measurements = all_measurements[:, idx, :]
        training_data = np.concatenate((training_data, chosen_measurements))
with open('processed.p', 'wb') as file:
    pickle.dump(training_data, file)
# with open('processed.p', 'rb') as file:
#     inp, angles_diff = pickle.load(file)


training_data = training_data[1:, :]
print(training_data)
print("Training set generated")
# Use GPU?
device = 'cpu'
if torch.cuda.is_available():
    print("Using GPU")
    device = 'cuda'
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

x = torch.from_numpy(training_data[:, :-2, :]).float()
diff = training_data[:, 2:, :2] - training_data[:, 1:-1, :2]
curr_pos = training_data[:, 1:-1, 2:]
x2 = torch.from_numpy(np.concatenate((diff, curr_pos), 2)).float()
y = torch.from_numpy(training_data[:, 2:, 2:]).float()

if device == 'cuda':
    x = x.cuda()
    x2 = x2.cuda()
    y = y.cuda()
# DONE split the training set and test set
print(len(x.cpu()))
train, test = torch.utils.data.random_split(range(len(x)), [0.8, 0.2])
x_train, x2_train, y_train = x[train.indices], x2[train.indices], y[train.indices]
x_test, x2_test, y_test = x[test.indices], x2[test.indices], y[test.indices]
# Eventually normalize the data

# Define neural network - an example
model = torch_model.RNNNet()
model.to(device)
# model = torch_model.Net(n_feature=2, n_hidden1=h, n_hidden2=h, n_output=2)
# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
regime = 0
loss_func = torch.nn.MSELoss()
num_epochs = 20000

# h = 16
g = 0.5
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=g)

l_vec = np.zeros(num_epochs)
l_test_vec = np.zeros(num_epochs)
l_compare_vec = np.zeros(num_epochs)

for t in tqdm(range(num_epochs)):
    # TODO Train network
    if regime==0 and t > 300:
        print("Decreasing learning rate")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        regime = 1

    model.train()
    x, x2, y = x_train, x2_train, y_train

    prediction = model.forward(x, x2)  # Forward pass prediction. Saves intermediary values required for backwards pass
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
    y_pred = model(x_test, x2_test)
    test_loss = loss_func(y_pred, y_test)
    l_test_vec[t] = test_loss
    y_compare = model.decode(x2_test)
    test_loss = loss_func(y_compare, y_test)
    l_compare_vec[t] = test_loss

# TODO Test the network
model.eval()
y_pred = model(x_test, x2_test)
test_loss = loss_func(y_pred, y_test)
print(f'Test loss: {test_loss}')

plt.plot(l_vec)
plt.plot(l_test_vec)
plt.plot(l_compare_vec)
plt.legend(['train', 'test', 'no RNN'])
plt.yscale('log')

torch.save(model.state_dict(), 'closed_loop_trained_RNN_model4.pth')
plt.show()

# Parameter TIPS - Try
# Adam with lr = 0.001
# StepLR scheduler with step_size=100, and gamma = 0.999
# Two hidden layers with 18 units each
