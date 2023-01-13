import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_model
from project.adaptive_filter.cerebellum import AdaptiveFilterCerebellum

from tqdm import tqdm  # progress bar

angles_diff = np.array([[0, 0]])
inp = np.array([[0, 0, 0, 0]])

# datasets = ["training_data.p"]
# for i in range(2, 32):
#     datasets.append("training_data"+str(i)+".p")
#
# for dataset in datasets:
#     print("Processing {}".format(dataset))
#     # Load data
#     data = pickle.load(open(dataset, "rb"))
#     angles = data[:, :2]
#     angles /= 90.0
#     end_pos = data[:, 2:]
#     end_pos = (end_pos - 200) / 200
#
#     for i, angle in tqdm(enumerate(angles)):
#         for j, angle2 in enumerate(angles):
#             if i!=j:
#                 angles_diff = np.concatenate((angles_diff, [angle2]))
#                 inp = np.concatenate((inp, [np.concatenate((end_pos[j] - end_pos[i], angle))]))
# with open('processed.p', 'wb') as file:
#     pickle.dump((inp, angles_diff), file)
with open('processed.p', 'rb') as file:
    inp, angles_diff = pickle.load(file)


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
model.load_state_dict(torch.load('closed_loop_trained_model.pth'))
model.to(device)
model.eval()

num_epochs = 500

l_vec = np.zeros(num_epochs)
l_test_vec = np.zeros(num_epochs)

n_inputs = 2
n_outputs = 2
n_bases = 8
beta = 1e-2

c = AdaptiveFilterCerebellum(n_inputs, n_outputs, n_bases, beta)

for epoch in tqdm(range(num_epochs)):
    x, y = x_train, y_train
    outp = model(x)
    t = outp.numpy()[0] * 90
    error = t - y
    c.step(t, error)


plt.plot(l_vec)
plt.plot(l_test_vec)
plt.legend(['train', 'test'])
plt.yscale('log')

plt.show()
