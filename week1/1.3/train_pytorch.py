import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_model

from tqdm import tqdm # progress bar

# Load data
data = pickle.load( open( "training_data.p", "rb" ) )

print(data.shape)
angles = data[:,:2].T
end_pos = data[:,2:].T

# Use GPU?
device = 'cpu'
if torch.cuda.is_available():
    print("Using GPU")
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

x = torch.from_numpy(end_pos.T).float()
y = torch.from_numpy(angles.T).float()
# TODO split the training set and test set
# Eventually normalize the data

if device == 'cuda':
    x = x.cuda()
    y = y.cuda()

# Define neural network - an example
model = torch_model.MLPNet(2, 16, 2)
# model = torch_model.Net(n_feature=2, n_hidden1=h, n_hidden2=h, n_output=2)
#print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()
num_epochs = 500000

#h = 16
#g = 0.999
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=g)

l_vec = np.zeros(num_epochs)

for t in tqdm(range(num_epochs)):
    # TODO Train network
    prediction = model(x) # Forward pass prediction. Saves intermediary values required for backwards pass
    loss = loss_func(prediction, y) # Computes the loss for each example, using the loss function defined above
    optimizer.zero_grad() # Clears gradients from previous iteration
    loss.backward() # Backpropagation of errors through the network
    optimizer.step() # Updating weights
    scheduler.step()

    l = loss.data
    if device == 'cuda':
        l = l.cpu()
    #print(l.numpy())
    l_vec[t] = l.numpy()

    # TODO Test the network

plt.plot(l_vec)
plt.yscale('log')

torch.save(model.state_dict(), 'trained_model.pth')
plt.show()


## Parameter TIPS - Try
# Adam with lr = 0.001
# StepLR scheduler with step_size=100, and gamma = 0.999
# Two hidden layers with 18 units each