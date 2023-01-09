import numpy as np
import matplotlib.pyplot as plt

## Initialization
# Length of simulation (time steps)
simlen = 30
# Output
y = np.zeros((simlen))
# Target
target = 0.0

# Controller gain
K = 1

# Set first output
y[0] = 1

# TODO define the time delay

## Simulation
for t in range(simlen-1):
    # Compute output
    # TODO include the time delay
    u = K * (target - y[t])
    y[t+1]=0.5*y[t] + 0.4*u # 1st order dynamics

## Plot
time = range(simlen)
plt.plot(time, y)
plt.xlabel('time step')
plt.ylabel('y')
plt.show()