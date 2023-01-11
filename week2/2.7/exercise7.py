import numpy as np
import matplotlib.pyplot as plt

from adaptive_filter.cerebellum import AdaptiveFilterCerebellum
from robot import SingleLink

Ts = 1e-2
n_inputs = 1
n_outputs = 1
n_bases = 4
beta = 1e-4

c = AdaptiveFilterCerebellum(Ts, n_inputs, n_outputs, n_bases, beta)

# DONE: Paste your experiment code from exercise 2.6

# Initialize simulation
# Ts = 1e-2
T_end = 10  # in one trial
n_steps = int(T_end / Ts)  # in one trial
n_trials = 50

plant = SingleLink(Ts)

# Logging variables
t_vec = np.array([Ts * i for i in range(n_steps * n_trials)])

theta_vec = np.zeros(n_steps * n_trials)
theta_ref_vec = np.zeros(n_steps * n_trials)

# Feedback controller variables
Kp = 30
Kv = 2

# DONE: Define parameters for periodic reference trajectory
A = np.pi
T = 10

# DONE: Change the code to the recurrent architecture
# You can update the cerebellum with: C = c.step(u, error)

# Simulation loop
for i in range(n_steps * n_trials):
    t = i * Ts
    # DONE: Calculate the reference at this time step
    # theta_ref = np.pi/4
    theta_ref = A * np.sin(2 * np.pi * t / T)

    # Measure
    theta = plant.theta
    omega = plant.omega

    # Feedback controller
    C = c.output
    error = (theta_ref - theta)
    error_fb = error + C
    tau = Kp * error_fb + Kv * (-omega)

    # Teach filter
    c.step(tau, error)

    # Iterate simulation dynamics
    plant.step(tau)

    theta_vec[i] = plant.theta
    theta_ref_vec[i] = theta_ref

mse = ((theta_vec - theta_ref_vec) ** 2).sum()
print(f'MSE: {mse}')

# DONE: Plot results
# Plotting
plt.plot(t_vec, theta_vec, label='theta')
plt.plot(t_vec, theta_ref_vec, '--', label='reference')
plt.legend()
plt.savefig("2_7.png")
plt.show()

# Plot trial error
error_vec = theta_ref_vec - theta_vec
l = int(T / Ts)
trial_error = np.zeros(n_trials)
for t in range(n_trials):
    trial_error[t] = np.sqrt(np.mean(error_vec[t * l:(t + 1) * l] ** 2))
plt.figure()
plt.plot(trial_error)
plt.title("beta = {}".format(beta))
plt.savefig("2_7.png")

plt.show()
