import numpy as np
import matplotlib.pyplot as plt

## Initialization
# Mass of the arm
m = 1
# Length of the arm
arm_length = 0.3
# Inertia
I = 1/3 * m *(arm_length)**2

# Gravity
g = 9.81

# Damping coefficient for the forward model
dampCoeff =0.5
# Sensory delay (s)
delay =  0
#delay =  0.1
# Simulation time step (s)
dt = 0.001
# Simulation duration (s)
max_time = 2
tvec = np.arange(0, max_time, dt)

# Time to start the movement (s)
mvt_start_time = 0.5

# Delayed versions of joint angle and velocity
delay_index = int(delay/dt)
delayed_ang = np.zeros(delay_index+1)
delayed_vel = np.zeros(delay_index+1)

# Plant - current actual joint angle and velocity
ang_actual = 0
vel_actual = 0

# Forward model - estimated current joint angle and velocity
ang_estimated = 0
vel_estimated = 0

# Target joint angle
final_target_ang = np.pi/2

# History of joint angle, delayed joint angle, and torque
ang_history = []
ang_est_history = []
ang_del_history = []
ang_target_history = []
torque_history = []
pert_history = []

# Perturbation start time (s)
pert_start_time = 1.2
# Perturbation end time (s)
pert_end_time = 1.5
# Perturbation amplitude
pert_amp =1
#pert_amp =0

# Gain and damping parameters for forward model
Kp_forward = 5
#Kp_forward = 0
Kd_forward = 0.3
#Kd_forward = 0

# Gain and damping parameters for feedback model
Kp_feedback = 5
#Kp_feedback = 0
Kd_feedback = 0.3
#Kd_feedback = 0


## Simulation
for t in tvec:

    # Set the desired joint angle once the movement start time is reached
    if t <  mvt_start_time:
        target_ang = 0
    else:
        target_ang = final_target_ang

    # Forward model torque
    forward_torque = Kp_forward * (target_ang - ang_estimated) + Kd_forward * ( - vel_estimated)
    # Feedback torque delayed
    feedback_torque = Kp_feedback * (target_ang - delayed_ang[delay_index]) + Kd_feedback * ( - delayed_vel[delay_index])
    
    # Total torque
    total_torque=forward_torque + feedback_torque

    # Forward model of arm_dynamics
    acc_estimated = (total_torque - dampCoeff*vel_estimated ) / I 
    vel_estimated = vel_estimated + dt * acc_estimated
    ang_estimated = ang_estimated + dt * vel_estimated

    # Apply perturbation
    if ((t >  pert_start_time) and (t < pert_end_time)):
        pert = pert_amp
    else:
        pert = 0

    # Plant - forward model of the arm
    acc = (total_torque - dampCoeff*vel_actual + pert) / I 
    vel_actual = vel_actual + dt * acc
    ang_actual = ang_actual + dt * vel_actual

    # Update delayed copy of joint angle and velocity
    for i in range(delay_index-1, -1, -1):
        delayed_ang[i+1] = delayed_ang[i]
        delayed_vel[i+1] = delayed_vel[i]
    
    delayed_ang[0] = ang_actual
    delayed_vel[0] = vel_actual
    
    # Update histories
    ang_history.append(ang_actual)
    ang_est_history.append(ang_estimated)
    ang_del_history.append(delayed_ang[delay_index])
    torque_history.append(total_torque)
    ang_target_history.append(target_ang)
    pert_history.append(pert)


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(tvec, ang_history, label='angle')

if delay != 0:
    ax1.plot(tvec, ang_del_history, 'k', label='delayed angle')
if Kp_forward != 0:
    ax1.plot(tvec, ang_est_history, 'g', label='estimated angle')
ax1.plot(tvec, ang_target_history, '--', label='target angle')
ax1.plot(tvec, pert_history, 'r', label='perturbation')
ax1.set_ylabel('angle')
ax1.legend()

ax2.plot(tvec, torque_history)
ax2.set_xlabel('time')
ax2.set_ylabel('motor command')

plt.show()