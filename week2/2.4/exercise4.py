import numpy as np
import matplotlib.pyplot as plt

# time parameters
dt = 0.005
max_time = 2
mvt_start_time = 0.5

# arm model
m = 1
arm_length = 0.3
I = m *(arm_length/2)**2
dampCoeff =0.5

# controller parameters
Kp = 30
Kv = 2
desired_end_pos = np.pi/2

# learning model parameters
w1 = 0
w2 = 0 
# learning rate
alpha = 1e-5

# weight histories
w1_hist=[]
w2_hist=[]

delay_true = 10 # time steps

# TODO: change delay_estimate
delay_estimate = 10


n_trials = 40

# learning loop
for i in range(n_trials):

    vel = 0
    pos = 0
    vel_estimated = 0
    pos_estimated = 0
    
    acc_delayed = np.zeros(delay_true+1)
    vel_delayed = np.zeros(delay_true+1)
    pos_delayed = np.zeros(delay_true+1)
    
    feedback_torque_delayed = np.zeros(delay_estimate+1)
    acc_estimated_delayed = np.zeros(delay_estimate+1)
    vel_estimated_delayed = np.zeros(delay_estimate+1)
    pos_estimated_delayed = np.zeros(delay_estimate+1)
    
    index_time =0

    # movement loop
    tvec = np.arange(0, max_time, dt)

    acc_estim_plot = np.zeros(tvec.size)
    acc_plot = np.zeros(tvec.size)
    vel_estim_plot = np.zeros(tvec.size)
    vel_plot = np.zeros(tvec.size)
    pos_estim_plot = np.zeros(tvec.size)
    pos_plot = np.zeros(tvec.size)
    torque_plot = np.zeros(tvec.size)


    for t in tvec:
        
        for k in range(delay_true-1, -1, -1):
            acc_delayed[k+1] = acc_delayed[k]
            vel_delayed[k+1] = vel_delayed[k]
            pos_delayed[k+1] = pos_delayed[k]
        
        for k in range(delay_estimate-1, -1, -1):
           acc_estimated_delayed[k+1] = acc_estimated_delayed[k] 
           vel_estimated_delayed[k+1] = vel_estimated_delayed[k]
           pos_estimated_delayed[k+1] = pos_estimated_delayed[k]
           feedback_torque_delayed[k+1] = feedback_torque_delayed[k]
        
        if t < mvt_start_time:
            desired_pos = 0
        else:
            desired_pos = desired_end_pos

        # feedback torque
        pos_err = (desired_pos - pos_estimated) - pos_delayed[-1] + pos_estimated_delayed[-1]
        vel_err = (- vel_estimated) - vel_delayed[-1] + vel_estimated_delayed[-1]
        
        feedback_torque =  Kp * pos_err  + Kv * vel_err
        
        feedback_torque_delayed[0] = feedback_torque
        
        # arm forward dynamics 
        acc = (feedback_torque - dampCoeff*vel) / I 
        vel = vel + dt * acc
        pos = pos + dt * vel
        
        acc_delayed[0] = acc
        vel_delayed[0] = vel
        pos_delayed[0] = pos

        #forward model of arm_dynamics
        acc_estimated = w1*feedback_torque + w2*vel_estimated
        vel_estimated = vel_estimated + dt * acc_estimated
        pos_estimated = pos_estimated + dt * vel_estimated
        
        acc_estimated_delayed[0] = acc_estimated
        vel_estimated_delayed[0] = vel_estimated
        pos_estimated_delayed[0] = pos_estimated

        # learning the weights with gradient descent
        error = (acc_delayed[-1] - acc_estimated_delayed[-1])
        w1 = w1 + alpha*error*feedback_torque_delayed[-1]
        w2 = w2 + alpha*error*vel_estimated_delayed[-1]
    
        # plotting    
        acc_estim_plot[index_time] = acc_estimated
        acc_plot[index_time] = acc
        vel_estim_plot[index_time] = vel_estimated
        vel_plot[index_time] = vel
        pos_estim_plot[index_time] = pos_estimated
        pos_plot[index_time] = pos    
        torque_plot[index_time] = feedback_torque        
        index_time = index_time + 1
    

    if i==1:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, num=1)

        ax1.plot(tvec, pos_plot, label='pos')
        ax1.set_xlabel('time')
        ax1.set_ylabel('pos')
        
        ax1.plot(tvec, pos_estim_plot, 'k', label='estimated pos')
        ax1.legend()
        ax1.title.set_text('First trial')

        ax2.plot(tvec, vel_plot, label='vel')
        ax2.set_xlabel('time')
        ax2.set_ylabel('vel')

        ax2.plot(tvec, vel_estim_plot, 'k', label='estimated vel')
        ax2.legend()


        ax3.plot(tvec, acc_plot, label='acc')
        ax3.set_xlabel('time')
        ax3.set_ylabel('acc')
        
        ax3.plot(tvec, acc_estim_plot, 'k', label='estimated acc')
        ax3.legend()

    w1_hist.append(w1)
    w2_hist.append(w2)

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, num=2)

ax1.plot(tvec, pos_plot, label='pos')
ax1.set_xlabel('time')
ax1.set_ylabel('pos')

ax1.plot(tvec, pos_estim_plot, 'k', label='estimated pos')
ax1.legend()
ax1.title.set_text('Last trial')


ax2.plot(tvec, vel_plot, label='vel')
ax2.set_xlabel('time')
ax2.set_ylabel('vel')

ax2.plot(tvec, vel_estim_plot, 'k', label='estimated vel')
ax2.legend()


ax3.plot(tvec, acc_plot, label='acc')
ax3.set_xlabel('time')
ax3.set_ylabel('acc')

ax3.plot(tvec, acc_estim_plot, 'k', label='estimated acc')
ax3.legend()

## TODO: Plot change in weights


plt.show()