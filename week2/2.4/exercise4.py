import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

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
w1_hist=np.array([])
w2_hist=np.array([])

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
        ax1.title.set_text('First trial with alpha = ' + str(alpha))

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

    w1_hist = np.append(w1_hist, w1)
    w2_hist = np.append(w2_hist, w2)

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, num=2)

ax1.plot(tvec, pos_plot, label='pos')
ax1.set_xlabel('time')
ax1.set_ylabel('pos')

ax1.plot(tvec, pos_estim_plot, 'k', label='estimated pos')
ax1.legend()
ax1.title.set_text('Last trial with alpha = ' + str(alpha))


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
w1_inter = interp.interp1d(np.arange(w1_hist.size), w1_hist)
w1_ = w1_inter(np.linspace(0,w1_hist.size-1,tvec.size))
w2_inter = interp.interp1d(np.arange(w2_hist.size), w2_hist)
w2_ = w2_inter(np.linspace(0,w2_hist.size-1,tvec.size))

f, (ax4, ax5) = plt.subplots(2, 1, sharex=True, num=3)
ax4.plot(tvec, w1_, label='weights')
ax5.plot(tvec, w2_, label='weights')
ax4.set_xlabel('time')
ax4.set_ylabel('weight')
ax4.legend
ax5.set_xlabel('time')
ax5.set_ylabel('weight')
ax5.legend

ax4.title.set_text('trial with alpha = ' + str(alpha))
#print(w1_hist)
#print(tvec)
plt.show()