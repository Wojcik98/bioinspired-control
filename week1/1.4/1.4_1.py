import numpy as np
import matplotlib.pyplot as plt

# Point 1.1
def LIF(um_0, I, T):
    Rm = 10*1e6 # Mega Ohm
    Cm = 1*1e-9 # Nano Farad
    u_thresh = -50*1e-3 # milli Volt
    u_rest = -65*1e-3 # milli Volt

    delta_t = 1e-5 
    um_t = np.zeros((int(T//delta_t)))
    um_t[0] = um_0

    dum_dt = lambda um_t: (u_rest - um_t + Rm*I)/(Rm*Cm)
   
    # TODO Calculate the um_t
    return um_t

# Point 1.2
# TODO calculate the membrane potential using the LIF function
# membrane_potential = LIF()

plt.figure(figsize=(7,5))
#plt.plot(list(range(int(0.1//1e-5))), membrane_potential)
#plt.show()


# Point 1.3
# TODO define a function to calculate the interspike intervals
#calculate_isi 

# TODO define a function to calculate the spiking frequency of a whole experiment
#spiking_frequency =


# Point 1.4
plt.figure(figsize=(7,5))
spikes = []
# TODO write the code to accumulate the spikes

#plt.plot(list(np.arange(0,5.5e-9, 0.5e-9)), spikes)
plt.xlabel('Constant current')
plt.ylabel('Spiking frequency')
#plt.show()