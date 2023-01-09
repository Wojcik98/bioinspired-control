import numpy as np
import math

class SimulationFunctions:



    def __init__(self,Var):

        self.T = Var[0]
        self.dt = Var[1]
        self.L = Var[2]
        self.kp = Var[3]
        self.kd = Var[4]
        self.le1 = Var[5]
        self.le2 = Var[6]
        self.m1 = Var[7]
        self.m2 = Var[8]
        self.g = Var[9]



    def minjerk(self,init,final,t):

        desired = np.zeros((2))
        desired[0]=init[0]+(final[0]-init[0])*(10*(t/self.T)**3-15*(t/self.T)**4+6*(t/self.T)**5)
        desired[1]=init[1]+(final[1]-init[1])*(10*(t/self.T)**3-15*(t/self.T)**4+6*(t/self.T)**5)

        return desired

    def invkinematics(self,position):

        q=np.zeros((2,1))
        q[1]=math.acos((sum(np.square(position))-(self.le1**2+self.le2**2))/(2*self.le1*self.le2))
        q[0]=math.atan2(position[1], position[0])-q[1]/2
        
        return q


    def pdcontroller(self,desired_angle, delayed_angle, delayed_velocity):

        torque = np.zeros((2))
        torque[0]=self.kp*(desired_angle[0]-delayed_angle[0])+self.kd*(0-delayed_velocity[0])
        torque[1]=self.kp*(desired_angle[1]-delayed_angle[1])+self.kd*(0-delayed_velocity[1])

        return torque


    
    def plant(self,ang,vel,acc,torque):
        
        #calculate inertia
        inertia = np.zeros((2))
        inertia[0]=self.m1*(self.le1/2)**2
        inertia[1]=self.m2*(self.le2/2)**2
        
        # calculate load torques
        t_load = np.zeros((2))
        t_load[0]=(self.m1*self.g*(self.le1/2)*np.sin(ang[0])) + (acc[0]*(inertia[0]+inertia[1]+self.m2*self.le1*self.le2*np.cos(ang[1])+(self.m1*self.le1**2+self.m2*self.le2**2)/4+self.m2*self.le1**2)) + (acc[1]*(inertia[1]+(self.m2*self.le2**2)/4+(self.m2*self.le1*self.le2)/2*np.cos(ang[1]))) + ((-(self.m2*self.le1*self.le2)/2)*vel[1]**2) + -self.m2*self.le1*self.le2*vel[0]*vel[1]*np.sin(ang[1])
        t_load[1]=(self.m2*self.g*(self.le2/2)*np.sin(ang[1])) + (acc[1]*(inertia[1]+(self.m1*self.le2**2)/4)) + (acc[0]*(inertia[1]+((self.m2*self.le1*self.le2)/2)*np.cos(ang[1])+(self.m2*self.le2**2)/4)+((self.m2*self.le1*self.le2)/2)*vel[0]**2*np.sin(ang[1]))
        
        #calculate acceleration
        centripetal_torque1=inertia[1]+((self.m2*self.le1*self.le2)/2)*np.cos(ang[1])+((self.m2*self.le2**2)/4)
        centripetal_torque2=((self.m2*self.le1*self.le2)/2)*np.sin(ang[1])
        coriolis_torque=self.m2*self.le1*self.le2*vel[0]*vel[1]*np.sin(ang[1])
        inertial_component1=inertia[0]+inertia[1]+(self.m2*self.le1*self.le2*np.cos(ang[1]))+((self.m1*self.le1**2+self.m2*self.le2**2)/4)+self.m2*self.le1**2
        inertial_component2=inertia[1]+((self.m1*self.le2**2)/4)
        interaction_inertial_torque=inertia[1]+((self.m2*self.le2**2)/4)+((self.m2*self.le1*self.le2)/2)*np.cos(ang[1])
        acc[0]=(torque[0]-(((vel[0]**2*centripetal_torque2)/inertial_component2)*centripetal_torque1+vel[1]**2*centripetal_torque2-coriolis_torque) / (inertial_component1-((interaction_inertial_torque/inertial_component2)*centripetal_torque1)))
        acc[1]=(torque[1]-acc[0]*centripetal_torque1-vel[0]**2*centripetal_torque2)/inertial_component2
        
        #calculate velocity
        vel[0]=vel[0]+self.dt*acc[0]
        vel[1]=vel[1]+self.dt*acc[1]
        
        #calculate shoulder angle
        ang[0]=ang[0]+self.dt*vel[0]
        #calculate elbow angle
        ang[1]=ang[1]+self.dt*vel[1]
        
        return ang, vel, acc
    
    def fkinematics(self,ang):
        
        elbow_pos=[[self.le1*np.cos(ang[0])], [self.le1*np.sin(ang[0])]]
        p1=[[np.cos(ang[0]), -np.sin(ang[0])], [np.sin(ang[0]), np.cos(ang[0])]]
        p2=[[self.le2*np.cos(ang[1])], [self.le2*np.sin(ang[1])]]
        wrist_pos=list(elbow_pos+np.matmul(np.array(p1),np.array(p2)))

        return elbow_pos, wrist_pos
