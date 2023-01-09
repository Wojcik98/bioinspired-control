function torque = pdcontroller(desired_angle, delayed_angle, delayed_velocity)
    global kp
    global kd
    torque(1)=kp*(desired_angle(1)-delayed_angle(1))+kd*(0-delayed_velocity(1));
    torque(2)=kp*(desired_angle(2)-delayed_angle(2))+kd*(0-delayed_velocity(2));
return