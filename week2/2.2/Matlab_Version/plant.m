function plant(torque)
    global le1
    global le2
    global m1
    global m2
    global ang
    global acc
    global vel
    global dt
    global g
    
    %calculate inertia
    inertia(1)=m1*(le1/2)^2;
    inertia(2)=m2*(le2/2)^2;
    
    % calculate load torques
    t_load(1)=(m1*g*(le1/2)*sin(ang(1)))+...
        (acc(1)*(inertia(1)+inertia(2)+m2*le1*le2*cos(ang(2))+(m1*le1^2+m2*le2^2)/4+m2*le1^2))+...
        (acc(2)*(inertia(2)+(m2*le2^2)/4+(m2*le1*le2)/2*cos(ang(2))))+...
        ((-(m2*le1*le2)/2)*vel(2)^2)+...
        -m2*le1*le2*vel(1)*vel(2)*sin(ang(2));
    t_load(2)=(m2*g*(le2/2)*sin(ang(2)))+...
        (acc(2)*(inertia(2)+(m1*le2^2)/4))+...
        (acc(1)*(inertia(2)+((m2*le1*le2)/2)*cos(ang(2))+(m2*le2^2)/4)+((m2*le1*le2)/2)*vel(1)^2*sin(ang(2)));
    
    %calculate acceleration
    centripetal_torque1=inertia(2)+((m2*le1*le2)/2)*cos(ang(2))+((m2*le2^2)/4);
    centripetal_torque2=((m2*le1*le2)/2)*sin(ang(2));
    coriolis_torque=m2*le1*le2*vel(1)*vel(2)*sin(ang(2));
    inertial_component1=inertia(1)+inertia(2)+(m2*le1*le2*cos(ang(2)))+((m1*le1^2+m2*le2^2)/4)+m2*le1^2;
    inertial_component2=inertia(2)+((m1*le2^2)/4);
    interaction_inertial_torque=inertia(2)+((m2*le2^2)/4)+((m2*le1*le2)/2)*cos(ang(2));
    acc(1)=(torque(1)-(((vel(1)^2*centripetal_torque2)/inertial_component2)*centripetal_torque1+vel(2)^2*centripetal_torque2-coriolis_torque)/...
        (inertial_component1-((interaction_inertial_torque/inertial_component2)*centripetal_torque1)));
    acc(2)=(torque(2)-acc(1)*centripetal_torque1-vel(1)^2*centripetal_torque2)/inertial_component2;
    
    %calculate velocity
    vel(1)=vel(1)+dt*acc(1);
    vel(2)=vel(2)+dt*acc(2);
    
    %calculate shoulder angle
    ang(1)=ang(1)+dt*vel(1);
    %calculate elbow angle
    ang(2)=ang(2)+dt*vel(2);
return