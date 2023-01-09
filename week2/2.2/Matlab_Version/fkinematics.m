function fkinematics(ang)
    global le1
    global le2
    global elbow_pos
    global wrist_pos
    
    elbow_pos=[le1*cos(ang(1)) ; le1*sin(ang(1))];
    p1=[cos(ang(1)) -sin(ang(1)) ; sin(ang(1)) cos(ang(1))];
    p2=[le2*cos(ang(2)) ; le2*sin(ang(2))];
    
    wrist_pos=elbow_pos+p1*p2;
return