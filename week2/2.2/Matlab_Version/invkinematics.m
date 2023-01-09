function q = invkinematics(position)
    global le1
    global le2
    q=zeros(2,1);
    q(2)=acos((sum(position.^2)-(le1^2+le2^2))/(2*le1*le2));
    q(1)=atan2(position(2), position(1))-q(2)/2;
return