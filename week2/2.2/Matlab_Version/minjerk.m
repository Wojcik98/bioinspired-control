function desired = minjerk(init,final,t)
    global T
    desired(1)=init(1)+(final(1)-init(1))*(10*(t/T)^3-15*(t/T)^4+6*(t/T)^5);
    desired(2)=init(2)+(final(2)-init(2))*(10*(t/T)^3-15*(t/T)^4+6*(t/T)^5);
return