clear all;
close all;

%% Parameters
% Movement duration
global T
T=.6;
% Time step
global dt
dt=.01;
% Simulation duration
L=6.0;
% Proportional parameter
global kp
kp=100.0;
% Derivative parameter
global kd
kd=10.0;
% Upper arm length
global le1
le1=.3;
% Lower arm length
global le2
le2=.3;
% Upper arm mass
global m1
m1=3;
% Lower arm mass
global m2
m2=3;
% Gravity
global g
g=-9.8;

%% Variables
% Joint angles [shoulder elbow]
global ang
ang=[-pi/4 pi];
ang_rec=zeros(L/dt+1,2);
delayed_ang=ang(:);
% Joint velocity [shoulder elbow]
global vel
vel=[0 0];
vel_rec=zeros(L/dt+1,2);
delayed_vel=vel(:);
% Joint acceleration [shoulder elbow]
global acc
acc=[0 0];
acc_rec=zeros(L/dt+1,2);
% Jerk [shoulder elbow]
jerk_rec=zeros(L/dt+1,2);
% Shoulder position
shoulder_pos=[0 0];
% Elbow position
global elbow_pos
elbow_pos=[0 0];
% Wrist position
global wrist_pos
wrist_pos=[0 0];
wrist_pos_rec=zeros(L/dt+1,2);
% Initial wrist position for current movement
init_wrist_pos=wrist_pos;
% Desired wrist position
final_wrist_pos=[0.3 0.0 ; 0.0 0.0 ; .3*cos(pi/4) .3*sin(pi/4) ; 0.0 0.0 ;
                 0.0 .3 ; 0.0 0.0 ; .3*cos(3*pi/4) .3*sin(3*pi/4) ; 0.0 0.0];
% Current target index
curr_target=1;
% Movement start_time
start_t=0;

% TODO define Noise on torque
% TODO define time steps of delay

%% Simulation

%% Reset variables
figure(1);
for t=0:dt:L

    % Update records
    ang_rec(round(t/dt)+1,:)=ang;
    vel_rec(round(t/dt)+1,:)=vel;
    acc_rec(round(t/dt)+1,:)=acc;
    if t>0
        jerk_rec(round(t/dt)+1,:)=acc-acc_rec(round(t/dt),:);
    end
            
    %% Current wrist target
    current_wrist_target=final_wrist_pos(curr_target,:);

    if curr_target<=8
%% Planner
        % Get desired position from planner
        if t-start_t<T
            desired_pos=minjerk(init_wrist_pos, current_wrist_target, t-start_t);
        end

%% Inverse kinematics
        % Get desired angle from inverse kinematics
        desired_ang=real(invkinematics(desired_pos));

%% Inverse dynamics
        % TODO calculate the torque with a delayed angles and velocities

        % Desired torque from PD controller -  the following does not consider the delay
        desired_torque=pdcontroller(desired_ang, ang, vel);
            
%% Forward dynamics
        % TODO include the noise to the plant
        % Pass torque to plant
        plant(desired_torque);

%% Forward kinematics
        % Calculate new joint positions
        fkinematics(ang);

        % Record wrist position
        wrist_pos_rec(round(t/dt)+1,:)=wrist_pos;

%% Next target
        if t-start_t>=T+.02 && curr_target<8
            curr_target=curr_target+1
            init_wrist_pos=wrist_pos;
            start_t=t;
        end
    end

%% Plot arm, wrist path, and targets
    plot(final_wrist_pos(:,1), final_wrist_pos(:,2), 'o', 'Color', 'green');
    xlabel('meters');
    xlabel('meters');
    axis([-.5 .5 -.50 .50]);
    line([shoulder_pos(1) elbow_pos(1)], [shoulder_pos(2) elbow_pos(2)], 'Color', 'blue');
    line([elbow_pos(1) wrist_pos(1)], [elbow_pos(2) wrist_pos(2)], 'Color', 'blue');   
    for t2=dt:dt:t
        line([wrist_pos_rec(round(t2/dt),1) wrist_pos_rec(round(t2/dt)+1,1)], [wrist_pos_rec(round(t2/dt),2) wrist_pos_rec(round(t2/dt)+1,2)], 'Color', 'red');
    end
    M(round(t/dt+1))=getframe;
end

figure(2);
subplot(3,1,1), plot([0.0:dt:L-dt], vel_rec(1:1:L/dt,1),[0.0:dt:L-dt], vel_rec(1:1:L/dt,2));
legend('shoulder', 'elbow');
xlabel('time (ms)');
ylabel('velocity');
subplot(3,1,2), plot([0.0:dt:L-dt], acc_rec(1:1:L/dt,1),[0.0:dt:L-dt], acc_rec(1:1:L/dt,2));
legend('shoulder', 'elbow');
xlabel('time (ms)');
ylabel('acceleration');
subplot(3,1,3), plot([0.0:dt:L-dt], jerk_rec(1:1:L/dt,1),[0.0:dt:L-dt], jerk_rec(1:1:L/dt,2));
legend('shoulder', 'elbow');
xlabel('time (ms)');
ylabel('jerk');


