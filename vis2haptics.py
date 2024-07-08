import roboticstoolbox as rt
import numpy as np
import scipy as sp
import spatialmath as sm
import rtde_receive
import rtde_control
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import scipy.io as scio
import time
import keyboard
from CSRL_math import *
from CSRL_orientation import *

from surgery_threads import *


# blocking
data_id = input('Data ID: ')

# Define follower (UR3)
rtde_c_follower = rtde_control.RTDEControlInterface("192.168.1.66")
rtde_r_follower = rtde_receive.RTDEReceiveInterface("192.168.1.66")


# Define leader (UR3e)
rtde_c_leader = rtde_control.RTDEControlInterface("192.168.1.60")
rtde_r_leader = rtde_receive.RTDEReceiveInterface("192.168.1.60")

# Get initial configuration
q0_leader = np.array(rtde_r_leader.getActualQ())
q0_follower = np.array(rtde_r_follower.getActualQ())

# Declare math pi
pi = math.pi

# Create the robots 


# Define the kinematics of the leader robot (UR3e)
robot_l = rt.DHRobot([
    rt.RevoluteDH(d = 0.15185, alpha = pi/2),
    rt.RevoluteDH(a = -0.24355),
    rt.RevoluteDH(a = -0.2132),
    rt.RevoluteDH(d = 0.13105, alpha = pi/2),
    rt.RevoluteDH(d = 0.08535, alpha = -pi/2),
    rt.RevoluteDH(d = 0.0921)
], name='UR3e')

# Define the kinematics of the follower robot (UR3)
robot_f = rt.DHRobot([
    rt.RevoluteDH(d = 0.1519, alpha = pi/2),
    rt.RevoluteDH(a = -0.24365),
    rt.RevoluteDH(a = -0.21325),
    rt.RevoluteDH(d = 0.11235, alpha = pi/2),
    rt.RevoluteDH(d = 0.08535, alpha = -pi/2),
    rt.RevoluteDH(d = 0.0819)
], name='UR3')

# Control cycle of the leader
dt = 0.002

# Init time
t = 0.0

# Start logging
qlog_leader = q0_leader
tlog = t

# get time now
t_now = time.time()

# Tool mass
tool_mass = 0.145

# gravity acceleration
gAcc = 9.81

# initialize qdot for leader (admittance simulation)
qddot_leader = np.zeros(6)
qdot_leader = np.zeros(6)


# for mapping sigma to length of tool
p0 = np.array([427.92, 435.41])
pT = np.array([336.72, 224.95])

# Define the admittance inertia
M_leader = np.identity(6)
M_leader[np.ix_([0, 1], [0, 1])] = 0.3 * M_leader[np.ix_([0, 1], [0, 1])]
M_leader[np.ix_([2, 3], [2, 3])] = 0.1 * M_leader[np.ix_([2, 3], [2, 3])]
M_leader[np.ix_([4, 5], [4, 5])] = 0.05 * M_leader[np.ix_([4, 5], [4, 5])]

# Compute inverse (constant)
Minv_leader = np.linalg.inv(M_leader)

# Define the admittance damping
D_leader = np.identity(6)
D_leader[np.ix_([0, 1], [0, 1])] = 0.6 * D_leader[np.ix_([0, 1], [0, 1])]
D_leader[np.ix_([2, 3], [2, 3])] = 0.6 * D_leader[np.ix_([2, 3], [2, 3])]
D_leader[np.ix_([4, 5], [4, 5])] = 0.6 * D_leader[np.ix_([4, 5], [4, 5])]

#----------------------------------------------------------------
# This is the gain of the AI guidance 
kAI = 0.005
# kAI = 0.0
# ----------------------------------------------------------------

# initialize leader and follower Jacobian
J_leader = np.zeros([6,6])
J_follower = np.zeros([6,6])

# The rotation of the FT sensor with respect to the flange (UR3e)
Rts = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])

# The rotation of the camera with respect to the flange
phiC = 10.0 * pi / 180.0 # This is the angle of the camera based on the CAD of the surgical tool 
Rwc = np.array([[0, -math.cos(phiC), -math.sin(phiC)], [1, 0, 0], [0, -math.sin(phiC), math.cos(phiC)]])

# get force measurement bias
f_leader_offset_tool = np.array(rtde_r_leader.getActualTCPForce()) #- tool_mass * gAcc

# BLOCKING -----------------------------------
# Initial mirroring ---> Move leader to the initial follower's pose
rtde_c_leader.moveJ(q0_follower, 0.5, 0.5)
#---------------------------------------------

# read the leader's joint variables
q0_leader = np.array(rtde_r_leader.getActualQ())

# Get pose of the leader's wrist  
g_leader = robot_l.fkine(q0_leader)
R_leader = np.array(g_leader.R)
p_leader = np.array(g_leader.t)
print('---------Leader:')
print('R: ', R_leader)
print('p: ', p_leader)

# Get pose of the follower's wrist 
g_follower = robot_l.fkine(q0_follower)
R_follower = np.array(g_follower.R)
p_follower = np.array(g_follower.t)
print('----------Follower:')
print('R: ', R_follower)
print('p: ', p_follower)

# Initialize logging data -----------
pl_log = p_leader
pf_log = p_follower
Ql_log = rot2quat(R_leader) # Quaternion format
Qf_log = rot2quat(R_follower) # Quaternion format
def_log = np.zeros(2) # Deformation
Fh_log = np.zeros(6) # Force applied by human
uf_log = np.zeros(6) # Torques on the joints by the human
Fv_log = np.zeros(6) # Virtual force applied by the AI guidance


# This variable is used to communicate to the UR3 every 8ms 
k_follower = 0

# Initialization of the Pytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# define cut image size 
image_size: Tuple[int, int, int] = (3, 620, 620)

# Load the file of the model
filename = 'model_epoch_800.pth'

# Load the model
model = load_model(image_size=image_size, filename=filename)
model.to(device)

# This is for the thread
RTproc = RealTimeProcessor(model, device, image_size)
RTproc.start()

# Energy tank concept
# Initialization of the tank's level
L = 1.0
L_dot = 0.0

# Energy tank params
a_tank = 0.9

# This is the gain for the trajectory tracking of the follower robot 
k_traj = 4.00
# initialize the pose error 
e = np.zeros(6)

# initialize the deformation variable in the pizel space
d_thres = np.zeros(2)



# Main control loop ..... runs 50000 times (100s)
for i in range(50000):

    # this is to synchronise with the UR3e 
    t_start = rtde_c_leader.initPeriod()

    # The loop stops if you push 'a'
    if keyboard.is_pressed('a') or rtde_r_leader.isEmergencyStopped() or rtde_r_leader.isProtectiveStopped() or rtde_r_follower.isEmergencyStopped() or rtde_r_follower.isProtectiveStopped():
        print('Stopping robot')
        break

    # Get the results from the DCNN
    ptool, d_thres = RTproc.get_results()

    # Get joint values
    q_leader = np.array(rtde_r_leader.getActualQ())
    q_follower = np.array(rtde_r_follower.getActualQ())

    # Calculate the pose of the leader's wrist 
    g_leader = robot_l.fkine(q_leader)
    R_leader = np.array(g_leader.R)
    p_leader = np.array(g_leader.t)

    # Calculate the pose of the follower's wrist
    g_follower = robot_l.fkine(q_follower)
    R_follower = np.array(g_follower.R)
    p_follower = np.array(g_follower.t)

    # This is the z axis of the leader
    # wrt the base frame
    zw = R_leader[:, 2]
    # wrt the wrist
    z = np.array([0,0,1])

    # Initialize the DCNN force 
    Fext = np.zeros(6)

    # If the values that we got from the DCNN are not None, then ...
    if ptool is not None and d_thres is not None:

        # Compute the distance of the contact point from the wrist of the follower (d variable)
        sigma = np.sqrt((ptool[0] - p0[0]) * (ptool[0] - p0[0]) + (ptool[1] - p0[1]) * (ptool[1] - p0[1])) / np.sqrt(
            (pT[0] - p0[0]) * (pT[0] - p0[0]) + (pT[1] - p0[1]) * (pT[1] - p0[1]))
        
        # Quadratic mapping 
        dtool = 0.13 + 0.083 * (sigma + 0.2) * (sigma + 0.2) + 0.06

        # The virtual force provided by the DCNN wrt the Camera frame 
        cFd = - np.array([d_thres[0], d_thres[1], 0])

        # Chnage the frame of reference to base frame 
        wFd = Rwc @ cFd

        # Gamma matrix (rigid object)
        Fext[:3] = wFd
        Fext[-3:] = skewSymmetric( z * dtool) @ wFd

    # print(time.time() - t_now)
    # t_now =  time.time()
    
    # Integrate time
    t = t + dt

    # End-effector Jacobian of the leader 
    Je_leader = np.array(robot_l.jacobe(q_leader))

    # Manipulator Jacobian of the leader 
    J_leader[:3] = R_leader @ Je_leader[:3]
    J_leader[-3:] = R_leader @ Je_leader[-3:]

    # End-effector Jacobian of the follower 
    Je_follower = np.array(robot_f.jacobe(q_follower))

    # Manipulator Jacobian of the leader 
    J_follower[:3] = R_follower @ Je_follower[:3]
    J_follower[-3:] = R_follower @ Je_follower[-3:]

    # Get the force/torque measurement from the sensor
    f_leader = np.array(rtde_r_leader.getActualTCPForce()) - f_leader_offset_tool

    # The force/torque with respect to the wrist of the leader robot 
    f_leader[:3] = - Rts @ f_leader[:3]
    f_leader[-3:] = - Rts @ f_leader[-3:]
    
    # Dead-zone (thresholding) of the the measurement of the force
    fnorm = np.linalg.norm(f_leader[:3])
    nF = f_leader[:3] / fnorm
    if fnorm<4.0:
        f_leader[:3] = np.zeros(3)
    else:
        f_leader[:3] = f_leader[:3] - 4.0 * nF

    # Dead-zone (thresholding) of the the measurement of the torqeu
    taunorm = np.linalg.norm(f_leader[-3:])
    nTau = f_leader[-3:] / taunorm
    if taunorm < 0.5:
        f_leader[-3:] = np.zeros(3)
    else:
        f_leader[-3:] = f_leader[-3:] - 0.5 * nTau

  
    # Compute the velocity of the leader's wrist 
    v_leader = J_leader @ qdot_leader

    # Compute the pose error between the follower and the leader pose 
    e[:3] = p_leader - p_follower
    e[-3:] = logError(R_leader, R_follower)

    #  Integrate the joint veloities of the leader (admittance control)
    qdot_leader = qdot_leader + qddot_leader * dt

    # Velocity of the leader's wrist wrt to the wrist (body velocity)
    v_w_wrist= Je_leader @ qdot_leader

    # Velocity of the leader wrt the base frame 
    v_0_wrist = np.zeros(6)
    v_0_wrist[:3] = R_leader @ v_w_wrist[:3]
    v_0_wrist[-3:] = R_leader @ v_w_wrist[-3:]

    # Energy tank level update 
    L = L + L_dot * dt
    s = sigmoid(L,0.5, 1)
    L_dot = a_tank * v_0_wrist @ (D_leader @ v_0_wrist) - kAI * s * (v_w_wrist @ Fext)
  
    # The torque applied due to the virtual force from the DCNN ... for guidance (Vis2Haptics)
    uf = kAI * s * np.transpose(Je_leader) @ Fext

    # Admittance control model 
    tau_leader =  0.7 * np.transpose( J_leader) @ f_leader + uf
    qddot_leader = Minv_leader @ (- D_leader @ qdot_leader + tau_leader)
    
    # Set the joint velocity of the leader 
    rtde_c_leader.speedJ(qdot_leader, 3.0, dt)

    # Log variables
    pl_log = np.vstack((pl_log, p_leader))
    pf_log = np.vstack((pl_log, p_follower))
    Ql_log = np.vstack((Ql_log, rot2quat(R_leader)))
    Qf_log = np.vstack((Qf_log, rot2quat(R_follower)))
    if d_thres is not None:
        def_log = np.vstack((def_log, d_thres))
    else:
        def_log = np.vstack((def_log, np.zeros(2)))
    Fh_log = np.vstack((Fh_log, f_leader))
    uf_log =  np.vstack((uf_log ,uf))
    Fv_log = np.vstack((Fv_log, Fext))
    tlog = np.vstack((tlog,t))

    # wait for the period synch of the UR3e 
    rtde_c_leader.waitPeriod(t_start)

    # synchronization of the UR3 (every 4 cycles of UR3e). 
    # UR3: 8ms 
    # UR3e: 2ms
    k_follower = k_follower + 1
    if k_follower == 4:

        # CLIK
        qdot_follower = np.linalg.pinv(J_follower) @ (v_leader + k_traj * e)
    
        # Set joint velocity of the follower robot 
        rtde_c_follower.speedJ(qdot_follower, 0.5, 4.0 * dt)

        k_follower = 0


# write the data to a file 
data = {'pl_log': pl_log, 'pf_log': pf_log, 'Ql_log': Ql_log, 'Qf_log': Qf_log,
                'def_log': def_log, 'Fh_log': Fh_log, 'uf_log': uf_log, 'Fv_log': Fv_log, 'tlog': tlog}
scio.savemat('Logging_' + str(data_id) + '.mat', data)

# Stop robot 
rtde_c_leader.speedStop()
rtde_c_leader.stopScript()

rtde_c_follower.speedStop()
rtde_c_follower.stopScript()


