import math
import numpy as np
from CSRL_math import *
import roboticstoolbox as rt
import numpy.linalg as la


# returns the vector-part error between two rotation matrices or quaternions A and Ad
def vectorError(A, Ad):
     
    Q = enforceQuat(A) 
    Qd = enforceQuat(Ad) 

    eta = Q[0]
    epsilon = Q[1:3+1]

    eta_d = Qd[0]
    epsilon_d = Qd[1:3+1]

    S = skewSymmetric(epsilon_d)
    
    return -eta * epsilon_d + eta_d * epsilon + S @ epsilon

# returns the logarithmic error between two rotation matrices or quaternions A and Ad 
def logError(A, Ad):

    Q = enforceQuat(A) 
    Qd = enforceQuat(Ad)

    e1 = 2 * quatLog(quatProduct(Q, quatInv(Qd)))
    e2 = 2 * quatLog(quatProduct(-Q, quatInv(Qd)))

    if np.linalg.norm(e1) < np.linalg.norm(e2):
        return e1
    else:
        return e2



# returns the logarithmic error time derivative, based on omega and omegads  
def logErrorDerivative(A, Ad, omega, omegad):

    R = enforceRot(A) 
    Rd = enforceRot(Ad) 

    return omega - R @ np.transpose(Rd) @ omegad


# returns the quaternion form of a rotation matrix R
def rot2quat(R):

    R = np.array(R)

    Q = np.zeros(4)

    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
  
        s = np.sqrt(tr + 1.0) * 2  
        Q[0] = 0.25 * s
        Q[1] = (R[2, 1] - R[1, 2]) / s
        Q[2] = (R[0, 2] - R[2, 0]) / s
        Q[3] = (R[1, 0] - R[0, 1]) / s
  
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
  
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  
        Q[0] = (R[2, 1] - R[1, 2]) / s
        Q[1] = 0.25 * s
        Q[2] = (R[0, 1] + R[1, 0]) / s
        Q[3] = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
  
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  
        Q[0] = (R[0, 2] - R[2, 0]) / s
        Q[1] = (R[0, 1] + R[1, 0]) / s
        Q[2] = 0.25 * s
        Q[3] = (R[1, 2] + R[2, 1]) / s
  
    else:
  
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        Q[0] = (R[1, 0] - R[0, 1]) / s
        Q[1] = (R[0, 2] + R[2, 0]) / s
        Q[2] = (R[1, 2] + R[2, 1]) / s
        Q[3] = 0.25 * s
  

    return Q / la.norm(Q)

    
    
# returns the rotation matrix form of a quaternion Q
def quat2rot(Q):

    n = Q[0]
    ex = Q[1]
    ey = Q[2]
    ez = Q[3]
    
    R = np.identity(3)

    R[0, 0] = 2 * (n * n + ex * ex) - 1
    R[0, 1] = 2 * (ex * ey - n * ez)
    R[0, 2] = 2 * (ex * ez + n * ey)

    R[1, 0] = 2 * (ex * ey + n * ez)
    R[1, 1] = 2 * (n * n + ey * ey) - 1
    R[1, 2] = 2 * (ey * ez - n * ex)

    R[2, 0] = 2 * (ex * ez - n * ey)
    R[2, 1] = 2 * (ey * ez + n * ex)
    R[2, 2] = 2 * (n * n + ez * ez) - 1

    return R

# returns the quarternion conugate of Q
def quatConjugate(Q):

  Qout = -Q
  Qout[0] = Q[0]
  return Qout

# returns the quarternion inverse of Q
def quatInv(Q):
  return quatConjugate(Q) / la.norm(Q)


# returns the quarternion product between Q1 and Q2
def quatProduct(Q1, Q2):

    S = skewSymmetric(Q1[1:3+1])
    prod = np.zeros(4)

    prod[0] = Q1[0] * Q2[0] - Q1[1:3+1] @ Q2[1:3+1]
    prod[1:3+1] = Q1[0] * Q2[1:3+1] +  Q2[0] * Q1[1:3+1] + S @ Q2[1:3+1]

    return prod

# returns the quaternion logarithm =k*theta
def quatLog(Q, eps=0.001):

    Qlog = np.zeros(3)
    
    v = np.max([np.min([ Q[0] , 1]), -1])

    nrm = la.norm(Q[1:3+1])

    if nrm > nrm * eps:
        Qlog = np.arctan2(nrm, v) * Q[1:3+1] / nrm
    
    return Qlog

# returns the inverse of the quaternion logarithm (exponential mapping)
def quatExp(w, eps=0.001):

    w = np.array(w)
    Qexp = np.zeros(4)
    nrm = la.norm(w)
    if nrm > nrm * eps:
  
        Qexp[0] = math.cos(nrm)
        Qexp[1:3+1] = math.sin(nrm) * w / nrm
    
    else:
        Qexp[0] = 1  

    return Qexp

# returns the rotation matrix of a rotatino around the x-axis
def rotX(theta):

    R=np.zeros((3,3))

    R[0, 0] = 1
    R[1, 1] = math.cos(theta); R[1, 2] = -math.sin(theta)
    R[2, 1] = math.sin(theta); R[2, 2] = math.cos(theta)

    return R

# returns the rotation matrix of a rotatino around the y-axis
def rotY(theta):

    R=np.zeros((3,3))
   
    R[0, 0] = math.cos(theta); R[0, 2] = math.sin(theta)
    R[1, 1] = 1
    R[2, 0] = -math.sin(theta); R[2, 2] = math.cos(theta)

    return R

# returns the rotation matrix of a rotatino around the Z-axis
def rotZ(theta):

    R=np.zeros((3,3))
   
    R[0, 0] = math.cos(theta); R[0, 1] = -math.sin(theta)
    R[1, 0] = math.sin(theta); R[1, 1] = math.cos(theta)
    R[2, 2] = 1

    return R

# ensures that the returned expression is quaternion (given either a quaternion or a Rotation matrix)
def enforceQuat(A):

    A = np.array(A)

    if A.size > 4: # A is a rotation matrix
        Q = rot2quat(A) 
    else: # A is quaternion
        Q = A

    return Q

# ensures that the returned expression is quaternion (given either a quaternion or a Rotation matrix)
def enforceRot(A):

    A = np.array(A)

    if A.size > 4: # A is a rotation matrix
        R = A
    else: # A is quaternion
        R = quat2rot(A) 

    return R

# theis computes the angle axis error between two vectors a and b
def angle_axis_vectors(a, b):


    theta = math.acos((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    k = skewSymmetric(a) @ b / np.linalg.norm(skewSymmetric(a) @ b)

    # k.shape = (3, 1)
    return theta * k

def rxryrz_to_rotation(rx, ry, rz):

    return  rotY(ry) @ rotX(rx) @ rotZ(rz)




def angle2rot(w):

    w = np.array(w)

    rx = w[0]
    ry = w[1]
    rz = w[2]

    # Tool orientation as unit quaternion
    angle = np.sqrt(rx**2 + ry**2 + rz**2)
    if angle < 1e-16:
        w = 1
        x = 0
        y = 0
        z = 0
    else:
        w = np.cos(angle / 2)
        sin_a = np.sin(angle / 2)
        x = sin_a * rx / angle
        y = sin_a * ry / angle
        z = sin_a * rz / angle
    return quat2rot([x, y, z, w])

