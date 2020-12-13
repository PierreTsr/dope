import numpy as np
import mathutils as bmath
import math

#Trunk keypoints: 13 points
#RFoot, LFoot, RKnee, LKnee, RAnkle, LAnkle, RWrist, LWrist, RElbow, Lelbow, RShoulder, LShoulder, Head
#Missing :
#   Pelvis : 13 => interpolate ankles
#   Ribcage : 14 => interpolate shoulders
trunkHierarchy = [2, 3, 4, 5, 13, 13, 8, 9, 10, 11, 14, 14, 14, 13, 13]


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def getQuaternionRotations(positions):
    """
    Computes rotations for a basic 15 keypoints rig.

    input: numpy array with the following positions: RFoot, LFoot, RKnee, LKnee, RAnkle, LAnkle, RWrist, LWrist, RElbow, Lelbow, RShoulder, LShoulder, Head

    output: dictionnary of quaternions with params "RKnee", "LKnee", "RAnkle", "LAnkle", "RElbow", "LElbow", "RShoulder", "LShoulder"
    """
    ribcage = (positions[10, :] + positions[11, :])/2
    pelvis = (positions[4, :] + positions[5, :])/2
    completePositions = np.vstack((positions, np.expand_dims(pelvis,0), np.expand_dims(ribcage, 0)))

    print(completePositions)
    rotations = dict({})
    rotations["Base HumanRCalf"] = kneeRotation(completePositions[4,:], completePositions[2,:], completePositions[0,:])
    rotations["Base HumanLCalf"] = kneeRotation(completePositions[5,:], completePositions[3,:], completePositions[1,:])
    rotations["Base HumanRThigh"] = ankleRotation(completePositions[13,:], completePositions[4,:], completePositions[2,:], completePositions[0,:], True)
    rotations["Base HumanLThigh"] = ankleRotation(completePositions[13,:], completePositions[5,:], completePositions[3,:], completePositions[1,:])
    rotations["Base HumanRForearm"] = elbowRotation(completePositions[10,:], completePositions[8,:], completePositions[6,:])
    rotations["Base HumanLForearm"] = elbowRotation(completePositions[11,:], completePositions[9,:], completePositions[7,:])
    rotations["Base HumanRUpperarm"] = shoulderRotation(completePositions[14,:], completePositions[10,:], completePositions[8,:], completePositions[6,:], True)
    rotations["Base HumanLUpperarm"] = shoulderRotation(completePositions[14,:], completePositions[11,:], completePositions[9,:], completePositions[7,:])

    return rotations


def shoulderRotation(ribcage, shoulder, elbow, wrist, right=True):
    """
    compute the quaternion rotation of the shoulder based on four keypoints.
    set right to True for the right side and to False for the left side
    """
    if right :
        initialY = np.expand_dims(normalize(shoulder - ribcage), 1)
        initialX = np.expand_dims(np.array([0, 0, -1]),1)
        initialZ = np.expand_dims(normalize(np.cross(initialX.ravel(), initialY.ravel())), 1)

        finalY = np.expand_dims(normalize(elbow - shoulder), 1)
        finalX = - np.expand_dims(normalize(np.cross(wrist - shoulder, elbow - shoulder)), 1)
        finalZ = np.expand_dims(normalize(np.cross(finalX.ravel(), finalY.ravel())), 1)

    else:
        initialY = np.expand_dims(normalize(shoulder - ribcage), 1)
        initialZ = np.expand_dims(np.array([0, 0, 1]),1)
        initialX = np.expand_dims(normalize(np.cross(initialY.ravel(), initialZ.ravel())), 1)

        finalY = np.expand_dims(normalize(elbow - shoulder), 1)
        finalZ = np.expand_dims(normalize(np.cross(wrist - shoulder, elbow - shoulder)), 1)
        finalX = np.expand_dims(normalize(np.cross(finalY.ravel(), final.ravel())), 1)
    
    initialBase = np.hstack((initialX, initialY, initialZ))
    
    finalBase = np.hstack((finalX, finalY, finalZ))
    print(initialBase,"\n", finalBase)
    rotation = finalBase.dot(initialBase.T)
    print(rotation)
    print(np.linalg.det(rotation))
    quat = bmath.Matrix(rotation).to_quaternion()
    print(quat)
    return rotation

def ankleRotation(pelvis, ankle, knee, foot, right=False):
    """
    compute the quaternion rotation of the ankle based on four keypoints.
    set right to True for the right side and to False for the left side
    """
    initialX = np.expand_dims(normalize(ankle - pelvis), 1)
    initialZ = np.expand_dims(np.array([0, 0, -1]), 1)
    initialY = np.expand_dims(normalize(np.cross(initialZ.ravel(), initialX.ravel())), 1)
    initialBase = np.hstack((initialX, initialY, initialZ))
    
    finalX = np.expand_dims(normalize(np.cross(knee - ankle, foot - ankle)), 1)
    if right:
        finalX = - finalX
    finalZ = np.expand_dims(normalize(knee - ankle), 1)
    finalY = np.expand_dims(np.cross(finalZ.ravel(), finalX.ravel()), 1)
    finalBase = np.hstack((finalX, finalY, finalZ))

    rotation = finalBase.dot(initialBase.T)
    quat = bmath.Matrix(rotation).to_quaternion()
    return quat

def kneeRotation(ankle, knee, foot):
    """
    compute the quaternion rotation of the knee based on three keypoints.
    """

    initialX = normalize(knee - ankle)
    finalX = normalize(foot - knee)
    axis = np.array([-1,0,0])
    angle = math.acos(initialX.dot(finalX))
    quat = bmath.Quaternion(axis, angle)

    return quat

def elbowRotation(shoulder, elbow, wrist):
    """
    compute the quaternion rotation of the elbow based on three keypoints.
    """

    initialX = normalize(elbow - shoulder)
    finalX = normalize(wrist - elbow)
    axis = np.cross(initialX, finalX)
    angle = math.acos(initialX.dot(finalX))
    quat = bmath.Quaternion(axis, angle)

    return quat

completePositions = np.array(
    [[ 1.05972715e-01, -1.91484168e-01, -9.87268388e-01],
    [-1.22449167e-01, -2.20937118e-01, -9.77727592e-01],
    [ 1.14813454e-01, -2.44370565e-01, -5.69621325e-01],
    [-1.09872602e-01, -2.70632267e-01, -5.58402002e-01],
    [ 9.01385844e-02, -2.47281659e-02, -2.34129876e-01],
    [-9.13349167e-02, -2.88566854e-02, -2.34967083e-01],
    [ 3.71678740e-01, -1.35432780e-01, -4.79045883e-02],
    [-2.35885561e-01, -2.05345720e-01, -4.08929251e-02],
    [ 3.08316708e-01, -5.69299376e-03,  5.63857565e-03],
    [-2.73237437e-01, -4.46720421e-02,  2.24210482e-04],
    [ 1.91548601e-01,  9.01740044e-03,  2.30540872e-01],
    [-1.88258335e-01, -5.49487676e-03,  2.33912423e-01],
    [-4.47874918e-04, -2.46447362e-02,  4.44248855e-01],
    [-5.98166138e-04, -2.67924257e-02, -2.34548479e-01],
    [ 1.64513290e-03,  1.76126184e-03,  2.32226640e-01]])

q = shoulderRotation(completePositions[14,:], completePositions[10,:], completePositions[8,:], completePositions[6,:], True)
blenderBase = np.array([[0.03 , 0.999, .008],
    [-0.035, 0.001, -0.99],
    [-0.99, 0.038, 0.019]])
print(rotation.dot(blenderBase))