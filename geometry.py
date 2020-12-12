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

    rotations = dict({})
    rotations["RKnee"] = kneeRotation(completePositions[4,:], completePositions[2,:], completePositions[0,:])
    rotations["LKnee"] = kneeRotation(completePositions[5,:], completePositions[3,:], completePositions[1,:])
    rotations["RAnkle"] = ankleRotation(completePositions[13,:], completePositions[4,:], completePositions[2,:], completePositions[0,:], True)
    rotations["LAnkle"] = ankleRotation(completePositions[13,:], completePositions[5,:], completePositions[3,:], completePositions[1,:])
    rotations["RElbow"] = elbowRotation(completePositions[10,:], completePositions[8,:], completePositions[6,:])
    rotations["LElbow"] = elbowRotation(completePositions[11,:], completePositions[9,:], completePositions[7,:])
    rotations["RShoulder"] = shoulderRotation(completePositions[14,:], completePositions[10,:], completePositions[8,:], completePositions[6,:], True)
    rotations["LShoulder"] = shoulderRotation(completePositions[14,:], completePositions[11,:], completePositions[9,:], completePositions[7,:])

    return rotations


def shoulderRotation(ribcage, shoulder, elbow, wrist, right=False):
    """
    compute the quaternion rotation of the shoulder based on four keypoints.
    set right to True for the right side and to False for the left side
    """
    initialX = np.expand_dims(normalize(shoulder - ribcage), 0)
    initialZ = np.expand_dims(np.array([0, 1, 0]),0)
    initialY = normalize(np.cross(initialZ, initialX))
    initialBase = np.vstack((initialX, initialY, initialZ))
    
    finalX = np.expand_dims(normalize(elbow - shoulder), 0)
    finalZ = np.expand_dims(normalize(np.cross(wrist - shoulder, elbow - shoulder)), 0)
    if right:
        finalZ = -finalZ
    finalY = normalize(np.cross(finalZ, finalX))
    finalBase = np.vstack((finalX, finalY, finalZ))

    rotation = finalBase.dot(initialBase.T)
    quat = bmath.Matrix(rotation).to_quaternion()
    return quat

def ankleRotation(pelvis, ankle, knee, foot, right=False):
    """
    compute the quaternion rotation of the ankle based on four keypoints.
    set right to True for the right side and to False for the left side
    """
    initialX = np.expand_dims(normalize(ankle - pelvis), 0)
    initialZ = np.expand_dims(np.array([0, -1, 0]),0)
    initialY = normalize(np.cross(initialZ, initialX))
    initialBase = np.vstack((initialX, initialY, initialZ))
    
    finalX = np.expand_dims(normalize(np.cross(knee - ankle, foot - ankle)), 0)
    if right:
        finalX = - finalX
    finalZ = np.expand_dims(normalize(knee - ankle), 0)
    finalY = np.cross(finalZ, finalX)
    finalBase = np.vstack((finalX, finalY, finalZ))
    print(finalBase.shape, initialBase.shape)

    rotation = finalBase.dot(initialBase.T)
    quat = bmath.Matrix(rotation).to_quaternion()
    return quat

def kneeRotation(ankle, knee, foot):
    """
    compute the quaternion rotation of the knee based on three keypoints.
    """

    initialX = normalize(knee - ankle)
    finalX = normalize(foot - knee)
    axis = np.cross(initialX, finalX)
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