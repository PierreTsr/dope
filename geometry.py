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
    completePositions = np.hstack((np.expand_dims(completePositions[:,0],1), np.expand_dims(completePositions[:,2],1), np.expand_dims(completePositions[:,1], 1)))

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
        initialX = np.expand_dims(np.array([0, 1, 0]),1)
        initialZ = np.expand_dims(normalize(np.cross(initialX.ravel(), initialY.ravel())), 1)

        finalY = np.expand_dims(normalize(elbow - shoulder), 1)
        finalZ = - np.expand_dims(normalize(np.cross(wrist - shoulder, elbow - shoulder)), 1)
        finalX = np.expand_dims(normalize(np.cross(finalY.ravel(), finalZ.ravel())), 1)

    else:
        initialY = np.expand_dims(normalize(shoulder - ribcage), 1)
        initialX = np.expand_dims(np.array([0, 1, 0]),1)
        initialZ = np.expand_dims(normalize(np.cross(initialX.ravel(), initialY.ravel())), 1)

        finalY = np.expand_dims(normalize(elbow - shoulder), 1)
        finalX = np.expand_dims(normalize(np.cross(wrist - shoulder, elbow - shoulder)), 1)
        finalZ = np.expand_dims(normalize(np.cross(finalX.ravel(), finalY.ravel())), 1)
    
    initialBase = np.array(
        [[0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]])
    initialBase = np.hstack((initialX, initialY, initialZ))

    finalBase = np.array(
        [[0, 0, -1],
        [1, 0, 0],
        [0, -1, 0]])

    finalBase = np.hstack((finalX, finalY, finalZ))
    print(initialBase,"\n", finalBase)
    rotation = (finalBase.T).dot(initialBase)
    rotation = rotation.T
    #rotation = initialBase
    print(rotation)
    print(np.linalg.det(rotation))
    quat = bmath.Matrix(rotation).to_quaternion()
    print(bmath.Matrix(rotation).to_euler())
    print(quat)
    return bmath.Matrix(rotation).to_euler()

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
    return bmath.Matrix(rotation).to_euler()

def kneeRotation(ankle, knee, foot):
    """
    compute the quaternion rotation of the knee based on three keypoints.
    """

    initialX = normalize(knee - ankle)
    finalX = normalize(foot - knee)
    axis = np.array([-1,0,0])
    angle = math.acos(initialX.dot(finalX))
    quat = bmath.Quaternion(axis, angle)

    return quat.to_euler()

def elbowRotation(shoulder, elbow, wrist, right = True):
    """
    compute the quaternion rotation of the elbow based on three keypoints.
    """
    if right:
        parentY = np.expand_dims(normalize(elbow - shoulder), 1)
        parentZ = - np.expand_dims(normalize(np.cross(wrist - shoulder, elbow - shoulder)), 1)
        parentX = np.expand_dims(normalize(np.cross(parentY.ravel(), parentZ.ravel())), 1)
        parentBase = np.hstack((parentX, parentY, parentZ))
        print(parentBase)


    initialY = normalize(elbow - shoulder).dot(parentBase)
    finalY = normalize(wrist - elbow).dot(parentBase)
    axis = normalize(np.cross(initialY, finalY))
    print(initialY)
    print(finalY)
    print(axis)
    angle = math.acos(initialY.dot(finalY))
    print(angle)
    rot = bmath.Quaternion(axis, angle).to_euler()
    print(rot)
    return rot

positions = np. array(
    [[ 0.10183189, -0.99292105,  0.33507288],
    [-0.13642786, -0.9890006 ,  0.32888356],
    [ 0.09715796, -0.6364533 ,  0.10153572],
    [-0.12420492, -0.63063073,  0.10392337],
    [ 0.08191029, -0.2443048 ,  0.04699538],
    [-0.10647766, -0.23837054,  0.04459823],
    [ 0.33463606, -0.20243369, -0.03211017],
    [-0.32894036, -0.17874508,  0.06295507],
    [ 0.28297815, -0.01894949,  0.01688627],
    [-0.24777538,  0.00328757,  0.0892858 ],
    [ 0.1802957 ,  0.21587019, -0.02095461],
    [-0.15435581,  0.23491085,  0.00785058],
    [ 0.0152513 ,  0.42819002, -0.05184324]])
ribcage = (positions[10, :] + positions[11, :])/2
pelvis = (positions[4, :] + positions[5, :])/2
completePositions = np.vstack((positions, np.expand_dims(pelvis,0), np.expand_dims(ribcage, 0)))
completePositions = np.hstack((np.expand_dims(completePositions[:,0],1), np.expand_dims(completePositions[:,2],1), np.expand_dims(completePositions[:,1], 1)))

q = shoulderRotation(completePositions[14,:], completePositions[10,:], completePositions[8,:], completePositions[6,:], True)

shoulderRotation(completePositions[14,:], completePositions[11,:], completePositions[9,:], completePositions[7,:], False)

elbowRotation(completePositions[10,:], completePositions[8,:], completePositions[6,:], True)