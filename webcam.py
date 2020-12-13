import cv2
import dope_custom
import numpy as np
import sys
import argparse
import time

parser = argparse.ArgumentParser(description="Run and display the result of the DOPE model on the webcam.")
parser.add_argument("-d", "--dim3", action="store_true")
parser.add_argument("-s", "--save")
args = parser.parse_args()

def setupCapture():
    dope_custom.setup("DOPErealtime_v1_0_0")
    vc = cv2.VideoCapture(0)
    cv2.namedWindow("DOPE")
    return vc

def capturePose(videoCapture):
    assert(videoCapture.isOpened())
    ret, frame = videoCapture.read()
    if ret:
        start = time.time()
        image, poses3d = dope_custom.runModel(frame, parts=["body"])
        image = image[:,:,::-1]
        cv2.imshow("DOPE", image)
        cv2.waitKey(1)
        return poses3d
    else:
        raise ValueError("Failed to read camera input")

def closeCapture(videoCapture):
    videoCapture.release()
    cv2.destroyWindow("DOPE")

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import art3d
    import geometrySimple
    
    dope_custom.setup("DOPErealtime_v1_0_0")
    vc = cv2.VideoCapture(0)

    if args.dim3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()
    else:
        cv2.namedWindow("DOPE")

    if args.save:  
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 12 #vc.get(cv2.CAP_PROP_FPS)
        print(width, height, fps)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter("output/" + args.save + ".avi", fourcc, fps, (width, height))

    frameTime = []

    while(vc.isOpened()):
        ret, frame = vc.read()
        if ret:
            start = time.time()
            image, poses3d = dope_custom.runModel(frame, parts=["body"])
            #if len(poses3d["body"]) > 0:
                #print(geometry .getQuaternionRotations(poses3d["body"][0,:,:]))
            print(poses3d)
            image = image[:,:,::-1]
            cv2.imshow("DOPE", image)
            if args.dim3 and len(poses3d["body"]) > 0:
                ax.clear()
                body = geometrySimple.getCompletePositions(poses3d["body"][0]) + np.array([0.5,0.5,0.6])
                legs = art3d.Line3DCollection([[body[0,:], body[2,:]],[body[2,:], body[4,:]],[body[4,:],body[5,:]], [body[5,:],body[3,:]],[body[3,:], body[1,:]]])
                arms = art3d.Line3DCollection([[body[13,:], body[14,:]],[body[6,:], body[8,:]],[body[8,:], body[10,:]],[body[10,:],body[11,:]], [body[11,:],body[9,:]],[body[9,:], body[7,:]]])
                ax.add_collection(legs)
                ax.add_collection(arms)
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)
            if args.save:
                image = image[:height,:width,:]
                output.write(image)
            rval, frame = vc.read()
            key = cv2.waitKey(5)
            if key == 27: # exit on ESC
                break
            frameTime.append(time.time() - start)
        else:
            break
    
    if args.save:
        output.release()
    vc.release()
    cv2.destroyWindow("DOPE")
    meanTime = np.mean(np.array(frameTime))
    print("Mean computation time per frame : ", int(meanTime*1000), "ms")