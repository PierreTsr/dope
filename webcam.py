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
    ret, frame = vc.read()
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

    dope_custom.setup("DOPErealtime_v1_0_0")
    vc = cv2.VideoCapture(0)

    if args.dim3:
        viewer3d = visu3d.Viewer3d()
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
            #print(poses3d)
            image = image[:,:,::-1]
            cv2.imshow("DOPE", image)
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