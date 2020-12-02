import cv2
import visu3d
import dope_custom
import numpy as np
import sys
import argparse
import time

parser = argparse.ArgumentParser(description="Run and display the result of the DOPE model on the webcam.")
parser.add_argument("-d", "--dim3", action="store_true")
parser.add_argument("-s", "--save")
args = parser.parse_args()

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
        fps = 20 #vc.get(cv2.CAP_PROP_FPS)
        print(width, height, fps)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter("output/" + args.save + ".avi", fourcc, fps, (width, height))

    frameTime = []

    while(vc.isOpened()):
        ret, frame = vc.read()
        if ret:
            start = time.time()
            if args.dim3:
                detections, body_with_wrists, body_with_head  = dope_custom.runModel3D(frame, viewer3d) 
                img3d, img2d = viewer3d.plot3d(frame[:,:,::-1], 
                    bodies={
                        'pose3d': np.stack([d['pose3d'] for d in detections['body']]) if len(detections["body"])>0 else np.empty( (0,0,3), dtype=np.float32),
                        'pose2d': np.stack([d['pose2d'] for d in detections['body']]) if len(detections["body"])>0 else np.empty( (0,0,2), dtype=np.float32),
                    },
                    hands={
                        'pose3d': np.stack([d['pose3d'] for d in detections['hand']]) if len(detections["hand"])>0 else np.empty( (0,0,3), dtype=np.float32),
                        'pose2d': np.stack([d['pose2d'] for d in detections['hand']]) if len(detections["hand"])>0 else np.empty( (0,0,2), dtype=np.float32),
                    },
                    faces={
                        'pose3d': np.stack([d['pose3d'] for d in detections['face']]) if len(detections["face"])>0 else np.empty( (0,0,3), dtype=np.float32),
                        'pose2d': np.stack([d['pose2d'] for d in detections['face']]) if len(detections["face"])>0 else np.empty( (0,0,2), dtype=np.float32),
                    },
                    body_with_wrists=body_with_wrists,
                    body_with_head=body_with_head,
                    interactive=False)

                if args.save:
                    output.write(img3d)
            else:
                image = dope_custom.runModel(frame, parts=["body"])
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