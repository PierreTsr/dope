import cv2
import visu3d
import dope_custom
import numpy as np


dope_custom.setup("DOPErealtime_v1_0_0")
viewer3d = visu3d.Viewer3d()
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    detections, body_with_wrists, body_with_head  = dope_custom.runModel3D(frame, viewer3d) 
    # display results in 3D
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
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
vc.release()
cv2.destroyWindow("DOPE")