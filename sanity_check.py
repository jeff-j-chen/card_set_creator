# Used to mark detections onto images for manual verification, to make sure they are actually correct

import cv2
import json
import numpy as np

# Load detections from file
detections = {}
with open("dets_fixed.txt", 'r') as f:
    for line in f:
        filename, json_part = line.split('\t', 1)
        detections[filename] = json.loads(json_part)

stop = False
# Draw rectangles and transcriptions on each image 
for filename, file_detections in detections.items():
    img = cv2.imread("nitin_fullset/" + filename)
    for detection in file_detections:
        points = np.array(detection['points'], np.int32)
        qp = points
        center = qp.mean(axis=0)
        angles = np.arctan2(qp[:,1] - center[1], qp[:,0] - center[0])
        sorted_points = qp[np.argsort(angles)]
        cv2.polylines(img, [sorted_points], True, (0, 255, 0), 2)
        cv2.putText(img, detection['transcription'], tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Image', img)
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') and not stop:
            break

cv2.destroyAllWindows()