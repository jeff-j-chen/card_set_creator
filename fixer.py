import os
import numpy as np
import cv2
import json
import math

card_dir = "/home/jeff/SSD_2/vis/"
all_dets = {}
with open('dets.txt', 'r') as f:
    for line in f:
        filename, json_part = line.split('\t', 1)
        detections = json.loads(json_part)
        all_dets[filename] = detections

for (file_path, detections) in all_dets.items():
    img = cv2.imread(os.path.join(card_dir, file_path))
    s = max(img.shape[1]/1600, img.shape[0]/900)
    img = cv2.resize(img, (int(img.shape[1]//s), int(img.shape[0]//s)))
    # for every set of points, check if any of the points fall onto the right half of the image. 
    # if so, subtract half the image width from that set of points
    # and print that you are doing so
    # use numpy to speed things up
    for detection in detections:
        if any(np.array(detection['points'])[:, 0] > img.shape[1]//2):
            print(f"Fixing {file_path}")
            detection['points'] = [(x - img.shape[1]//2, y) for (x, y) in detection['points']]

    for i, detection in enumerate(detections):
        # Get the points
        points = np.array(detection['points'])

        # Fit a minimum area rectangle around the points
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # Get the points of the rotated rectangle
        rotated_points = box.tolist()

        # Rearrange the points to start from the top-left and move clockwise
        rotated_points = [rotated_points[i] for i in [2, 3, 0, 1]]

        # Update the points in the detection
        detection['points'] = rotated_points


    # save the left half of the image to the folder nitin_fullset
    cv2.imwrite(f"nitin_fullset/{file_path}", img[:, :img.shape[1]//2])

# recreate dets.txt with the new data
with open('dets_fixed.txt', 'w') as f:
    for (file_path, detections) in all_dets.items():
        for detection in detections:
            qp = np.array(detection['points'])
            center = qp.mean(axis=0)
            angles = np.arctan2(qp[:,1] - center[1], qp[:,0] - center[0])
            sorted_points = qp[np.argsort(angles)]
            detection['points'] = sorted_points.tolist()
        f.write(f"{file_path}\t{json.dumps(detections)}\n")

cv2.destroyAllWindows()