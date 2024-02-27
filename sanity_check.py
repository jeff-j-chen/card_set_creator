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
bad=["eb-115044431801_cropped.jpg", "eb-115774546898_cropped.jpg", "eb-115845598534_cropped.jpg", "eb-115845601368_cropped.jpg", "eb-115845603874_cropped.jpg", "eb-115845605358_cropped.jpg", "eb-115845689624_cropped.jpg", "eb-115849609391_cropped.jpg", "eb-115849682283_cropped.jpg", "eb-123531992080_cropped.jpg", "eb-125596100149_cropped.jpg", "eb-125670969760_cropped.jpg", "eb-125708445256_cropped.jpg", "eb-125832985798_cropped.jpg", "eb-125872973791_cropped.jpg", "eb-125875806561_cropped.jpg", "eb-125892443679_cropped.jpg", "eb-134050566428_cropped.jpg", "eb-134410467684_cropped.jpg", "eb-134469026892_cropped.jpg", "eb-134522529538_cropped.jpg", "eb-134551414801_cropped.jpg", "eb-134555328405_cropped.jpg", "eb-134555335192_cropped.jpg", "eb-134559030061_cropped.jpg", "eb-134561738126_cropped.jpg", "eb-134574690855_cropped.jpg", "eb-134583252681_cropped.jpg", "eb-134584237872_cropped.jpg", "eb-134600169419_cropped.jpg"]
stop = False
# Draw rectangles and transcriptions on each image 
i = 0
while i < len(detections):
    filename, file_detections = list(detections.items())[i]
    # if filename not in bad:
    #     i+=1
    #     continue
    img = cv2.imread("nitin_fullset/" + filename)
    print(f"image dim: {img.shape}")
    for detection in file_detections:
        points = np.array(detection['points'], np.int32)
        print(f"reading points: {points}")
        qp = points
        center = qp.mean(axis=0)
        angles = np.arctan2(qp[:,1] - center[1], qp[:,0] - center[0])
        sorted_points = qp[np.argsort(angles)]
        cv2.polylines(img, [sorted_points], True, (0, 255, 0), 2)
        cv2.putText(img, detection['transcription'], tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # draw a test rectangle 
        cv2.rectangle(img, (113, 100), (240, 640), (255, 0, 0), 2)

    cv2.imshow('Image', img)
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') and not stop:
            i += 1
            break
        elif key == ord('x'):
            with open('bad.txt', 'r') as f:
                already_marked = f.read().split('\n')
            if filename not in already_marked:
                with open('bad.txt', 'a') as f:
                    f.write(filename + '\n')
                print(f"Marked {filename} as bad")
                i += 1
                break
        elif key == ord('z'):
            i -= 1
            break
            

cv2.destroyAllWindows()