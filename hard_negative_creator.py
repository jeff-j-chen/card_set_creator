# creates a set of hard negative crops from the baseball test set

import os
import cv2
import json
import numpy as np

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        for detection in params['detections']:
            points = np.array(detection['points'])
            contour = points.reshape((-1, 1, 2))
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                cv2.polylines(params['flipped'], [points], True, (76, 76, 225), 4, cv2.LINE_AA)
                cv2.imshow('image', params['flipped'])
                crop = params['original'][
                    min(points[:, 1]):max(points[:, 1]),
                    min(points[:, 0]):max(points[:, 0])
                ]
                part_name = f"{os.path.basename(file_path)[:-4]}_{params['i']}.jpg"
                name = f"rec_crops/baseball_test_crops/{part_name}"
                cv2.imwrite(name, crop)
                with open('pred_labels/baseball_test_preds.txt', 'a') as f:
                    f.write(f"{part_name}\t{detection['transcription']}\n")
                params['i'] += 1
                break

cv2.namedWindow('image')

card_dir = "/home/jeff/SSD_2/vis/"
with open(os.path.join(card_dir, 'results.json'), 'r') as f:
    results = json.load(f)

if not os.path.exists('last_hn.txt'): open('last_hn.txt', 'w').write('0')
with open('last_hn.txt', 'r') as f: last = int(f.read())
for (i, (file_path, detections)) in enumerate(results.items()):
    if i < last: continue
    with open('last_hn.txt', 'w') as f: f.write(str(i))
    img = cv2.imread(os.path.join(card_dir, file_path))
    s = max(img.shape[1]/1600, img.shape[0]/900)
    img = cv2.resize(img, (int(img.shape[1]//s), int(img.shape[0]//s)))
    img_flipped = np.concatenate((img[:, img.shape[1]//2:], img[:, :img.shape[1]//2]), axis=1)
    for detection in detections:
        detection['points'] = [(int(x//s), int(y//s)) for (x, y) in detection['points']]
    params = {
        'detections': detections,
        'i': 0,
        'original': img,
        'flipped': img_flipped
    }
    # keep track of the number of crops
    cv2.setMouseCallback('image', click_event, params)
    cv2.imshow('image', img_flipped)
    while True:
        # only continue if the space key is pressed
        key = cv2.waitKey(1)
        if key == ord(' '):
            break
        # if i press the enter key, then add the 'last' to last.txt
        elif key == 13:
            with open('bad.txt', 'a') as f: f.write(f"\n{i}")
            break

cv2.destroyAllWindows()