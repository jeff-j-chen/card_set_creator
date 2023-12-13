import os
import numpy as np
import cv2
import json
import time

def draw_rectangles(event, x, y, flags, params):
    # check if the point is on the left or right half of the image
    # left half is used for annotating correct bounding boxes
    if x < params['image'].shape[1]//2:
        if event == cv2.EVENT_LBUTTONDOWN:
            params['start_pt'] = (x, y)
            params['drawing'] = True
        elif event == cv2.EVENT_LBUTTONUP:
            params['end_pt'] = (x, y)
            params['drawing'] = False
            if np.linalg.norm(np.array(params['start_pt']) - np.array(params['end_pt'])) <= 10:
                transc_add = add_quad_point(params)
            else:
                transc_add = add_rect(params)
            cv2.imshow('image', params['image'])
            cv2.waitKey(100)
            if len(transc_add) > 0:
                transc = input('Transcription: ')
                if transc != '' and transc != ' ':
                    mark = {
                        'transcription': transc,
                        'points': list(transc_add),
                    }
                    output.append(mark)
        elif event == cv2.EVENT_MOUSEMOVE:
            if params['drawing']:
                img_copy = params['og_img'].copy()
                params['image'] = cv2.rectangle(img_copy, params['start_pt'], (x, y), (134, 199, 56), 3)
                cv2.imshow('image', params['image'])
    # if click is on the right half, the user wants to discard originally detected bounding boxes
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            discard_det_at(x, y, params)

def discard_det_at(x, y, params):
    for detection in params['detections']:
        points = np.array(detection['points'])
        points[:, 0] += params['image'].shape[1]//2
        contour = points.reshape((-1, 1, 2))
        if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
            params['og_img'] = cv2.polylines(params['image'], [points], True, (76, 76, 225), 4, cv2.LINE_AA)
            cv2.imshow('image', params['image'])
            detection['transcription'] = ''

def add_rect(params):
    x1, y1 = min(params['start_pt'][0], params['end_pt'][0]), min(params['start_pt'][1], params['end_pt'][1])
    x2, y2 = max(params['start_pt'][0], params['end_pt'][0]), max(params['start_pt'][1], params['end_pt'][1])
    cv2.rectangle(params['image'], params['start_pt'], params['end_pt'], (81, 219, 66), 3)
    params['og_img'] = params['image'].copy()
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

def add_quad_point(params):
    params['og_img'] = cv2.circle(params['image'], params['start_pt'], 4, (51, 67, 232), -1)
    cv2.imshow('image', params['image'])
    params['quad_points'].append(params['start_pt'])
    if len(params['quad_points']) == 4:
        qp = np.array(params['quad_points'])
        center = qp.mean(axis=0)
        angles = np.arctan2(qp[:,1] - center[1], qp[:,0] - center[0])
        sorted_points = qp[np.argsort(angles)]
        params['og_img'] = cv2.polylines(params['image'], [sorted_points], True, (79, 92, 227), 3)
        params['quad_points'].clear()
        return sorted_points
    return []

cv2.namedWindow('image')

card_dir = "/home/jeff/SSD_2/vis/"
with open(os.path.join(card_dir, 'results.json'), 'r') as f:
    results = json.load(f)

output = []
if not os.path.exists('last_det.txt'): open('last_det.txt', 'w').write('0')
with open('last_det.txt', 'r') as f: last = int(f.read())

for (i, (file_path, detections)) in enumerate(results.items()):
    if i < last: continue
    with open('last_det.txt', 'w') as f: f.write(str(i))
    img = cv2.imread(os.path.join(card_dir, file_path))
    s = max(img.shape[1]/1600, img.shape[0]/900)
    img = cv2.resize(img, (int(img.shape[1]//s), int(img.shape[0]//s)))
    img_flipped = np.concatenate((img[:, img.shape[1]//2:], img[:, :img.shape[1]//2]), axis=1)
    for detection in detections:
        detection['points'] = [(int(x//s), int(y//s)) for (x, y) in detection['points']]
    cv2.imshow('image', img)
    cv2.setMouseCallback(
        'image',
        draw_rectangles, 
        {
            'image': img,
            'og_img': img.copy(),
            'start_pt': None,
            'end_pt': None,
            'drawing': False,
            'filename': os.path.basename(file_path),
            'detections': detections,
            'quad_points': [],
        }
    )
    while True:
        key = cv2.waitKey(1)
        if key == ord(' '):
            # combined = output
            combined = output + detections
            combined = [d for d in combined if d['transcription'] != '']
            for d in combined:
                for i in range(len(d['points'])):
                    d['points'][i] = [int(x) for x in d['points'][i]]
            with open('dets.txt', 'a') as f:
                json_combined = json.dumps(combined)
                f.write(f"{os.path.basename(file_path)}\t{json_combined}\n")
            output.clear()
            break

cv2.destroyAllWindows()
