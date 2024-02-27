# ADD A WAY TO LOOK THROUGH EVERY CARD THAT HAS IS TALLER THAN IT IS WIDE, THEN HAVE AN OPTION TO ROTATE IT 90

#DBACKS #4 marte -> starling marte




# Use detections from ocrapp as the base annotations, enabling quick labelling for new data
LOAD_FROM_RESULTS = True
CHECKING_MODE = False
CREATING_SLIDES = False
HN_MODE = False
hn_ocr_path = "/home/jeff/SSD_2/1000_sample_ocr/gpt_results.json"
hn_gpt4_path = "/home/jeff/SSD_2/1000_sample_gpt4/results.json"

import os
import cv2
import json
import time
import numpy as np
from PIL import Image
from utility import draw_ocr_box_txt

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
            cv2.waitKey(10)
            if len(transc_add) > 0:
                cv2.waitKey(10)
                time.sleep(0.1)
                cv2.imshow('image', params['image'])
                transc = input('Transcription: ')
                if transc != '' and transc != ' ':
                    mark = {
                        'transcription': transc,
                        'points': list(transc_add),
                    }
                    output.append(mark)
                    # annotate the image with the text at the given locatin
                    # use the size of the drawn bounding box to determine the location and size of the text
                    mid_center = (transc_add[0][0], (transc_add[0][1] + transc_add[2][1])//2 + 10)
                    font_size = 0.5
                    params['og_img'] = cv2.putText(params['image'], transc, mid_center, cv2.FONT_HERSHEY_SIMPLEX, font_size, (81, 219, 66), 1, cv2.LINE_AA)
                    cv2.waitKey(10)
                    time.sleep(0.1)
                    cv2.imshow('image', params['og_img'])
                    
        elif event == cv2.EVENT_MOUSEMOVE: 
            if params['drawing']:
                img_copy = params['og_img'].copy()
                params['image'] = cv2.rectangle(img_copy, params['start_pt'], (x, y), (134, 199, 56), 2)
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
    cv2.rectangle(params['image'], params['start_pt'], params['end_pt'], (81, 219, 66), 2)
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


if LOAD_FROM_RESULTS:
    # card_dir = "/home/jeff/SSD_2/1000_sample_ocr"
    card_dir = "/home/jeff/SSD_2/feb11_rec_feb7_det_output"
    with open(os.path.join(card_dir, 'results.json'), 'r') as f:
        results = json.load(f)
    all_cards = "/home/jeff/SSD_2/Downloads/all_cards/Baseball_data"
else:
    card_dir = "/home/jeff/SSD_2/Downloads/all_cards/Baseball_data"
    results = {}
    read = 'dets_fixed.txt' if not CHECKING_MODE else 'dets_new.txt'
    with open(read, 'r') as f:
        for line in f:
            filename, json_part = line.split('\t', 1)
            results[filename] = json.loads(json_part)

output = []
if not os.path.exists('last_det.txt'): open('last_det.txt', 'w').write('0')
with open(f"last{'_hn' if HN_MODE else '_det'}.txt", 'r') as f: 
    last = int(f.read())
    print(f"Hardnegative mode is {HN_MODE}, last is {last}")
    if CHECKING_MODE: last -= 2

seen_files = []
with open("dets.txt", 'r') as f:
    for line in f:
        filename, json_part = line.split('\t', 1)
        seen_files.append(filename)

# load the json files from the HN ocr and gpt4 results
discrep_dict = {}
if HN_MODE:
    with open(hn_ocr_path, 'r') as f:
        hn_ocr = json.load(f)
    with open(hn_gpt4_path, 'r') as f:
        hn_gpt4 = json.load(f)
        for file in hn_gpt4.keys():
            for key in hn_gpt4[file].keys():
                if type(hn_gpt4[file][key]) == list and len(hn_gpt4[file][key]) > 1:
                    hn_gpt4[file][key] = hn_gpt4[file][key][0]
    
    for file in hn_ocr.keys():
        discrep_dict[file] = []
        if file not in hn_gpt4.keys():
            continue
        for key in hn_ocr[file].keys():
            if hn_ocr[file][key].lower() != hn_gpt4[file][key].lower() and key == "name":
                discrep_dict[file] = [key, hn_ocr[file][key], hn_gpt4[file][key]]

for (i, (file_path, dts)) in enumerate(results.items()):
    if file_path != "eb-115044431801_cropped.jpg": 
        continue
    # if i < last: continue #1121
    # if file_path in seen_files and not HN_MODE: 
    #     print(f"skipping file {file_path} at index {i}")
    #     continue   
    # if HN_MODE:
    #     if len(discrep_dict[file_path]) <= 0: continue
    #     print(f"{discrep_dict[file_path][0]}: got {discrep_dict[file_path][1].lower()} (expected {discrep_dict[file_path][2].lower()})")
    # if not CHECKING_MODE:
    #     with open(f"last{'_hn' if HN_MODE else '_det'}.txt", 'w') as f:  f.write(str(i))
            
    img = cv2.imread(os.path.join(card_dir, file_path))
    s = max(img.shape[1]/1600, img.shape[0]/900)
    img = cv2.resize(img, (int(img.shape[1]//s), int(img.shape[0]//s)))
    print(f"image dim: {img.shape}")
    # for detection in dts:
    #     detection['points'] = [(int(x//s//s//0.9), int(y//s//s//0.9)) for (x, y) in detection['points']]
    
    if LOAD_FROM_RESULTS:
        for detection in dts:
            detection['points'] = [(int(x//s), int(y//s)) for (x, y) in detection['points']]
        # take the left of the image, draw bounding boxes, then stitch it horizontally with the right side of the original
        left = img[:, :img.shape[1]//2]
        right = img[:, img.shape[1]//2:]
        left = draw_ocr_box_txt(
            image=Image.fromarray(cv2.cvtColor(left, cv2.COLOR_BGR2RGB)),
            boxes=[det['points'] for det in dts],
            txts=[det['transcription'] for det in dts],
            scores=[0 for _ in dts],
            drop_score=0,
            font_path="/home/jeff/SSD_2/card_set_creator/simfang.ttf",
        )
        img = cv2.cvtColor(np.array(left), cv2.COLOR_RGB2BGR)
    else:
        img = draw_ocr_box_txt(
            image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
            boxes=[det['points'] for det in dts],
            txts=[det['transcription'] for det in dts],
            scores=[0 for _ in dts],
            drop_score=0,
            font_path="/home/jeff/SSD_2/card_set_creator/simfang.ttf",
        )
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    

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
            'detections': dts,
            'quad_points': [],
        }
    )
    while True:
        key = cv2.waitKey(1)
        if key == ord(' '):
            # combined = output
            combined = output + dts
            combined = [d for d in combined if d['transcription'] != '']
            for d in combined:
                for i in range(len(d['points'])):
                    d['points'][i] = [int(x) for x in d['points'][i]]
            mod = '_new' if not LOAD_FROM_RESULTS else ''
            mod2 = '_double' if CHECKING_MODE else ''
            with open(f"dets{mod}{mod2}.txt", 'a') as f:
                json_combined = json.dumps(combined)
                f.write(f"{os.path.basename(file_path)}\t{json_combined}\n")
            output.clear()
            break
        # if the key is x, simply break and move on
        elif key == ord('x'):
            break
        # if the key is z, write the current file name to good.txt, and draw a green rectangle around the image
        elif key == ord('z'):
            # make sure that the file name is not already in good.txt
            with open('good.txt', 'r') as f:
                already_marked = f.read().split('\n')
            if file_path not in already_marked:
                with open('good.txt', 'a') as f:
                    f.write(file_path + '\n')
                print(f"Marked {file_path} as good")
            img = cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (81, 219, 66), 10)
            cv2.imshow('image', img)
            if CREATING_SLIDES:
                break

cv2.destroyAllWindows()
