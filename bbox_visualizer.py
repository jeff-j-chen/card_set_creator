# simply show every image, one after the other

import cv2
import os
import json

card_dir = "/home/jeff/SSD_2/vis/"
with open(os.path.join(card_dir, 'results.json'), 'r') as f:
    results = json.load(f)


def visualize():
    with open('bad.txt', 'r') as f:
        indices = f.readlines()
        indices = [int(i) for i in indices]
        

    for (i, (file_path, detections)) in enumerate(results.items()):
        if i not in indices: continue
        img = cv2.imread(os.path.join(card_dir, file_path))
        s = max(img.shape[1]/1600, img.shape[0]/900)
        img = cv2.resize(img, (int(img.shape[1]//s), int(img.shape[0]//s)))
        cv2.imshow('image', img)
        while True:
            key = cv2.waitKey(1)
            if key == ord(' '):
                break
            elif key == 13:
                print("/home/jeff/SSD_2/Downloads/all_cards/Baseball_data/" + file_path)

# Call the function
visualize()