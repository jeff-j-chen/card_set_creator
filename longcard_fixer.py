import os
import cv2
import json
import numpy as np

class DetScaler:
    def __init__(self):
        self.all_cards = "/home/jeff/SSD_2/Downloads/all_cards/Baseball_data"
        with open('dets.txt', 'r') as f:
            line = f.readline()
            filename, json_part = line.split('\t', 1)
            img = cv2.imread(os.path.join(self.all_cards, filename))
            s = max(img.shape[1]*2/1600, img.shape[0]/900)
            self.img = cv2.resize(img, (int(img.shape[1]//s), int(img.shape[0]//s)))
            self.dets = json.loads(json_part)
            # ignore the rest of the lines
            for line in f:
                pass
        self.all_dets = {}

    def scale_points(self, scale, update_dets=False):
        update = self.img.copy()
        for detection in self.dets:
            points = np.array(detection['points'], np.float64)
            points[:, 0] *= scale
            points[:, 1] *= scale
            points = points.astype(np.int32)
            if update_dets:
                detection['points'] = points.tolist()
            else:
                cv2.polylines(update, [points], True, (0, 255, 0), 2)
        return update

    def update_image(self, trackbar_inp):
        scale = trackbar_inp / 1000
        update = self.scale_points(scale, False)
        cv2.imshow('image', update)

    def main(self):
        cv2.namedWindow('image')
        cv2.createTrackbar('s', 'image', 655, 1100, self.update_image)
        cv2.setTrackbarMin('s', 'image', 500)

        with open('dets.txt', 'r') as f:
            for line in f:
                filename, json_part = line.split('\t', 1)
                img = cv2.imread(os.path.join(self.all_cards, filename))
                s = max(img.shape[1]*2/1600, img.shape[0]/900)
                self.img = cv2.resize(img, (int(img.shape[1]//s), int(img.shape[0]//s)))
                self.dets = json.loads(json_part)
                print(f"width: {img.shape[1]}, height: {img.shape[0]}")
                if img.shape[1] < img.shape[0] + 50:
                    self.all_dets[filename] = self.dets
                    continue
                self.update_image(cv2.getTrackbarPos('s', 'image'))
                while True:
                    key = cv2.waitKey(1)
                    if key == ord('x'):
                        break
                    elif key == ord(' '):
                        self.scale_points(cv2.getTrackbarPos('s', 'image') / 1000, True)
                        self.all_dets[filename] = self.dets
                        break
        cv2.destroyAllWindows()

    def write_fixes(self):
        with open('dets_fixed.txt', 'w') as f:
            for file_path, detections in self.all_dets.items():
                for detection in detections:
                    qp = np.array(detection['points'])
                    center = qp.mean(axis=0)
                    angles = np.arctan2(qp[:,1] - center[1], qp[:,0] - center[0])
                    sorted_points = qp[np.argsort(angles)]
                    detection['points'] = sorted_points.tolist()
                f.write(f"{file_path}\t{json.dumps(detections)}\n")

det_scaler = DetScaler()
det_scaler.main()
det_scaler.write_fixes()