import os
import cv2
import json
import numpy as np
from tqdm import tqdm

class Exporter:
    def __init__(self):
        self.card_dir = "/home/jeff/SSD_2/Downloads/all_cards/Baseball_data"
        self.nitin_crops_folder = "nitin_crops"
        self.rec_lines = []
        self.det_lines = []
        self.rounded_rot = 0
        cv2.namedWindow('crop')
        cv2.createTrackbar('angle', 'crop', -0, 90, self.update_rotation)
        cv2.setTrackbarMin('angle', 'crop', -90)


    def rotate_image(self, crop, angle):
        h, w = crop.shape[:2]
        center = (w // 2, h // 2)

        # Compute the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute the new size of the image
        abs_cos = abs(rotation_matrix[0,0])
        abs_sin = abs(rotation_matrix[0,1])

        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to take into account the new image size
        rotation_matrix[0, 2] += new_w / 2 - center[0]
        rotation_matrix[1, 2] += new_h / 2 - center[1]

        # Perform the rotation and return the result
        rotated_crop = cv2.warpAffine(crop, rotation_matrix, (new_w, new_h), borderValue=(255,255,255))
        return rotated_crop

    def update_rotation(self, value):
        increment = 45
        self.rounded_rot = round(value / increment) * increment
        cv2.setTrackbarPos('angle', 'crop', self.rounded_rot)
        if self.current_crop is None: return
        self.rotated_crop = self.rotate_image(self.current_crop, self.rounded_rot)
        
        scaled = cv2.resize(self.rotated_crop, (0, 0), fx=2, fy=2)
        cv2.imshow('crop', scaled)

    def process_crops(self):
        with open('dets.txt', 'r') as input_file:
            for i, line in tqdm(enumerate(input_file), total=sum(1 for _ in open('dets.txt', 'r'))):
                file_name, json_part = line.strip().split('\t')
                detections = json.loads(json_part)

                img = cv2.imread(os.path.join(self.card_dir, file_name))
                s = max(img.shape[1]*2/1600, img.shape[0]/900)
                img = cv2.resize(img, (int(img.shape[1]//s), int(img.shape[0]//s)))
                cv2.imwrite(f"nitin_fullset/{file_name}", img)

                for i, detection in enumerate(detections):
                    points = np.array(detection['points'])
                    rect = cv2.minAreaRect(np.array(points))
                    box = np.intp(cv2.boxPoints(rect))

                    angle = rect[2]
                    if angle > 45: angle -= 90

                    M = cv2.getRotationMatrix2D(rect[0], angle, 1)
                    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

                    rotated_box = cv2.transform(box[None, :, :], M)
                    x_min, y_min = np.min(rotated_box, axis=1)[0]
                    x_max, y_max = np.max(rotated_box, axis=1)[0]
                    x_min, x_max = max(0, min(x_min, x_max)), min(img.shape[1], max(x_min, x_max))
                    y_min, y_max = max(0, min(y_min, y_max)), min(img.shape[0], max(y_min, y_max))

                    # check if the crop is empty beforehand using a numpy function
                    if not np.any(rotated[int(y_min):int(y_max), int(x_min):int(x_max)]): continue

                    crop = rotated[int(y_min):int(y_max), int(x_min):int(x_max)]
                    crop_file_name = f'{os.path.basename(file_name)[:-4]}_crop{i}.jpg'
                    self.current_crop = crop

                    if crop.shape[0] > crop.shape[1] * 1.25 and 'rotation' not in detection:
                        self.update_rotation(0)
                        cv2.setTrackbarPos('angle', 'crop', 0)
                        while True:
                            key = cv2.waitKey(1)
                            if key == ord(' '):
                                detection['rotation'] = self.rounded_rot
                                break
                        
                        cv2.imwrite(os.path.join('nitin_crops', crop_file_name), self.rotated_crop)
                    else:
                        cv2.imwrite(os.path.join('nitin_crops', crop_file_name), self.current_crop)
                    self.rec_lines.append(f'nitin_crops/{crop_file_name}\t{detection["transcription"]}\n')
                
                self.det_lines.append(f"{file_name}\t{json.dumps(detections)}\n")

    def export_to_files(self):
        rec_cutoff = int(len(self.rec_lines) * 0.75)
        with open('./export/rec_train_set.txt', 'w') as f:
            f.writelines(self.rec_lines[:rec_cutoff])
        with open('./export/rec_val_set.txt', 'w') as f:
            f.writelines(self.rec_lines[rec_cutoff:])

        det_cutoff = int(len(self.det_lines) * 0.75)
        with open('./export/det_train_set.txt', 'w') as f:
            f.writelines(self.det_lines[:det_cutoff])
        with open('./export/det_val_set.txt', 'w') as f:
            f.writelines(self.det_lines[det_cutoff:])

        with open("dets.txt", 'w') as f:
            f.writelines(self.det_lines)


exporter = Exporter()
exporter.process_crops()
exporter.export_to_files()
