# given a set of detections in dets_fixed.txt: takes image from nitin_fullset, generates crops to export them to nitin_crops
# as well as four files for the rec and detection models, two for training and two for validation


import json
import cv2
import numpy as np
import os
from tqdm import tqdm

# Create the output directory if it doesn't exist
os.makedirs('nitin_crops', exist_ok=True)
# Open the output text file
rec_lines = []
det_lines = []
with open('dets_fixed.txt', 'r') as input_file:
    # Read each line
    # check the number of lines 
    for i, line in tqdm(enumerate(input_file), total=sum(1 for _ in open('dets_fixed.txt', 'r'))):
        det_lines.append(line)
        # Split the line into the file name and the JSON part
        file_name, json_part = line.strip().split('\t')
        # Parse the JSON part
        detections = json.loads(json_part)
        # Read the image
        image = cv2.imread(os.path.join('nitin_fullset', file_name))
        # For each detection
        for i, detection in enumerate(detections):
            # Get the points
            points = np.array(detection['points'])

            # Fit a minimum area rectangle around the points
            rect = cv2.minAreaRect(np.array(points))
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Calculate the angle of the rectangle
            angle = rect[2]

            # Correct the angle
            if angle > 45:
                angle -= 90

            # Rotate the image around the center of the rectangle
            M = cv2.getRotationMatrix2D(rect[0], angle, 1)
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

            # Calculate the axis-aligned bounding box of the rotated rectangle
            rotated_box = cv2.transform(box[None, :, :], M)
            x_min, y_min = np.min(rotated_box, axis=1)[0]
            x_max, y_max = np.max(rotated_box, axis=1)[0]

            # Ensure the coordinates are in the correct order and within the image dimensions
            x_min, x_max = max(0, min(x_min, x_max)), min(image.shape[1], max(x_min, x_max))
            y_min, y_max = max(0, min(y_min, y_max)), min(image.shape[0], max(y_min, y_max))

            # Check if the crop is not empty
            if x_min < x_max and y_min < y_max:
                # Crop the rotated image
                crop = rotated[int(y_min):int(y_max), int(x_min):int(x_max)]
            else:
                print("The crop is empty.")

            # Save the cropped image
            crop_file_name = f'{os.path.basename(file_name)[:-4]}_crop{i}.jpg'
            cv2.imwrite(os.path.join('nitin_crops', crop_file_name), crop)
            rec_lines.append(f'nitin_crops/{crop_file_name}\t{detection["transcription"]}\n')

rec_cutoff = int(len(rec_lines) * 0.75)
with open('./export/rec_train_set.txt', 'w') as f:
    f.writelines(rec_lines[:rec_cutoff])
with open('./export/rec_val_set.txt', 'w') as f:
    f.writelines(rec_lines[rec_cutoff:])

det_cutoff = int(len(det_lines) * 0.75)
with open('./export/det_train_set.txt', 'w') as f:
    f.writelines(det_lines[:det_cutoff])
with open('./export/det_val_set.txt', 'w') as f:
    f.writelines(det_lines[det_cutoff:])