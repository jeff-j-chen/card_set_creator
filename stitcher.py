import os
import cv2
import numpy as np
from tqdm import tqdm

A_path = '/home/jeff/SSD_2/feb11_sample'
B_path = '/home/jeff/SSD_2/feb7_sample'
output = '/home/jeff/SSD_2/feb7_vs_feb11'

if not os.path.exists(output):
    os.mkdir(output)

# 1. Collect the last 150 images from directory A
A_images = sorted(os.listdir(A_path))[-150:]

# 3. Use OpenCV to stitch the images together, side by side
for A_image in tqdm(A_images):
    # check if it is actually an image
    if not A_image.endswith('.jpg'):
        continue
    # get the basename of the image
    A_image_name = os.path.basename(A_image)
    img1 = cv2.imread(os.path.join(A_path, A_image))
    # check if A_image_name is in B_images
    if A_image_name not in os.listdir(B_path):
        continue 
    img2 = cv2.imread(os.path.join(B_path, A_image_name))
    stitched_img = np.hstack((img1, img2))
    cv2.imwrite(os.path.join(output, A_image_name), stitched_img)