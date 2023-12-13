# evaluates the performance of the model by seeing the total number of detections versus the incorrect detections
import os
import json

def quantify_model_performance(results):
    # Read the last index processed
    with open('last.txt', 'r') as f:
        last = int(f.read())

    # Count total detections up to the last index
    total_detections = sum(len(detections) for i, (file_path, detections) in enumerate(results.items()) if i < last)

    # Count lines in baseball_test_preds.txt
    with open('pred_labels/baseball_test_preds.txt', 'r') as f:
        test_preds_lines = len(f.readlines())

    # Count lines in bad.txt
    with open('bad.txt', 'r') as f:
        bad_lines = len(f.readlines())

    print(f"Incorrect detections: {test_preds_lines}")
    print(f"Total detections: {total_detections}")
    print(f"Percentage of correct detections: {round((1 - test_preds_lines / total_detections) * 100, 2)}")
    print(f"Cards with bad bounding boxes: {bad_lines}")

card_dir = "/home/jeff/SSD_2/vis/"
with open(os.path.join(card_dir, 'results.json'), 'r') as f:
    results = json.load(f)
quantify_model_performance(results)