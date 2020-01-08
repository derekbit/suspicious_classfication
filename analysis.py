import cv2
import os
import csv

labels = ['normal', 'abnormal']
with open('dataset/data_file.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)

    top_dir = 'dataset'

    for label in labels:
        dir_path = os.path.join(top_dir, label)

        for filename in os.listdir(dir_path):
            path = os.path.join(dir_path, filename)

            cap = cv2.VideoCapture(path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            writer.writerow([label, filename, frame_count, fps])
