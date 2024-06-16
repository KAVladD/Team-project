import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

from draw_signal import draw_signal

dots_path = 'dots'
signals_path = 'signals'
videos_path = 'fingers_vids'
data_path = 'data'
train_ratio = 0.7

task = 1
output = "norm_dist"
height = 1280
width = 720

dots_folder = os.listdir(dots_path)
for i, file_name in enumerate(dots_folder):

    task_type = file_name.split('_')[2]
    if task_type != '1': continue

    if i/len(dots_folder) <= train_ratio:
        folder_name = 'train'
    else:
        folder_name = 'val'
        
    path_parts = file_name.split('_')
    vid_name = path_parts[2] + '_' + path_parts[3].split('.')[0] + '.MOV'

    dots_sample_path = os.path.join(dots_path, file_name)
    video_path = os.path.join(videos_path, path_parts[0], path_parts[1], vid_name)   
    signal_path = os.path.join(signals_path, "signal_" + file_name)
    frames_path = os.path.join(signals_path, "frames_" + file_name)
    images_path = os.path.join(data_path, folder_name)
    
    current_dots = np.load(dots_sample_path, allow_pickle=True)
    signal, frames = draw_signal(current_dots, task=task, output=output, height=height, width=width)
    signal = np.array(signal)
    np.save(signal_path, signal)
    np.save(signal_path, frames)

    print(video_path)

    cap = cv2.VideoCapture(video_path)
    i = 0
    k = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        image_value = 0 
        
        if i in frames: 
            image_value = signal[k] 
            k+=1

        if image_value == 0: continue
        
        image_value = f'{image_value:.4f}'
        image_name = f'{file_name.split(".")[0]}_{i}_{image_value}.jpg'
        image_path = os.path.join(images_path, image_name) 
        cv2.imwrite(image_path, frame)

        i+=1

    cap.release()
    
cv2.destroyAllWindows()