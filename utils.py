import os

import numpy as np
import cv2
from PIL import Image
from scipy.signal import find_peaks


def peaks_position_me(peaks1, peaks2):

    if np.abs(peaks2[1] - peaks1[0]) < np.abs(peaks2[0] - peaks1[0]):
        peaks2 = peaks2[1:]

    n = min(len(peaks1), len(peaks2))
    peaks_dif = peaks1[:n] - peaks2[:n]

    return np.mean(peaks_dif)


def peaks_position_mae(peaks1, peaks2):

    if np.abs(peaks2[1] - peaks1[0]) < np.abs(peaks2[0] - peaks1[0]):
        peaks2 = peaks2[1:]

    n = min(len(peaks1), len(peaks2))
    peaks_dif = peaks1[:n] - peaks2[:n]

    return np.mean(np.abs(peaks_dif))


def peaks_position_mse(peaks1, peaks2):

    if np.abs(peaks2[1] - peaks1[0]) < np.abs(peaks2[0] - peaks1[0]):
        peaks2 = peaks2[1:]

    n = min(len(peaks1), len(peaks2))
    peaks_dif = peaks1[:n] - peaks2[:n]

    return np.std(peaks_dif, ddof=1)


def norm_signal(signal):

    new_signal = signal.copy()
    new_signal -= np.mean(new_signal)
    new_signal /= np.max(new_signal)

    return new_signal


def pars_video_path(video_path):

    video_path = video_path.split("/")
    patient = video_path[1]
    if patient == "patient0":
        patient = "Patient0"
    m = video_path[2]
    task = video_path[3].split(".")[0]
    signal_name = f"signal_{patient}_{m}_{task}.npy"
    frames_name = f"frames_{patient}_{m}_{task}.npy"

    signal_path = os.path.join("signals", signal_name)
    frames_path = os.path.join("signals", frames_name)

    return signal_path, frames_path


def calc_metrics(
    norm_gt_impr_signal,
    norm_net1_signal,
    norm_net2_signal,
    start_point=0.3,
    end_point=0.7,
    step=0.05,
):

    steps_n = int((end_point - start_point + 0.00001) / step)

    start_point = int(len(norm_gt_impr_signal) * start_point)
    end_point = int(len(norm_gt_impr_signal) * end_point)
    step = int(len(norm_gt_impr_signal) * step)

    me1, mae1, mse1 = [], [], []
    me2, mae2, mse2 = [], [], []

    for i in range(steps_n):

        gt_signal_part = norm_gt_impr_signal[start_point + i * step : start_point + (i + 1) * step]
        net1_signal_part = norm_net1_signal[start_point + i * step : start_point + (i + 1) * step]
        net2_signal_part = norm_net2_signal[start_point + i * step : start_point + (i + 1) * step]

        gt_peaks = find_peaks(gt_signal_part, distance=5)[0]
        net1_peaks = find_peaks(net1_signal_part, distance=5)[0]
        net2_peaks = find_peaks(net2_signal_part, distance=5)[0]

        me1.append(peaks_position_me(gt_peaks, net1_peaks))
        mae1.append(peaks_position_mae(gt_peaks, net1_peaks))
        mse1.append(peaks_position_mse(gt_peaks, net1_peaks))

        me2.append(peaks_position_me(gt_peaks, net2_peaks))
        mae2.append(peaks_position_mae(gt_peaks, net2_peaks))
        mse2.append(peaks_position_mse(gt_peaks, net2_peaks))

    return np.mean(me1), np.mean(mae1), np.mean(mse1), np.mean(me2), np.mean(mae2), np.mean(mse2)


def process_video(video_path, net1, net2, val_transform, device="cuda"):

    signal_path, frames_path = pars_video_path(video_path)

    cap = cv2.VideoCapture(video_path)
    net1_signal = []
    net2_signal = []
    n = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        n += 1

        image = Image.fromarray(frame)
        image = val_transform(image).unsqueeze(0).to(device)

        res1 = net1(image)
        net1_signal.append(res1.to("cpu").detach().numpy()[0][0])

        res2 = net2(image)
        net2_signal.append(res2.to("cpu").detach().numpy()[0][0])

    gt_signal = np.load(signal_path, allow_pickle=True)
    frames = np.load(frames_path, allow_pickle=True)

    gt_impr_signal = []
    k = 0

    for i in range(n):
        if i in frames:
            gt_impr_signal.append(gt_signal[k])
            k += 1
        else:
            gt_impr_signal.append(0)

    norm_gt_impr_signal = norm_signal(gt_impr_signal)
    norm_net1_signal = norm_signal(net1_signal)
    norm_net2_signal = norm_signal(net2_signal)

    start_point = 0.3
    end_point = 0.7
    step = 0.05

    me1, mae1, mse1, me2, mae2, mse2 = calc_metrics(
        norm_gt_impr_signal, norm_net1_signal, norm_net2_signal
    )

    cap.release()

    return me1, mae1, mse1, me2, mae2, mse2
