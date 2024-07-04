import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
from draw_signal import draw_signal
from scipy.signal import find_peaks, argrelextrema
import os

class Signal():
    def __init__(self, t, s):
        self.t = t
        self.s = s

def distances(link_json):
  file_name = link_json.split(sep='\\')[-1]
  task, side = file_name.split(sep='_')[0][-1], file_name.split(sep='_')[1]

  if (side=='L'):
    hand = 'left hand'
    noise_hand = 'right hand'
  else:
    hand = 'right hand'
    noise_hand = 'left hand'
  
  with open(link_json, 'r', encoding = 'utf8') as f:
    text = json.load(f)

  frame = 0
  d = []
  if (task=='1'):
    for i in range(len(text)):
      if noise_hand in text[i].keys():
        pass
      else:
        sum_sqr = (float(text[i][hand]['FORE_TIP']['X1']) -
                           float(text[i][hand]['THUMB_TIP']['X1'])) ** 2 + (
                                  float(text[i][hand]['FORE_TIP']['Y1']) -
                                  float(text[i][hand]['THUMB_TIP']['Y1'])) ** 2 + (
                                      float(text[i][hand]['FORE_TIP']['Z1']) -
                                      float(text[i][hand]['THUMB_TIP']['Z1'])) ** 2
        d.append(math.sqrt(sum_sqr))
        frame += 1
      
      

  elif (task=='2'):
    for i in range(len(text)):
       sum_sqr = (float(text[i][hand]['MIDDLE_TIP']['X1']) -
                           float(text[i][hand]['CENTRE']['X1'])) ** 2 + (
                                  float(text[i][hand]['MIDDLE_TIP']['Y1']) -
                                  float(text[i][hand]['CENTRE']['Y1'])) ** 2 + (
                                      float(text[i][hand]['MIDDLE_TIP']['Z1']) -
                                      float(text[i][hand]['CENTRE']['Z1'])) ** 2
       d.append(math.sqrt(sum_sqr))

  elif (task=='3'):
    for i in range(len(text)):
       sum_sqr = float(text[i][hand]['CENTRE']['Angle'])
       d.append(sum_sqr)

  else:
    sum_sqr = 0
    d.append(math.sqrt(sum_sqr))
  return np.arange(frame), np.array(d)


def equate_arrays_shape(arr1, arr2):
    arr1 = arr1[:min(arr1.shape[0], arr2.shape[0])]
    arr2 = arr2[:min(arr1.shape[0], arr2.shape[0])]
    return arr1, arr2

def calc_signal_properties(signal: Signal):
    # Calculate synchronization parameters
    max_peaks = argrelextrema(signal.s, np.greater, order=4)[0]
    min_peaks = argrelextrema(signal.s, np.less, order=4)[0]

    # Equate peaks arrays shape
    max_peaks, min_peaks = equate_arrays_shape(max_peaks, min_peaks)

    signal.amplitudes = signal.s[max_peaks] - signal.s[min_peaks]
    signal.periods = signal.t[max_peaks][1:] - signal.t[max_peaks][:-1]


def sync_two_signals(path_to_signal_from_LM: str, path_to_signal_from_MP: str):
    # Get signal from Leap Motion points
    frames_lm, dist_lm = distances(path_to_signal_from_LM)
    lm_fps = 115
    t_lm = frames_lm * (1/lm_fps)

    # Get signal from Mediapipe points
    s = np.load(path_to_signal_from_MP, allow_pickle=True)
    dist_mp, frames_mp = draw_signal(s, 1, grap=False)
    dist_mp, frames_mp = np.array(dist_mp), np.array(frames_mp)
    mp_fps = 30
    t_mp = frames_mp * (1/mp_fps)

    # Normalize signals
    dist_lm -= min(dist_lm)
    dist_lm /= max(dist_lm)
    dist_mp -= min(dist_mp)
    dist_mp /= max(dist_mp)

    # Sync timelines by first and last lowest peaks
    low_peaks_lm = argrelextrema(dist_lm, np.less, order=14)[0]
    low_peaks_mp = argrelextrema(dist_mp, np.less, order=5)[0]
    print(path_to_signal_from_LM)
    first_low_peak_lm, last_low_peak_lm = low_peaks_lm[dist_lm[low_peaks_lm] < 0.5][0], low_peaks_lm[dist_lm[low_peaks_lm] < 0.5][-1]
    first_low_peak_mp, last_low_peak_mp = low_peaks_mp[dist_mp[low_peaks_mp] < 0.5][0], low_peaks_mp[dist_mp[low_peaks_mp] < 0.5][-1]

    # Time lag between signals
    time_lag = t_mp[first_low_peak_mp] - t_lm[first_low_peak_lm]

    # Select periods of doing an exercise
    t_lm_sync = t_lm[first_low_peak_lm:last_low_peak_lm]
    dist_lm_sync = dist_lm[first_low_peak_lm:last_low_peak_lm]
    t_mp_sync = t_mp[first_low_peak_mp:last_low_peak_mp] - time_lag
    dist_mp_sync = dist_mp[first_low_peak_mp:last_low_peak_mp]

    # Creating Signal classes to collect their properties in the future
    lm = Signal(t_lm_sync, dist_lm_sync)
    mp = Signal(t_mp_sync, dist_mp_sync)

    # Calculate properties of each signal (amplitude, period)
    calc_signal_properties(lm)
    calc_signal_properties(mp)

    # Calculate synchronization properties
    lm.amplitudes, mp.amplitudes = equate_arrays_shape(lm.amplitudes, mp.amplitudes)
    delta_ampl = mp.amplitudes - lm.amplitudes
    delta_ampl_avg = np.mean(delta_ampl)

    lm.periods, mp.periods = equate_arrays_shape(lm.periods, mp.periods)
    delta_period = mp.periods - lm.periods
    delta_period_avg = np.mean(delta_period)

    return delta_ampl_avg, delta_period_avg
    


