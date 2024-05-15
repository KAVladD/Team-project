import matplotlib.pyplot as plt
import numpy as np

def draw_signal (result_points, task, grap):
    frame = []
    znach = []
    if task == 1:
        for i in range(result_points.shape[1]):
            if result_points[4,i] == 0:
                pass
            else:
                sum_sqr = (result_points[4,i]['x'] - result_points[8,i]['x']) ** 2 
                + (result_points[4,i]['y'] - result_points[8,i]['y']) ** 2
                + (result_points[4,i]['z'] - result_points[8,i]['z']) ** 2 
                distance = np.sqrt(sum_sqr)
                znach.append(distance)
                frame.append(i)
        
    if task == 2:
         for i in range(result_points.shape[1]):
            if result_points[4,i] == 0:
                pass
            else:
                sum_sqr = (result_points[0,i]['x'] - result_points[12,i]['x']) ** 2 
                + (result_points[0,i]['y'] - result_points[12,i]['y']) ** 2
                + (result_points[0,i]['z'] - result_points[12,i]['z']) ** 2 
                distance = np.sqrt(sum_sqr)
                znach.append(distance)
                frame.append(i)

    if grap == True:
        plt.plot(frame, znach)
        plt.show()

    return znach, frame