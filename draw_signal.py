import matplotlib.pyplot as plt
import numpy as np

def compute_distance(points, point1, point2, y_norm=1):

    x_comp = (points[point1]['x'] - points[point2]['x']) ** 2
    y_comp = y_norm * (points[point1]['y'] - points[point2]['y']) ** 2
    z_comp = (points[point1]['z'] - points[point2]['z']) ** 2 

    return np.sqrt(x_comp + y_comp + z_comp)

def compute_angle(points, point1, point2, vertex_point=1, y_norm=1):

    x1 = points[point1]['x'] - points[vertex_point]['x']
    y1 = points[point1]['y'] - points[vertex_point]['y']
    z1 = points[point1]['z'] - points[vertex_point]['z']

    x2 = points[point2]['x'] - points[vertex_point]['x']
    y2 = points[point2]['y'] - points[vertex_point]['y']
    z2 = points[point2]['z'] - points[vertex_point]['z']

    len1 = np.sqrt(x1**2 + y_norm*y1**2 + z1**2)
    len2 = np.sqrt(x2**2 + y_norm*y2**2 + z2**2)

    cos_val = (x1*x2 + y1*y2 + z1*z2) / (len1 * len2)

    return np.arccos(cos_val)

def draw_signal (result_points, task=1, output="dist", grap=False, height=1, width=1):
    frames = []
    vals = []
    y_norm = (height/width)**2

    norm_points = [1, 0]
    vertex_point = 1

    if task == 1:
        task_points = [4, 8]

    if task == 2:
        task_points = [0, 12]

    for i in range(result_points.shape[1]):
            if result_points[task_points[0],i] == 0:
                pass
            else:
                if output == "dist":
                    distance = compute_distance(result_points[:,i], task_points[0], task_points[1], y_norm)
                elif output == "norm_dist":
                    distance = compute_distance(result_points[:,i], task_points[0], task_points[1], y_norm)
                    norm = compute_distance(result_points[:,i], norm_points[0], norm_points[1], y_norm)
                    distance = distance / norm
                elif output == "angle":
                    distance = compute_angle(result_points[:,i], task_points[0], task_points[1], vertex_point, y_norm)

                vals.append(distance)
                frames.append(i)

    if grap == True:
        plt.plot(frames, vals)
        plt.show()

    return vals, frames

if __name__=="__main__":
    dot_path = "new_dots/patient0_m1_1_L.npy"
    dots = np.load(dot_path, allow_pickle=True,)

    draw_signal(dots, task=1, output="angle", grap=True)
