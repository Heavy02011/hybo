import cv2
import time
import hybo
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.tri as tri

SERIAL_DEV = '/dev/ttyUSB0'

# simple xy-plot
def visualize_points(points): 
    plt.clf()
    x_values = points[:, 0]
    y_values = points[:, 1]
    z_values = points[:, 2]

    #plt.scatter(points[:, 0], points[:, 1], c='b', marker='o', s=10)
    plt.scatter(x_values, y_values, c='b', marker='o', s=10)
    plt.xlim(0, 400)
    plt.ylim(0, 400)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('LIDAR Points Visualization')
    plt.grid(True)
    plt.pause(0.001)

# colored xy-plot with z as depth
def visualize_points2(points): 
    plt.clf()
    x_values = points[:, 0]
    y_values = points[:, 1]
    z_values = points[:, 2] / 100. # mm --> cm

    plt.scatter(x_values, y_values, c=z_values, marker='o', s=10, cmap='viridis')
    plt.xlim(0, 400)
    plt.ylim(0, 400)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('LIDAR Points Visualization')
    plt.grid(True)
    plt.colorbar(label='Z Value')
    plt.pause(0.001)

# 3 contour plot
def visualize_points3(points):
    plt.clf()
    
    x_values = points[:, 0]
    y_values = points[:, 1]
    z_values = points[:, 2]

    triang = tri.Triangulation(x_values, y_values)
    plt.tricontourf(triang, z_values, cmap=plt.get_cmap("viridis"))
    
    plt.xlim(0, 400)
    plt.ylim(0, 400)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('LIDAR Points Visualization')
    plt.colorbar(label='Z')
    plt.grid(True)
    plt.pause(0.001)

hybo = hybo.Lidar(SERIAL_DEV)
hybo.start()

# waiting for first frame
time.sleep(1)

plt.figure(figsize=(10, 10))

try:
    while True:
        raw_scan = hybo.get_latest_frame()
        time.sleep(0.01)

        if raw_scan is not None:
            sequence  = raw_scan["sequence"]
            time_peak = raw_scan["time_peak"]
            new_scan  = raw_scan["points"]

            visualize_points3(np.array(new_scan))
            time.sleep(0.05)

except KeyboardInterrupt:
    print("Visualization stopped.")
finally:
    hybo.close()

