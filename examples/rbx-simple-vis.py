import time
import hybo
import numpy as np
import matplotlib.pyplot as plt

SERIAL_DEV = '/dev/ttyUSB0'

def visualize_points(points):
    plt.clf()
    plt.scatter(points[:, 0], points[:, 1], c='b', marker='o', s=10)
    plt.xlim(0, 4000)
    plt.ylim(0, 4000)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('LIDAR Points Visualization')
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

            visualize_points(np.array(new_scan))

except KeyboardInterrupt:
    print("Visualization stopped.")
finally:
    hybo.close()

