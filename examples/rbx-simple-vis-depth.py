import time
import hybo
import numpy as np
import matplotlib.pyplot as plt
import cv2

SERIAL_DEV = '/dev/ttyUSB0'

def visualize_depth(points):
    points = points.astype(np.int32)

    min_depth = np.min(points[:, 2])
    max_depth = np.max(points[:, 2])
    normalized_depth = ((points[:, 2] - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    image_size = (512, 512)
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    x_img = ((points[:, 0] - np.min(points[:, 0])) / (np.max(points[:, 0]) - np.min(points[:, 0])) * (image_size[1] - 1)).astype(int)
    y_img = ((points[:, 1] - np.min(points[:, 1])) / (np.max(points[:, 1]) - np.min(points[:, 1])) * (image_size[0] - 1)).astype(int)

    colormap = plt.get_cmap('jet')
    color_depth = (colormap(normalized_depth / 255)[:, :3] * 255).astype(np.uint8)

    for i in range(len(points)):
        if 0 <= x_img[i] < image_size[1] and 0 <= y_img[i] < image_size[0]:
            image[y_img[i], x_img[i]] = color_depth[i]

    cv2.imshow('Color Depth Image', image)
    cv2.waitKey(1)


hybo = hybo.Lidar(SERIAL_DEV)
hybo.start()

time.sleep(1)

try:
    while True:
        raw_scan = hybo.get_latest_frame()
        time.sleep(0.01)

        if raw_scan is not None:
            sequence  = raw_scan["sequence"]
            time_peak = raw_scan["time_peak"]
            new_scan  = raw_scan["points"]

            visualize_depth(np.array(new_scan))

except KeyboardInterrupt:
    print("Visualization stopped.")
finally:
    hybo.close()
    cv2.destroyAllWindows()

