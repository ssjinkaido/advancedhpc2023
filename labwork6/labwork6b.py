from numba import cuda
import numpy as np
import cv2
import time
import math

image = cv2.imread("../images/image.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape
input_device = cuda.to_device(image)
output_device = cuda.device_array(
    (height, width, 3),
    np.uint8,
)


@cuda.jit
def brightness(src, dst, threshold, brightness_type):
    x, y = cuda.grid(2)
    if y < dst.shape[0] and x < dst.shape[1]:
        if brightness_type == 0:
            r = min(src[y, x, 0] + threshold, 255)
            g = min(src[y, x, 1] + threshold, 255)
            b = min(src[y, x, 2] + threshold, 255)
        elif brightness_type == 1:
            r = max(src[y, x, 0] - threshold, 0)
            g = max(src[y, x, 1] - threshold, 0)
            b = max(src[y, x, 2] - threshold, 0)
    dst[y, x, 0] = r
    dst[y, x, 1] = g
    dst[y, x, 2] = b


block_size = [(8, 8), (16, 16), (24, 24), (32, 32)]
time_processed_per_block = []
threshold = 20


def show_brightness_image(func, brightness_type):
    for bs in block_size:
        grid_size_x = math.ceil(width / bs[0])
        grid_size_y = math.ceil(height / bs[1])
        grid_size = (grid_size_x, grid_size_y)
        start = time.time()
        func[grid_size, bs](input_device, output_device, threshold, brightness_type)

        time_processed = time.time() - start
        time_processed_per_block.append(time_processed)

        output_host = output_device.copy_to_host()
        brightness_str = "increase" if brightness_type == 0 else "decrease"
        cv2.imwrite(f"{brightness_str}_image.png", output_host)


show_brightness_image(brightness, 0)
show_brightness_image(brightness, 1)
