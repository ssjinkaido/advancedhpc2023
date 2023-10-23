from numba import cuda
import numpy as np
import cv2
import time
import math

image = cv2.imread("../images/image.jpeg")
height, width, _ = image.shape
input_device = cuda.to_device(image)
output_device = cuda.device_array(
    (height, width),
    np.uint8,
)


@cuda.jit
def grayscale_gpu(src, dst):
    x, y = cuda.grid(2)
    if y < dst.shape[0] and x < dst.shape[1]:
        g = np.uint8((src[y, x, 0] + src[y, x, 1] + src[y, x, 2]) // 3)
        dst[y, x] = g


@cuda.reduce
def find_max(value, value1):
    return max(value, value1)


@cuda.reduce
def find_min(value, value1):
    return min(value, value1)


@cuda.jit
def recalculate_intensity(src, min_value, max_value):
    x, y = cuda.grid(2)

    if y < src.shape[0] and x < src.shape[1]:
        value_recalculated = (src[y, x] - min_value) / (max_value - min_value) * 255
        src[y, x] = value_recalculated


block_size = [(8, 8), (16, 16), (24, 24), (32, 32)]
time_processed_per_block = []

for bs in block_size:
    grid_size_x = math.ceil(width / bs[0])
    grid_size_y = math.ceil(height / bs[1])
    grid_size = (grid_size_x, grid_size_y)
    start = time.time()
    grayscale_gpu[grid_size, bs](input_device, output_device)
    output_host = output_device.copy_to_host()
    min_value = find_min(output_host.flatten())
    max_value = find_max(output_host.flatten())
    output_grayscale = cuda.to_device(output_host)
    recalculate_intensity[grid_size, bs](output_grayscale, min_value, max_value)
    time_processed = time.time() - start
    time_processed_per_block.append(time_processed)

    output_host = output_device.copy_to_host()
    cv2.imwrite(f"gray_stretch_image.png", output_host)
