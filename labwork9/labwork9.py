from numba import cuda, float32
import numba
import numpy as np
import cv2
import time
import math

image = cv2.imread("../images/image.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = image.shape
input_device = cuda.to_device(image)
output_device = cuda.device_array(
    (height, width),
    np.uint8,
)


@cuda.jit
def histogram(src, hist):
    x, y = cuda.grid(2)
    if y < src.shape[0] and x < src.shape[1]:
        cuda.atomic.add(hist, src[y, x], 1)


@cuda.jit
def equalization(src, cdf, dst):
    x, y = cuda.grid(2)
    if y < src.shape[0] and x < src.shape[1]:
        dst[y, x] = cdf[src[y, x]]


block_size = [(16, 16)]
time_processed_per_block = []

for bs in block_size:
    grid_size_x = math.ceil(width / bs[0])
    grid_size_y = math.ceil(height / bs[1])
    grid_size = (grid_size_x, grid_size_y)
    start = time.time()
    hist = np.zeros((256,), np.uint32)
    hist_device = cuda.to_device(hist)
    histogram[grid_size, bs](input_device, hist_device)
    hist = hist_device.copy_to_host()
    cdf = (np.cumsum(hist) / (width * height)) * 255
    cdf_device = cuda.to_device(cdf)
    equalization[grid_size, bs](input_device, cdf_device, output_device)
    output_host = output_device.copy_to_host()
    cv2.imwrite("he.png", output_host)
