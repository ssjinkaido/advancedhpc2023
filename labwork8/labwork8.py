from numba import cuda
import numpy as np
import cv2
import time
import math

image = cv2.imread("../images/image.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape
image = np.float32(image)
input_device = cuda.to_device(image)
output_device = cuda.device_array((height, width, 3), np.float32)


@cuda.jit
def rgb_to_hsv(src, dst):
    x, y = cuda.grid(2)
    if y < dst.shape[0] and x < dst.shape[1]:
        max_value = max(src[y, x, 0], src[y, x, 1], src[y, x, 2])
        min_value = min(src[y, x, 0], src[y, x, 1], src[y, x, 2])
        delta = max_value - min_value
        if delta == 0:
            h_value = 0
        elif max_value == src[y, x, 0]:
            h_value = 60 * (((src[y, x, 1] - src[y, x, 2]) / delta) % 6)
        elif max_value == src[y, x, 1]:
            h_value = 60 * (((src[y, x, 2] - src[y, x, 0]) / delta) + 2)
        elif max_value == src[y, x, 2]:
            h_value = 60 * (((src[y, x, 0] - src[y, x, 1]) / delta) + 4)

        if max_value == 0:
            s_value = 0
        else:
            s_value = delta / max_value
        v_value = max_value
        dst[y, x, 0] = h_value
        dst[y, x, 1] = s_value
        dst[y, x, 2] = v_value


@cuda.jit
def hsv_to_rgb(src, dst):
    x, y = cuda.grid(2)

    if y < src.shape[0] and x < src.shape[1]:
        d = src[y, x, 0] / 60
        hi = int(d) % 6
        f = d - hi
        l = src[y, x, 2] * (1 - src[y, x, 1])
        m = src[y, x, 2] * (1 - f * src[y, x, 1])
        n = src[y, x, 2] * (1 - (1 - f) * src[y, x, 1])

        if 0 <= src[y, x, 0] < 60:
            r, g, b = src[y, x, 2], n, l
        elif 60 <= src[y, x, 0] < 120:
            r, g, b = m, src[y, x, 2], l
        elif 120 <= src[y, x, 0] < 180:
            r, g, b = l, src[y, x, 2], n
        elif 180 <= src[y, x, 0] < 240:
            r, g, b = l, m, src[y, x, 2]
        elif 240 <= src[y, x, 0] < 300:
            r, g, b = n, l, src[y, x, 2]
        elif 300 <= src[y, x, 0] < 360:
            r, g, b = src[y, x, 2], l, m

        dst[y, x, 0] = r
        dst[y, x, 1] = g
        dst[y, x, 2] = b


block_size = [(8, 8), (16, 16), (24, 24), (32, 32)]
time_processed_per_block = []

for bs in block_size:
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    cv2.imwrite(f"hsvcv2_image.png", image_hsv)
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    cv2.imwrite(f"rgbcv2_image.png", image_rgb)
    grid_size_x = math.ceil(width / bs[0])
    grid_size_y = math.ceil(height / bs[1])
    grid_size = (grid_size_x, grid_size_y)
    start = time.time()
    rgb_to_hsv[grid_size, bs](input_device, output_device)
    output_host = output_device.copy_to_host()
    cv2.imwrite(f"hsv_image.png", output_host)
    hsv_to_rgb[grid_size, bs](output_device, input_device)
    output_host = np.uint8(input_device.copy_to_host())
    cv2.imwrite(f"rgb_image.png", output_host)
    time_processed = time.time() - start
    time_processed_per_block.append(time_processed)
