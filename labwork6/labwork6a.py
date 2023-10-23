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
def binarize(src, dst, threshold):
    x, y = cuda.grid(2)
    if y < dst.shape[0] and x < dst.shape[1]:
        g = np.uint8((src[y, x, 0] + src[y, x, 1] + src[y, x, 2]) // 3)
        g = 255 if g > threshold else 0
    dst[y, x, 0] = g
    dst[y, x, 1] = g
    dst[y, x, 2] = g


block_size = [(8, 8), (16, 16), (24, 24), (32, 32)]
time_processed_per_block = []
threshold = 50
for bs in block_size:
    grid_size_x = math.ceil(width / bs[0])
    grid_size_y = math.ceil(height / bs[1])
    grid_size = (grid_size_x, grid_size_y)
    start = time.time()
    binarize[grid_size, bs](input_device, output_device, threshold)

    time_processed = time.time() - start
    time_processed_per_block.append(time_processed)

    output_host = output_device.copy_to_host()
    # output_host = output_host * 255
    cv2.imwrite(f"binarize_image.png", output_host)

for bs, t in zip(block_size, time_processed_per_block):
    print(f"Block size: {bs}, time processed: {t}")
