from numba import cuda
import numpy as np
import cv2
import time
import math

image = cv2.imread("../images/image.jpeg")
image1 = cv2.imread("../images/castle.jpeg")
height, width, _ = image.shape
input_device = cuda.to_device(image)
input_device1 = cuda.to_device(image1)
output_device = cuda.device_array(
    (height, width, 3),
    np.uint8,
)


@cuda.jit
def blend(src, src1, dst, c):
    x, y = cuda.grid(2)
    if y < dst.shape[0] and x < dst.shape[1]:
        dst[y, x, 0] = src[y, x, 0] * c + (1 - c) * src1[y, x, 0]
        dst[y, x, 1] = src[y, x, 1] * c + (1 - c) * src1[y, x, 1]
        dst[y, x, 2] = src[y, x, 2] * c + (1 - c) * src1[y, x, 2]


block_size = [(8, 8), (16, 16), (24, 24), (32, 32)]
time_processed_per_block = []
c = 0.5


def blend_image(func):
    for bs in block_size:
        grid_size_x = math.ceil(width / bs[0])
        grid_size_y = math.ceil(height / bs[1])
        grid_size = (grid_size_x, grid_size_y)
        start = time.time()
        func[grid_size, bs](input_device, input_device1, output_device, c)

        time_processed = time.time() - start
        time_processed_per_block.append(time_processed)

        output_host = output_device.copy_to_host()
        cv2.imwrite(f"blend_image.png", output_host)


blend_image(blend)
