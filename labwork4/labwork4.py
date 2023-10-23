from numba import cuda
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import math

image = cv2.imread("../images/image.jpeg")
height, width, _ = image.shape
pixel_count = height * width
output_device = cuda.device_array(
    (width, height),
    np.uint8,
)
input_device = cuda.to_device(image)


def grayscale_cpu(image):
    height = image.shape[0]
    width = image.shape[1]
    image_grayscale = np.zeros((height, width), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            image_grayscale[h, w] = (
                int(image[h][w][0]) + int(image[h][w][1]) + int(image[h][w][2])
            ) // 3
    return image_grayscale


@cuda.jit
def grayscale_gpu(src, dst):
    x, y = cuda.grid(2)
    # x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < dst.shape[0] and y < dst.shape[1]:
        g = np.uint8((src[y, x, 0] + src[y, x, 1] + src[y, x, 2]) // 3)
        dst[x, y] = g


start = time.time()
grayscale_cpu_image = grayscale_cpu(image)
print(f"Time processed on CPU: {time.time() - start}s")
cv2.imwrite(f"grayscale_image_cpu.png", grayscale_cpu_image)

block_size = [(8, 8), (16, 16), (24, 24), (32, 32)]
time_processed_per_block = []
for bs in block_size:
    grid_size_x = math.ceil(width / bs[0])
    grid_size_y = math.ceil(height / bs[1])
    grid_size = (grid_size_x, grid_size_y)
    start = time.time()
    grayscale_gpu[grid_size, bs](input_device, output_device)

    time_processed = time.time() - start
    time_processed_per_block.append(time_processed)

    output_host = output_device.copy_to_host()
    grayscale_image = np.reshape(output_host, (width, height))
    grayscale_image = np.transpose(grayscale_image)

    cv2.imwrite(f"grayscale_image_gpu_{bs}.png", grayscale_image)

for bs, t in zip(block_size, time_processed_per_block):
    print(f"Block size: {bs}, time processed: {t}")

plt.figure(figsize=(10, 5))
plt.plot(block_size, time_processed_per_block, marker="o", linestyle="-", color="b")
plt.title("Time Processed Per Block vs 2D Block Size")
plt.xlabel("Block Size")
plt.ylabel("Time Processed Per Block")
x_ticks = np.arange(4, 34, 4)
plt.xticks(x_ticks)
y_ticks = np.arange(0, 0.22, 0.02)
plt.yticks(y_ticks)
plt.grid(True)
plt.savefig("comparison.png")
plt.show()
