from numba import cuda
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

image = cv2.imread("../images/image.jpeg")
image_1d = np.reshape(image, (-1, 3))
height, width, _ = image.shape
pixel_count = height * width
output_device = cuda.device_array((pixel_count,), np.uint8)
input_device = cuda.to_device(image_1d)


def grayscale_cpu(image):
    height = image.shape[0]
    width = image.shape[1]
    image_greyscale = np.zeros((height, width), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            image_greyscale[h, w] = (
                image[h][w][0] + image[h][w][1] + image[h][w][2]
            ) // 3
    return image_greyscale


@cuda.jit
def grayscale_gpu(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) // 3)
    dst[tidx] = g


start = time.time()
grayscale_image = grayscale_cpu(image)
print(f"Time processed on CPU: {time.time() - start}s")
cv2.imwrite(f"grayscale_image_cpu.png", grayscale_image)

block_size = [64, 128, 256, 512, 1024]
time_processed_per_block = []
for bs in block_size:
    if pixel_count % bs == 0:
        grid_size = pixel_count / bs
    else:
        grid_size = pixel_count // bs + 1

    start = time.time()
    grayscale_gpu[grid_size, bs](input_device, output_device)

    time_processed = time.time() - start
    time_processed_per_block.append(time_processed)

    output_host = output_device.copy_to_host()
    grayscale_image = np.reshape(output_host, (height, width))
    cv2.imwrite(f"grayscale_image_gpu_{bs}.png", grayscale_image)

for bs, t in zip(block_size, time_processed_per_block):
    print(f"Block size: {bs}, time processed: {t}")

plt.figure(figsize=(10, 5))
plt.plot(block_size, time_processed_per_block, marker="o", linestyle="-", color="b")
plt.title("Time Processed Per Block vs Block Size")
plt.xlabel("Block Size")
plt.ylabel("Time Processed Per Block")
x_ticks = np.arange(0, 1088, 64)
plt.xticks(x_ticks)
y_ticks = np.arange(0, 0.16, 0.02)
plt.yticks(y_ticks)
plt.grid(True)
plt.savefig("comparison.png")
plt.show()
