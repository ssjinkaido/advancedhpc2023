from numba import cuda, types
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import math

image = cv2.imread("../images/image.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape
pixel_count = height * width

kernel_size = 7
h = kernel_size // 2
w = kernel_size // 2

image = np.pad(
    image,
    pad_width=(
        (h, h),
        (w, w),
        (0, 0),
    ),
    mode="constant",
    constant_values=0,
)

input_device = cuda.to_device(image)
output_device = cuda.device_array(
    (height + h * 2, width + w * 2, 3),
    np.uint8,
)
gaussian_kernel = np.array(
    [
        [0, 0, 1, 2, 1, 0, 0],
        [0, 3, 13, 22, 13, 3, 0],
        [1, 13, 59, 97, 59, 13, 1],
        [2, 22, 97, 159, 97, 22, 2],
        [1, 13, 59, 97, 59, 13, 1],
        [0, 3, 13, 22, 13, 3, 0],
        [0, 0, 1, 2, 1, 0, 0],
    ],
    dtype=np.float32,
)
gaussian_kernel /= gaussian_kernel.sum()


@cuda.jit
def gaussian_convolution_without_sm(src, dst, gaussian_kernel):
    x, y = cuda.grid(2)

    if x < dst.shape[1] and y < dst.shape[0]:
        convolution_sum_r = 0.0
        convolution_sum_g = 0.0
        convolution_sum_b = 0.0
        for i in range(-3, 4):
            for j in range(-3, 4):
                x_pos = x + i
                y_pos = y + j
                if 0 <= x_pos < src.shape[1] and 0 <= y_pos < src.shape[0]:
                    convolution_sum_r += (
                        src[y_pos, x_pos, 0] * gaussian_kernel[i + 3, j + 3]
                    )
                    convolution_sum_g += (
                        src[y_pos, x_pos, 1] * gaussian_kernel[i + 3, j + 3]
                    )
                    convolution_sum_b += (
                        src[y_pos, x_pos, 2] * gaussian_kernel[i + 3, j + 3]
                    )

        dst[y, x, 0] = convolution_sum_r
        dst[y, x, 1] = convolution_sum_g
        dst[y, x, 2] = convolution_sum_b


@cuda.jit
def gaussian_convolution_with_sm(src, dst, gaussian_kernel):
    x, y = cuda.grid(2)
    shared_kernel = cuda.shared.array(shape=(7, 7), dtype=types.float32)
    if cuda.threadIdx.x < 7 and cuda.threadIdx.y < 7:
        shared_kernel[cuda.threadIdx.x, cuda.threadIdx.y] = gaussian_kernel[
            cuda.threadIdx.x, cuda.threadIdx.y
        ]

    cuda.syncthreads()
    if x < dst.shape[1] and y < dst.shape[0]:
        convolution_sum_r = 0.0
        convolution_sum_g = 0.0
        convolution_sum_b = 0.0
        for i in range(-3, 4):
            for j in range(-3, 4):
                x_pos = x + i
                y_pos = y + j
                if 0 <= x_pos < src.shape[1] and 0 <= y_pos < src.shape[0]:
                    convolution_sum_r += (
                        src[y_pos, x_pos, 0] * shared_kernel[i + 3, j + 3]
                    )
                    convolution_sum_g += (
                        src[y_pos, x_pos, 1] * shared_kernel[i + 3, j + 3]
                    )
                    convolution_sum_b += (
                        src[y_pos, x_pos, 2] * shared_kernel[i + 3, j + 3]
                    )

        dst[y, x, 0] = convolution_sum_r
        dst[y, x, 1] = convolution_sum_g
        dst[y, x, 2] = convolution_sum_b


def gaussian_convolution(func, title="sm"):
    block_size = [(8, 8), (16, 16), (24, 24), (32, 32)]
    time_processed_per_block = []
    for bs in block_size:
        grid_size_x = math.ceil(width / bs[0])
        grid_size_y = math.ceil(height / bs[1])
        grid_size = (grid_size_x, grid_size_y)
        start = time.time()
        func[grid_size, bs](input_device, output_device, gaussian_kernel)

        time_processed = time.time() - start
        time_processed_per_block.append(time_processed)

        blurred_image = output_device.copy_to_host()
        cv2.imwrite(f"{title}_blurred_image_gpu_{bs}.png", blurred_image)

    for bs, t in zip(block_size, time_processed_per_block):
        print(f"Block size: {bs}, time processed: {t}")

    plt.figure(figsize=(10, 5))
    plt.plot(block_size, time_processed_per_block, marker="o", linestyle="-", color="b")
    plt.title(f"Time Processed Per Block vs 2D Block Size ({title})")
    plt.xlabel("Block Size")
    plt.ylabel("Time Processed Per Block")
    x_ticks = np.arange(4, 34, 4)
    plt.xticks(x_ticks)
    y_ticks = np.arange(0, 0.22, 0.02)
    plt.yticks(y_ticks)
    plt.grid(True)
    plt.savefig(f"comparison_{title}.png")
    plt.show()


gaussian_convolution(gaussian_convolution_without_sm, title="without_sm")
gaussian_convolution(gaussian_convolution_with_sm, title="sm")
