from numba import cuda, float32
import numba
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


@cuda.jit
def find_max(d_input, d_maximum):
    gtid = cuda.grid(1)
    ltid = cuda.threadIdx.x
    bdim = cuda.blockDim.x

    shared = cuda.shared.array(shape=256, dtype=d_input.dtype)
    if gtid < len(d_input):
        shared[ltid] = d_input[gtid]
    cuda.syncthreads()
    stride = bdim // 2
    while stride > 0:
        if ltid < stride and shared[ltid] < shared[ltid + stride]:
            shared[ltid] = shared[ltid + stride]
        cuda.syncthreads()
        stride //= 2
    if ltid == 0:
        d_maximum[cuda.blockIdx.x] = shared[0]


@cuda.jit
def find_min(d_input, d_maximum):
    gtid = cuda.grid(1)
    ltid = cuda.threadIdx.x
    bdim = cuda.blockDim.x

    shared = cuda.shared.array(shape=256, dtype=d_input.dtype)
    if gtid < len(d_input):
        shared[ltid] = d_input[gtid]
    cuda.syncthreads()
    stride = bdim // 2
    while stride > 0:
        if ltid < stride and shared[ltid] > shared[ltid + stride]:
            shared[ltid] = shared[ltid + stride]
        cuda.syncthreads()
        stride //= 2
    if ltid == 0:
        d_maximum[cuda.blockIdx.x] = shared[0]


@cuda.jit
def recalculate_intensity(src, min_value, max_value):
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    shared_memory = cuda.shared.array((16, 16), dtype=np.float32)
    if y < src.shape[0] and x < src.shape[1]:
        shared_memory[ty, tx] = src[y, x]
    cuda.syncthreads()
    if y < src.shape[0] and x < src.shape[1]:
        value_recalculated = (
            (shared_memory[ty, tx] - min_value) / (max_value - min_value) * 255
        )
        src[y, x] = value_recalculated


block_size = [(16, 16)]
time_processed_per_block = []

for bs in block_size:
    grid_size_x = math.ceil(width / bs[0])
    grid_size_y = math.ceil(height / bs[1])
    grid_size = (grid_size_x, grid_size_y)
    start = time.time()
    grayscale_gpu[grid_size, bs](input_device, output_device)
    output_host = output_device.copy_to_host()
    cv2.imwrite(f"gray_shared.png", output_host)
    d_maximum = cuda.device_array(shape=1, dtype=np.float32)
    d_minimum = cuda.device_array(shape=1, dtype=np.float32)
    buffer = output_host.flatten()
    nb_threads = 256

    while buffer.size > nb_threads:
        nb_blocks = (buffer.size + nb_threads - 1) // nb_threads

        temp = cuda.device_array(shape=nb_blocks, dtype=buffer.dtype)

        find_max[nb_blocks, nb_threads](buffer, temp)

        cuda.synchronize()

        buffer = temp
    find_max[1, buffer.size](buffer, d_maximum)
    max_value = d_maximum.copy_to_host()[0]

    buffer = output_host.flatten()

    while buffer.size > nb_threads:
        nb_blocks = (buffer.size + nb_threads - 1) // nb_threads

        temp = cuda.device_array(shape=nb_blocks, dtype=buffer.dtype)

        find_min[nb_blocks, nb_threads](buffer, temp)

        cuda.synchronize()

        buffer = temp

    find_min[1, buffer.size](buffer, d_minimum)
    min_value = d_minimum.copy_to_host()[0]

    output_grayscale = cuda.to_device(output_host)
    recalculate_intensity[grid_size, bs](output_grayscale, min_value, max_value)
    time_processed = time.time() - start
    time_processed_per_block.append(time_processed)

    output_host = output_device.copy_to_host()
    cv2.imwrite(f"gray_stretch_image.png", output_host)
