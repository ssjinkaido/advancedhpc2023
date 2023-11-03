from numba import cuda
import numba
import numpy as np
import cv2
import time
import math
import argparse


@cuda.jit
def rgb_to_hsv(src, dst):
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    shared_hsv = cuda.shared.array(shape=(8, 8, 3), dtype=numba.float32)

    if y < dst.shape[0] and x < dst.shape[1]:
        for i in range(3):
            shared_hsv[ty, tx, i] = src[y, x, i]
    cuda.syncthreads()
    if ty < 8 and tx < 8:
        max_value = max(
            shared_hsv[ty, tx, 0], shared_hsv[ty, tx, 1], shared_hsv[ty, tx, 2]
        )
        min_value = min(
            shared_hsv[ty, tx, 0], shared_hsv[ty, tx, 1], shared_hsv[ty, tx, 2]
        )
        delta = max_value - min_value
        if delta == 0:
            h_value = 0
        elif max_value == shared_hsv[ty, tx, 0]:
            h_value = 60 * (
                ((shared_hsv[ty, tx, 1] - shared_hsv[ty, tx, 2]) / delta) % 6
            )
        elif max_value == shared_hsv[ty, tx, 1]:
            h_value = 60 * (
                ((shared_hsv[ty, tx, 2] - shared_hsv[ty, tx, 0]) / delta) + 2
            )
        elif max_value == shared_hsv[ty, tx, 2]:
            h_value = 60 * (
                ((shared_hsv[ty, tx, 0] - shared_hsv[ty, tx, 1]) / delta) + 4
            )

        if max_value == 0:
            s_value = 0
        else:
            s_value = delta / max_value

        v_value = max_value
        dst[y, x, 0] = h_value
        dst[y, x, 1] = s_value
        dst[y, x, 2] = v_value
    cuda.syncthreads()


@cuda.jit
def kuwahara_gpu_with_sm(
    src_hsv, src_rgb, dst, small_window_height, small_window_width
):
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # i set 26 because window_size = 9-> 8+8+8 = 24<26, if you want larger window size, you have to increase the shared_array shape)
    shared_hsv = cuda.shared.array(shape=(26, 26, 1), dtype=numba.float32)
    shared_rgb = cuda.shared.array(shape=(26, 26, 3), dtype=numba.float32)
    pad = small_window_width - 1
    if x < src_hsv.shape[1] and y < src_hsv.shape[0]:
        shared_hsv[ty, tx] = src_hsv[
            y - pad,
            x - pad,
        ]
        shared_hsv[ty + pad * 2, tx] = src_hsv[
            y + pad,
            x - pad,
        ]
        shared_hsv[ty, tx + pad * 2] = src_hsv[
            y - pad,
            x + pad,
        ]
        shared_hsv[ty + pad * 2, tx + pad * 2] = src_hsv[
            y + pad,
            x + pad,
        ]
    cuda.syncthreads()
    if x < src_rgb.shape[1] and y < src_rgb.shape[0]:
        for i in range(3):
            shared_rgb[ty, tx, i] = src_rgb[y - pad, x - pad, i]
            shared_rgb[ty + pad * 2, tx, i] = src_rgb[y + pad, x - pad, i]
            shared_rgb[ty, tx + pad * 2, i] = src_rgb[y - pad, x + pad, i]
            shared_rgb[ty + pad * 2, tx + pad * 2, i] = src_rgb[y + pad, x + pad, i]

    cuda.syncthreads()
    if x < src_hsv.shape[1] and y < src_hsv.shape[0]:
        tops = cuda.local.array(4, numba.int64)
        heights = cuda.local.array(4, numba.int64)
        lefts = cuda.local.array(4, numba.int64)
        widths = cuda.local.array(4, numba.int64)

        tops[0] = ty
        tops[1] = ty
        tops[2] = ty + small_window_height - 1
        tops[3] = ty + small_window_height - 1

        heights[0] = small_window_height
        heights[1] = small_window_height
        heights[2] = small_window_height
        heights[3] = small_window_height

        lefts[0] = tx
        lefts[1] = tx + small_window_width - 1
        lefts[2] = tx
        lefts[3] = tx + small_window_width - 1

        widths[0] = small_window_width
        widths[1] = small_window_width
        widths[2] = small_window_width
        widths[3] = small_window_width

        smallest_std_window = np.inf
        smallest_window_idx = -1

        for window in range(4):
            total_sum_window = 0.0
            top = tops[window]
            left = lefts[window]
            height = heights[window]
            width = widths[window]
            for i in range(top, top + height):
                for j in range(left, left + width):
                    total_sum_window += shared_hsv[i, j, 0]
            mean_window = total_sum_window / (height * width)
            sum_of_squared_diff_window = 0.0

            for i in range(top, top + height):
                for j in range(left, left + width):
                    diff = shared_hsv[i, j, 0] - mean_window
                    sum_of_squared_diff_window += diff * diff

            std_window = math.sqrt(sum_of_squared_diff_window / (height * width))
            if std_window < smallest_std_window:
                smallest_std_window = std_window
                smallest_window_idx = window

        top = tops[smallest_window_idx]
        left = lefts[smallest_window_idx]
        height = heights[smallest_window_idx]
        width = widths[smallest_window_idx]
        total_sum_window_r = 0.0
        total_sum_window_g = 0.0
        total_sum_window_b = 0.0

        for i in range(top, top + height):
            for j in range(left, left + width):
                total_sum_window_r += shared_rgb[i, j, 0]
                total_sum_window_g += shared_rgb[i, j, 1]
                total_sum_window_b += shared_rgb[i, j, 2]
        dst[y, x, 0] = total_sum_window_r / (height * width)
        dst[y, x, 1] = total_sum_window_g / (height * width)
        dst[y, x, 2] = total_sum_window_b / (height * width)
    cuda.syncthreads()


def pad_image_to_divisible_by_8(image, n):
    padded_height = image.shape[0] + 2 * n
    padded_width = image.shape[1] + 2 * n

    left_pad = top_pad = n
    right_pad = bottom_pad = n

    if padded_width % 8 != 0:
        additional_width = 8 - (padded_width % 8)
        left_pad += additional_width // 2
        right_pad += additional_width - (additional_width // 2)

    if padded_height % 8 != 0:
        additional_height = 8 - (padded_height % 8)
        top_pad += additional_height // 2
        bottom_pad += additional_height - (additional_height // 2)

    padded_image = np.pad(
        image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode="constant"
    )

    return padded_image, left_pad, right_pad, top_pad, bottom_pad


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labwork")
    parser.add_argument(
        "--window_size",
        "-w",
        type=int,
        default=7,
        help="Window size (must be odd number)",
    )
    args = parser.parse_args()
    if args.window_size % 2 != 1:
        print("Window size must be odd number")
        raise ValueError("Invalid window size provided!")

    image = cv2.imread("../images/scene.jpg")

    # pad image
    h_pad = args.window_size // 2
    w_pad = args.window_size // 2
    image, left_pad, right_pad, top_pad, bottom_pad = pad_image_to_divisible_by_8(
        image, h_pad
    )
    height, width, _ = image.shape
    image = np.float32(image)

    input_rgb_device = cuda.to_device(image)
    output = np.zeros((height, width, 3), np.float32)

    output_hsv_device = cuda.to_device(output)
    output_sm_device = cuda.to_device(output)
    output_without_sm_device = cuda.to_device(output)

    block_size = (8, 8)

    grid_size_x = math.ceil(width / block_size[0])
    grid_size_y = math.ceil(height / block_size[1])
    grid_size = (grid_size_x, grid_size_y)

    rgb_to_hsv[grid_size, block_size](input_rgb_device, output_hsv_device)

    image_hsv = output_hsv_device.copy_to_host()[:, :, 2]
    input_hsv_device = output_hsv_device[:, :, 2]

    small_window_height = math.ceil(args.window_size / 2)
    small_window_width = math.ceil(args.window_size / 2)
    height, width = image_hsv.shape

    start = time.time()
    kuwahara_gpu_with_sm[grid_size, block_size](
        input_hsv_device,
        input_rgb_device,
        output_sm_device,
        small_window_height,
        small_window_width,
    )
    image_gpu_sm = np.uint8(output_sm_device.copy_to_host())
    image_gpu_sm = image_gpu_sm[
        top_pad : height - bottom_pad,
        left_pad : width - right_pad,
        :,
    ]
    cv2.imwrite(f"kuwahara_gpu_with_sm_{args.window_size}.png", image_gpu_sm)
    time_processed = time.time() - start
    print(f"Time processed on GPU (with sm): {time_processed}s")
