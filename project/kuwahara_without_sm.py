from numba import cuda
import numba
import numpy as np
import cv2
import time
import math
import argparse


@cuda.jit
def kuwahara_gpu_without_sm(
    src_rgb, src_hsv, dst, small_window_height, small_window_width
):
    x, y = cuda.grid(2)
    image_v = src_hsv
    if (
        x < dst.shape[1] - small_window_width + 1
        and y < dst.shape[0] - small_window_height + 1
        and x > small_window_width - 2
        and y > small_window_height - 2
    ):
        tops = cuda.local.array(4, numba.int64)
        heights = cuda.local.array(4, numba.int64)
        lefts = cuda.local.array(4, numba.int64)
        widths = cuda.local.array(4, numba.int64)

        tops[0] = y - small_window_height + 1
        tops[1] = y - small_window_height + 1
        tops[2] = y
        tops[3] = y

        heights[0] = small_window_height
        heights[1] = small_window_height
        heights[2] = small_window_height
        heights[3] = small_window_height

        lefts[0] = x - small_window_width + 1
        lefts[1] = x
        lefts[2] = x - small_window_width + 1
        lefts[3] = x

        widths[0] = small_window_width
        widths[1] = small_window_width
        widths[2] = small_window_width
        widths[3] = small_window_width

        smallest_std_window = np.inf
        smallest_window_idx = -1
        for window in range(4):
            total_sum_window = 0
            top = tops[window]
            left = lefts[window]
            height = heights[window]
            width = widths[window]
            for i in range(top, top + height):
                for j in range(left, left + width):
                    total_sum_window += image_v[i, j]

            mean_window = total_sum_window / (width * height)
            sum_of_squared_diff_window = 0

            for i in range(top, top + height):
                for j in range(left, left + width):
                    diff = image_v[i, j] - mean_window
                    sum_of_squared_diff_window += diff * diff

            std_window = math.sqrt(sum_of_squared_diff_window / (width * height))
            if std_window < smallest_std_window:
                smallest_std_window = std_window
                smallest_window_idx = window

        top = tops[smallest_window_idx]
        left = lefts[smallest_window_idx]
        height = heights[smallest_window_idx]
        width = widths[smallest_window_idx]
        total_sum_window_r = 0
        total_sum_window_g = 0
        total_sum_window_b = 0

        for i in range(top, top + height):
            for j in range(left, left + width):
                total_sum_window_r += src_rgb[i, j, 0]
                total_sum_window_g += src_rgb[i, j, 1]
                total_sum_window_b += src_rgb[i, j, 2]

        dst[y, x, 0] = total_sum_window_r / (width * height)
        dst[y, x, 1] = total_sum_window_g / (width * height)
        dst[y, x, 2] = total_sum_window_b / (width * height)


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

    h_pad = args.window_size // 2
    w_pad = args.window_size // 2
    image = np.pad(
        image,
        pad_width=(
            (h_pad, h_pad),
            (w_pad, w_pad),
            (0, 0),
        ),
        mode="constant",
        constant_values=0,
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
    start = time.time()
    rgb_to_hsv[grid_size, block_size](input_rgb_device, output_hsv_device)

    image_hsv = output_hsv_device.copy_to_host()[:, :, 2]
    input_hsv_device = output_hsv_device[:, :, 2]

    small_window_height = math.ceil(args.window_size / 2)
    small_window_width = math.ceil(args.window_size / 2)

    start = time.time()
    kuwahara_gpu_without_sm[grid_size, block_size](
        input_rgb_device,
        input_hsv_device,
        output_without_sm_device,
        small_window_height,
        small_window_width,
    )
    image_gpu1 = np.uint8(output_without_sm_device.copy_to_host())
    image_gpu1 = image_gpu1[
        small_window_height - 1 : height - small_window_height + 1,
        small_window_width - 1 : width - small_window_width + 1,
        :,
    ]
    cv2.imwrite(f"kuwahara_gpu_without_sm_{args.window_size}.png", image_gpu1)
    time_processed = time.time() - start
    print(f"Time processed on GPU (without sm): {time_processed}s")
