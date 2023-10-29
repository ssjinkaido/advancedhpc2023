from numba import cuda
import numba
import numpy as np
import cv2
import time
import math
import argparse


def extract_window(img, top, height, left, width):
    return img[top : top + height, left : left + width]


def kuwahara_cpu(image_rgb, image_hsv, small_window_height, small_window_width):
    image_height = image_rgb.shape[0]
    image_width = image_rgb.shape[1]
    image_output = np.zeros(
        (
            image_height - (small_window_height - 1) * 2,
            image_width - (small_window_width - 1) * 2,
            3,
        )
    )
    image_v = image_hsv[:, :, 2]
    image_r = image_rgb[:, :, 0]
    image_g = image_rgb[:, :, 1]
    image_b = image_rgb[:, :, 2]
    for h in range(small_window_height - 1, image_height - small_window_height + 1):
        for w in range(small_window_width - 1, image_width - small_window_height + 1):
            window_coordinates = [
                [
                    h - small_window_height + 1,
                    small_window_height,
                    w - small_window_width + 1,
                    small_window_width,
                ],
                [
                    h - small_window_height + 1,
                    small_window_height,
                    w,
                    small_window_width,
                ],
                [
                    h,
                    small_window_height,
                    w - small_window_width + 1,
                    small_window_width,
                ],
                [h, small_window_height, w, small_window_width],
            ]

            window1 = extract_window(image_v, *window_coordinates[0])
            window2 = extract_window(image_v, *window_coordinates[1])
            window3 = extract_window(image_v, *window_coordinates[2])
            window4 = extract_window(image_v, *window_coordinates[3])

            std_dev1 = np.std(window1)
            std_dev2 = np.std(window2)
            std_dev3 = np.std(window3)
            std_dev4 = np.std(window4)
            min_std = min(std_dev1, std_dev2, std_dev3, std_dev4)
            if std_dev1 == min_std:
                mean_r = np.mean(extract_window(image_r, *window_coordinates[0]))
                mean_g = np.mean(extract_window(image_g, *window_coordinates[0]))
                mean_b = np.mean(extract_window(image_b, *window_coordinates[0]))
            elif std_dev2 == min_std:
                mean_r = np.mean(extract_window(image_r, *window_coordinates[1]))
                mean_g = np.mean(extract_window(image_g, *window_coordinates[1]))
                mean_b = np.mean(extract_window(image_b, *window_coordinates[1]))
            elif std_dev3 == min_std:
                mean_r = np.mean(extract_window(image_r, *window_coordinates[2]))
                mean_g = np.mean(extract_window(image_g, *window_coordinates[2]))
                mean_b = np.mean(extract_window(image_b, *window_coordinates[2]))
            elif std_dev4 == min_std:
                mean_r = np.mean(extract_window(image_r, *window_coordinates[3]))
                mean_g = np.mean(extract_window(image_g, *window_coordinates[3]))
                mean_b = np.mean(extract_window(image_b, *window_coordinates[3]))
            image_output[
                h - small_window_height + 1, w - small_window_height + 1, 0
            ] = mean_r
            image_output[
                h - small_window_height + 1, w - small_window_height + 1, 1
            ] = mean_g
            image_output[
                h - small_window_height + 1, w - small_window_height + 1, 2
            ] = mean_b
    return image_output


def rgb_to_hsv(image):
    y, x = image.shape[0], image.shape[1]
    output = np.zeros((image.shape), np.float32)
    for i in range(y):
        for j in range(x):
            max_value = max(image[i, j, 0], image[i, j, 1], image[i, j, 2])
            min_value = min(image[i, j, 0], image[i, j, 1], image[i, j, 2])
            delta = max_value - min_value
            if delta == 0:
                h_value = 0
            elif max_value == image[i, j, 0]:
                h_value = 60 * (((image[i, j, 1] - image[i, j, 2]) / delta) % 6)
            elif max_value == image[i, j, 1]:
                h_value = 60 * (((image[i, j, 2] - image[i, j, 0]) / delta) + 2)
            elif max_value == image[i, j, 2]:
                h_value = 60 * (((image[i, j, 0] - image[i, j, 1]) / delta) + 4)

            if max_value == 0:
                s_value = 0
            else:
                s_value = delta / max_value
            v_value = max_value
            output[i, j, 0] = h_value
            output[i, j, 1] = s_value
            output[i, j, 2] = v_value
    return output


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

    output = np.zeros((height, width, 3), np.float32)
    time_processed_per_method = []

    start = time.time()
    image_hsv = rgb_to_hsv(image)

    small_window_height = math.ceil(args.window_size / 2)
    small_window_width = math.ceil(args.window_size / 2)
    start = time.time()

    image_cpu = np.uint8(
        kuwahara_cpu(image, image_hsv, small_window_height, small_window_width)
    )

    time_processed = time.time() - start
    print(f"Time processed on CPU (with sm): {time_processed}s")
    cv2.imwrite(f"kuwahara_cpu.png", image_cpu)
