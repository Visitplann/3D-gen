"""Simple image cleanup used before reconstruction."""

import os

import cv2
import numpy as np


DEBUG_VISUALS = os.environ.get("PIPELINE_VISUAL_DEBUG") == "1"


def prepare_image_for_reconstruction(image_bgr):
    """Return a clean RGB image and a grayscale copy of the same object."""
    show_debug_image("Original", image_bgr)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.bilateralFilter(image_rgb, d=6, sigmaColor=10, sigmaSpace=20)
    clean_rgb = image_rgb.copy()
    show_debug_image("Clean", clean_rgb)

    image_rgb = cv2.convertScaleAbs(image_rgb, alpha=1.2, beta=10)
    show_debug_image("Contrast", image_rgb)

    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    show_debug_image("Gray", gray_image)
    return gray_image, clean_rgb


def convert_height_map_to_normal_map(gray_image, strength=2.0, invert_y=True):
    """Create a simple normal map from a grayscale image."""
    gray_image = gray_image.astype(np.float32) / 255.0

    gradient_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    if invert_y:
        gradient_y = -gradient_y

    normal_x = -gradient_x * strength
    normal_y = -gradient_y * strength
    normal_z = np.ones_like(gray_image)

    normal_length = np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
    normal_x /= normal_length
    normal_y /= normal_length
    normal_z /= normal_length

    normal_map = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
    normal_map[:, :, 0] = ((normal_x + 1) * 0.5 * 255).astype(np.uint8)
    normal_map[:, :, 1] = ((normal_y + 1) * 0.5 * 255).astype(np.uint8)
    normal_map[:, :, 2] = ((normal_z + 1) * 0.5 * 255).astype(np.uint8)
    return normal_map


def show_debug_image(name, image):
    if not DEBUG_VISUALS:
        return

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 800, 600)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Backwards-compatible names used by older code.
preprocess_image = prepare_image_for_reconstruction
height_map_to_normal_map = convert_height_map_to_normal_map
