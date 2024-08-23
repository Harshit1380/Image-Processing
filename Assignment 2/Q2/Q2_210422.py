import cv2
import numpy as np

def solution(image_path_a, image_path_b):

    def decouple_intensity_color(image):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        intensity_channel = lab_image[:, :, 0]
        color_image = image / intensity_channel[:, :, np.newaxis]
        color_image *= 255
        return intensity_channel, color_image

    def combine_color_intensity(color_image, intensity_image):
        normalized_intensity = intensity_image.astype(float) / 255.0
        normalized_color_image = color_image * normalized_intensity[:, :, np.newaxis]
        normalized_color_image = np.clip(normalized_color_image, 0, 255).astype(np.uint8)
        return normalized_color_image

    def gaussian(x, sigma):
        return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-x ** 2 / (2 * sigma ** 2))

    def joint_bilateral_filter(image, image2, diameter, sigma_color, sigma_space):
        height, width, _ = image.shape
        result = np.zeros_like(image, dtype=np.float64)
        for i in range(height):
            for j in range(width):
                intensity = image2[i, j]
                x_coords, y_coords = np.meshgrid(
                    np.arange(max(0, i - diameter), min(height, i + diameter + 1)),
                    np.arange(max(0, j - diameter), min(width, j + diameter + 1))
                )
                spatial_distance = np.sqrt((i - x_coords) ** 2 + (j - y_coords) ** 2)
                intensity_difference = intensity - image2[x_coords, y_coords]
                spatial_weight = gaussian(spatial_distance, sigma_space)
                intensity_weight = gaussian(intensity_difference, sigma_color)
                weight = spatial_weight * intensity_weight
                weighted_sum = np.sum(image[x_coords, y_coords] * weight[:, :, np.newaxis], axis=(0, 1))
                total_weight = np.sum(weight, axis=(0, 1))
                result[i, j] = weighted_sum / total_weight
        return result.astype(np.uint8)

    no_flash = cv2.imread(image_path_a)
    flash = cv2.imread(image_path_b)
    sigma_s = 7
    sigma_r = 8
    diameter = 15
    flash_intensity, _ = decouple_intensity_color(flash)
    no_flash_intensity, _ = decouple_intensity_color(no_flash)
    flash_intensity = flash_intensity * (np.sum(no_flash_intensity) / np.sum(flash_intensity))
    smoothed = joint_bilateral_filter(no_flash, flash_intensity, diameter, sigma_r, sigma_s)
    smoothed_intensity, smoothed_color = decouple_intensity_color(smoothed)
    combined = combine_color_intensity(smoothed_color, flash_intensity)
    intensity_difference = smoothed_intensity - flash_intensity
    _, binary_intensity_difference = cv2.threshold(intensity_difference, 1, 255, cv2.THRESH_BINARY)
    binary_intensity_difference_inv = (cv2.bitwise_not(binary_intensity_difference.astype(np.uint8))) / 255
    smoothed_mask = smoothed * binary_intensity_difference_inv.astype(np.uint8)[..., None]
    masked_combined = combined * (binary_intensity_difference / 255).astype(np.uint8)[..., None]
    final_blended_image = cv2.addWeighted(masked_combined, 1.3, smoothed_mask, 1, 0)
    return final_blended_image
