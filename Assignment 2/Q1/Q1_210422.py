import cv2
import numpy as np

def solution(image_path):
    def color_threshold_mask(img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        smoothed_img = cv2.GaussianBlur(gray_img, (9, 9), 0)
        min_color = np.array([180, 0, 0], dtype=np.uint8)
        max_color = np.array([255, 200, 100], dtype=np.uint8)
        mask = cv2.inRange(smoothed_img, min_color, max_color)
        return mask

    def biggest_contour(img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(image)
        if contours:
            outer_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(contour_img, [outer_contour], -1,(255, 255, 255), 2)
        kernel = np.ones((2, 2), np.uint8)
        contour_img = cv2.dilate(contour_img, kernel, iterations=1)
        cv2.drawContours(contour_img, [outer_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        return contour_img

    def morphological_processes(img):
        kernel = np.ones((31, 31), np.uint8)
        closing_result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        closing_result2 = cv2.morphologyEx(closing_result, cv2.MORPH_CLOSE, kernel)
        closing_result3 = cv2.morphologyEx(closing_result2, cv2.MORPH_CLOSE, kernel)
        return closing_result3

    image = cv2.imread(image_path)
    mask = color_threshold_mask(image)
    lava_initial = biggest_contour(mask)
    lava_final = morphological_processes(lava_initial)
    cv2.imshow('output', lava_final)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return lava_final
