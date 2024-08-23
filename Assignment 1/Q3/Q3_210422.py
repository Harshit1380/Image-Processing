import cv2
import numpy as np

def solution(image_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass\
    image = cv2.imread(image_path)
    try:
        bng_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bng_img = cv2.bitwise_not(bng_img)
        bng_threshold_img = cv2.threshold(bng_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        bounding_box_coords = np.column_stack(np.where(bng_threshold_img > 0))
        bounding_box_angle = cv2.minAreaRect(bounding_box_coords)[-1]
        if bounding_box_angle < -45:
            bounding_box_angle = -(90 + bounding_box_angle)
        elif bounding_box_angle >= 45:
            bounding_box_angle = 90 - bounding_box_angle
        else:
            bounding_box_angle = -bounding_box_angle
        if(bounding_box_angle<0):
            bounding_box_angle+=180
        (height, width) = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, bounding_box_angle, 1)
        rotated_img = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated_img
    except:
        return image
