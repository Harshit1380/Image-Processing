import cv2
import numpy as np

# Usage
def solution(image_path):
    image= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    try:
        width, height = 600,600
        bnw_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bng_threshold_img = cv2.threshold(bnw_img, 1, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(bng_threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        largest_contour = max(contours, key=cv2.contourArea)
        ep = 0.02 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, ep, True)
        center = np.mean(approx_polygon, axis=0)[0]
        sorted_poly = sorted(approx_polygon, key=lambda point: (np.arctan2(point[0][1]-center[1], point[0][0]-center[0])))
        source_points = np.array(sorted_poly, dtype=np.float32)
        dest_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        if  source_points.shape[0] != dest_points.shape[0] or source_points.shape[2] != dest_points.shape[1]:    
            return image
        perspective_matrix = cv2.getPerspectiveTransform(source_points.astype(np.float32), dest_points)
        corrected_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
        ######################################################################

        return corrected_image
    except:
        return image
