import cv2
import numpy as np

# Open both camera devices
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

# Stereo block matching setup
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        break

    # Convert to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Compute disparity (depth map)
    disparity = stereo.compute(gray_left, gray_right)
    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    cv2.imshow('Left', gray_left)
    cv2.imshow('Right', gray_right)
    cv2.imshow('Depth Map', disp_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
