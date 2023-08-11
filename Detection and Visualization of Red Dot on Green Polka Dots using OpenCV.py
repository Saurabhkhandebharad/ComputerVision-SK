# Draw Red Dot On Green Polka Dots : Saurabh Khandebharad

import cv2
import numpy as np

cap = cv2.VideoCapture('input_video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video_sk.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Converting the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Defining a range of green colors in HSV
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    # Mask that only includes green pixels
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    
    # Applying the mask to the original image to extract the green regions
    green_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Converting the green frame to grayscale
    gray_frame = cv2.cvtColor(green_frame, cv2.COLOR_BGR2GRAY)
    
    # Applying thresholding to the grayscale image
    thresh = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY)[1]
    
    # Searching the contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Drawing a red circle at the center of each contour
    dot_size = 5
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), dot_size, (0, 0, 255), -1)
    
    # Putting the processed frame in the output video
    out.write(frame)
    
    # Final processed frame displayed
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
