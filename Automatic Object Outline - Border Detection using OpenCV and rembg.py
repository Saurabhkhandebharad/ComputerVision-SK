# Automatic Object Outline/Border Detection :Saurabh Khandebharad

import cv2
import rembg
import os 

# Folder path 
folder_path = r"C:\Users\HP_owner\Downloads\TEST IMAGES"
# Input image filename
image_filename = "1.jpg"

# Construct the full image path
image_path = os.path.join(folder_path, image_filename)

while True:
    # Reading image
    img = cv2.imread(image_path)

    # normally, image goes outside of the screen, so to adjust it, we resize the image to standard screen size
    screen_width = 1366
    screen_height = 720

    # Resizing the image while maintaining aspect ratio
    height, width, _ = img.shape
    if width > screen_width or height > screen_height:
        scale = min(screen_width/width, screen_height/height)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    # input image to draw upon
    cv2.imshow("Input Image", img)

    # Select the ROI and press SPACE or ENTER button.
    # Cancel the selection process by pressing c button.

    # drawing a rectangle around the object of interest
    roi = cv2.selectROI("Input Image", img, fromCenter=False, showCrosshair=True)

    # Passing the ROI (Region Of Interest) to rembg to remove the background
    cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

    # Removing the background using rembg
    mask = rembg.remove(cropped)

    # Finding the contours
    contours, _ = cv2.findContours(mask[:, :, 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Outlining over the object
    outline = cv2.drawContours(cropped, contours, -1, (0, 0, 255), 2)

    # Superimposing object outline on the input image
    img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = outline
    cv2.imshow("Output Image", img)

    # Wait for user input
    key = cv2.waitKey(0) & 0xFF

    # If the user presses "q", quit the loop
    if key == ord('q'):
        break

    # If the user presses "c", clear the outline
    if key == ord('c'):
        img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = cropped
        cv2.imshow("Output Image", img)


cv2.destroyAllWindows()
