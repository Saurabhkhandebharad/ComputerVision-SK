# License card Recognition and Extraction using EasyOCR :Saurabh Khandebharad

import cv2
import easyocr
import os # to navigate the code towards data

input_folder = 'Licence Inputs'
output_folder = 'Licence Outputs'

# Searching and reading input images from folder
image_files = os.listdir(input_folder)
image_files = [f for f in image_files if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
reader = easyocr.Reader(['en'])

for file_name in image_files:
    file_path = os.path.join(input_folder, file_name)
    image = cv2.imread(file_path)
    results = reader.readtext(file_path)

    if results:
        print(f"Warning, text detected in {file_name}- could be sensitive")

    # Creating a box for each text detected
    for (bbox, text, prob) in results:
        x_min, y_min = [int(val) for val in bbox[0]]
        x_max, y_max = [int(val) for val in bbox[2]]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
        
        # Creating another box for better visual over processed text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, 2)[0] 
        cv2.rectangle(image, (x_min, y_min - 20), (x_min + text_size[0] + 2, y_min - 2), (220,220,220), cv2.FILLED)     

        # Writing detected text in box
        cv2.putText(image, text, (x_min, y_min -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0),2)

    # Writing and pushing processed image to the output folder    
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path,  image)

cv2.destroyAllWindows()
