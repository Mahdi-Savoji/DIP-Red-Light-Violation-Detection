import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import time
import string
import re
from difflib import SequenceMatcher
from collections import Counter
import pandas as pd
from datetime import datetime


lisenceplate = []
temp_list = []


# Get Video From "Github Repository"
cap = cv2.VideoCapture("trafficVideo_original.mp4")
# Get the original video's frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the video codec and other parameters
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # FourCC code for MP4 video format
fps = 30.0  # Frames per second

# Create a VideoWriter object with the original frame size
video_writer = cv2.VideoWriter('output_file2.avi', fourcc, fps, (640, 480))

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
x, y, width, height = cv2.getWindowImageRect('Video')


num_LisencePlate = 0
change_camera_h = 0
change_camera_w = 0
x_main = 0
y_main = 0
w_main = 0
h_main = 0
duration = 9

# Define the traffic light parameters
light_radius = 50
light_spacing = 100
light_center = (1800, 200)
timer_duration = 7  # in seconds

# Define the colors for each traffic light state
red_color = (0, 0, 255)
yellow_color = (0, 255, 255)
green_color = (0, 255, 0)
gray_color = (100,100,100)

# Calculate the positions of the traffic lights
red_light_pos = (light_center[0], light_center[1])
yellow_light_pos = (light_center[0], light_center[1] + light_spacing)
green_light_pos = (light_center[0], light_center[1] + 2 * light_spacing)

# Get the current time
start_time = time.time()
predicted_result = {}

timer = 0
custom_config = r'--oem 3 --psm 9'


# Find Most Repeated Item in List
def find_most_repeated_item(lst):
    item_counts = Counter(lst)
    max_count = max(item_counts.values())
    most_repeated_items = [item for item, count in item_counts.items() if count == max_count]

    return most_repeated_items

# Find Similarity Between Two Item
def is_similar_to_other_items(string, item, similarity_threshold=0.7):
    similarity_ratio = SequenceMatcher(None, string, item).ratio()
    if similarity_ratio >= similarity_threshold:
        return True
    return False

# Remove Character after Number For Cleaning Data LisencePlate
def remove_chars_after_digits(string):
    match = re.search(r'\d+', string)
    
    if match:
        first_group = match.group()
        string = string[:match.start() + len(first_group)]

    return string


def clearing_plate_detection(lst):
    for main_index, i in enumerate(lst):
        if i != 'none':
            temp = []
            check = True
        
            for index, j in enumerate(lst[main_index+1:]):
                if is_similar_to_other_items(i, j):

                    lst[index + main_index + 1] = 'none'
                    temp.append(j)
                    check = False
            if check:
                lisenceplate.append(i)
            else:
                lst[main_index] = 'none'
                temp.append(i)
                lisenceplate.extend(find_most_repeated_item(temp))
    return(lisenceplate)




while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get the current position (in milliseconds) in the video
    current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

    # Convert the time to seconds and minutes
    current_time_seconds = current_time_ms / 1000
    
    # Traffic light
    if  2 * duration > current_time_seconds >= duration:
        current_state = 0
        current_color = red_color
        light_on_pos = red_light_pos

        color_red = red_color
        color_yellow = gray_color
        color_green = gray_color

        timer = duration * 2 - current_time_seconds

    elif  duration > current_time_seconds >= 0 :
        current_state = 1
        current_color = yellow_color
        light_on_pos = yellow_light_pos

        color_yellow = red_color
        color_red = gray_color
        color_green = gray_color
        
        timer = duration - current_time_seconds

    elif current_time_seconds >= 2 * duration :
        current_state = 2
        current_color = green_color
        light_on_pos = green_light_pos

        color_green = red_color
        color_yellow = gray_color
        color_red = gray_color

        timer = 3 * duration - current_time_seconds

        

    # Draw the traffic lights on the frame
    cv2.circle(frame, red_light_pos, light_radius, color_red, -1)
    cv2.circle(frame, yellow_light_pos, light_radius, color_yellow, -1)
    cv2.circle(frame, green_light_pos, light_radius, color_green, -1)

    # Draw the current light on
    cv2.circle(frame, light_on_pos, light_radius, current_color, -1)

    # Draw the timer
    timer_text = "Timer: {:.1f}s".format(max(0, timer))
    cv2.putText(frame, timer_text, (light_center[0] - 220, light_center[1] -  light_spacing),
                cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2, cv2.LINE_AA)
    

    # Denoising 
    denoised_frame = cv2.GaussianBlur(frame, (5,5), 0)
    denoised_frame = cv2.medianBlur(denoised_frame, 3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_x1, roi_y1 = 200, 250
    roi_x2, roi_y2 = 450, 450
    img = gray[roi_y1:roi_y2, roi_x1:roi_x2]

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours of edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and aspect ratio to get rectangular shapes
    filtered_contours = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(contour)
        if len(approx) == 4 and area > 600:
            filtered_contours.append(approx)
            x, y, w, h = cv2.boundingRect(contour)
            if x_main == 0:
                x_main, y_main, w_main, h_main = x, y, w, h
                x_first = x
                y_first = y
            else:
                if abs((y_main + h_main /2) - (y + h /2)) < 20:
                    change_camera_h = y_first - y
                    change_camera_w = x_first - x
                    x_main, y_main, w_main, h_main = x, y, w, h

    if current_state == 0:
        line_color = (0, 0, 255)
    else:
        line_color = (0, 255, 0)

    line_thickness = 3

    # Draw the Stop line with Consistnet with Camera Movement
    cv2.line(denoised_frame, (600 - change_camera_w , 930 - change_camera_h), (1920 - change_camera_w, 980 -  change_camera_h), line_color , 3)
    
    
    if True:
        cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
        scale_factor = 1.05
        min_neighbors = 3
        min_plate_size = (30, 30)
        max_plate_size = (200, 200)

        stop_region = denoised_frame[930 - change_camera_h:, :]

        plates = cascade.detectMultiScale(denoised_frame, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        filtered_plates = []
        for plate in plates:
            x, y, w, h = plate
            plate_center = (x + w // 2, y + h // 2)
            aspect_ratio = w / h
            plate_area = w *h
            if 2 < aspect_ratio < 5 and y > 930 - change_camera_h and plate_area > 4000:
                filtered_plates.append((x,y,w,h))


        for i, (x,y,w,h) in enumerate(filtered_plates):
            plate_img = denoised_frame[y:y+h, x:x+w]


            plate_img_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
          
            contours, _ = cv2.findContours(plate_img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            license_plate_contour = max(contours, key=cv2.contourArea)
            x_plate, y_plate, w_plate, h_plate = cv2.boundingRect(license_plate_contour)
            license_plate = plate_img[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]

            # Image to Text
            predicted = pytesseract.image_to_string(license_plate, lang="eng", config=custom_config)

            # Preprocess on Text of LisencePlate For Cleaning Data 
            special_characters = r"¥£\|-_=+/[]{}()!@#$%^&*(\\:;\'\"?><, "
            translation_table = str.maketrans("", "", special_characters)
            predicted = predicted.translate(translation_table)
            predicted = predicted.replace('\n', '')
            predicted = remove_chars_after_digits(predicted)

            if re.search(r'\d{3}', predicted) is not None and len(re.findall(r'\d', predicted)) < 5 and len(re.findall(r'[A-Za-z]', predicted)) == 2:
                if current_state == 0:
                    predicted_result[predicted] = [1, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), False]
                else:
                    predicted_result[predicted] = [0, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), False]
                num_LisencePlate += 1       
                cv2.rectangle(denoised_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite(f'LisencePlate_image/{num_LisencePlate}.png', plate_img)

            elif re.search(r'\d{3}', predicted) is not None and re.search(r'[A-Za-z]', predicted) is None:
                middle_index = len(predicted) // 2
                predicted = predicted[middle_index-2:middle_index+2]
                if current_state == 0:
                    predicted_result[predicted] = [1, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True]
                else:
                    predicted_result[predicted] = [0, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True]
                num_LisencePlate += 1  
                cv2.rectangle(denoised_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite(f'LisencePlate_image/{num_LisencePlate}.png', plate_img)
    
    video_writer.write(denoised_frame)
    cv2.imshow('Video', denoised_frame)
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()


Fined_Cars = clearing_plate_detection(list(predicted_result.keys()))


# Remove Incorrect LisencePlate 
Filterd_Cars = []
for i in Fined_Cars:
    check = True
    number = re.sub(r'[A-Za-z]', '', i)
    for j in Fined_Cars:
        if j != i:
            if number == j:
                check = False
    if check:
        Filterd_Cars.append(i)

selected_dict = {key: predicted_result[key] for key in Filterd_Cars}

# Create DataFrame with Final Result
df = pd.DataFrame.from_dict(selected_dict, orient='index', columns=['penalty', 'datetime', 'taxi'])
df.reset_index(inplace=True)
df.columns = ['licensePlate', 'penlaty', 'datetime', 'taxi']
print(df)
df.to_csv("Cars.csv", index=True)