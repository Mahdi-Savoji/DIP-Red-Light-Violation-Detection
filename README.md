# Traffic Video License Plate Detection

This project is designed to detect license plates in traffic videos and extract the license plate numbers using optical character recognition (OCR). It utilizes computer vision techniques, including image processing and text recognition, to identify and analyze license plates in real-time.
## Requirements

- Python 3.x
- OpenCV
- Numpy
- PyTesseract
- Matplotlib
- String
- Re
- Difflib
- Collections
- Pandas
- Datetime

## Installation

1. Clone the repository:

   ```shell
   git https://github.com/Mahdi-Savoji/Traffic-Red-Light-Runing-Violatino-Detection-and-Recognition-Using-DIP.git

2. Install the required dependencies:

   ```shell
    pip install opencv-python numpy pytesseract matplotlib pandas

3. Download the traffic video from the project directory and save it as trafficVideo_original.mp4.

## Usage

1. Run the license_plate_detection.py script:

   ```shell
    python license_plate_detection.py

2. The script will process the video and detect license plates in real-time.

3. The detected license plate numbers will be extracted using optical character recognition (OCR) and displayed on the screen.

4. The processed video with the license plate numbers highlighted will be saved as output_file2.avi.



License Plate Detection and Traffic Light Processing
This code snippet demonstrates the process of detecting license plates and processing traffic lights in a video stream or frame. It performs the following steps:

1. Draws circles on the frame to represent the positions of the traffic lights and the currently active light.
2. Displays a timer indicating the remaining time for the current light.
3. Applies denoising techniques such as Gaussian and median blur to reduce noise in the frame.
4. Selects a region of interest (ROI) and converts it to grayscale.
5. Performs Canny edge detection on the ROI to detect license plate edges.
6. Filters the detected contours based on area and aspect ratio to identify rectangular license plate shapes.
7. Draws a stop line on the frame, considering camera movement.
8. Uses a cascade classifier to detect license plates in the frame.
9. Filters the detected plates based on size and location to remove false positives.
10. Extracts the license plate region and applies image preprocessing.
11. Performs optical character recognition (OCR) using PyTesseract to extract the text from the license plate.
12. Cleans up the extracted text by removing special characters and unnecessary characters.
13. Stores valid license plates along with associated information in a dictionary.
14. Annotates the frame with bounding boxes around the detected license plates.
15. Writes the annotated frame to a video file and displays it.
16. Continues the process until the user presses the Escape key.
17. Performs additional filtering on the detected license plates to remove incorrect plates.
18. Creates a Pandas DataFrame with the final results.
19. Prints the DataFrame and saves it as a CSV file named "Cars.csv".

## Example Output

This section showcases the original and output GIFs to demonstrate the license plate detection and processing.

- The Input GIF below demonstrates the input video:

   <img src="Test/Original.gif" alt="Input" width="400" height="500">

- The output GIF below showcases the processed video with highlighted license plate numbers:

   <img src="Test/Output.gif" alt="Output" width="400" height="500">

This section displays the extracted license plate numbers along with associated information.

| License Plate Number | Penalty |    DateTime    | taxi|
|----------------------|---------|----------------|-----|
|        KW527         |    0    | 9/1/2023 12:38 |FALSE|
|        8525          |    0    | 9/1/2023 12:38 |TRUE |
|        3867          |    0    | 9/1/2023 12:38 |TRUE |
|        YB6433        |    1    | 9/1/2023 12:39 |FALSE|
|        NN773         |    1    | 9/1/2023 12:39 |FALSE|
