🚗 Classical Computer Vision & Lane Line Extractor
📌 Project Overview

This project implements a Lane Line Extractor using Python and OpenCV to detect road lane boundaries in static road images using classical computer vision techniques.

The system processes road images through HSV color filtering, Gaussian blur, Canny edge detection, and Region of Interest (ROI) masking. It then uses the Probabilistic Hough Line Transform to detect and visualize the left and right lane lines.

The objective is to understand the fundamental image processing techniques used in road perception systems.

🎯 Objectives
Load and process static road images using OpenCV.
Convert images from BGR to HSV color space.
Extract white and yellow lane markings using color filtering.
Apply Gaussian blur to reduce image noise.
Use Canny edge detection to identify lane boundaries.
Apply ROI masking to isolate the road region.
Detect and draw lane lines on the original road images.
Save intermediate processing stages and final results.
🛠️ Technologies Used
Python
OpenCV
NumPy
Classical Computer Vision
📂 Project Structure
text
lane-line-extractor/
│
├── lane_line_extractor.py
├── README.md
├── requirements.txt
├── test1.jpeg
├── test2.jpeg
├── lane_detection_1.png
├── lane_detection_2.png
└── output/

⚙️ Installation
1. Clone the repository
bash
git clone https://github.com/kotharideev-sys/lane-extractor.git
cd lane-line-extractor

2. Install the dependencies

Create a requirements.txt file containing:

text
opencv-python
numpy


Install the required libraries:

bash
pip install -r requirements.txt

▶️ How to Run

Execute the Python script:

bash
python lane_line_extractor.py


When prompted, enter the path to an image or a folder containing road images.

For example:

text
Enter image or folder path (q to quit): .


The program processes the selected images and saves the results in the output/ directory.

🔬 Computer Vision Pipeline
1. Image Loading and Preprocessing

The input road images are loaded using OpenCV and resized to a standard width of 960 pixels while maintaining their aspect ratio.

2. HSV Color Filtering

The images are converted from BGR to HSV color space.

Color masks are applied to identify potential white and yellow lane markings. An automatic brightness threshold is used to help identify white lane markings.

3. Gaussian Blur

Gaussian blur is applied to reduce noise and smooth the image before edge detection.

4. Canny Edge Detection

The Canny edge detector identifies strong intensity changes that may correspond to road lane boundaries.

5. Region of Interest (ROI) Masking

A trapezoidal mask isolates the road area in front of the camera, reducing unwanted edges from trees, mountains, and surrounding scenery.

6. Lane Line Detection

The Probabilistic Hough Line Transform identifies line segments within the ROI. The segments are classified according to their slopes and positions, and suitable segments are combined to estimate the left and right lane boundaries.

The detected lane lines are overlaid on the original image in red.

7. Visualization

The program saves individual processing stages and a combined image showing the complete pipeline.

📸 Results and Screenshots

The following screenshots demonstrate the road images used in the project and the lane line detection visualization.

1. Input Road Image — Test 1

The first image shows a highway with distant mountain scenery and visible road boundaries.

!Input Road Image 1

2. Input Road Image — Test 2

The second image shows a two-lane mountain road surrounded by forest and snow-covered mountains.

!Input Road Image 2

3. Lane Line Detection — Test 1

The red lines illustrate the estimated left and right lane boundaries on the highway image.

!Lane Detection Result 1

4. Lane Line Detection — Test 2

The red lines illustrate the estimated lane boundaries on the mountain road image.

!Lane Detection Result 2

🎛️ Adjustable Parameters

The program allows several settings to be customized interactively.

Parameter	Default Value	Description
Image width	960 pixels	Standard processing width
ROI horizon	0.60	Controls the upper boundary of the road region
White brightness threshold	Automatic	Controls white lane-marking detection
Gaussian blur kernel	5 × 5	Controls image smoothing
Canny low threshold	50	Lower edge detection threshold
Canny high threshold	150	Upper edge detection threshold

These parameters can be adjusted to improve detection under different road and lighting conditions.

⚠️ Limitations
Detection accuracy depends on the visibility of lane markings.
Shadows and reflections can affect color filtering.
Faded lane markings may not be detected correctly.
Incorrect ROI settings can exclude relevant road boundaries.
Curved roads may require more advanced lane modeling.
The system is designed for static images and is not a production-ready autonomous driving solution.
🚀 Future Improvements
Improve lane detection on curved roads.
Add perspective transformation for a bird's-eye view.
Implement video-based lane tracking.
Improve detection under varying lighting conditions.
Evaluate accuracy using labeled road images.
📚 Learning Resources
OpenCV Official Image Processing Tutorials
OpenCV Canny Edge Detection
OpenCV Hough Line Transform
freeCodeCamp OpenCV Resources
Towards Data Science
🎓 Learning Outcomes

Through this project, I gained practical experience with:

Image loading and resizing using OpenCV.
HSV color space transformations.
Color-based lane marking extraction.
Gaussian smoothing and noise reduction.
Canny edge detection.
ROI masking.
Hough Transform-based line detection.
Visualization of classical computer vision pipelines.
👨‍💻 Author

Name: Your Name

GitHub: YOUR-USERNAME

📅 Assignment Details

Assignment: Classical Computer Vision & Lane Line Extractor

Submission Date: Monday, 26 October 2026

Built using Python, OpenCV, and NumPy.
