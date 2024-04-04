# Object Detection using TensorFlow Hub

This repository contains code for performing object detection using TensorFlow and TensorFlow Hub. The code utilizes a pre-trained model from TensorFlow Hub to detect objects in images.

## Setup

1. Install the required libraries:
pip install tensorflow tensorflow_hub matplotlib pillow


2. Run the code in a Python environment that supports TensorFlow.

## Usage

1. Run the code in a Python environment to perform object detection on images.
2. The code downloads images from specified URLs, resizes them, and runs object detection using the pre-trained model.
3. Detected objects are displayed with bounding boxes and labels on the images.

## Code Structure

- `object_detection.py`: Contains the main code for object detection.


## How to Run

1. Clone the repository.
2. Run the `object_detection.py` script.
3. Specify the image URLs for object detection.

## Sample Images

![Image 1](https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg)
![Image 2](https://upload.wikimedia.org/wikipedia/commons/1/1b/The_Coleoptera_of_the_British_islands_%28Plate_125%29_%288592917784%29.jpg)
![Image 3](https://upload.wikimedia.org/wikipedia/commons/0/09/The_smaller_British_birds_%288053836633%29.jpg)

## Acknowledgements

- TensorFlow: https://www.tensorflow.org/
- TensorFlow Hub: https://tfhub.dev/
- PIL (Python Imaging Library): https://python-pillow.org/

