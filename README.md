# Handshape Library
A python library to analyze and annotate handshape clusters. The tool receives as inputs the extracted json files from [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and lets the user run k means on the normalized and scaled hand configurations.

## Getting started

### Prerequisites
You need to install the following python libraries in order to use the library
'''
os,
PIL,
cv2,
natsort,
pandas,
matplotlib,
sklearn,
numpy,
bokeh,
moviepy
'''

## Running your own data
To run the framework with your own data, you simply need to run [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) in the video you want to analyze. Get the json files and the output video and place them in a folder along with this library. See Example.ipynb for the supported functions. If you wish run:

'''
from HSL import Annotations
Annotations.annotate("json_directory/", "frame_directory/").video
'''
to get everything with the predifined values
