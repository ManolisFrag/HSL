import cv2
import time
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pympi
import re
from fnmatch import fnmatch
import argparse
import natsort


def video_to_frames(input_loc, output_loc):
    """
    Function to extract frames from a video.
    Specify: path to video and output path
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1
    print(video_length)
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        resized_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), resized_frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds for conversion." % (time_end-time_start) + str("\n\n"))
            break
            
    return(video_length)

def frames_to_video(dir_path, ext, output):
    """
    Function to compile frames into a video.
    Specify directory of images, extention of images ex. png, output name followed by codec ex.mp4
    """
#     dir_path = './Outputs/final/'
#     ext = 'png'
#     output = 'output_video.mp4'

    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)
    print(len(images))
    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))
    
    images = natsort.natsorted(images)
    for image in images:

        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)

        out.write(frame) # Write out frame to video

        cv2.imshow('video',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))
    